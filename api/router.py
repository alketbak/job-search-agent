"""FastAPI router — wires all REST endpoints and the dashboard.

We keep the router small and delegate real work to agents/ and memory/.
Every endpoint returns typed Pydantic responses for OpenAPI clarity.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import desc, func, select

from agents.orchestrator import run_pipeline
from api.schemas import (
    ApplicationResponse,
    GraphResponse,
    HealthResponse,
    JobListResponse,
    JobResponse,
    JobStatusUpdate,
    PipelineRunResponse,
    RunListResponse,
    RunRequest,
    RunResponse,
    SkillGapResponse,
)
from database import Application, Job, JobStatus, SkillGap, get_session
from database.models import PipelineRun
from memory.knowledge_graph import get_full_graph, store_application

log = logging.getLogger(__name__)

router = APIRouter()

# Resolve the dashboard HTML once — relative to this file so it works whether
# the server is launched from the repo root or inside Docker.
_DASHBOARD_HTML = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    return HealthResponse()


@router.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    """Serve the D3 dashboard at the root URL."""
    if not _DASHBOARD_HTML.exists():
        raise HTTPException(status_code=500, detail="dashboard/index.html is missing")
    return FileResponse(_DASHBOARD_HTML, media_type="text/html")


@router.post("/run", response_model=RunResponse, tags=["pipeline"])
async def trigger_run(payload: RunRequest, background: BackgroundTasks) -> RunResponse:
    """Run the pipeline synchronously and return a summary.

    We intentionally block the HTTP request on the pipeline run (rather than
    kicking it off in the background) so the caller gets real counts back.
    If you prefer fire-and-forget, uncomment the background-task branch.
    """
    final = await run_pipeline(query=payload.query, location=payload.location)
    return RunResponse(
        run_id=final.get("run_id", ""),
        jobs_found=len(final.get("raw_jobs") or []),
        jobs_analyzed=len(final.get("analyzed_jobs") or []),
        digest_path=(final.get("delivery_result") or {}).get("path"),
        errors=final.get("errors") or [],
    )


@router.get("/jobs", response_model=JobListResponse, tags=["jobs"])
async def list_jobs(
    status: str | None = Query(default=None, description="Filter by status"),
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> JobListResponse:
    async with get_session() as session:
        stmt = select(Job).order_by(desc(Job.found_at))
        if status:
            try:
                stmt = stmt.where(Job.status == JobStatus(status))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await session.execute(count_stmt)).scalar_one()
        rows = (await session.execute(stmt.offset(offset).limit(limit))).scalars().all()
    jobs = [JobResponse(**j.to_dict()) for j in rows]
    return JobListResponse(total=int(total), jobs=jobs)


@router.get("/jobs/{job_id}", response_model=JobResponse, tags=["jobs"])
async def get_job(job_id: str) -> JobResponse:
    async with get_session() as session:
        obj = await session.get(Job, job_id)
        if not obj:
            raise HTTPException(status_code=404, detail="Job not found")
        return JobResponse(**obj.to_dict())


@router.patch("/jobs/{job_id}/status", response_model=JobResponse, tags=["jobs"])
async def update_job_status(job_id: str, update: JobStatusUpdate) -> JobResponse:
    try:
        new_status = JobStatus(update.status)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid status value")

    async with get_session() as session:
        obj = await session.get(Job, job_id)
        if not obj:
            raise HTTPException(status_code=404, detail="Job not found")
        obj.status = new_status
        if new_status == JobStatus.APPLIED:
            obj.applied_at = datetime.utcnow()
            # Also persist an Application row for audit + mirror to the graph.
            session.add(Application(job_id=obj.id, applied_at=obj.applied_at))
            try:
                await store_application(
                    company=obj.company,
                    role=obj.title,
                    date=obj.applied_at.strftime("%Y-%m-%d"),
                    status=new_status.value,
                )
            except Exception as e:  # never fail the HTTP request on graph errors
                log.warning("Graph write for application failed: %s", e)
        out = JobResponse(**obj.to_dict())
    return out


@router.get("/applications", response_model=list[ApplicationResponse], tags=["jobs"])
async def list_applications() -> list[ApplicationResponse]:
    async with get_session() as session:
        rows = (
            (await session.execute(select(Application).order_by(desc(Application.applied_at))))
            .scalars()
            .all()
        )
    return [ApplicationResponse(**a.to_dict()) for a in rows]


@router.get("/graph", response_model=GraphResponse, tags=["graph"])
async def get_graph() -> JSONResponse:
    try:
        graph = await get_full_graph()
    except Exception as e:
        log.error("Failed to read knowledge graph: %s", e)
        graph = {"nodes": [], "links": []}
    return JSONResponse(graph)


@router.get("/skill-gaps", response_model=list[SkillGapResponse], tags=["graph"])
async def list_skill_gaps() -> list[SkillGapResponse]:
    async with get_session() as session:
        rows = (
            (await session.execute(select(SkillGap).order_by(desc(SkillGap.frequency))))
            .scalars()
            .all()
        )
    return [SkillGapResponse(**g.to_dict()) for g in rows]


@router.get("/runs", response_model=RunListResponse, tags=["pipeline"])
async def list_runs(limit: int = Query(default=50, ge=1, le=500)) -> RunListResponse:
    async with get_session() as session:
        rows = (
            (
                await session.execute(
                    select(PipelineRun).order_by(desc(PipelineRun.started_at)).limit(limit)
                )
            )
            .scalars()
            .all()
        )
    return RunListResponse(runs=[PipelineRunResponse(**r.to_dict()) for r in rows])
