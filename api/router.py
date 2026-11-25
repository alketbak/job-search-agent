"""FastAPI router - wires all REST endpoints and the dashboard.

We keep the router small and delegate real work to agents/ and memory/.
Every endpoint returns typed Pydantic responses for OpenAPI clarity.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import desc, func, or_, select

from agents.dedup import canonical_url, content_hash, find_duplicate
from agents.worker import get_arq_pool
from api.rate_limit import rate_limiter
from api.schemas import (
    ApplicationCreateRequest,
    ApplicationDetailResponse,
    ApplicationResponse,
    ApplicationUpdateRequest,
    CandidateProfileResponse,
    CandidateSkillResponse,
    CVResponse,
    FeedbackCreateRequest,
    FeedbackResponse,
    GraphResponse,
    HealthResponse,
    InterviewCreateRequest,
    InterviewResponse,
    JobListResponse,
    JobResponse,
    JobStatusUpdate,
    KanbanCard,
    KanbanResponse,
    ManualJobCreateRequest,
    ManualJobCreateResponse,
    MatchResultResponse,
    MeDeletionResponse,
    MeExportResponse,
    MeUsageResponse,
    PipelineRunResponse,
    RunListResponse,
    RunRequest,
    RunResponse,
    ScoringResponse,
    SkillGapResponse,
)
from auth.users import current_active_user
from config import settings
from database import Application, Job, JobStatus, SkillGap, get_session
from database.models import (
    CV,
    ApplicationStatus,
    CandidateProfile,
    CandidateSkill,
    CVChunk,
    Feedback,
    Interview,
    JobEmbedding,
    JobRequirement,
    MatchResult,
    PipelineRun,
    User,
    job_visible_to,
)
from extraction.cv_extractor import load_uploaded_candidate_profile
from extraction.cv_file import InvalidCVFile, validate_and_extract_text
from memory.knowledge_graph import get_full_graph, store_application
from observability.quota import LLMQuotaExceeded, check_and_increment
from observability.quota import current_usage as current_llm_usage

log = logging.getLogger(__name__)

router = APIRouter()

# Resolve the dashboard HTML once - relative to this file so it works whether
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


async def _get_visible_job(session: Any, job_id: str, user_id: str) -> Job:
    """The job if `user_id` may see it; 404 otherwise - the same answer as
    "doesn't exist", so another account's private job ids can't be probed."""
    job = await session.get(Job, job_id)
    if job is None or (job.owner_user_id is not None and job.owner_user_id != user_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return job


async def _start_scoring(job_id: str, user_id: str) -> ScoringResponse:
    """Queue scoring of one job against `user_id`'s own uploaded CV.

    Charged as one unit of the daily LLM quota: unlike a shared scraped job,
    this extraction + explanation only ever benefits this one user."""
    if await load_uploaded_candidate_profile(user_id) is None:
        return ScoringResponse(scoring_status="no_cv")
    try:
        await check_and_increment(user_id)
    except LLMQuotaExceeded:
        return ScoringResponse(scoring_status="quota_exceeded")
    run_id = str(uuid.uuid4())
    async with get_session() as session:
        session.add(PipelineRun(id=run_id, user_id=user_id, status="pending"))
    pool = await get_arq_pool()
    await pool.enqueue_job("score_job_job", run_id, job_id, user_id, _job_id=run_id)
    return ScoringResponse(scoring_status="queued", scoring_run_id=run_id)


@router.post(
    "/run",
    response_model=RunResponse,
    status_code=202,
    tags=["pipeline"],
    dependencies=[Depends(rate_limiter("5/minute", "run"))],
)
async def trigger_run(
    payload: RunRequest,
    current_user: User = Depends(current_active_user),
) -> RunResponse:
    """Enqueue a per-user matching run in the background worker -
    scrape, dedupe, extract, retrieve, score, explain, persist, notify,
    scored against `current_user`'s own uploaded CV.

    Returns 202 with a `run_id` immediately; poll `GET /runs/{run_id}`
    for the real outcome once `status` reaches `success`/`failed`. The
    pipeline makes an LLM call per new job, far too slow to run inside
    the request.

    Rate-limited: each run enqueues a scrape + score + explain pass over every job
    found, real LLM calls included; nothing should be able to trigger
    that in a tight loop.

    409 if the account has no processed CV: there would be nothing to
    score against.
    """
    if await load_uploaded_candidate_profile(current_user.id) is None:
        raise HTTPException(
            status_code=409,
            detail="Upload a CV first (POST /cv) - matching scores jobs against your own CV.",
        )
    run_id = str(uuid.uuid4())
    async with get_session() as session:
        session.add(
            PipelineRun(
                id=run_id,
                user_id=current_user.id,
                status="pending",
                query=payload.query,
                location=payload.location,
            )
        )
    pool = await get_arq_pool()
    await pool.enqueue_job("run_matching_job", run_id, current_user.id, payload.query, payload.location)
    return RunResponse(run_id=run_id, status="pending")


@router.get("/jobs", response_model=JobListResponse, tags=["jobs"])
async def list_jobs(
    status: str | None = Query(default=None, description="Filter by status"),
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(current_active_user),
) -> JobListResponse:
    """Shared jobs (from sources, cached across all users since
    extraction is per job content) plus the caller's own hand-added jobs.
    Other accounts' hand-added jobs are never included."""
    async with get_session() as session:
        stmt = select(Job).where(job_visible_to(current_user.id)).order_by(desc(Job.found_at))
        if status:
            try:
                stmt = stmt.where(Job.status == JobStatus(status))
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}") from e
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await session.execute(count_stmt)).scalar_one()
        rows = (await session.execute(stmt.offset(offset).limit(limit))).scalars().all()
    jobs = [JobResponse(**j.to_dict()) for j in rows]
    return JobListResponse(total=int(total), jobs=jobs)


@router.post(
    "/jobs",
    response_model=ManualJobCreateResponse,
    status_code=201,
    tags=["jobs"],
    dependencies=[Depends(rate_limiter("30/minute", "add_job"))],
)
async def add_job_manually(
    payload: ManualJobCreateRequest,
    current_user: User = Depends(current_active_user),
) -> ManualJobCreateResponse:
    """Add a job the user found themselves, track it, and queue scoring it
    against their own CV.

    The job is private to the caller (`Job.owner_user_id`). Dedup runs the
    pipeline's exact-match stages (canonical URL, then normalized
    title/company/location) over shared jobs and the caller's own jobs
    only: pasting a job the scrapers already found reuses that shared row;
    nothing ever matches another account's private job. Returns 409 if the
    caller already tracks the matched job.

    Scoring runs in the background (`scoring_status="queued"`); it's
    skipped with `no_cv` if the account has no processed CV yet, or
    `quota_exceeded` - the job and application are saved either way.
    """

    def _clean(value: str | None) -> str | None:
        value = (value or "").strip()
        return value or None

    title = payload.title.strip()
    company = payload.company.strip()
    if not title or not company:
        raise HTTPException(status_code=422, detail="Title and company must not be blank")
    location = _clean(payload.location)
    url = _clean(payload.url)

    async with get_session() as session:
        job = await find_duplicate(
            session,
            {"title": title, "company": company, "location": location or "", "url": url},
            owner_user_id=current_user.id,
        )
        if job is None:
            job = Job(
                title=title,
                company=company,
                location=location,
                url=url,
                description=_clean(payload.description),
                source="manual",
                owner_user_id=current_user.id,
                canonical_url=canonical_url(url),
                content_hash=content_hash(title, company, location or ""),
            )
            session.add(job)
            await session.flush()
        else:
            existing = (
                await session.execute(
                    select(Application).where(
                        Application.job_id == job.id, Application.user_id == current_user.id
                    )
                )
            ).scalars().first()
            if existing:
                raise HTTPException(
                    status_code=409,
                    detail=f"You already track this job (application {existing.id}).",
                )

        status = ApplicationStatus(payload.status)
        now = datetime.utcnow()
        application = Application(
            user_id=current_user.id,
            job_id=job.id,
            status=status,
            status_changed_at=now,
            applied_at=now if status == ApplicationStatus.APPLIED else None,
            notes=payload.notes,
        )
        session.add(application)
        await session.flush()
        created = application.to_dict()
        job_id = job.id

    # After the session commits, so the worker can see the new rows.
    scoring = await _start_scoring(job_id, current_user.id)
    return ManualJobCreateResponse(**created, **scoring.model_dump())


@router.post(
    "/jobs/{job_id}/score",
    response_model=ScoringResponse,
    status_code=202,
    tags=["jobs"],
    dependencies=[Depends(rate_limiter("30/minute", "score_job"))],
)
async def score_job(job_id: str, current_user: User = Depends(current_active_user)) -> ScoringResponse:
    """(Re)score one job against the caller's current CV - e.g. a job
    added before a CV was uploaded, or after uploading a new CV."""
    async with get_session() as session:
        await _get_visible_job(session, job_id, current_user.id)
    return await _start_scoring(job_id, current_user.id)


@router.get("/jobs/{job_id}", response_model=JobResponse, tags=["jobs"])
async def get_job(job_id: str, current_user: User = Depends(current_active_user)) -> JobResponse:
    async with get_session() as session:
        obj = await _get_visible_job(session, job_id, current_user.id)
        return JobResponse(**obj.to_dict())


@router.patch("/jobs/{job_id}/status", response_model=JobResponse, tags=["jobs"])
async def update_job_status(
    job_id: str,
    update: JobStatusUpdate,
    current_user: User = Depends(current_active_user),
) -> JobResponse:
    """Legacy, global status (docs/DEVELOPMENT.md: `Job.status` is a pre-existing,
    NOT user-scoped field kept separate from the real per-user
    `Application.status` lifecycle). Any authenticated user can call this
    on any shared job (and on their own hand-added ones). Prefer
    `POST /applications` / `PATCH /applications/{id}` for real per-user
    tracking; this endpoint mutates a row every user sees (or, for a
    hand-added private job, only its owner sees)."""
    try:
        new_status = JobStatus(update.status)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid status value") from e

    async with get_session() as session:
        obj = await _get_visible_job(session, job_id, current_user.id)
        obj.status = new_status
        if new_status == JobStatus.APPLIED:
            obj.applied_at = datetime.utcnow()
            # Also persist an Application row for audit + mirror to the graph.
            session.add(Application(user_id=current_user.id, job_id=obj.id, applied_at=obj.applied_at))
        # The knowledge graph is global - a private hand-added job stays out of it.
        if new_status == JobStatus.APPLIED and obj.owner_user_id is None:
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


@router.get("/applications", response_model=list[ApplicationResponse], tags=["applications"])
async def list_applications(
    status: str | None = Query(default=None, description="Filter by application status"),
    current_user: User = Depends(current_active_user),
) -> list[ApplicationResponse]:
    async with get_session() as session:
        stmt = (
            select(Application)
            .where(Application.user_id == current_user.id)
            .order_by(desc(Application.status_changed_at))
        )
        if status:
            try:
                stmt = stmt.where(Application.status == ApplicationStatus(status))
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}") from e
        rows = (await session.execute(stmt)).scalars().all()
    return [ApplicationResponse(**a.to_dict()) for a in rows]


@router.post("/applications", response_model=ApplicationResponse, status_code=201, tags=["applications"])
async def create_application(
    payload: ApplicationCreateRequest,
    current_user: User = Depends(current_active_user),
) -> ApplicationResponse:
    """Start tracking a job (status lifecycle: interested through
    withdrawn). One application per (user, job): re-posting for a
    job that's already tracked returns 409 pointing at the existing one."""
    try:
        new_status = ApplicationStatus(payload.status)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid status: {payload.status}") from e

    user_id = current_user.id
    async with get_session() as session:
        await _get_visible_job(session, payload.job_id, user_id)

        existing = (
            await session.execute(
                select(Application).where(
                    Application.job_id == payload.job_id, Application.user_id == user_id
                )
            )
        ).scalars().first()
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"An application already exists for this job: {existing.id}",
            )

        now = datetime.utcnow()
        application = Application(
            user_id=user_id,
            job_id=payload.job_id,
            status=new_status,
            status_changed_at=now,
            applied_at=now if new_status == ApplicationStatus.APPLIED else None,
            notes=payload.notes,
        )
        session.add(application)
        await session.flush()
        return ApplicationResponse(**application.to_dict())


@router.get("/applications/kanban", response_model=KanbanResponse, tags=["applications"])
async def applications_kanban(current_user: User = Depends(current_active_user)) -> KanbanResponse:
    """Applications grouped by status, for the dashboard Kanban board."""
    async with get_session() as session:
        rows = (
            await session.execute(
                select(Application, Job)
                .join(Job, Job.id == Application.job_id)
                .where(Application.user_id == current_user.id)
            )
        ).all()
        job_ids = [job.id for _, job in rows]
        latest_scores: dict[str, float] = {}
        if job_ids:
            score_rows = (
                await session.execute(
                    select(MatchResult.job_id, MatchResult.score_final, MatchResult.created_at)
                    .where(MatchResult.job_id.in_(job_ids), MatchResult.user_id == current_user.id)
                    .order_by(desc(MatchResult.created_at))
                )
            ).all()
            for job_id, score_final, _ in score_rows:
                latest_scores.setdefault(job_id, score_final)

    columns: dict[str, list[KanbanCard]] = {s.value: [] for s in ApplicationStatus}
    for application, job in rows:
        card = KanbanCard(
            application_id=application.id,
            job_id=job.id,
            job_title=job.title,
            job_company=job.company,
            status=application.status.value,
            status_changed_at=application.status_changed_at,
            score_final=latest_scores.get(job.id),
        )
        columns[application.status.value].append(card)
    for cards in columns.values():
        cards.sort(key=lambda c: c.status_changed_at or datetime.min, reverse=True)
    return KanbanResponse(columns=columns)


@router.get("/applications/{application_id}", response_model=ApplicationDetailResponse, tags=["applications"])
async def get_application(
    application_id: str, current_user: User = Depends(current_active_user)
) -> ApplicationDetailResponse:
    async with get_session() as session:
        application = await session.get(Application, application_id)
        # 404, not 403, for someone else's application - same response as
        # "doesn't exist" so this endpoint can't be used to enumerate
        # other users' application ids.
        if not application or application.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Application not found")
        job = await session.get(Job, application.job_id)
        interviews = (
            await session.execute(select(Interview).where(Interview.application_id == application_id))
        ).scalars().all()
        feedback_entries = (
            await session.execute(select(Feedback).where(Feedback.application_id == application_id))
        ).scalars().all()
        match_result = (
            await session.execute(
                select(MatchResult)
                .where(MatchResult.job_id == application.job_id, MatchResult.user_id == current_user.id)
                .order_by(desc(MatchResult.created_at))
                .limit(1)
            )
        ).scalars().first()

        return ApplicationDetailResponse(
            **application.to_dict(),
            job=JobResponse(**job.to_dict()) if job else None,
            interviews=[InterviewResponse(**i.to_dict()) for i in interviews],
            feedback_entries=[FeedbackResponse(**f.to_dict()) for f in feedback_entries],
            match_result=MatchResultResponse(**match_result.to_dict()) if match_result else None,
        )


@router.patch("/applications/{application_id}", response_model=ApplicationResponse, tags=["applications"])
async def update_application(
    application_id: str,
    payload: ApplicationUpdateRequest,
    current_user: User = Depends(current_active_user),
) -> ApplicationResponse:
    """Update status and/or other fields. A status change bumps
    `status_changed_at`, and sets `applied_at` the first time status
    reaches "applied" - never resets it on a later transition."""
    async with get_session() as session:
        application = await session.get(Application, application_id)
        if not application or application.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Application not found")

        if payload.status is not None:
            try:
                new_status = ApplicationStatus(payload.status)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid status: {payload.status}") from e
            if new_status != application.status:
                application.status = new_status
                application.status_changed_at = datetime.utcnow()
                if new_status == ApplicationStatus.APPLIED and application.applied_at is None:
                    application.applied_at = application.status_changed_at

        if payload.notes is not None:
            application.notes = payload.notes
        if payload.salary_offered is not None:
            application.salary_offered = payload.salary_offered
        if payload.response_received is not None:
            application.response_received = payload.response_received

        await session.flush()
        return ApplicationResponse(**application.to_dict())


@router.post(
    "/applications/{application_id}/interviews",
    response_model=InterviewResponse,
    status_code=201,
    tags=["applications"],
)
async def create_interview(
    application_id: str,
    payload: InterviewCreateRequest,
    current_user: User = Depends(current_active_user),
) -> InterviewResponse:
    async with get_session() as session:
        application = await session.get(Application, application_id)
        if not application or application.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Application not found")
        interview = Interview(
            application_id=application_id,
            interview_date=payload.interview_date,
            stage=payload.stage,
            notes=payload.notes,
        )
        session.add(interview)
        await session.flush()
        return InterviewResponse(**interview.to_dict())


@router.get(
    "/applications/{application_id}/interviews",
    response_model=list[InterviewResponse],
    tags=["applications"],
)
async def list_interviews(
    application_id: str, current_user: User = Depends(current_active_user)
) -> list[InterviewResponse]:
    async with get_session() as session:
        application = await session.get(Application, application_id)
        if not application or application.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Application not found")
        rows = (
            await session.execute(select(Interview).where(Interview.application_id == application_id))
        ).scalars().all()
    return [InterviewResponse(**i.to_dict()) for i in rows]


@router.post(
    "/applications/{application_id}/feedback",
    response_model=FeedbackResponse,
    status_code=201,
    tags=["applications"],
)
async def create_feedback(
    application_id: str,
    payload: FeedbackCreateRequest,
    current_user: User = Depends(current_active_user),
) -> FeedbackResponse:
    """Outcome feedback - a signal for the user's own records, never an
    automatic relevance label: rejections have many causes unrelated to
    fit. Nothing in evaluation/ reads from this table."""
    async with get_session() as session:
        application = await session.get(Application, application_id)
        if not application or application.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Application not found")
        feedback = Feedback(application_id=application_id, outcome=payload.outcome, notes=payload.notes)
        session.add(feedback)
        await session.flush()
        return FeedbackResponse(**feedback.to_dict())


@router.get(
    "/applications/{application_id}/feedback",
    response_model=list[FeedbackResponse],
    tags=["applications"],
)
async def list_feedback(
    application_id: str, current_user: User = Depends(current_active_user)
) -> list[FeedbackResponse]:
    async with get_session() as session:
        application = await session.get(Application, application_id)
        if not application or application.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Application not found")
        rows = (
            await session.execute(select(Feedback).where(Feedback.application_id == application_id))
        ).scalars().all()
    return [FeedbackResponse(**f.to_dict()) for f in rows]


@router.get("/graph", response_model=GraphResponse, tags=["graph"])
async def get_graph(current_user: User = Depends(current_active_user)) -> JSONResponse:
    """Known limitation: the MCP knowledge graph is not
    partitioned per user - `current_user` gates access but every
    authenticated user currently sees the SAME graph, built from every
    user's pipeline runs. Listed under known issues in
    docs/DEVELOPMENT.md."""
    try:
        graph = await get_full_graph()
    except Exception as e:
        log.error("Failed to read knowledge graph: %s", e)
        graph = {"nodes": [], "links": []}
    return JSONResponse(graph)


@router.get("/skill-gaps", response_model=list[SkillGapResponse], tags=["graph"])
async def list_skill_gaps(current_user: User = Depends(current_active_user)) -> list[SkillGapResponse]:
    async with get_session() as session:
        rows = (
            (await session.execute(select(SkillGap).order_by(desc(SkillGap.frequency))))
            .scalars()
            .all()
        )
    return [SkillGapResponse(**g.to_dict()) for g in rows]


@router.get("/runs", response_model=RunListResponse, tags=["pipeline"])
async def list_runs(
    limit: int = Query(default=50, ge=1, le=500),
    current_user: User = Depends(current_active_user),
) -> RunListResponse:
    """`PipelineRun.user_id` is nullable (a run isn't always tied to one
    candidate's matching) - shows runs owned by `current_user` plus
    unowned (global) runs, never another user's owned run."""
    async with get_session() as session:
        rows = (
            (
                await session.execute(
                    select(PipelineRun)
                    .where(
                        or_(PipelineRun.user_id == current_user.id, PipelineRun.user_id.is_(None))
                    )
                    .order_by(desc(PipelineRun.started_at))
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
    return RunListResponse(runs=[PipelineRunResponse(**r.to_dict()) for r in rows])


@router.get("/runs/{run_id}", response_model=PipelineRunResponse, tags=["pipeline"])
async def get_run(run_id: str, current_user: User = Depends(current_active_user)) -> PipelineRunResponse:
    """Poll a single run's status - what `POST /run`'s 202 response's
    `run_id` is for. Same ownership rule as `GET /runs`:
    visible if it's yours or unowned, 404 for someone else's."""
    async with get_session() as session:
        run = await session.get(PipelineRun, run_id)
        if not run or (run.user_id is not None and run.user_id != current_user.id):
            raise HTTPException(status_code=404, detail="Run not found")
    return PipelineRunResponse(**run.to_dict())


# --- CV upload + structured extraction ---

async def _build_cv_response(
    session,
    cv: CV,
    extraction_status: str | None,
    failure_reason: str | None,
    profile: CandidateProfile | None,
) -> CVResponse:
    profile_response = None
    if profile:
        skills = (
            await session.execute(
                select(CandidateSkill).where(CandidateSkill.candidate_profile_id == profile.id)
            )
        ).scalars().all()
        profile_response = CandidateProfileResponse(
            id=profile.id,
            cv_id=profile.cv_id,
            skills=[
                CandidateSkillResponse(
                    name=s.skill_name,
                    category=s.category,
                    evidence_level=s.evidence_level,
                    evidence_snippet=s.evidence_snippet,
                )
                for s in skills
            ],
            languages=profile.languages or [],
            experience_entries=profile.experience_entries or [],
            education=profile.education or [],
            certifications=profile.certifications or [],
            location_preference=profile.location_preference,
            remote_preference=profile.remote_preference,
            employment_type=profile.employment_type,
            seniority_target=profile.seniority_target,
            prompt_version=profile.prompt_version,
            created_at=profile.created_at,
        )
    return CVResponse(
        id=cv.id,
        version=cv.version,
        file_type=cv.file_type,
        content_hash=cv.content_hash,
        uploaded_at=cv.uploaded_at,
        consent_given_at=cv.consent_given_at,
        extraction_status=extraction_status,
        extraction_failure_reason=failure_reason,
        profile=profile_response,
    )


@router.post(
    "/cv",
    response_model=CVResponse,
    status_code=202,
    tags=["cv"],
    dependencies=[Depends(rate_limiter("10/minute", "cv_upload"))],
)
async def upload_cv(
    file: UploadFile = File(...),
    consent: bool = Form(
        ...,
        description="Must be true - confirms the uploader consents to this CV being "
        "stored and processed (GDPR consent).",
    ),
    current_user: User = Depends(current_active_user),
) -> CVResponse:
    """Upload a CV (PDF or plain text); extraction runs as a background
    job (agents/worker.py::extract_cv_job). Rate-limited.

    Returns 202 immediately with `extraction_status="pending"` - poll
    `GET /cv/{id}` for `"success"`/`"failed"` once the worker picks it up.
    A successful upload and a successful extraction are tracked
    separately: invalid LLM output gets one retry, then the CV is marked
    failed rather than the request crashing.
    """
    if settings.demo_mode:
        raise HTTPException(
            status_code=403,
            detail="CV upload is disabled in demo mode (DEMO_MODE=true).",
        )
    if not consent:
        raise HTTPException(
            status_code=400,
            detail="Consent is required to upload and process a CV (consent=true).",
        )
    try:
        await check_and_increment(current_user.id)
    except LLMQuotaExceeded as e:
        raise HTTPException(status_code=429, detail=str(e)) from e
    content = await file.read()
    try:
        text = validate_and_extract_text(file.filename or "", content)
    except InvalidCVFile as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    file_type = "pdf" if (file.filename or "").lower().endswith(".pdf") else "text"
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    async with get_session() as session:
        user_id = current_user.id
        existing_count = (
            await session.execute(
                select(func.count()).select_from(CV).where(CV.user_id == user_id)
            )
        ).scalar_one()
        cv = CV(
            user_id=user_id,
            version=existing_count + 1,
            raw_text=text,
            file_type=file_type,
            content_hash=content_hash,
            consent_given_at=datetime.utcnow(),
        )
        cv.last_extraction_status = "pending"
        cv.last_extraction_failure_reason = None
        session.add(cv)
        await session.flush()

        # Extraction runs in the background worker, not inline.
        # `_job_id=cv.id` is the idempotency key -
        # re-uploading the identical pending CV can't double-enqueue
        # (double-spend the LLM quota) while the first job is still
        # queued or running.
        pool = await get_arq_pool()
        await pool.enqueue_job("extract_cv_job", cv.id, user_id, _job_id=cv.id)

        return await _build_cv_response(session, cv, "pending", None, None)


@router.get("/me/cv", response_model=CVResponse | None, tags=["cv"])
async def get_my_latest_cv(current_user: User = Depends(current_active_user)) -> CVResponse | None:
    """The caller's most recently uploaded CV (any extraction status), or
    null if they never uploaded one."""
    async with get_session() as session:
        cv = (
            await session.execute(
                select(CV).where(CV.user_id == current_user.id).order_by(desc(CV.uploaded_at)).limit(1)
            )
        ).scalars().first()
        if cv is None:
            return None
        profile = (
            await session.execute(
                select(CandidateProfile)
                .where(CandidateProfile.cv_id == cv.id)
                .order_by(desc(CandidateProfile.created_at))
                .limit(1)
            )
        ).scalars().first()
        return await _build_cv_response(
            session, cv, cv.last_extraction_status, cv.last_extraction_failure_reason, profile
        )


@router.get("/cv/{cv_id}", response_model=CVResponse, tags=["cv"])
async def get_cv(cv_id: str, current_user: User = Depends(current_active_user)) -> CVResponse:
    async with get_session() as session:
        cv = await session.get(CV, cv_id)
        if not cv or cv.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="CV not found")
        profile = (
            await session.execute(
                select(CandidateProfile)
                .where(CandidateProfile.cv_id == cv_id)
                .order_by(desc(CandidateProfile.created_at))
                .limit(1)
            )
        ).scalars().first()
        return await _build_cv_response(
            session, cv, cv.last_extraction_status, cv.last_extraction_failure_reason, profile
        )


# --- deterministic hybrid scoring ---

@router.get("/match-results", response_model=list[MatchResultResponse], tags=["scoring"])
async def list_match_results(
    job_id: str | None = Query(default=None),
    candidate_profile_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
    current_user: User = Depends(current_active_user),
) -> list[MatchResultResponse]:
    """Score breakdown + blockers + explanation for a job, a candidate
    profile, or both - whatever filters are given narrows the results.
    Always scoped to `current_user` - a match result is this user's CV
    scored against a (shared) job, never visible to anyone else."""
    async with get_session() as session:
        stmt = (
            select(MatchResult)
            .where(MatchResult.user_id == current_user.id)
            .order_by(desc(MatchResult.created_at))
        )
        if job_id:
            stmt = stmt.where(MatchResult.job_id == job_id)
        if candidate_profile_id:
            stmt = stmt.where(MatchResult.candidate_profile_id == candidate_profile_id)
        rows = (await session.execute(stmt.limit(limit))).scalars().all()
    return [MatchResultResponse(**r.to_dict()) for r in rows]


@router.get("/me/usage", response_model=MeUsageResponse, tags=["account"])
async def get_my_usage(current_user: User = Depends(current_active_user)) -> MeUsageResponse:
    """Per-user daily LLM quota usage and cost.

    `estimated_cost_usd` is always `None` today - no local record of
    per-call token counts exists outside Langfuse (observability/
    tracing.py sends usage there, it isn't persisted in this app's own
    DB), and Langfuse itself is often unconfigured. Reporting "unavailable"
    here rather than guessing matches the rule the evaluation harness
    follows for cost.
    """
    calls_today = await current_llm_usage(current_user.id)
    quota = settings.llm_daily_quota_per_user
    return MeUsageResponse(
        calls_today=calls_today,
        daily_quota=quota,
        quota_remaining=max(0, quota - calls_today),
        estimated_cost_usd=None,
    )


# --- GDPR basics ---

@router.get("/me/export", response_model=MeExportResponse, tags=["account"])
async def export_my_data(current_user: User = Depends(current_active_user)) -> MeExportResponse:
    """Full export of everything `current_user` owns (GDPR data export).
    See docs/PRIVACY.md for retention policy and what
    this does/doesn't cover (Langfuse traces are external - see
    `DELETE /me`'s same caveat)."""
    async with get_session() as session:
        user_row = await session.get(User, current_user.id)
        cvs = (
            (await session.execute(select(CV).where(CV.user_id == current_user.id))).scalars().all()
        )
        profiles = (
            (
                await session.execute(
                    select(CandidateProfile).where(CandidateProfile.user_id == current_user.id)
                )
            )
            .scalars()
            .all()
        )
        applications = (
            (
                await session.execute(
                    select(Application).where(Application.user_id == current_user.id)
                )
            )
            .scalars()
            .all()
        )
        application_ids = [a.id for a in applications]
        interviews = []
        feedback_entries = []
        if application_ids:
            interviews = (
                (
                    await session.execute(
                        select(Interview).where(Interview.application_id.in_(application_ids))
                    )
                )
                .scalars()
                .all()
            )
            feedback_entries = (
                (
                    await session.execute(
                        select(Feedback).where(Feedback.application_id.in_(application_ids))
                    )
                )
                .scalars()
                .all()
            )
        match_results = (
            (
                await session.execute(
                    select(MatchResult).where(MatchResult.user_id == current_user.id)
                )
            )
            .scalars()
            .all()
        )
        jobs_added = (
            (await session.execute(select(Job).where(Job.owner_user_id == current_user.id)))
            .scalars()
            .all()
        )

    return MeExportResponse(
        user=user_row.to_dict(),
        cvs=[c.to_dict() for c in cvs],
        candidate_profiles=[p.to_dict() for p in profiles],
        applications=[a.to_dict() for a in applications],
        interviews=[i.to_dict() for i in interviews],
        feedback_entries=[f.to_dict() for f in feedback_entries],
        match_results=[m.to_dict() for m in match_results],
        jobs_added=[j.to_dict() for j in jobs_added],
        exported_at=datetime.utcnow(),
    )


@router.delete("/me", response_model=MeDeletionResponse, tags=["account"])
async def delete_my_account(current_user: User = Depends(current_active_user)) -> MeDeletionResponse:
    """Account deletion (GDPR): CVs, embeddings, matches, applications and
    hand-added jobs. Deletes every row explicitly,
    table by table, rather than relying solely on the DB's `ON DELETE
    CASCADE` - the FK cascades exist too (Postgres enforces them
    regardless), but SQLite (used in fast tests) doesn't enforce FKs by
    default, and deletion must not silently depend on the database engine.

    Langfuse traces are NOT deleted - they live in an external service
    this app has no delete-by-trace-id integration with
    (observability/tracing.py has none) - documented in docs/PRIVACY.md.
    """
    async with get_session() as session:
        applications = (
            (
                await session.execute(
                    select(Application.id).where(Application.user_id == current_user.id)
                )
            )
            .scalars()
            .all()
        )
        profiles = (
            (
                await session.execute(
                    select(CandidateProfile.id).where(CandidateProfile.user_id == current_user.id)
                )
            )
            .scalars()
            .all()
        )

        deleted_counts: dict[str, int] = {}

        if applications:
            for model, key in ((Interview, "interviews"), (Feedback, "feedback_entries")):
                rows = (
                    (await session.execute(select(model).where(model.application_id.in_(applications))))
                    .scalars()
                    .all()
                )
                deleted_counts[key] = len(rows)
                for row in rows:
                    await session.delete(row)

        if profiles:
            skills = (
                (
                    await session.execute(
                        select(CandidateSkill).where(CandidateSkill.candidate_profile_id.in_(profiles))
                    )
                )
                .scalars()
                .all()
            )
            deleted_counts["candidate_skills"] = len(skills)
            for row in skills:
                await session.delete(row)

        for model, key, column in (
            (Application, "applications", Application.user_id),
            (MatchResult, "match_results", MatchResult.user_id),
            (CandidateProfile, "candidate_profiles", CandidateProfile.user_id),
            (CVChunk, "cv_chunks", CVChunk.user_id),
            (CV, "cvs", CV.user_id),
        ):
            rows = (await session.execute(select(model).where(column == current_user.id))).scalars().all()
            deleted_counts[key] = len(rows)
            for row in rows:
                await session.delete(row)
        await session.flush()

        # Jobs this user added by hand are private to them, so they go too,
        # with everything hanging off them.
        owned_job_ids = (
            (await session.execute(select(Job.id).where(Job.owner_user_id == current_user.id)))
            .scalars()
            .all()
        )
        if owned_job_ids:
            for model in (JobRequirement, JobEmbedding, MatchResult, Application):
                for row in (
                    (await session.execute(select(model).where(model.job_id.in_(owned_job_ids))))
                    .scalars()
                    .all()
                ):
                    await session.delete(row)
            await session.flush()
            for job in (
                (await session.execute(select(Job).where(Job.id.in_(owned_job_ids)))).scalars().all()
            ):
                await session.delete(job)
        deleted_counts["jobs_added"] = len(owned_job_ids)

        user = await session.get(User, current_user.id)
        await session.delete(user)
        deleted_counts["user"] = 1

    return MeDeletionResponse(
        deleted_counts=deleted_counts,
        langfuse_traces_note=(
            "Langfuse traces (if LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY were ever "
            "configured) are not deleted by this endpoint - see docs/PRIVACY.md."
        ),
    )
