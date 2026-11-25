"""LangGraph orchestrator - wires all agents into a state machine.

Pipeline shape:

    ingest -> dedupe -> extract -> retrieve -> score -> explain -> persist -> notify

Each node:
- accepts the full state, mutates a subset, returns the updated state
- catches its own exceptions and appends to `state["errors"]` rather than crashing
- is async so LangGraph's async runtime can interleave them

State carries one evolving `jobs` list, progressively enriched by each
node, rather than separate raw/scored/analyzed snapshots - a strictly
linear pipeline doesn't need them.
Pydantic objects (CandidateProfileExtraction, JobProfileExtraction,
ScoreBreakdown) are always stored in state as `.model_dump()` dicts, never
as model instances or numpy arrays - LangGraph's `MemorySaver` checkpointer
msgpack-serializes every node's output (see docs/DEVELOPMENT.md's "LangGraph state
must stay msgpack-serializable" note).

We compile with an in-memory checkpointer so `/runs` can retrieve run history.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from sqlalchemy import select

from agents.dedup import canonical_url as job_canonical_url
from agents.dedup import content_hash as job_content_hash
from agents.dedup import find_duplicate
from agents.delivery_agent import deliver
from agents.retrieval import embed_texts, ensure_chunks_for_cv, ensure_cv_chunks, get_cv_chunks
from agents.scraper_agent import scrape
from config import settings
from database import Job, JobStatus, get_session
from database.models import JobEmbedding, JobRequirement, MatchResult, PipelineRun
from database.seed import ensure_default_user
from extraction.cv_extractor import (
    ensure_candidate_profile_from_file,
    load_uploaded_candidate_profile,
)
from extraction.job_extractor import extract_job_profile
from extraction.schemas import CandidateProfileExtraction, ExtractionFailed, JobProfileExtraction
from memory.knowledge_graph import store_job_run, store_skill_gap
from observability import tracing
from scoring.explainer import generate_explanation
from scoring.hybrid_scorer import score_match
from scoring.schemas import ScoreBreakdown

log = logging.getLogger(__name__)


class JobRadarState(TypedDict, total=False):
    """State shared across all orchestrator nodes.

    Using `total=False` means every key is optional - LangGraph merges per-node
    partial updates rather than requiring a full state every time.
    """

    query: str
    location: str
    run_id: str
    started_at: str
    # Whose run this is. With a user, extract/retrieve score against that
    # account's own uploaded CV and persist writes its match results.
    # `None` (the scheduler's daily run) means the default user and the
    # operator's `CV_PATH` file.
    user_id: str | None
    jobs: list[dict[str, Any]]
    candidate_profile: dict[str, Any] | None
    candidate_profile_id: str | None
    # The uploaded CV the profile came from (signed-in users only); None
    # means the operator's `CV_PATH` file (runs with no user_id).
    cv_id: str | None
    cv_chunks: list[list[Any]]  # [[chunk_text, [floats]], ...]
    requirement_embeddings: dict[str, list[float]]
    skill_gaps: list[str]
    delivery_result: dict[str, Any]
    jobs_found: int
    jobs_scored: int
    jobs_explained: int
    errors: list[str]


def _log_error(state: JobRadarState, node: str, err: Exception) -> list[str]:
    """Append a structured error marker to state.errors, return updated list."""
    msg = f"{node}: {type(err).__name__}: {err}"
    log.exception("Pipeline node %s failed: %s", node, err)
    return list(state.get("errors") or []) + [msg]


def _job_text(job: dict[str, Any]) -> str:
    return f"{job.get('title','')} at {job.get('company','')}. {job.get('description','')}"


def _requirement_dicts_from_rows(rows: list[JobRequirement]) -> list[dict[str, Any]]:
    return [{"text": r.text, "mandatory": r.mandatory, "category": r.requirement_type} for r in rows]


async def _cached_job_profile(session: Any, job: Job) -> dict[str, Any] | None:
    """The job profile already extracted for `job` on an earlier run, or
    None if it was never extracted (no requirement rows)."""
    req_rows = (
        await session.execute(select(JobRequirement).where(JobRequirement.job_id == job.id))
    ).scalars().all()
    if not req_rows:
        return None
    return {
        "title": job.title,
        "company": job.company,
        "location": job.location,
        "remote": job.remote,
        "salary_min": job.salary_min,
        "salary_max": job.salary_max,
        "salary_currency": job.salary_currency,
        "seniority": job.seniority,
        "years_experience_min": job.years_experience_min,
        "employment_type": job.employment_type,
        "requirements": _requirement_dicts_from_rows(list(req_rows)),
    }


# ---------------- nodes ----------------

async def ingest_node(state: JobRadarState) -> dict[str, Any]:
    query = state.get("query") or settings.default_query
    location = state.get("location") or settings.default_location
    try:
        jobs = await scrape(query=query, location=location)
        log.info("ingest_node: %s jobs", len(jobs))
        return {"jobs": jobs, "jobs_found": len(jobs)}
    except Exception as e:
        return {"jobs": [], "jobs_found": 0, "errors": _log_error(state, "ingest", e)}


async def dedupe_node(state: JobRadarState) -> dict[str, Any]:
    """Classify each job as new or a duplicate of an existing persisted job
    (canonical URL -> normalized triple -> embedding similarity, see
    agents/dedup.py) BEFORE extraction, so duplicates skip the expensive
    LLM extraction call entirely (extract once per job, cached by
    content). Read-only against the DB; persist_node does
    the actual writes later.
    """
    jobs = state.get("jobs") or []
    if not jobs:
        return {"jobs": []}
    try:
        texts = [_job_text(j) for j in jobs]
        embeddings = embed_texts(texts)
        async with get_session() as session:
            for job, embedding in zip(jobs, embeddings, strict=True):
                job["_embedding"] = embedding
                existing = await find_duplicate(session, job, embedding)
                if existing is None:
                    job["existing_job_id"] = None
                    job["cached_job_profile"] = None
                    continue
                job["existing_job_id"] = existing.id
                job["cached_job_profile"] = await _cached_job_profile(session, existing)
        new_count = sum(1 for j in jobs if j["existing_job_id"] is None)
        log.info("dedupe_node: %s new, %s duplicates (of %s)", new_count, len(jobs) - new_count, len(jobs))
        return {"jobs": jobs}
    except Exception as e:
        return {"jobs": jobs, "errors": _log_error(state, "dedupe", e)}


async def extract_node(state: JobRadarState) -> dict[str, Any]:
    """Load the candidate profile and extract a `JobProfileExtraction` for
    every job that doesn't already have one cached from a prior run (see
    dedupe_node).

    Whose CV: with a `user_id`, the newest CV that account uploaded
    (`POST /cv`); without one (the scheduler's run), the operator's
    `CV_PATH` file, cached by content hash."""
    jobs = state.get("jobs") or []
    candidate_profile: dict[str, Any] | None = None
    candidate_profile_id: str | None = None
    cv_id: str | None = None
    errors = list(state.get("errors") or [])
    run_id = state.get("run_id")
    user_id = state.get("user_id")
    try:
        if user_id:
            uploaded = await load_uploaded_candidate_profile(user_id)
            if uploaded is not None:
                extraction, candidate_profile_id, cv_id = uploaded
                candidate_profile = extraction.model_dump()
            else:
                errors.append("extract: no_cv: this account has no processed CV - upload one first")
                log.warning("extract_node: user has no processed CV - jobs will not be scored")
        else:
            result = await ensure_candidate_profile_from_file(pipeline_run_id=run_id)
            if result is not None:
                extraction, candidate_profile_id = result
                candidate_profile = extraction.model_dump()
            else:
                log.warning("extract_node: no candidate profile available - jobs will score against nothing")
    except Exception as e:
        log.error("extract_node: candidate profile step failed: %s", e)

    for job in jobs:
        cached = job.get("cached_job_profile")
        if cached:
            job["job_profile"] = cached
            job["extraction_failed"] = False
            continue
        try:
            profile = await extract_job_profile(_job_text(job), pipeline_run_id=run_id)
            job["job_profile"] = profile.model_dump()
            job["extraction_failed"] = False
        except ExtractionFailed as e:
            log.warning("extract_node: job extraction failed for %r: %s", job.get("title"), e.reason)
            job["job_profile"] = JobProfileExtraction(title=job.get("title", "")).model_dump()
            job["extraction_failed"] = True

    return {
        "jobs": jobs,
        "candidate_profile": candidate_profile,
        "candidate_profile_id": candidate_profile_id,
        "cv_id": cv_id,
        "errors": errors,
    }


async def retrieve_node(state: JobRadarState) -> dict[str, Any]:
    """Ensure CV chunks are embedded, fetch them, and embed every job
    requirement text so scoring/hybrid_scorer.py's semantic-similarity
    fallback has vectors to compare against - the "retrieval" step
    between extraction and scoring."""
    jobs = state.get("jobs") or []
    try:
        cv_id = state.get("cv_id")
        if state.get("user_id"):
            # Signed-in account: only its own uploaded CV, never CV_PATH.
            cv_chunks = []
            if cv_id:
                await ensure_chunks_for_cv(cv_id)
                cv_chunks = await get_cv_chunks(state["user_id"], cv_id=cv_id)
        else:
            user_id = await ensure_default_user()
            await ensure_cv_chunks(user_id)
            cv_chunks = await get_cv_chunks(user_id)

        all_req_texts = sorted(
            {req["text"] for job in jobs for req in (job.get("job_profile") or {}).get("requirements", [])}
        )
        req_vecs = embed_texts(all_req_texts) if all_req_texts else []
        requirement_embeddings = dict(zip(all_req_texts, req_vecs, strict=True))

        # Legacy-compat `raw_score` (semantic similarity is used inside
        # components, not as the final score - this is a secondary
        # diagnostic signal kept for the existing /jobs API field, not used
        # by hybrid_scorer at all).
        for job in jobs:
            embedding = job.get("_embedding")
            if embedding is None or not cv_chunks:
                job["raw_score"] = 0.0
                continue
            job["raw_score"] = max(
                (_cosine(embedding, chunk_emb) for _, chunk_emb in cv_chunks), default=0.0
            )

        return {
            "jobs": jobs,
            "cv_chunks": [[text, emb] for text, emb in cv_chunks],
            "requirement_embeddings": requirement_embeddings,
        }
    except Exception as e:
        return {
            "jobs": jobs,
            "cv_chunks": [],
            "requirement_embeddings": {},
            "errors": _log_error(state, "retrieve", e),
        }


def _cosine(a: list[float], b: list[float]) -> float:
    import numpy as np

    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _derive_legacy_fields(breakdown: ScoreBreakdown) -> dict[str, Any]:
    """claude_score/skill_gaps/highlights/apply_recommendation are kept so
    the existing `/jobs` API response shape doesn't break, but now
    derived from the deterministic breakdown instead of an LLM decision."""
    skill_gaps = [
        c.text
        for c in breakdown.requirement_classifications
        if c.category in ("skill", "responsibility") and c.status in ("missing", "partial")
    ]
    highlights = [
        c.text
        for c in breakdown.requirement_classifications
        if c.category in ("skill", "responsibility") and c.status == "met"
    ]
    apply_recommendation = (
        breakdown.score_final >= settings.score_apply_recommendation_threshold and not breakdown.blockers
    )
    return {
        "claude_score": round(breakdown.score_final * 10),
        "skill_gaps": skill_gaps[:20],
        "highlights": highlights[:20],
        "apply_recommendation": apply_recommendation,
    }


async def score_node(state: JobRadarState) -> dict[str, Any]:
    """Deterministic hybrid score for every job - no LLM involved. Pure
    function (scoring/hybrid_scorer.py), so this node is just plumbing."""
    jobs = state.get("jobs") or []
    candidate_dict = state.get("candidate_profile")
    if candidate_dict is None and state.get("user_id"):
        # A signed-in account without a CV: a score against an empty
        # profile would look real but mean nothing - store none instead.
        return {"jobs": jobs, "jobs_scored": 0}
    try:
        if candidate_dict is None:
            log.warning("score_node: no candidate profile - all jobs score as fully unverifiable")
            candidate = CandidateProfileExtraction()
        else:
            candidate = CandidateProfileExtraction(**candidate_dict)

        cv_chunks = [(text, emb) for text, emb in (state.get("cv_chunks") or [])]
        requirement_embeddings = state.get("requirement_embeddings") or {}

        for job in jobs:
            job_profile = JobProfileExtraction(**(job.get("job_profile") or {"title": job.get("title", "")}))
            breakdown = score_match(candidate, job_profile, cv_chunks, requirement_embeddings)
            job["score_breakdown"] = breakdown.model_dump()
            job.update(_derive_legacy_fields(breakdown))

        jobs.sort(key=lambda j: j["score_breakdown"]["score_final"], reverse=True)
        log.info("score_node: scored %s jobs", len(jobs))
        return {"jobs": jobs, "jobs_scored": len(jobs)}
    except Exception as e:
        return {"jobs": jobs, "jobs_scored": 0, "errors": _log_error(state, "score", e)}


async def explain_node(state: JobRadarState) -> dict[str, Any]:
    """Only the top `settings.explanation_top_n` jobs get an LLM
    explanation - cost control, and the only step in this pipeline that
    calls an LLM at all besides extraction."""
    jobs = state.get("jobs") or []
    run_id = state.get("run_id")
    try:
        top_n = [j for j in jobs if j.get("score_breakdown")][: settings.explanation_top_n]
        semaphore = asyncio.Semaphore(max(1, settings.analyst_concurrency))

        async def _explain_one(job: dict[str, Any]) -> None:
            async with semaphore:
                breakdown = ScoreBreakdown(**job["score_breakdown"])
                job["explanation"] = await generate_explanation(
                    breakdown, job.get("title", ""), job.get("company"), pipeline_run_id=run_id
                )

        await asyncio.gather(*(_explain_one(j) for j in top_n))
        for job in jobs:
            job.setdefault("explanation", None)
            job["claude_reasoning"] = job["explanation"]

        log.info("explain_node: explained %s of %s jobs", len(top_n), len(jobs))
        return {"jobs": jobs, "jobs_explained": len(top_n)}
    except Exception as e:
        for job in jobs:
            job.setdefault("explanation", None)
            job["claude_reasoning"] = job.get("explanation")
        return {"jobs": jobs, "jobs_explained": 0, "errors": _log_error(state, "explain", e)}


async def persist_node(state: JobRadarState) -> dict[str, Any]:
    """Write everything to Postgres: the run, each job (new or updated -
    dedup decision already made in dedupe_node), its embedding, newly
    extracted requirements (cached ones are already in the DB), and the
    `MatchResult` row with the full score breakdown + explanation."""
    try:
        jobs = state.get("jobs") or []
        run_id = state.get("run_id") or str(uuid.uuid4())
        user_id = state.get("user_id") or await ensure_default_user()
        new_jobs = 0
        updated_jobs = 0

        async with get_session() as session:
            # agents/worker.py's background job pre-creates a `pending`
            # row with this exact run_id before the pipeline even starts, so the caller has something to
            # poll immediately - upsert here rather than blind-insert, or
            # this would hit a duplicate-primary-key error for every
            # background-triggered run. The old fully-synchronous callers
            # (the scheduler) never pre-create a row, so `existing_run` is
            # None for them and this falls back to the original insert.
            existing_run = await session.get(PipelineRun, run_id)
            run_errors = state.get("errors") or []
            run_status = "failed" if run_errors else "success"
            if existing_run:
                existing_run.user_id = user_id
                existing_run.status = run_status
                existing_run.finished_at = datetime.utcnow()
                existing_run.query = state.get("query")
                existing_run.location = state.get("location")
                existing_run.jobs_found = state.get("jobs_found") or len(jobs)
                existing_run.jobs_scored = state.get("jobs_scored") or 0
                existing_run.jobs_analyzed = state.get("jobs_explained") or 0
                existing_run.errors = run_errors
                run = existing_run
            else:
                run = PipelineRun(
                    id=run_id,
                    user_id=user_id,
                    status=run_status,
                    started_at=datetime.fromisoformat(state["started_at"])
                    if state.get("started_at")
                    else datetime.utcnow(),
                    finished_at=datetime.utcnow(),
                    query=state.get("query"),
                    location=state.get("location"),
                    jobs_found=state.get("jobs_found") or len(jobs),
                    jobs_scored=state.get("jobs_scored") or 0,
                    jobs_analyzed=state.get("jobs_explained") or 0,
                    errors=run_errors,
                )
                session.add(run)
            await session.flush()  # jobs.run_id FK needs this row to exist first

            for j in jobs:
                job_profile = j.get("job_profile") or {}
                existing_id = j.get("existing_job_id")
                existing = await session.get(Job, existing_id) if existing_id else None

                common_fields = dict(
                    seniority=job_profile.get("seniority"),
                    remote=job_profile.get("remote"),
                    employment_type=job_profile.get("employment_type"),
                    salary_min=job_profile.get("salary_min"),
                    salary_max=job_profile.get("salary_max"),
                    salary_currency=job_profile.get("salary_currency"),
                    years_experience_min=job_profile.get("years_experience_min"),
                    raw_score=j.get("raw_score"),
                    claude_score=j.get("claude_score"),
                    claude_reasoning=j.get("claude_reasoning"),
                    skill_gaps=j.get("skill_gaps") or [],
                    highlights=j.get("highlights") or [],
                    apply_recommendation=bool(j.get("apply_recommendation")),
                    run_id=run_id,
                )

                if existing:
                    for key, value in common_fields.items():
                        setattr(existing, key, value)
                    # Deliberately NOT touching `status` - a rescrape must not
                    # reset a job the user already marked applied/interviewing.
                    job_row = existing
                    updated_jobs += 1
                else:
                    job_row = Job(
                        title=j.get("title", ""),
                        company=j.get("company", ""),
                        location=j.get("location", ""),
                        description=j.get("description", ""),
                        url=j.get("url", ""),
                        source=j.get("source", "unknown"),
                        canonical_url=job_canonical_url(j.get("url")),
                        content_hash=job_content_hash(
                            j.get("title", ""), j.get("company", ""), j.get("location", "")
                        ),
                        status=JobStatus.NEW,
                        **common_fields,
                    )
                    session.add(job_row)
                    new_jobs += 1
                await session.flush()  # job_row.id needed for the FKs below

                embedding = j.get("_embedding")
                if embedding is not None:
                    existing_emb = (
                        await session.execute(
                            select(JobEmbedding).where(
                                JobEmbedding.job_id == job_row.id,
                                JobEmbedding.embedding_model == settings.embedding_model,
                            )
                        )
                    ).scalars().first()
                    if existing_emb:
                        existing_emb.embedding = embedding
                    else:
                        session.add(
                            JobEmbedding(job_id=job_row.id, embedding_model=settings.embedding_model, embedding=embedding)
                        )

                # Only write JobRequirement rows for freshly extracted jobs -
                # a cache hit (j["cached_job_profile"] was set) already has
                # them in the DB from a previous run.
                if not j.get("cached_job_profile") and not j.get("extraction_failed"):
                    for req in job_profile.get("requirements", []):
                        session.add(
                            JobRequirement(
                                job_id=job_row.id,
                                requirement_type=req["category"],
                                text=req["text"],
                                mandatory=req["mandatory"],
                                prompt_version="job_extraction_v2",
                            )
                        )

                breakdown = j.get("score_breakdown")
                if breakdown:
                    session.add(
                        MatchResult(
                            user_id=user_id,
                            candidate_profile_id=state.get("candidate_profile_id"),
                            job_id=job_row.id,
                            pipeline_run_id=run_id,
                            score_technical_skills=breakdown["score_technical_skills"],
                            score_experience=breakdown["score_experience"],
                            score_responsibilities=breakdown["score_responsibilities"],
                            score_seniority=breakdown["score_seniority"],
                            score_location_language=breakdown["score_location_language"],
                            score_education=breakdown["score_education"],
                            score_final=breakdown["score_final"],
                            blockers=breakdown["blockers"],
                            requirement_classifications=breakdown["requirement_classifications"],
                            llm_explanation=j.get("explanation"),
                            llm_model=settings.llm_model_smart if j.get("explanation") else None,
                            prompt_version="job_extraction_v2",
                        )
                    )

        log.info("persist_node: run %s - %s new jobs, %s deduped/updated", run_id, new_jobs, updated_jobs)
        return {}
    except Exception as e:
        return {"errors": _log_error(state, "persist", e)}


async def notify_node(state: JobRadarState) -> dict[str, Any]:
    """Memory-graph write + digest delivery - both are post-persist
    notification side effects, so they share one graph node (the pipeline's
    "... -> persist -> notify" tail), each still independently
    error-tolerant internally."""
    jobs = state.get("jobs") or []
    run_id = state.get("run_id") or str(uuid.uuid4())
    errors = list(state.get("errors") or [])

    try:
        await store_job_run(run_id=run_id, jobs_found=state.get("jobs_found") or len(jobs), top_jobs=jobs)
        from collections import Counter

        gap_counts: Counter[str] = Counter(gap for j in jobs for gap in (j.get("skill_gaps") or []))
        for skill, freq in gap_counts.most_common():
            await store_skill_gap(skill, freq)
    except Exception as e:
        errors = _log_error(state, "notify.memory", e)

    try:
        result = await deliver(
            scraped_count=state.get("jobs_found") or len(jobs),
            scored_count=state.get("jobs_scored") or len(jobs),
            analyzed_jobs=jobs,
            run_id=run_id,
        )
    except Exception as e:
        result = {}
        errors = _log_error(state, "notify.delivery", e)

    return {"delivery_result": result, "errors": errors}


async def score_job_for_user(job_id: str, user_id: str, run_id: str) -> JobRadarState:
    """Score one already-saved job against `user_id`'s CV - used for jobs
    a user adds by hand, which no scrape run would ever pick up.

    Runs the same extract -> retrieve -> score -> explain -> persist steps
    as a full run, minus ingest/dedupe (the job is already a row) and
    notify (one job isn't a digest). persist_node then updates that job
    row, stores its requirements/embedding, and writes this user's
    `MatchResult` plus a `PipelineRun` row under `run_id` to poll.
    """
    async with get_session() as session:
        job = await session.get(Job, job_id)
        if job is None:
            raise ValueError(f"Job {job_id} no longer exists")
        job_dict: dict[str, Any] = {
            "title": job.title,
            "company": job.company,
            "location": job.location or "",
            "description": job.description or "",
            "url": job.url or "",
            "source": job.source,
            "existing_job_id": job.id,
            "cached_job_profile": await _cached_job_profile(session, job),
        }
    job_dict["_embedding"] = embed_texts([_job_text(job_dict)])[0]

    state: JobRadarState = {
        "run_id": run_id,
        "user_id": user_id,
        "started_at": datetime.utcnow().isoformat(),
        "jobs": [job_dict],
        "jobs_found": 1,
        "errors": [],
    }
    for node in (extract_node, retrieve_node, score_node, explain_node, persist_node):
        state.update(await node(state))  # type: ignore[typeddict-item]
    tracing.flush()
    return state


# ---------------- graph assembly ----------------

def _build_graph():
    graph = StateGraph(JobRadarState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("dedupe", dedupe_node)
    graph.add_node("extract", extract_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("score", score_node)
    graph.add_node("explain", explain_node)
    graph.add_node("persist", persist_node)
    graph.add_node("notify", notify_node)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "dedupe")
    graph.add_edge("dedupe", "extract")
    graph.add_edge("extract", "retrieve")
    graph.add_edge("retrieve", "score")
    graph.add_edge("score", "explain")
    graph.add_edge("explain", "persist")
    graph.add_edge("persist", "notify")
    graph.add_edge("notify", END)

    # Checkpointer lets us replay / inspect past runs from /runs.
    return graph.compile(checkpointer=MemorySaver())


# Compile once at import time - the graph is stateless between runs.
_compiled = _build_graph()


async def run_pipeline(
    query: str | None = None,
    location: str | None = None,
    user_id: str | None = None,
    run_id: str | None = None,
) -> JobRadarState:
    """Run the full pipeline once. Returns the final state dict.

    `user_id`: whoever this run's matching is for. `None` (the scheduler)
    means the bootstrap default user and the operator's `CV_PATH` file;
    per-user matching (agents/worker.py's background job) always passes
    the authenticated caller's id.

    `run_id`: lets a caller pre-create a `PipelineRun` row (status
    `pending`) before the pipeline starts, so there's something to poll
    immediately - agents/worker.py's background job does this; persist_node
    upserts by this id instead of blind-inserting. `None` generates a
    fresh one, the original behavior.
    """
    run_id = run_id or str(uuid.uuid4())
    initial: JobRadarState = {
        "query": query or settings.default_query,
        "location": location or settings.default_location,
        "run_id": run_id,
        "started_at": datetime.utcnow().isoformat(),
        "jobs": [],
        "errors": [],
        "user_id": user_id,
    }
    log.info(
        "Starting pipeline run %s (query=%s location=%s)",
        run_id,
        initial["query"],
        initial["location"],
    )
    final = await _compiled.ainvoke(
        initial,
        # LangGraph requires a thread_id to checkpoint - we use run_id so every
        # run is its own thread / trace.
        config={"configurable": {"thread_id": run_id}},
    )
    log.info(
        "Pipeline %s done - found=%s scored=%s errors=%s",
        run_id,
        len(final.get("jobs") or []),
        final.get("jobs_scored") or 0,
        len(final.get("errors") or []),
    )
    tracing.flush()  # ensure this run's traces are sent before returning
    return final  # type: ignore[return-value]
