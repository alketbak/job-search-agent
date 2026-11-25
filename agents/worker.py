"""arq background worker: CV processing and per-user matching off the
request path, with retries, timeouts, idempotency keys and a durable
failure status.

Run locally:  arq agents.worker.WorkerSettings
In Docker:    docker compose up worker   (see docker-compose.yml)

Three job types:
- `extract_cv_job`: the CV-extraction LLM call `POST /cv` used to make
  inline (blocking the HTTP response on an Anthropic call). Enqueued with
  `_job_id=cv_id` - arq treats a second enqueue with the same job id as a
  no-op while the first is still queued/running, which is the idempotency
  key required (re-POSTing the same upload while it's still
  processing doesn't double-charge the LLM quota or double-extract).
- `run_matching_job`: a full per-user run - scrape, dedupe, extract,
  retrieve, score, explain, persist, notify - scored against `user_id`'s
  own uploaded CV. Not idempotency-keyed - each trigger is a deliberate
  "run it again," not a duplicate of the same intent.
- `score_job_job`: score one existing job (e.g. one the user added by
  hand) against `user_id`'s CV. Keyed by its pre-created run id.

Failure status: both jobs update a durable DB row (`CV.last_extraction_status`
/ `PipelineRun.status`) themselves, in a `finally`-equivalent (the
`except` branch), so a caller polling `GET /cv/{id}` or `GET /runs/{id}`
sees `failed` (with a reason) rather than the row just sitting at
`pending` forever if the job errors out. Retries/timeouts are arq
`Worker`-level settings (`max_tries`, `job_timeout` below) - if all
retries are exhausted, arq's own `on_job_end`... is not relied upon here;
the job function's own `except` branch runs on every failed attempt
including the last, so the DB status is current after any one of them,
not only after the final one.
"""

from __future__ import annotations

import logging
from typing import Any

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from config import configure_logging, settings
from database.db import get_session
from database.models import CV, PipelineRun
from extraction.cv_extractor import extract_candidate_profile, persist_candidate_profile
from extraction.schemas import ExtractionFailed

configure_logging()
log = logging.getLogger("jobradar.worker")


async def extract_cv_job(ctx: dict[str, Any], cv_id: str, user_id: str) -> dict[str, Any]:
    async with get_session() as session:
        cv = await session.get(CV, cv_id)
        if cv is None:
            log.warning("extract_cv_job: cv_id=%s no longer exists - skipping", cv_id)
            return {"cv_id": cv_id, "status": "skipped"}

        try:
            extraction = await extract_candidate_profile(cv.raw_text)
            await persist_candidate_profile(session, user_id, cv.id, extraction)
            cv.last_extraction_status = "success"
            cv.last_extraction_failure_reason = None
            log.info("extract_cv_job: cv_id=%s succeeded", cv_id)
            return {"cv_id": cv_id, "status": "success"}
        except ExtractionFailed as e:
            cv.last_extraction_status = "failed"
            cv.last_extraction_failure_reason = e.reason
            log.warning("extract_cv_job: cv_id=%s failed: %s", cv_id, e.reason)
            return {"cv_id": cv_id, "status": "failed", "reason": e.reason}
        except Exception as e:
            cv.last_extraction_status = "failed"
            cv.last_extraction_failure_reason = f"worker_error:{type(e).__name__}"
            log.exception("extract_cv_job: cv_id=%s unexpected error", cv_id)
            raise  # let arq's retry policy decide whether to try again


async def run_matching_job(
    ctx: dict[str, Any],
    run_id: str,
    user_id: str,
    query: str | None,
    location: str | None,
) -> dict[str, Any]:
    from agents.orchestrator import (
        run_pipeline,  # local import avoids a circular import at module load
    )

    async with get_session() as session:
        run = await session.get(PipelineRun, run_id)
        if run is not None:
            run.status = "running"

    try:
        final = await run_pipeline(query=query, location=location, user_id=user_id, run_id=run_id)
        log.info("run_matching_job: run_id=%s finished (jobs_found=%s)", run_id, final.get("jobs_found"))
        return {"run_id": run_id, "status": "success"}
    except Exception as e:
        log.exception("run_matching_job: run_id=%s unexpected error", run_id)
        async with get_session() as session:
            run = await session.get(PipelineRun, run_id)
            if run is not None:
                run.status = "failed"
                run.errors = [*(run.errors or []), f"worker_error:{type(e).__name__}"]
        raise  # let arq's retry policy decide whether to try again


async def score_job_job(ctx: dict[str, Any], run_id: str, job_id: str, user_id: str) -> dict[str, Any]:
    """Score one job (e.g. one the user added by hand) against `user_id`'s
    CV. `run_id`'s `PipelineRun` row is pre-created by the API as `pending`."""
    from agents.orchestrator import score_job_for_user  # avoids a circular import at module load

    async with get_session() as session:
        run = await session.get(PipelineRun, run_id)
        if run is not None:
            run.status = "running"

    try:
        await score_job_for_user(job_id, user_id, run_id)
        log.info("score_job_job: run_id=%s job_id=%s finished", run_id, job_id)
        return {"run_id": run_id, "status": "success"}
    except Exception as e:
        log.exception("score_job_job: run_id=%s unexpected error", run_id)
        async with get_session() as session:
            run = await session.get(PipelineRun, run_id)
            if run is not None:
                run.status = "failed"
                run.errors = [*(run.errors or []), f"worker_error:{type(e).__name__}"]
        raise


_pool: ArqRedis | None = None


async def get_arq_pool() -> ArqRedis:
    """Lazy singleton connection pool for enqueueing jobs from the API
    process - separate from the `Worker`'s own pool, which only exists in
    the worker process."""
    global _pool
    if _pool is None:
        _pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    return _pool


class WorkerSettings:
    functions = [extract_cv_job, run_matching_job, score_job_job]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    # 3 attempts (not arq's default 5 -
    # an LLM call that fails 3 times in a row is unlikely to succeed on a
    # 4th/5th identical retry, and this bounds API spend on a broken call).
    max_tries = 3
    # 5 minutes - generous for one LLM call (extract_cv_job) or a full
    # scrape+score+explain run (run_matching_job) without letting a truly
    # hung job occupy a worker slot indefinitely.
    job_timeout = 300
