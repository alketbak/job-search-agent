"""JobRadar entrypoint - FastAPI app + APScheduler daily cron + MCP lifecycle.

Run locally:   python main.py
Run via uvicorn: uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agents.orchestrator import run_pipeline
from agents.retrieval import warmup as retrieval_warmup
from api.router import router
from auth.backend import auth_backend
from auth.schemas import UserCreate, UserRead, UserUpdate
from auth.users import fastapi_users
from config import configure_logging, ensure_dirs, settings
from database import init_db
from database.db import dispose_engine
from database.seed import ensure_default_user
from memory.mcp_client import get_memory_client, shutdown_memory_client

configure_logging()
log = logging.getLogger("jobradar")


async def _scheduled_run() -> None:
    """Wrapper for the daily cron so it logs cleanly and never crashes."""
    try:
        log.info("Scheduled pipeline run starting (query=%s)", settings.default_query)
        await run_pipeline()
    except Exception as e:
        log.exception("Scheduled run failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup + shutdown sequence. Replaces the deprecated @on_event hooks."""
    ensure_dirs()
    init_db()
    await ensure_default_user()

    # Boot the MCP memory server. If it can't start (no npx, etc.), the client
    # silently falls back to an in-memory stub so nothing downstream breaks.
    try:
        await get_memory_client()
    except Exception as e:
        log.warning("Could not initialize MCP memory client: %s", e)

    # Warm the embedding model + CV chunks (pgvector) so the first /run isn't slow.
    await retrieval_warmup()

    # APScheduler - async mode so it uses the same event loop as FastAPI.
    # Only runs here if RUN_SCHEDULER is true (the default, for
    # `python main.py` standalone). docker-compose's `api` service sets
    # RUN_SCHEDULER=false and relies on the separate `scheduler` service
    # (scheduler_worker.py) instead.
    if settings.run_scheduler:
        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            _scheduled_run,
            CronTrigger(hour=settings.schedule_hour, minute=settings.schedule_minute),
            id="daily_pipeline",
            replace_existing=True,
        )
        scheduler.start()
        app.state.scheduler = scheduler
        log.info(
            "Scheduler started - daily pipeline at %02d:%02d",
            settings.schedule_hour,
            settings.schedule_minute,
        )
    else:
        log.info("RUN_SCHEDULER=false - scheduler disabled in this process")

    try:
        yield
    finally:
        log.info("Shutting down JobRadar")
        if getattr(app.state, "scheduler", None):
            app.state.scheduler.shutdown(wait=False)
        await shutdown_memory_client()
        await dispose_engine()


app = FastAPI(
    title="JobRadar",
    description=(
        "Multi-agent job search intelligence system - LangGraph orchestration, "
        "pgvector-based CV matching, Claude-powered analysis, and an MCP-backed "
        "knowledge graph visualized with D3.js."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting on upload and matching endpoints is enforced via api/rate_limit.py's `rate_limiter()`
# dependency, applied directly on the relevant routes in api/router.py
# (no app-level wiring needed; see that module's docstring for why this
# isn't slowapi's decorator-based approach).

# register/login/me via fastapi-users rather than hand-rolled auth routes.
# Password-reset and email verification routers are deliberately not wired
# up - both need a real email-sending flow that doesn't exist yet; registered users are
# auto-active (no email verification gate) until that's added.
app.include_router(
    fastapi_users.get_auth_router(auth_backend), prefix="/auth/jwt", tags=["auth"]
)
app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate), prefix="/auth", tags=["auth"]
)
app.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate), prefix="/users", tags=["auth"]
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level=settings.log_level.lower(),
    )
