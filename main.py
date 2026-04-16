"""JobRadar entrypoint — FastAPI app + APScheduler daily cron + MCP lifecycle.

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
from agents.rag_agent import warmup as rag_warmup
from api.router import router
from config import configure_logging, ensure_dirs, settings
from database import init_db
from database.db import dispose_engine
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

    # Boot the MCP memory server. If it can't start (no npx, etc.), the client
    # silently falls back to an in-memory stub so nothing downstream breaks.
    try:
        await get_memory_client()
    except Exception as e:
        log.warning("Could not initialize MCP memory client: %s", e)

    # Warm the embedding model + FAISS index so the first /run isn't slow.
    await rag_warmup()

    # APScheduler — async mode so it uses the same event loop as FastAPI.
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
        "Scheduler started — daily pipeline at %02d:%02d",
        settings.schedule_hour,
        settings.schedule_minute,
    )

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
        "Multi-agent job search intelligence system — LangGraph orchestration, "
        "FAISS-based CV matching, Claude-powered analysis, and an MCP-backed "
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
