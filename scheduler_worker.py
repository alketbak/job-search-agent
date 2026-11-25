"""Standalone scheduler process.

Runs the daily pipeline job with APScheduler in its own process, separate
from the API: it serves no HTTP and can be deployed/restarted
independently. `python main.py` can still run the same job in-process
(`RUN_SCHEDULER=true`) for a single-process setup.

Run locally:  python scheduler_worker.py
In Docker:    docker compose up scheduler   (see docker-compose.yml)

Writes a heartbeat timestamp to `settings.scheduler_heartbeat_file` every
60s so a container healthcheck can confirm the process's event loop is
still alive, not just that the process exists - a hung asyncio loop still
shows up as a running PID.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from datetime import UTC, datetime
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from agents.orchestrator import run_pipeline
from config import configure_logging, ensure_dirs, settings
from database import init_db
from database.db import dispose_engine
from database.seed import ensure_default_user
from memory.mcp_client import get_memory_client, shutdown_memory_client

configure_logging()
log = logging.getLogger("jobradar.scheduler")


async def _scheduled_run() -> None:
    try:
        log.info("Scheduled pipeline run starting (query=%s)", settings.default_query)
        await run_pipeline()
    except Exception as e:
        log.exception("Scheduled run failed: %s", e)


def _write_heartbeat() -> None:
    path = Path(settings.scheduler_heartbeat_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(datetime.now(UTC).isoformat(), encoding="utf-8")


async def main() -> None:
    ensure_dirs()
    init_db()
    await ensure_default_user()

    try:
        await get_memory_client()
    except Exception as e:
        log.warning("Could not initialize MCP memory client: %s", e)

    _write_heartbeat()  # heartbeat exists immediately, not just after the first tick

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        _scheduled_run,
        CronTrigger(hour=settings.schedule_hour, minute=settings.schedule_minute),
        id="daily_pipeline",
        replace_existing=True,
    )
    scheduler.add_job(_write_heartbeat, IntervalTrigger(seconds=60), id="heartbeat", replace_existing=True)
    scheduler.start()
    log.info(
        "Scheduler worker started - daily pipeline at %02d:%02d",
        settings.schedule_hour,
        settings.schedule_minute,
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass  # Windows; Ctrl+C still raises KeyboardInterrupt below

    try:
        await stop_event.wait()
    finally:
        log.info("Shutting down scheduler worker")
        scheduler.shutdown(wait=False)
        await shutdown_memory_client()
        await dispose_engine()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
