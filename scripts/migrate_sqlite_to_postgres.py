"""One-off migration: v1 SQLite data -> JobRadar 2.0 Postgres schema.

Reads the old `jobradar.db` (schema: jobs, applications,
skill_gaps, pipeline_runs - no user_id, no canonical_url/content_hash) via
the stdlib `sqlite3` module (NOT the new ORM models - they don't match the
old table shape) and backfills:

  - one default user (idempotent - reuses database.seed.ensure_default_user)
  - pipeline_runs -> pipeline_runs (+ user_id)
  - jobs -> jobs (+ canonical_url, content_hash, computed here)
  - applications -> applications (+ user_id)
  - skill_gaps -> skill_gaps (unchanged shape)

Never drops data: every row that can't be parsed cleanly is logged and
still migrated with best-effort defaults, never silently skipped. Idempotent: re-running skips rows
whose primary key already exists in Postgres, so it's safe to run twice
against the same source file.

Usage:
    python scripts/migrate_sqlite_to_postgres.py [--sqlite-path jobradar.db] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from config import settings  # noqa: E402
from database.models import Application, Job, JobStatus, PipelineRun, SkillGap  # noqa: E402
from database.seed import ensure_default_user  # noqa: E402

log = logging.getLogger("migrate_sqlite_to_postgres")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _parse_dt(value: str | None) -> datetime | None:
    """Parse SQLite's ISO-ish datetime strings. Never raises."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        log.warning("Could not parse datetime %r - leaving null", value)
        return None


def _parse_json_list(value: str | None) -> list[Any]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        log.warning("Could not parse JSON list %r - treating as empty", value)
        return []


def _canonical_url(url: str | None) -> str | None:
    url = (url or "").strip()
    return url or None


def _content_hash(title: str, company: str, location: str) -> str:
    """Normalized (title, company, location) hash - the dedup fallback key
    used by agents/dedup.py when canonical_url is missing or ambiguous."""
    norm = re.sub(r"\s+", " ", f"{title}|{company}|{location}".strip().lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _read_sqlite(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"SQLite source not found: {path}")
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    out: dict[str, list[dict[str, Any]]] = {}
    try:
        for table in ("pipeline_runs", "jobs", "applications", "skill_gaps"):
            try:
                rows = conn.execute(f"SELECT * FROM {table}").fetchall()  # noqa: S608
                out[table] = [dict(r) for r in rows]
            except sqlite3.OperationalError as e:
                log.warning("Table %s not found in source DB (%s) - skipping", table, e)
                out[table] = []
    finally:
        conn.close()
    return out


def migrate(sqlite_path: Path, dry_run: bool = False) -> dict[str, int]:
    data = _read_sqlite(sqlite_path)
    log.info(
        "Source counts - pipeline_runs=%s jobs=%s applications=%s skill_gaps=%s",
        len(data["pipeline_runs"]),
        len(data["jobs"]),
        len(data["applications"]),
        len(data["skill_gaps"]),
    )

    if dry_run:
        log.info("--dry-run: no writes performed (default user not created)")
        return {k: len(v) for k, v in data.items()}

    user_id = asyncio.run(ensure_default_user())
    log.info("Using default user %s", user_id)

    engine = create_engine(settings.database_url, future=True)
    counts = {"pipeline_runs": 0, "jobs": 0, "applications": 0, "skill_gaps": 0, "skipped": 0}

    with Session(engine) as session:
        for row in data["pipeline_runs"]:
            if session.get(PipelineRun, row["id"]):
                continue
            session.add(
                PipelineRun(
                    id=row["id"],
                    user_id=user_id,
                    started_at=_parse_dt(row.get("started_at")) or datetime.utcnow(),
                    finished_at=_parse_dt(row.get("finished_at")),
                    query=row.get("query"),
                    location=row.get("location"),
                    jobs_found=row.get("jobs_found") or 0,
                    jobs_scored=row.get("jobs_scored") or 0,
                    jobs_analyzed=row.get("jobs_analyzed") or 0,
                    errors=_parse_json_list(row.get("errors")),
                )
            )
            counts["pipeline_runs"] += 1
        session.flush()  # pipeline_runs must exist before jobs.run_id references them

        for row in data["jobs"]:
            if session.get(Job, row["id"]):
                continue
            title = row.get("title") or ""
            company = row.get("company") or ""
            location = row.get("location") or ""
            try:
                status = JobStatus[row["status"]] if row.get("status") else JobStatus.NEW
            except KeyError:
                log.warning("Unknown status %r on job %s - defaulting to NEW", row.get("status"), row["id"])
                status = JobStatus.NEW
            run_id = row.get("run_id")
            if run_id and not session.get(PipelineRun, run_id):
                log.warning("Job %s references missing run_id %s - clearing FK", row["id"], run_id)
                run_id = None
            session.add(
                Job(
                    id=row["id"],
                    title=title,
                    company=company,
                    location=location,
                    description=row.get("description"),
                    url=row.get("url"),
                    source=row.get("source") or "unknown",
                    canonical_url=_canonical_url(row.get("url")),
                    content_hash=_content_hash(title, company, location),
                    raw_score=row.get("raw_score"),
                    claude_score=row.get("claude_score"),
                    claude_reasoning=row.get("claude_reasoning"),
                    skill_gaps=_parse_json_list(row.get("skill_gaps")),
                    highlights=_parse_json_list(row.get("highlights")),
                    apply_recommendation=bool(row.get("apply_recommendation")),
                    status=status,
                    found_at=_parse_dt(row.get("found_at")) or datetime.utcnow(),
                    applied_at=_parse_dt(row.get("applied_at")),
                    run_id=run_id,
                )
            )
            counts["jobs"] += 1
        session.flush()  # jobs must exist before applications.job_id references them

        for row in data["applications"]:
            if session.get(Application, row["id"]):
                continue
            if not session.get(Job, row["job_id"]):
                log.warning("Application %s references missing job_id %s - skipping", row["id"], row["job_id"])
                counts["skipped"] += 1
                continue
            session.add(
                Application(
                    id=row["id"],
                    user_id=user_id,
                    job_id=row["job_id"],
                    applied_at=_parse_dt(row.get("applied_at")) or datetime.utcnow(),
                    notes=row.get("notes"),
                    response_received=bool(row.get("response_received")),
                )
            )
            counts["applications"] += 1

        for row in data["skill_gaps"]:
            if session.get(SkillGap, row["id"]):
                continue
            session.add(
                SkillGap(
                    id=row["id"],
                    skill_name=row.get("skill_name") or "unknown",
                    frequency=row.get("frequency") or 1,
                    last_seen=_parse_dt(row.get("last_seen")) or datetime.utcnow(),
                )
            )
            counts["skill_gaps"] += 1

        session.commit()

    log.info("Migrated: %s", counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite-path", default="jobradar.db")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    migrate(Path(args.sqlite_path), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
