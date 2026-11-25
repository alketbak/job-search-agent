"""Interactive job-relevance labeling CLI.

Shows one job at a time plus a summary of the current candidate profile,
asks the user - a real person judging their own fit - to label relevance:

    0 = not relevant   1 = somewhat relevant   2 = strong match

and appends each label to `evaluation/labels/<date>.jsonl`. This is the
ONLY place labels may be created: never generated, simulated or
inferred. `evaluation/run_eval.py` only ever *reads* this
directory; nothing writes to it except a human running this script.

Already-labeled jobs are skipped by default (pass --relabel to include
them again). Resumable: quitting mid-session (q) just stops; labels
written so far are already saved, one line at a time.

Usage:
    python evaluation/label_jobs.py [--limit N] [--relabel]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

# Make the repo importable whether launched as `python evaluation/label_jobs.py`
# or `python -m evaluation.label_jobs` - this module's deferred imports
# (database, extraction) need the repo root on sys.path either way.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_LABELS_DIR = Path(__file__).parent / "labels"

Decision = Literal[0, 1, 2, "skip", "quit"]


def load_existing_labels(labels_dir: Path) -> dict[str, dict]:
    """job_id -> most recent label entry, across all *.jsonl files in dir.
    Malformed lines are skipped, not fatal - a labeling session shouldn't
    crash because an earlier file has one bad line."""
    labels: dict[str, dict] = {}
    if not labels_dir.exists():
        return labels
    for path in sorted(labels_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "job_id" in entry:
                    labels[entry["job_id"]] = entry
    return labels


def prompt_relevance(job: dict[str, Any], candidate_summary: str, input_fn=input) -> Decision:
    """Show one job, ask for a label. `input_fn` is injectable so tests can
    script the interaction without a real terminal."""
    print("\n" + "=" * 70)
    print(f"Title:    {job.get('title', '')}")
    print(f"Company:  {job.get('company', '')}")
    print(f"Location: {job.get('location') or '-'}")
    print("-" * 70)
    print((job.get("description") or "")[:800])
    print("-" * 70)
    print("Your profile summary:")
    print(candidate_summary[:500])
    print("=" * 70)
    while True:
        raw = input_fn("Relevance [0/1/2, s=skip, q=quit]: ").strip().lower()
        if raw == "q":
            return "quit"
        if raw == "s":
            return "skip"
        if raw in ("0", "1", "2"):
            return int(raw)  # type: ignore[return-value]
        print("Please enter 0, 1, 2, s, or q.", file=sys.stderr)


def write_label(labels_dir: Path, entry: dict[str, Any]) -> Path:
    labels_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    path = labels_dir / f"{date_str}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return path


async def _fetch_jobs(limit: int) -> list[dict[str, Any]]:
    from sqlalchemy import desc, select

    from database.db import get_session
    from database.models import Job

    async with get_session() as session:
        rows = (
            await session.execute(
                # Shared jobs only: hand-added jobs are private to the account that added them.
                select(Job).where(Job.owner_user_id.is_(None)).order_by(desc(Job.found_at)).limit(limit)
            )
        ).scalars().all()
        return [r.to_dict() for r in rows]


async def _fetch_candidate_summary() -> tuple[str, str | None]:
    """Returns (summary_text, candidate_profile_id_or_None)."""
    from extraction.cv_extractor import ensure_candidate_profile_from_file

    result = await ensure_candidate_profile_from_file()
    if result is None:
        return "(no candidate profile available - run POST /cv or set CV_PATH)", None
    extraction, profile_id = result
    skills = ", ".join(s.name for s in extraction.skills[:15]) or "(none extracted)"
    summary = f"Skills: {skills}\nSeniority target: {extraction.seniority_target or '-'}"
    return summary, profile_id


async def run_labeling_session(
    limit: int,
    relabel: bool,
    labels_dir: Path = DEFAULT_LABELS_DIR,
    input_fn=input,
) -> int:
    """Core loop, factored out of main() so tests can drive it with a
    fake input_fn and a temp labels_dir - never the real one. Returns the
    number of labels written this session."""
    from database import init_db

    init_db()
    existing = {} if relabel else load_existing_labels(labels_dir)
    # Fetch extra, since some of the most-recent jobs may already be labeled.
    candidates = await _fetch_jobs(limit=max(limit * 3, limit))
    candidate_summary, candidate_profile_id = await _fetch_candidate_summary()

    written = 0
    for job in candidates:
        if written >= limit:
            break
        if job["id"] in existing:
            continue
        decision = prompt_relevance(job, candidate_summary, input_fn=input_fn)
        if decision == "quit":
            break
        if decision == "skip":
            continue
        entry = {
            "job_id": job["id"],
            "candidate_profile_id": candidate_profile_id,
            "relevance": decision,
            "notes": None,
            "labeled_at": datetime.now(UTC).isoformat(),
        }
        write_label(labels_dir, entry)
        written += 1
        print(f"Saved. ({written}/{limit} labeled this session)")

    print(f"\nDone - labeled {written} job(s) this session.")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20, help="Max jobs to label this session")
    parser.add_argument("--relabel", action="store_true", help="Include already-labeled jobs")
    args = parser.parse_args()
    asyncio.run(run_labeling_session(limit=args.limit, relabel=args.relabel))


if __name__ == "__main__":
    main()
