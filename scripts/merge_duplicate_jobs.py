"""One-off cleanup: merge duplicate shared jobs into one row per job.

Before dedup existed, the seed pipeline ran twice and stored every
job twice. This finds shared jobs (`owner_user_id IS NULL`) that the
pipeline's own dedup keys say are the same job - same `canonical_url` or
same `content_hash` - keeps the earliest-found row of each group, moves
everything that points at the other rows onto it, and deletes the rest.

What moves to the surviving job:
  - applications and match results (repointed)
  - embeddings / requirements (repointed only if the survivor has none for
    that embedding model / no requirements at all; otherwise the
    duplicate's copies are dropped as redundant)
  - the legacy `Job.status`/`applied_at`, if the survivor is still `new`
    and the duplicate isn't

A group where one account tracks more than one of the copies is skipped
and reported, not merged: picking which application (with its interviews
and feedback) to keep is a human decision.

Hand-added private jobs are never touched.

Dry run by default. Usage:
    python scripts/merge_duplicate_jobs.py            # show what would change
    python scripts/merge_duplicate_jobs.py --apply    # do it
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from database.models import (  # noqa: E402
    Application,
    Job,
    JobEmbedding,
    JobRequirement,
    JobStatus,
    MatchResult,
)


@dataclass
class MergeReport:
    groups: list[tuple[Job, list[Job]]] = field(default_factory=list)  # (survivor, duplicates)
    skipped: list[tuple[list[Job], str]] = field(default_factory=list)  # (group, reason)
    moved: dict[str, int] = field(default_factory=dict)

    @property
    def jobs_removed(self) -> int:
        return sum(len(dups) for _, dups in self.groups)

    def count(self, key: str, n: int = 1) -> None:
        self.moved[key] = self.moved.get(key, 0) + n


def _group_duplicates(jobs: list[Job]) -> list[list[Job]]:
    """Union jobs that share a canonical URL or a content hash."""
    parent = {j.id: j.id for j in jobs}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first_by_key: dict[tuple[str, str], str] = {}
    for job in jobs:
        for key in (("url", job.canonical_url), ("hash", job.content_hash)):
            if not key[1]:
                continue
            if key in first_by_key:
                parent[find(job.id)] = find(first_by_key[key])
            else:
                first_by_key[key] = job.id

    groups: dict[str, list[Job]] = {}
    for job in jobs:
        groups.setdefault(find(job.id), []).append(job)
    return [g for g in groups.values() if len(g) > 1]


async def merge_duplicate_jobs(session: AsyncSession, apply: bool) -> MergeReport:
    """Find (and with `apply=True`, merge) duplicate shared jobs. The
    caller owns the transaction: nothing is committed here."""
    report = MergeReport()
    jobs = list(
        (await session.execute(select(Job).where(Job.owner_user_id.is_(None)))).scalars().all()
    )
    for group in _group_duplicates(jobs):
        group.sort(key=lambda j: (j.found_at, j.id))
        survivor, duplicates = group[0], group[1:]
        ids = [j.id for j in group]

        apps = list(
            (await session.execute(select(Application).where(Application.job_id.in_(ids)))).scalars().all()
        )
        users = [a.user_id for a in apps]
        if len(users) != len(set(users)):
            report.skipped.append((group, "an account tracks more than one copy"))
            continue
        report.groups.append((survivor, duplicates))
        if not apply:
            continue

        dup_ids = [j.id for j in duplicates]
        for app in apps:
            if app.job_id != survivor.id:
                app.job_id = survivor.id
                report.count("applications")
        for mr in (
            await session.execute(select(MatchResult).where(MatchResult.job_id.in_(dup_ids)))
        ).scalars().all():
            mr.job_id = survivor.id
            report.count("match_results")

        survivor_models = set(
            (
                await session.execute(
                    select(JobEmbedding.embedding_model).where(JobEmbedding.job_id == survivor.id)
                )
            ).scalars().all()
        )
        for emb in (
            await session.execute(select(JobEmbedding).where(JobEmbedding.job_id.in_(dup_ids)))
        ).scalars().all():
            if emb.embedding_model in survivor_models:
                await session.delete(emb)
                report.count("embeddings_dropped")
            else:
                emb.job_id = survivor.id
                survivor_models.add(emb.embedding_model)
                report.count("embeddings_moved")

        survivor_has_reqs = (
            await session.execute(select(JobRequirement.id).where(JobRequirement.job_id == survivor.id).limit(1))
        ).first() is not None
        for dup in duplicates:
            reqs = (
                await session.execute(select(JobRequirement).where(JobRequirement.job_id == dup.id))
            ).scalars().all()
            for req in reqs:
                if survivor_has_reqs:
                    await session.delete(req)
                    report.count("requirements_dropped")
                else:
                    req.job_id = survivor.id
                    report.count("requirements_moved")
            if reqs:
                survivor_has_reqs = True

            if survivor.status == JobStatus.NEW and dup.status != JobStatus.NEW:
                survivor.status = dup.status
                survivor.applied_at = survivor.applied_at or dup.applied_at

        await session.flush()  # repoints land before the rows they pointed at go
        for dup in duplicates:
            await session.delete(dup)
        await session.flush()
    return report


def _print_report(report: MergeReport, apply: bool) -> None:
    verb = "Merged" if apply else "Would merge"
    print(f"{verb} {len(report.groups)} duplicate group(s), removing {report.jobs_removed} job row(s).")
    for survivor, dups in report.groups:
        print(f"  keep {survivor.id}  {survivor.title} @ {survivor.company}  (+{len(dups)} copy)")
    for group, reason in report.skipped:
        print(f"  SKIPPED {group[0].title} @ {group[0].company}: {reason}")
    if apply and report.moved:
        print("  " + ", ".join(f"{k}: {v}" for k, v in sorted(report.moved.items())))
    if not apply and report.groups:
        print("Dry run - nothing changed. Re-run with --apply to merge.")


async def _main(apply: bool) -> None:
    from database.db import dispose_engine, get_session

    async with get_session() as session:  # commits on exit; dry run changes nothing
        report = await merge_duplicate_jobs(session, apply=apply)
    _print_report(report, apply)
    await dispose_engine()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="actually merge (default: dry run)")
    asyncio.run(_main(parser.parse_args().apply))
