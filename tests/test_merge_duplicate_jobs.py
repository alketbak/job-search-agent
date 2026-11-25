"""scripts/merge_duplicate_jobs.py - one-off cleanup of duplicate shared jobs.

Each test works inside one session that is rolled back at the end, so the
merge never touches other tests' rows in the shared test database.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select

from agents.dedup import content_hash
from database import init_db
from database.db import _async_session_factory
from database.models import (
    Application,
    Job,
    JobEmbedding,
    JobRequirement,
    JobStatus,
    MatchResult,
    User,
)
from scripts.merge_duplicate_jobs import merge_duplicate_jobs


@pytest.fixture()
async def session() -> AsyncIterator:
    init_db()
    async with _async_session_factory() as s:
        yield s
        await s.rollback()


async def _user(session) -> str:
    user = User(email=f"merge-{uuid.uuid4().hex[:8]}@merge-test.local", hashed_password="!")
    session.add(user)
    await session.flush()
    return user.id


async def _pair(session, **dup_fields) -> tuple[Job, Job]:
    """Two copies of one job; the first was found earlier."""
    title = f"Dup {uuid.uuid4().hex[:8]}"
    h = content_hash(title, "Acme", "Munich")
    now = datetime.utcnow()
    first = Job(title=title, company="Acme", location="Munich", source="mock", content_hash=h,
                found_at=now - timedelta(minutes=1))
    second = Job(title=title, company="Acme", location="Munich", source="mock", content_hash=h,
                 found_at=now, **dup_fields)
    session.add_all([first, second])
    await session.flush()
    return first, second


def _my_groups(report, *job_ids: str):
    return [(s, d) for s, d in report.groups if s.id in job_ids]


@pytest.mark.asyncio
async def test_dry_run_reports_but_changes_nothing(session) -> None:
    first, second = await _pair(session)
    report = await merge_duplicate_jobs(session, apply=False)
    groups = _my_groups(report, first.id)
    assert [(s.id, [d.id for d in dups]) for s, dups in groups] == [(first.id, [second.id])]
    assert await session.get(Job, second.id) is not None


@pytest.mark.asyncio
async def test_apply_keeps_earliest_and_moves_everything(session) -> None:
    user_id = await _user(session)
    first, second = await _pair(session, status=JobStatus.APPLIED)
    session.add_all([
        Application(user_id=user_id, job_id=second.id),
        MatchResult(user_id=user_id, job_id=second.id, score_final=0.7),
        JobEmbedding(job_id=first.id, embedding_model="m1", embedding=[0.1]),
        JobEmbedding(job_id=second.id, embedding_model="m1", embedding=[0.2]),
        JobEmbedding(job_id=second.id, embedding_model="m2", embedding=[0.3]),
        JobRequirement(job_id=second.id, requirement_type="skill", text="Python", mandatory=True),
    ])
    await session.flush()

    report = await merge_duplicate_jobs(session, apply=True)
    assert _my_groups(report, first.id)
    assert await session.get(Job, second.id) is None

    def rows(model):
        return select(model).where(model.job_id.in_([first.id, second.id]))

    apps = (await session.execute(rows(Application))).scalars().all()
    mrs = (await session.execute(rows(MatchResult))).scalars().all()
    embs = (await session.execute(rows(JobEmbedding))).scalars().all()
    reqs = (await session.execute(rows(JobRequirement))).scalars().all()
    assert [a.job_id for a in apps] == [first.id]
    assert [m.job_id for m in mrs] == [first.id]
    # m1: survivor already had one, duplicate's dropped; m2: moved over.
    assert sorted((e.embedding_model, e.job_id) for e in embs) == [("m1", first.id), ("m2", first.id)]
    assert [(r.text, r.job_id) for r in reqs] == [("Python", first.id)]
    assert (await session.get(Job, first.id)).status == JobStatus.APPLIED


@pytest.mark.asyncio
async def test_group_tracked_twice_by_one_account_is_skipped(session) -> None:
    user_id = await _user(session)
    first, second = await _pair(session)
    session.add_all([Application(user_id=user_id, job_id=first.id), Application(user_id=user_id, job_id=second.id)])
    await session.flush()

    report = await merge_duplicate_jobs(session, apply=True)
    assert not _my_groups(report, first.id)
    assert any(group[0].id == first.id for group, _ in report.skipped)
    assert await session.get(Job, second.id) is not None


@pytest.mark.asyncio
async def test_private_jobs_are_never_merged(session) -> None:
    user_id = await _user(session)
    first, second = await _pair(session, owner_user_id=user_id)
    report = await merge_duplicate_jobs(session, apply=True)
    assert not _my_groups(report, first.id)
    assert await session.get(Job, second.id) is not None
