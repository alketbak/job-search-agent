"""SQLAlchemy ORM models.

We use the 2.0 declarative style (`Mapped`, `mapped_column`) — it gives nicer
typing and future-proofs for SQLAlchemy 2.x deprecations.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.db import Base


def _uuid_str() -> str:
    """UUID4 as a string — portable across SQLite/Postgres without extension loads."""
    return str(uuid.uuid4())


class JobStatus(str, enum.Enum):
    NEW = "new"
    APPLIED = "applied"
    INTERVIEWING = "interviewing"
    REJECTED = "rejected"
    OFFER = "offer"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    company: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    location: Mapped[str] = mapped_column(String(256), nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    url: Mapped[str] = mapped_column(String(1024), nullable=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # RAG score: max cosine similarity between this job's embedding and any CV chunk.
    raw_score: Mapped[float] = mapped_column(Float, nullable=True)
    # Analyst score: Claude's 1-10 assessment with reasoning.
    claude_score: Mapped[int] = mapped_column(Integer, nullable=True)
    claude_reasoning: Mapped[str] = mapped_column(Text, nullable=True)
    # JSON blobs for structured lists — SQLite stores as TEXT, but SQLA handles it.
    skill_gaps: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    highlights: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    apply_recommendation: Mapped[bool] = mapped_column(default=False)

    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, native_enum=False), default=JobStatus.NEW, index=True
    )

    found_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    applied_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    run_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)

    applications: Mapped[list["Application"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "description": self.description,
            "url": self.url,
            "source": self.source,
            "raw_score": self.raw_score,
            "claude_score": self.claude_score,
            "claude_reasoning": self.claude_reasoning,
            "skill_gaps": self.skill_gaps or [],
            "highlights": self.highlights or [],
            "apply_recommendation": self.apply_recommendation,
            "status": self.status.value if isinstance(self.status, JobStatus) else self.status,
            "found_at": self.found_at.isoformat() if self.found_at else None,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "run_id": self.run_id,
        }


class Application(Base):
    __tablename__ = "applications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    applied_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_received: Mapped[bool] = mapped_column(default=False)

    job: Mapped[Job] = relationship(back_populates="applications")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "notes": self.notes,
            "response_received": self.response_received,
        }


class SkillGap(Base):
    """Aggregated skill gaps across all analyzed jobs. Useful for dashboard."""

    __tablename__ = "skill_gaps"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    skill_name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    frequency: Mapped[int] = mapped_column(Integer, default=1)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "skill_name": self.skill_name,
            "frequency": self.frequency,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
        }


class PipelineRun(Base):
    """Audit log of each pipeline invocation — lets the /runs endpoint work."""

    __tablename__ = "pipeline_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    query: Mapped[str | None] = mapped_column(String(256), nullable=True)
    location: Mapped[str | None] = mapped_column(String(256), nullable=True)
    jobs_found: Mapped[int] = mapped_column(Integer, default=0)
    jobs_scored: Mapped[int] = mapped_column(Integer, default=0)
    jobs_analyzed: Mapped[int] = mapped_column(Integer, default=0)
    errors: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "query": self.query,
            "location": self.location,
            "jobs_found": self.jobs_found,
            "jobs_scored": self.jobs_scored,
            "jobs_analyzed": self.jobs_analyzed,
            "errors": self.errors or [],
        }
