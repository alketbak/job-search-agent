"""SQLAlchemy ORM models.

We use the 2.0 declarative style (`Mapped`, `mapped_column`) - it gives nicer
typing and future-proofs for SQLAlchemy 2.x deprecations.

The schema follows `docs/ARCHITECTURE.md` §3: users, cvs, cv_chunks,
candidate_profiles, candidate_skills, jobs, job_requirements,
job_embeddings, match_results, applications, interviews, feedback,
skill_gaps, pipeline_runs. Every user-owned table carries a `user_id`.
`jobs`/`job_requirements`/`job_embeddings` are shared across users because
extraction and embeddings are cached per job content - except jobs a user
added by hand, which are private to them (`Job.owner_user_id`).

`CVChunk` holds raw CV text chunks for embedding-based retrieval, a layer
below the structured `CandidateProfile` that extraction produces.

Vector columns use `pgvector.sqlalchemy.Vector`, which only compiles against
the `postgresql` dialect. `.with_variant(JSON(), "sqlite")` keeps
`Base.metadata.create_all()` working against the SQLite test database (fast
unit tests) - on SQLite the column is just a JSON blob of
floats, with no real similarity search. Anything that needs actual vector
search must run against Postgres.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from fastapi_users.db import SQLAlchemyBaseUserTable
from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Boolean, DateTime, Enum, Float, ForeignKey, Integer, String, Text, or_
from sqlalchemy.orm import Mapped, mapped_column, relationship

from config import settings
from database.db import Base


def _uuid_str() -> str:
    """UUID4 as a string - portable across SQLite/Postgres without extension loads."""
    return str(uuid.uuid4())


def _embedding_type() -> Any:
    """Vector column type: real `vector(n)` on Postgres, JSON fallback on SQLite."""
    return Vector(settings.embedding_dim).with_variant(JSON(), "sqlite")


class JobStatus(str, enum.Enum):
    NEW = "new"
    APPLIED = "applied"
    INTERVIEWING = "interviewing"
    REJECTED = "rejected"
    OFFER = "offer"


class ApplicationStatus(str, enum.Enum):
    """Richer lifecycle than `JobStatus` - a job can
    exist without an application; once one exists, its own status is the
    source of truth for where things stand. `JobStatus` is kept as-is for
    the pre-existing `/jobs` API shape (see `database/models.py::Job`'s
    legacy-column comment) rather than migrated, so the split didn't need a
    breaking API change."""

    INTERESTED = "interested"
    SAVED = "saved"
    APPLIED = "applied"
    SCREENING = "screening"
    INTERVIEW = "interview"
    TECHNICAL_ASSESSMENT = "technical_assessment"
    OFFER = "offer"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


class User(SQLAlchemyBaseUserTable[str], Base):
    """Auth user (`SQLAlchemyBaseUserTable[str]` supplies
    `email`/`hashed_password`/`is_active`/`is_superuser`/`is_verified` -
    the `[str]` type param matches our existing uuid4-as-string `id`
    column, not fastapi-users' default UUID type, so no primary-key
    migration was needed for the pre-existing `users` table).

    `is_default` predates auth (the single-default-user bootstrap, see `database/seed.py::ensure_default_user`) and is kept only for
    paths with no authenticated caller (the scheduler's legacy fallback,
    tests) - see docs/DEVELOPMENT.md's pipeline notes before adding a new caller of
    `ensure_default_user`.
    """

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "is_default": self.is_default,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class CV(Base):
    """A versioned CV, uploaded via `POST /cv` - or, for the default user
    only, read from `settings.cv_path` (see agents/retrieval.py)."""

    __tablename__ = "cvs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, default=1)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    file_type: Mapped[str] = mapped_column(String(32), default="text")
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    # So GET /cv/{id} can report *why* extraction failed after the fact,
    # not just that it did.
    last_extraction_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    last_extraction_failure_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # GDPR: consent on CV upload. Set by api/router.py::upload_cv at the moment of upload - the
    # endpoint rejects the request with 400 if consent isn't given, so a
    # CV row can never exist without this being populated. A timestamp,
    # not just a bool, so "when did they consent" is answerable later.
    consent_given_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    chunks: Mapped[list[CVChunk]] = relationship(back_populates="cv", cascade="all, delete-orphan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "version": self.version,
            "file_type": self.file_type,
            "content_hash": self.content_hash,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
            "consent_given_at": self.consent_given_at.isoformat() if self.consent_given_at else None,
        }


class CVChunk(Base):
    """One embedded chunk of a CV's raw text - the pgvector replacement for
    the old per-process FAISS index built in `agents/rag_agent.py`."""

    __tablename__ = "cv_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    cv_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("cvs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding = mapped_column(_embedding_type(), nullable=True)

    cv: Mapped[CV] = relationship(back_populates="chunks")


class CandidateProfile(Base):
    """Structured CV extraction (Pydantic `CandidateProfile` at the API
    boundary)."""

    __tablename__ = "candidate_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    cv_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("cvs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    languages: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    experience_entries: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    education: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    certifications: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    location_preference: Mapped[str | None] = mapped_column(String(256), nullable=True)
    remote_preference: Mapped[str | None] = mapped_column(String(64), nullable=True)
    employment_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    seniority_target: Mapped[str | None] = mapped_column(String(64), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    skills: Mapped[list[CandidateSkill]] = relationship(
        back_populates="candidate_profile", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "cv_id": self.cv_id,
            "languages": self.languages,
            "experience_entries": self.experience_entries,
            "education": self.education,
            "certifications": self.certifications,
            "location_preference": self.location_preference,
            "remote_preference": self.remote_preference,
            "employment_type": self.employment_type,
            "seniority_target": self.seniority_target,
            "prompt_version": self.prompt_version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class CandidateSkill(Base):
    """One skill claim extracted from a CV, with its evidence level.

    `evidence_level` is one of professional | academic | personal_project |
    mentioned_only; `category` is one of language | framework | database |
    cloud | tool | soft_skill | other - both match
    `extraction.schemas.SkillClaim` exactly. Not
    enforced as DB enums since the Pydantic `Literal` at the extraction
    boundary is where it matters; a CHECK constraint can be added later
    without a data migration if needed.
    """

    __tablename__ = "candidate_skills"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    candidate_profile_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("candidate_profiles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    skill_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(32), nullable=False, default="other")
    evidence_level: Mapped[str] = mapped_column(String(32), nullable=False)
    evidence_snippet: Mapped[str | None] = mapped_column(Text, nullable=True)

    candidate_profile: Mapped[CandidateProfile] = relationship(back_populates="skills")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    company: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    location: Mapped[str] = mapped_column(String(256), nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    url: Mapped[str] = mapped_column(String(1024), nullable=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # NULL = a shared job from a source connector, visible to every account.
    # Set = a job a user added by hand (`POST /jobs`), visible only to that
    # user and never matched by the pipeline's dedup (see agents/dedup.py).
    owner_user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True
    )

    # Dedup keys - canonical URL first, then a
    # normalized (company, title, location) hash; embedding similarity is
    # the final fallback and isn't a column (computed at dedup time against
    # job_embeddings). See agents/dedup.py.
    canonical_url: Mapped[str | None] = mapped_column(String(1024), nullable=True, index=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    # Scalar fields from JobProfileExtraction - added
    # now specifically so a deduped/cached job (which skips re-extraction,
    # see agents/dedup.py + agents/orchestrator.py's extract_node) still has
    # something for scoring/hybrid_scorer.py's seniority/experience/
    # location components to read, instead of losing this data between
    # runs.
    seniority: Mapped[str | None] = mapped_column(String(64), nullable=True)
    remote: Mapped[str | None] = mapped_column(String(16), nullable=True)
    employment_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    salary_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    salary_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    salary_currency: Mapped[str | None] = mapped_column(String(8), nullable=True)
    years_experience_min: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # RAG score: max cosine similarity between this job's embedding and any CV chunk.
    raw_score: Mapped[float] = mapped_column(Float, nullable=True)
    # Legacy column names from the old design where the LLM decided the score.
    # Repurposed rather than migrated (docs/ARCHITECTURE.md
    # goal 3: "Uses the LLM to explain results, never to decide the
    # score") - claude_score now mirrors the deterministic
    # `round(score_final * 10)`, claude_reasoning holds the LLM explanation
    # of that already-computed score, not an LLM-decided score itself. The
    # full breakdown lives in `match_results`; these two columns exist only
    # so the pre-existing `/jobs` API response shape doesn't break.
    claude_score: Mapped[int] = mapped_column(Integer, nullable=True)
    claude_reasoning: Mapped[str] = mapped_column(Text, nullable=True)
    # JSON blobs for structured lists - SQLite stores as TEXT, but SQLA handles it.
    skill_gaps: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    highlights: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    apply_recommendation: Mapped[bool] = mapped_column(default=False)

    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, native_enum=False), default=JobStatus.NEW, index=True
    )

    found_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    applied_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    run_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("pipeline_runs.id", ondelete="SET NULL"), nullable=True, index=True
    )

    applications: Mapped[list[Application]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    requirements: Mapped[list[JobRequirement]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    embeddings: Mapped[list[JobEmbedding]] = relationship(
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
            "owner_user_id": self.owner_user_id,
            "canonical_url": self.canonical_url,
            "content_hash": self.content_hash,
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


def job_visible_to(user_id: str | None) -> Any:
    """SQL filter for the jobs `user_id` may see: every shared job plus
    their own hand-added ones. None means shared jobs only."""
    if user_id is None:
        return Job.owner_user_id.is_(None)
    return or_(Job.owner_user_id.is_(None), Job.owner_user_id == user_id)


class JobRequirement(Base):
    """One extracted requirement from a job ad (mandatory or preferred),
    written by job extraction."""

    __tablename__ = "job_requirements"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    requirement_type: Mapped[str] = mapped_column(String(64), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    mandatory: Mapped[bool] = mapped_column(Boolean, default=False)
    prompt_version: Mapped[str | None] = mapped_column(String(32), nullable=True)

    job: Mapped[Job] = relationship(back_populates="requirements")


class JobEmbedding(Base):
    """Job description embedding(s) for pgvector retrieval.

    A job can have more than one row if more than one `embedding_model` is
    in use (e.g. the MiniLM-vs-e5 comparison in evaluation/).
    """

    __tablename__ = "job_embeddings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding = mapped_column(_embedding_type(), nullable=True)

    job: Mapped[Job] = relationship(back_populates="embeddings")


class MatchResult(Base):
    """Deterministic hybrid score + LLM explanation for one (candidate, job)
    pair."""

    __tablename__ = "match_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    candidate_profile_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("candidate_profiles.id", ondelete="SET NULL"), nullable=True, index=True
    )
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    pipeline_run_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("pipeline_runs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    score_technical_skills: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_experience: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_responsibilities: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_seniority: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_location_language: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_education: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_final: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    blockers: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    requirement_classifications: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    llm_explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "candidate_profile_id": self.candidate_profile_id,
            "job_id": self.job_id,
            "pipeline_run_id": self.pipeline_run_id,
            "score_technical_skills": self.score_technical_skills,
            "score_experience": self.score_experience,
            "score_responsibilities": self.score_responsibilities,
            "score_seniority": self.score_seniority,
            "score_location_language": self.score_location_language,
            "score_education": self.score_education,
            "score_final": self.score_final,
            "blockers": self.blockers or [],
            "requirement_classifications": self.requirement_classifications or [],
            "llm_explanation": self.llm_explanation,
            "llm_model": self.llm_model,
            "prompt_version": self.prompt_version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Application(Base):
    """`status` is the lifecycle owner (9 values, see
    `ApplicationStatus`) - `applied_at` only gets set when a transition
    actually reaches "applied", not at row creation (a user can create one
    at "interested" or "saved" long before applying).
    """

    __tablename__ = "applications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[ApplicationStatus] = mapped_column(
        Enum(ApplicationStatus, native_enum=False), default=ApplicationStatus.INTERESTED, index=True
    )
    status_changed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    applied_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    salary_offered: Mapped[float | None] = mapped_column(Float, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_received: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    job: Mapped[Job] = relationship(back_populates="applications")
    interviews: Mapped[list[Interview]] = relationship(
        back_populates="application", cascade="all, delete-orphan"
    )
    feedback_entries: Mapped[list[Feedback]] = relationship(
        back_populates="application", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "job_id": self.job_id,
            "status": self.status.value if isinstance(self.status, ApplicationStatus) else self.status,
            "status_changed_at": self.status_changed_at.isoformat() if self.status_changed_at else None,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "salary_offered": self.salary_offered,
            "notes": self.notes,
            "response_received": self.response_received,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Interview(Base):
    """One interview stage for an application."""

    __tablename__ = "interviews"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    application_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("applications.id", ondelete="CASCADE"), nullable=False, index=True
    )
    interview_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    application: Mapped[Application] = relationship(back_populates="interviews")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "application_id": self.application_id,
            "interview_date": self.interview_date.isoformat() if self.interview_date else None,
            "stage": self.stage,
            "notes": self.notes,
        }


class Feedback(Base):
    """Outcome feedback for an application - a signal, never an automatic
    relevance label: rejections have many causes unrelated to fit."""

    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    application_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("applications.id", ondelete="CASCADE"), nullable=False, index=True
    )
    outcome: Mapped[str | None] = mapped_column(String(64), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    application: Mapped[Application] = relationship(back_populates="feedback_entries")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "application_id": self.application_id,
            "outcome": self.outcome,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class SkillGap(Base):
    """Aggregated skill gaps across all analyzed jobs. Useful for dashboard.

    `docs/ARCHITECTURE.md` notes the target design computes this from
    `match_results.requirement_classifications` instead; the table is kept
    until that replacement exists so `/skill-gaps` keeps working.
    """

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
    """Audit log of each pipeline invocation - lets the /runs endpoint work.

    `status` (added when `POST /run` moved to the arq background worker): `pending` from the moment the API enqueues
    the job, `running` once the worker picks it up, then `success`/
    `failed` - lets `GET /runs/{run_id}` report real progress instead of
    the row simply not existing until the whole thing finishes. Rows
    written by the old fully-synchronous path (and this column's
    migration backfill for rows that predate it) go straight to
    `success`/`failed` since there was never an in-between state to
    observe.
    """

    __tablename__ = "pipeline_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    status: Mapped[str] = mapped_column(String(16), default="pending", nullable=False)
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
            "user_id": self.user_id,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "query": self.query,
            "location": self.location,
            "jobs_found": self.jobs_found,
            "jobs_scored": self.jobs_scored,
            "jobs_analyzed": self.jobs_analyzed,
            "errors": self.errors or [],
        }
