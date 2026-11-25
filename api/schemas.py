"""Pydantic request/response schemas for the API layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

JobStatusLiteral = Literal["new", "applied", "interviewing", "rejected", "offer"]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"


class RunRequest(BaseModel):
    query: str | None = Field(default=None, description="Free-text search query")
    location: str | None = Field(default=None, description="Location filter, e.g. 'Munich'")


class RunResponse(BaseModel):
    """`POST /run` enqueues a background job
    (agents/worker.py) instead of blocking on the pipeline - this is the
    202 response, not a finished result. Poll `GET /runs/{run_id}` for
    the real outcome (`jobs_found`/`jobs_analyzed`/`errors` live there,
    once `status` reaches `success`/`failed`)."""

    run_id: str
    status: Literal["pending"] = "pending"


class JobResponse(BaseModel):
    id: str
    title: str
    company: str
    location: str | None = None
    description: str | None = None
    url: str | None = None
    source: str
    # Set only on a job the caller added by hand (private to them).
    owner_user_id: str | None = None
    raw_score: float | None = None
    claude_score: int | None = None
    claude_reasoning: str | None = None
    skill_gaps: list[str] = []
    highlights: list[str] = []
    apply_recommendation: bool = False
    status: JobStatusLiteral = "new"
    found_at: datetime | None = None
    applied_at: datetime | None = None
    run_id: str | None = None


class JobListResponse(BaseModel):
    total: int
    jobs: list[JobResponse]


class JobStatusUpdate(BaseModel):
    status: JobStatusLiteral


ApplicationStatusLiteral = Literal[
    "interested",
    "saved",
    "applied",
    "screening",
    "interview",
    "technical_assessment",
    "offer",
    "rejected",
    "withdrawn",
]


class ApplicationResponse(BaseModel):
    id: str
    user_id: str | None = None
    job_id: str
    status: ApplicationStatusLiteral = "interested"
    status_changed_at: datetime | None = None
    applied_at: datetime | None = None
    salary_offered: float | None = None
    notes: str | None = None
    response_received: bool = False
    created_at: datetime | None = None


class SkillGapResponse(BaseModel):
    id: str
    skill_name: str
    frequency: int
    last_seen: datetime | None = None


class GraphNode(BaseModel):
    id: str
    name: str
    type: str
    observations: list[str] = []


class GraphLink(BaseModel):
    source: str
    target: str
    type: str


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    links: list[GraphLink]


class PipelineRunResponse(BaseModel):
    id: str
    status: Literal["pending", "running", "success", "failed"] = "success"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    query: str | None = None
    location: str | None = None
    jobs_found: int = 0
    jobs_scored: int = 0
    jobs_analyzed: int = 0
    errors: list[str] = []


class RunListResponse(BaseModel):
    runs: list[PipelineRunResponse]


class ErrorResponse(BaseModel):
    detail: str
    context: dict[str, Any] | None = None


# --- CV upload + structured extraction ---

ExtractionStatus = Literal["pending", "success", "failed"]


class CandidateSkillResponse(BaseModel):
    name: str
    category: str
    evidence_level: str
    evidence_snippet: str | None = None


class CandidateProfileResponse(BaseModel):
    id: str
    cv_id: str | None = None
    skills: list[CandidateSkillResponse] = []
    languages: list[dict[str, Any]] = []
    experience_entries: list[dict[str, Any]] = []
    education: list[dict[str, Any]] = []
    certifications: list[str] = []
    location_preference: str | None = None
    remote_preference: str | None = None
    employment_type: str | None = None
    seniority_target: str | None = None
    prompt_version: str | None = None
    created_at: datetime | None = None


class CVResponse(BaseModel):
    id: str
    version: int
    file_type: str
    content_hash: str
    uploaded_at: datetime | None = None
    consent_given_at: datetime | None = None
    extraction_status: ExtractionStatus | None = None
    extraction_failure_reason: str | None = None
    profile: CandidateProfileResponse | None = None


# --- deterministic hybrid scoring ---

class RequirementClassificationResponse(BaseModel):
    text: str
    mandatory: bool
    category: str
    status: Literal["met", "partial", "missing", "unverifiable"]
    evidence: str | None = None


class MatchResultResponse(BaseModel):
    id: str
    job_id: str
    candidate_profile_id: str | None = None
    pipeline_run_id: str | None = None
    score_technical_skills: float | None = None
    score_experience: float | None = None
    score_responsibilities: float | None = None
    score_seniority: float | None = None
    score_location_language: float | None = None
    score_education: float | None = None
    score_final: float | None = None
    blockers: list[str] = []
    requirement_classifications: list[RequirementClassificationResponse] = []
    llm_explanation: str | None = None
    llm_model: str | None = None
    prompt_version: str | None = None
    created_at: datetime | None = None


# --- application tracking ---

class ApplicationCreateRequest(BaseModel):
    job_id: str
    status: ApplicationStatusLiteral = "interested"
    notes: str | None = None


class ManualJobCreateRequest(BaseModel):
    """A job the user found themselves (not via a source connector) -
    saved as a shared `Job` row and tracked as the caller's application in
    one step. Length caps mirror the `jobs` table's column sizes."""

    title: str = Field(min_length=1, max_length=512)
    company: str = Field(min_length=1, max_length=256)
    location: str | None = Field(default=None, max_length=256)
    url: str | None = Field(default=None, max_length=1024)
    description: str | None = Field(default=None, max_length=50_000)
    status: ApplicationStatusLiteral = "interested"
    notes: str | None = None


ScoringStatusLiteral = Literal["queued", "no_cv", "quota_exceeded"]


class ScoringResponse(BaseModel):
    """Whether scoring a job against the caller's CV was started.

    `queued`: poll `GET /runs/{scoring_run_id}`. `no_cv`: the account has
    no processed CV yet. `quota_exceeded`: the daily LLM quota is used up."""

    scoring_status: ScoringStatusLiteral
    scoring_run_id: str | None = None


class ApplicationUpdateRequest(BaseModel):
    status: ApplicationStatusLiteral | None = None
    notes: str | None = None
    salary_offered: float | None = None
    response_received: bool | None = None


class InterviewCreateRequest(BaseModel):
    interview_date: datetime | None = None
    stage: str | None = None
    notes: str | None = None


class InterviewResponse(BaseModel):
    id: str
    application_id: str
    interview_date: datetime | None = None
    stage: str | None = None
    notes: str | None = None


class FeedbackCreateRequest(BaseModel):
    outcome: str | None = None
    notes: str | None = None


class FeedbackResponse(BaseModel):
    id: str
    application_id: str
    outcome: str | None = None
    notes: str | None = None
    created_at: datetime | None = None


class ApplicationDetailResponse(ApplicationResponse):
    job: JobResponse | None = None
    interviews: list[InterviewResponse] = []
    feedback_entries: list[FeedbackResponse] = []
    match_result: MatchResultResponse | None = None


class KanbanCard(BaseModel):
    application_id: str
    job_id: str
    job_title: str
    job_company: str
    status: ApplicationStatusLiteral
    status_changed_at: datetime | None = None
    score_final: float | None = None


class KanbanResponse(BaseModel):
    columns: dict[str, list[KanbanCard]]


# --- GDPR basics ---

class MeExportResponse(BaseModel):
    """Full export of everything this account owns (GDPR). Rows are each model's own `to_dict()` output
    verbatim (the real stored data, not a UI-trimmed shape) rather than a
    second, parallel set of export-specific fields to keep in sync."""

    user: dict[str, Any]
    cvs: list[dict[str, Any]]
    candidate_profiles: list[dict[str, Any]]
    applications: list[dict[str, Any]]
    interviews: list[dict[str, Any]]
    feedback_entries: list[dict[str, Any]]
    match_results: list[dict[str, Any]]
    jobs_added: list[dict[str, Any]] = []
    exported_at: datetime


class MeDeletionResponse(BaseModel):
    """Account deletion (GDPR). `langfuse_traces_note` names the one
    category this endpoint can't actually delete - see docs/PRIVACY.md."""

    deleted_counts: dict[str, int]
    langfuse_traces_note: str


class MeUsageResponse(BaseModel):
    """Per-user daily LLM quota usage and cost.
    `estimated_cost_usd` is `None` unless `ANTHROPIC_PRICING_JSON` is
    configured - cost tracking never guesses a price (same rule the
    evaluation harness follows)."""

    calls_today: int
    daily_quota: int
    quota_remaining: int
    estimated_cost_usd: float | None = None


class ManualJobCreateResponse(ApplicationResponse):
    """The new application, plus whether scoring it was started."""

    scoring_status: ScoringStatusLiteral
    scoring_run_id: str | None = None
