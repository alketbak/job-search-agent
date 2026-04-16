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
    run_id: str
    jobs_found: int
    jobs_analyzed: int
    digest_path: str | None = None
    errors: list[str] = []


class JobResponse(BaseModel):
    id: str
    title: str
    company: str
    location: str | None = None
    description: str | None = None
    url: str | None = None
    source: str
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


class ApplicationResponse(BaseModel):
    id: str
    job_id: str
    applied_at: datetime | None = None
    notes: str | None = None
    response_received: bool = False


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
