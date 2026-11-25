"""Pydantic boundary types for structured extraction.

These are the types the LLM is asked to produce and the types everything
downstream (persistence, scoring) consumes - never raw dicts.
Field shapes intentionally mirror the DB schema
(`database/models.py`'s `CandidateSkill`, `JobRequirement`) so mapping an
extraction result onto a row is a straight field-for-field copy.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EvidenceLevel = Literal["professional", "academic", "personal_project", "mentioned_only"]
SkillCategory = Literal["language", "framework", "database", "cloud", "tool", "soft_skill", "other"]
RemotePreference = Literal["remote", "hybrid", "onsite"]
EmploymentType = Literal["full_time", "part_time", "contract", "freelance", "internship"]
# "responsibility" is distinct from "experience"
# (years-of-experience requirements) - scoring/hybrid_scorer.py scores them
# with different logic (semantic similarity vs. years comparison).
RequirementCategory = Literal[
    "skill", "responsibility", "experience", "education", "language", "certification", "other"
]


class SkillClaim(BaseModel):
    """One skill claim extracted from a CV. Never invented - if the CV
    doesn't support it, it must not appear here."""

    name: str
    category: SkillCategory = "other"
    evidence_level: EvidenceLevel
    evidence_snippet: str = Field(description="Verbatim or near-verbatim CV text supporting this claim")


class LanguageProficiency(BaseModel):
    language: str
    level: str = Field(description="As stated in the CV — e.g. 'native', 'C1', 'fluent'. Never inferred.")


class ExperienceEntry(BaseModel):
    title: str
    company: str
    start_date: str | None = None
    end_date: str | None = None  # None/"present" for current roles
    description: str | None = None


class EducationEntry(BaseModel):
    degree: str
    institution: str
    field_of_study: str | None = None
    graduation_year: int | None = None


class CandidateProfileExtraction(BaseModel):
    """Structured CV extraction. Persisted onto `CandidateProfile` +
    `CandidateSkill` rows (see extraction/cv_extractor.py)."""

    skills: list[SkillClaim] = []
    languages: list[LanguageProficiency] = []
    experience: list[ExperienceEntry] = []
    education: list[EducationEntry] = []
    certifications: list[str] = []
    location_preference: str | None = None
    remote_preference: RemotePreference | None = None
    employment_type: EmploymentType | None = None
    seniority_target: str | None = None


class JobRequirementClaim(BaseModel):
    """One requirement extracted from a job ad. `mandatory=False` means
    "preferred"."""

    text: str
    mandatory: bool
    category: RequirementCategory = "other"


class JobProfileExtraction(BaseModel):
    """Structured job-ad extraction. Scalar fields are persisted onto `Job`
    columns, requirements as `JobRequirement` rows. A new scalar field here
    needs a matching `Job` column, or deduped jobs (which skip
    re-extraction) silently lose it."""

    title: str
    company: str | None = None
    location: str | None = None
    remote: RemotePreference | None = None
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    seniority: str | None = None
    years_experience_min: int | None = None
    employment_type: EmploymentType | None = None
    requirements: list[JobRequirementClaim] = []


class ExtractionFailed(Exception):
    """Raised when extraction fails after the one allowed retry.

    Carries enough to log/report without ever including raw CV or job-ad
    content.
    """

    def __init__(self, reason: str, prompt_version: str) -> None:
        self.reason = reason
        self.prompt_version = prompt_version
        super().__init__(f"Extraction failed ({prompt_version}): {reason}")
