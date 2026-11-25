"""Pydantic types for the deterministic hybrid score.

Field names mirror `database.models.MatchResult` exactly, so persisting a
`ScoreBreakdown` is a direct field-for-field copy.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

RequirementStatus = Literal["met", "partial", "missing", "unverifiable"]


class RequirementClassification(BaseModel):
    """One job requirement, classified against the candidate's evidence.

    `evidence` is a CV snippet or skill name that justifies the status -
    never populated for "missing", since there's nothing to point to.
    """

    text: str
    mandatory: bool
    category: str
    status: RequirementStatus
    evidence: str | None = None


class ScoreBreakdown(BaseModel):
    score_technical_skills: float
    score_experience: float
    score_responsibilities: float
    score_seniority: float
    score_location_language: float
    score_education: float
    score_final: float
    blockers: list[str] = []
    requirement_classifications: list[RequirementClassification] = []
