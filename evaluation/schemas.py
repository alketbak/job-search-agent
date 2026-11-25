"""Pydantic types for evaluation."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

RelevanceLabel = Literal[0, 1, 2]


class JobLabel(BaseModel):
    """One human-provided relevance label. The ONLY way these should ever
    be created is a real person running `evaluation/label_jobs.py` -
    never generated, simulated or inferred.
    """

    job_id: str
    candidate_profile_id: str | None = None
    relevance: RelevanceLabel
    notes: str | None = None
    labeled_at: datetime
