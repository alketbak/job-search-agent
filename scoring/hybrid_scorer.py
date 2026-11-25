"""Deterministic hybrid scoring.

Pure, synchronous, no I/O - no LLM, no DB, no model loading. Callers
precompute embeddings (CV chunks, job requirement texts) once per run and
pass them in as plain `list[float]` vectors; this module just does the
arithmetic. That split is what makes `score_match` trivially unit-testable
with canned embeddings and fully deterministic (same inputs -> same
score).

Scoring approach, per requirement category:
- "skill" / "certification" / "other": name-match against the candidate's
  skills (weighted by evidence_level) first; semantic similarity against
  CV chunks as a fallback (catches synonyms FAISS-style name matching
  would miss, e.g. "container orchestration" vs "Kubernetes").
- "responsibility": semantic similarity only - these are full sentences,
  not named entities, so there's no name pool to match against.
- "education" / "language": name-match against the candidate's education
  entries / spoken languages.
- "experience": years-of-experience comparison, not evidence matching -
  parses a "N years" figure from the requirement text and compares against
  years estimated from the candidate's experience entries.

Missing mandatory requirements become explicit blockers and cap the final
score, so they're never hidden in a high average.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

import numpy as np

from config import settings
from extraction.schemas import (
    CandidateProfileExtraction,
    ExperienceEntry,
    JobProfileExtraction,
    JobRequirementClaim,
)
from scoring.schemas import RequirementClassification, RequirementStatus, ScoreBreakdown

EVIDENCE_WEIGHT = {
    "professional": 1.0,
    "academic": 0.8,
    "personal_project": 0.6,
    "mentioned_only": 0.4,
}
_STATUS_SCORE = {"met": 1.0, "partial": 0.6, "missing": 0.0, "unverifiable": 0.5}
_SENIORITY_RANK = {
    "intern": 0,
    "junior": 1,
    "entry": 1,
    "mid": 2,
    "mid-level": 2,
    "intermediate": 2,
    "senior": 3,
    "staff": 4,
    "lead": 4,
    "principal": 5,
}

CVChunk = tuple[str, list[float]]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _best_semantic_match(
    requirement_embedding: list[float] | None, cv_chunks: list[CVChunk]
) -> tuple[str | None, float]:
    if requirement_embedding is None or not cv_chunks:
        return None, 0.0
    best_text, best_sim = None, 0.0
    for text, embedding in cv_chunks:
        sim = _cosine(requirement_embedding, embedding)
        if sim > best_sim:
            best_text, best_sim = text, sim
    return best_text, best_sim


def _evidence_pool(category: str, candidate: CandidateProfileExtraction) -> list[tuple[str, float, str]]:
    """(name, evidence_weight, snippet) tuples relevant to this category."""
    if category in ("skill", "other"):
        return [(s.name, EVIDENCE_WEIGHT[s.evidence_level], s.evidence_snippet) for s in candidate.skills]
    if category == "certification":
        return [(c, 1.0, c) for c in candidate.certifications] + [
            (s.name, EVIDENCE_WEIGHT[s.evidence_level], s.evidence_snippet) for s in candidate.skills
        ]
    if category == "education":
        pool = []
        for e in candidate.education:
            pool.append((e.degree, 1.0, f"{e.degree} — {e.institution}"))
            if e.field_of_study:
                pool.append((e.field_of_study, 1.0, f"{e.degree} in {e.field_of_study} — {e.institution}"))
        return pool
    if category == "language":
        return [(lang.language, 1.0, f"{lang.language}: {lang.level}") for lang in candidate.languages]
    return []  # "responsibility" / "experience" use dedicated logic, not a name pool


def _classify_by_name_match(
    req: JobRequirementClaim,
    pool: list[tuple[str, float, str]],
    cv_chunks: list[CVChunk],
    requirement_embedding: list[float] | None,
) -> RequirementClassification:
    req_norm = _normalize(req.text)
    best: tuple[float, str] | None = None
    for name, weight, snippet in pool:
        name_norm = _normalize(name)
        if name_norm and (name_norm in req_norm or req_norm in name_norm):
            if best is None or weight > best[0]:
                best = (weight, snippet)

    if best is not None:
        status: RequirementStatus = "met" if best[0] >= 0.6 else "partial"
        return RequirementClassification(
            text=req.text, mandatory=req.mandatory, category=req.category, status=status, evidence=best[1]
        )

    chunk_text, sim = _best_semantic_match(requirement_embedding, cv_chunks)
    if chunk_text is not None and sim >= settings.semantic_match_threshold:
        return RequirementClassification(
            text=req.text,
            mandatory=req.mandatory,
            category=req.category,
            status="partial",
            evidence=chunk_text[:200],
        )

    return RequirementClassification(
        text=req.text, mandatory=req.mandatory, category=req.category, status="missing", evidence=None
    )


def _classify_responsibility(
    req: JobRequirementClaim, cv_chunks: list[CVChunk], requirement_embedding: list[float] | None
) -> RequirementClassification:
    chunk_text, sim = _best_semantic_match(requirement_embedding, cv_chunks)
    if chunk_text is None:
        return RequirementClassification(
            text=req.text, mandatory=req.mandatory, category=req.category, status="unverifiable", evidence=None
        )
    threshold = settings.semantic_match_threshold
    status: RequirementStatus
    if sim >= threshold + 0.15:
        status = "met"
    elif sim >= threshold:
        status = "partial"
    else:
        status = "missing"
    evidence = chunk_text[:200] if status != "missing" else None
    return RequirementClassification(
        text=req.text, mandatory=req.mandatory, category=req.category, status=status, evidence=evidence
    )


def _extract_year(date_str: str | None) -> int | None:
    if not date_str:
        return None
    match = re.search(r"(19|20)\d{2}", date_str)
    return int(match.group(0)) if match else None


def estimate_total_experience_years(experience: list[ExperienceEntry]) -> int | None:
    """Rough span estimate: (latest end year) - (earliest start year) across
    all experience entries. Not a sum of non-overlapping tenures - a
    deliberately simple heuristic, not a full employment-history parser."""
    spans = []
    now_year = datetime.now(UTC).year
    for entry in experience:
        start = _extract_year(entry.start_date)
        if start is None:
            continue
        end = _extract_year(entry.end_date) or now_year
        spans.append((start, end))
    if not spans:
        return None
    return max(0, max(e for _, e in spans) - min(s for s, _ in spans))


def _extract_years_from_text(text: str) -> int | None:
    match = re.search(r"(\d+)\+?\s*year", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _classify_experience(
    req: JobRequirementClaim, candidate: CandidateProfileExtraction
) -> RequirementClassification:
    required_years = _extract_years_from_text(req.text)
    total_years = estimate_total_experience_years(candidate.experience)
    if required_years is None or total_years is None:
        return RequirementClassification(
            text=req.text, mandatory=req.mandatory, category=req.category, status="unverifiable", evidence=None
        )
    evidence: str | None = f"~{total_years} years of experience estimated from CV"
    status: RequirementStatus
    if total_years >= required_years:
        status = "met"
    elif total_years >= required_years - 1:
        status = "partial"
    else:
        status, evidence = "missing", None
    return RequirementClassification(
        text=req.text, mandatory=req.mandatory, category=req.category, status=status, evidence=evidence
    )


def classify_requirement(
    req: JobRequirementClaim,
    candidate: CandidateProfileExtraction,
    cv_chunks: list[CVChunk],
    requirement_embedding: list[float] | None,
) -> RequirementClassification:
    if req.category == "responsibility":
        return _classify_responsibility(req, cv_chunks, requirement_embedding)
    if req.category == "experience":
        return _classify_experience(req, candidate)
    pool = _evidence_pool(req.category, candidate)
    return _classify_by_name_match(req, pool, cv_chunks, requirement_embedding)


def _weighted_avg(classifications: list[RequirementClassification], category: str) -> float:
    """Average match score for one category, mandatory items weighted 2x.
    Neutral 0.5 (no signal either way) if the job has no such requirements -
    never 0, which would unfairly read as "fails every unstated criterion"."""
    items = [c for c in classifications if c.category == category]
    if not items:
        return 0.5
    total_weight = sum(2.0 if c.mandatory else 1.0 for c in items)
    total_score = sum((2.0 if c.mandatory else 1.0) * _STATUS_SCORE[c.status] for c in items)
    return total_score / total_weight if total_weight else 0.5


def _score_experience(job: JobProfileExtraction, candidate: CandidateProfileExtraction) -> float:
    total_years = estimate_total_experience_years(candidate.experience)
    if job.years_experience_min is None:
        return 0.5 if total_years is None else min(1.0, 0.5 + 0.1 * min(total_years, 5))
    if total_years is None:
        return 0.3  # stated requirement, unverifiable candidate side - not a hard fail
    if total_years >= job.years_experience_min:
        return 1.0
    gap = job.years_experience_min - total_years
    return max(0.0, 1.0 - gap * 0.25)


def _seniority_rank(text: str | None) -> int | None:
    if not text:
        return None
    norm = _normalize(text)
    for key, rank in _SENIORITY_RANK.items():
        if key in norm:
            return rank
    return None


def _score_seniority(job: JobProfileExtraction, candidate: CandidateProfileExtraction) -> float:
    job_rank = _seniority_rank(job.seniority)
    candidate_rank = _seniority_rank(candidate.seniority_target)
    if job_rank is None or candidate_rank is None:
        return 0.5
    diff = abs(job_rank - candidate_rank)
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.6
    return 0.2


def _score_location_language(
    job: JobProfileExtraction,
    candidate: CandidateProfileExtraction,
    classifications: list[RequirementClassification],
) -> float:
    if job.location and candidate.location_preference:
        a, b = _normalize(job.location), _normalize(candidate.location_preference)
        location_score = 1.0 if (a in b or b in a) else 0.3
    else:
        location_score = 0.5

    if job.remote and candidate.remote_preference:
        if job.remote == candidate.remote_preference:
            remote_score = 1.0
        elif "hybrid" in (job.remote, candidate.remote_preference):
            remote_score = 0.5
        else:
            remote_score = 0.2
    else:
        remote_score = 0.5

    language_score = _weighted_avg(classifications, "language")
    return 0.5 * location_score + 0.25 * remote_score + 0.25 * language_score


def score_match(
    candidate: CandidateProfileExtraction,
    job: JobProfileExtraction,
    cv_chunks: list[CVChunk],
    requirement_embeddings: dict[str, list[float]],
) -> ScoreBreakdown:
    """Score one (candidate, job) pair. Deterministic - same inputs, same
    output, every time. `requirement_embeddings` maps requirement text ->
    embedding for every job requirement that needs semantic matching
    (callers typically embed all of them; a missing entry just means that
    requirement falls back to "missing"/"unverifiable" instead of trying
    semantic matching)."""
    classifications = [
        classify_requirement(req, candidate, cv_chunks, requirement_embeddings.get(req.text))
        for req in job.requirements
    ]

    technical_skills = _weighted_avg(classifications, "skill")
    responsibilities = _weighted_avg(classifications, "responsibility")
    education = _weighted_avg(classifications, "education")
    experience = _score_experience(job, candidate)
    seniority = _score_seniority(job, candidate)
    location_language = _score_location_language(job, candidate, classifications)

    final = (
        settings.score_weight_technical_skills * technical_skills
        + settings.score_weight_experience * experience
        + settings.score_weight_responsibilities * responsibilities
        + settings.score_weight_seniority * seniority
        + settings.score_weight_location_language * location_language
        + settings.score_weight_education * education
    )

    blockers = [c.text for c in classifications if c.mandatory and c.status == "missing"]
    if blockers:
        final = min(final, settings.score_blocker_cap)

    return ScoreBreakdown(
        score_technical_skills=round(technical_skills, 4),
        score_experience=round(experience, 4),
        score_responsibilities=round(responsibilities, 4),
        score_seniority=round(seniority, 4),
        score_location_language=round(location_language, 4),
        score_education=round(education, 4),
        score_final=round(final, 4),
        blockers=blockers,
        requirement_classifications=classifications,
    )
