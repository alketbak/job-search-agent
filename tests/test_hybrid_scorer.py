"""Tests for scoring/hybrid_scorer.py.

Pure function, no DB, no LLM - every test here is a plain unit test with
canned embeddings. Covers a missing mandatory requirement and
determinism (same inputs -> same score, no LLM call involved).
"""

from __future__ import annotations

from extraction.schemas import (
    CandidateProfileExtraction,
    EducationEntry,
    ExperienceEntry,
    JobProfileExtraction,
    JobRequirementClaim,
    LanguageProficiency,
    SkillClaim,
)
from scoring.hybrid_scorer import estimate_total_experience_years, score_match


def _vec(seed: float, dim: int = 8) -> list[float]:
    """Tiny deterministic "embedding" for testing - real dim doesn't matter,
    cosine similarity math is dimension-agnostic."""
    return [seed] * dim


def test_scoring_is_deterministic() -> None:
    candidate = CandidateProfileExtraction(
        skills=[SkillClaim(name="Python", category="language", evidence_level="professional", evidence_snippet="x")]
    )
    job = JobProfileExtraction(
        title="AI Engineer",
        requirements=[JobRequirementClaim(text="Python", mandatory=True, category="skill")],
    )
    a = score_match(candidate, job, cv_chunks=[], requirement_embeddings={})
    b = score_match(candidate, job, cv_chunks=[], requirement_embeddings={})
    assert a.model_dump() == b.model_dump()


def test_missing_mandatory_requirement_creates_blocker_and_caps_score() -> None:
    from config import settings

    candidate = CandidateProfileExtraction(
        skills=[SkillClaim(name="Python", category="language", evidence_level="professional", evidence_snippet="x")],
        seniority_target="senior",
    )
    job = JobProfileExtraction(
        title="AI Engineer",
        seniority="senior",
        requirements=[
            JobRequirementClaim(text="Python", mandatory=True, category="skill"),
            JobRequirementClaim(text="Kubernetes", mandatory=True, category="skill"),
        ],
    )
    result = score_match(candidate, job, cv_chunks=[], requirement_embeddings={})
    assert "Kubernetes" in result.blockers
    assert result.score_final <= settings.score_blocker_cap


def test_preferred_missing_requirement_is_not_a_blocker() -> None:
    candidate = CandidateProfileExtraction(
        skills=[SkillClaim(name="Python", category="language", evidence_level="professional", evidence_snippet="x")]
    )
    job = JobProfileExtraction(
        title="AI Engineer",
        requirements=[
            JobRequirementClaim(text="Python", mandatory=True, category="skill"),
            JobRequirementClaim(text="Kubernetes", mandatory=False, category="skill"),
        ],
    )
    result = score_match(candidate, job, cv_chunks=[], requirement_embeddings={})
    assert result.blockers == []


def test_evidence_level_weighting_professional_is_met_mentioned_only_is_partial() -> None:
    job = JobProfileExtraction(
        title="AI Engineer",
        requirements=[
            JobRequirementClaim(text="Python", mandatory=True, category="skill"),
            JobRequirementClaim(text="Rust", mandatory=False, category="skill"),
        ],
    )
    candidate = CandidateProfileExtraction(
        skills=[
            SkillClaim(name="Python", category="language", evidence_level="professional", evidence_snippet="4 years"),
            SkillClaim(name="Rust", category="language", evidence_level="mentioned_only", evidence_snippet="listed"),
        ]
    )
    result = score_match(candidate, job, cv_chunks=[], requirement_embeddings={})
    by_text = {c.text: c for c in result.requirement_classifications}
    assert by_text["Python"].status == "met"
    assert by_text["Rust"].status == "partial"


def test_semantic_fallback_matches_skill_without_exact_name() -> None:
    """No candidate skill named "container orchestration" - but a CV chunk
    with a near-identical embedding should still register as a (weaker)
    partial match via the semantic fallback."""
    candidate = CandidateProfileExtraction(skills=[])
    job = JobProfileExtraction(
        title="Platform Engineer",
        requirements=[JobRequirementClaim(text="container orchestration", mandatory=True, category="skill")],
    )
    cv_chunks = [("Ran Kubernetes clusters in production for 3 years.", _vec(1.0))]
    requirement_embeddings = {"container orchestration": _vec(1.0)}  # identical vector -> similarity 1.0
    result = score_match(candidate, job, cv_chunks, requirement_embeddings)
    assert result.requirement_classifications[0].status == "partial"
    assert "Kubernetes" in (result.requirement_classifications[0].evidence or "")


def test_semantic_fallback_below_threshold_is_missing() -> None:
    candidate = CandidateProfileExtraction(skills=[])
    job = JobProfileExtraction(
        title="Platform Engineer",
        requirements=[JobRequirementClaim(text="container orchestration", mandatory=False, category="skill")],
    )
    cv_chunks = [("Baked bread as a hobby.", _vec(-1.0))]  # opposite vector -> similarity -1.0
    requirement_embeddings = {"container orchestration": _vec(1.0)}
    result = score_match(candidate, job, cv_chunks, requirement_embeddings)
    assert result.requirement_classifications[0].status == "missing"


def test_responsibility_classified_via_semantic_similarity() -> None:
    candidate = CandidateProfileExtraction()
    job = JobProfileExtraction(
        title="AI Engineer",
        requirements=[
            JobRequirementClaim(text="own the LLM evaluation pipeline", mandatory=True, category="responsibility")
        ],
    )
    cv_chunks = [("Built and maintained an LLM evaluation pipeline end to end.", _vec(1.0))]
    requirement_embeddings = {"own the LLM evaluation pipeline": _vec(1.0)}
    result = score_match(candidate, job, cv_chunks, requirement_embeddings)
    assert result.requirement_classifications[0].status == "met"
    assert result.requirement_classifications[0].category == "responsibility"


def test_responsibility_without_embedding_is_unverifiable() -> None:
    candidate = CandidateProfileExtraction()
    job = JobProfileExtraction(
        title="AI Engineer",
        requirements=[JobRequirementClaim(text="mentor junior engineers", mandatory=False, category="responsibility")],
    )
    result = score_match(candidate, job, cv_chunks=[], requirement_embeddings={})
    assert result.requirement_classifications[0].status == "unverifiable"


def test_experience_years_requirement_met() -> None:
    candidate = CandidateProfileExtraction(
        experience=[ExperienceEntry(title="Engineer", company="Acme", start_date="2018", end_date="2024")]
    )
    job = JobProfileExtraction(
        title="AI Engineer",
        requirements=[JobRequirementClaim(text="5+ years experience", mandatory=True, category="experience")],
    )
    result = score_match(candidate, job, cv_chunks=[], requirement_embeddings={})
    assert result.requirement_classifications[0].status == "met"
    assert result.blockers == []


def test_experience_years_requirement_missing() -> None:
    candidate = CandidateProfileExtraction(
        experience=[ExperienceEntry(title="Engineer", company="Acme", start_date="2023", end_date="2024")]
    )
    job = JobProfileExtraction(
        title="AI Engineer",
        requirements=[JobRequirementClaim(text="5+ years experience", mandatory=True, category="experience")],
    )
    result = score_match(candidate, job, cv_chunks=[], requirement_embeddings={})
    assert result.requirement_classifications[0].status == "missing"
    assert "5+ years experience" in result.blockers


def test_estimate_total_experience_years() -> None:
    entries = [
        ExperienceEntry(title="A", company="X", start_date="2018", end_date="2020"),
        ExperienceEntry(title="B", company="Y", start_date="2020", end_date="2024"),
    ]
    assert estimate_total_experience_years(entries) == 6


def test_estimate_total_experience_years_no_dates() -> None:
    assert estimate_total_experience_years([ExperienceEntry(title="A", company="X")]) is None


def test_education_requirement_matched_by_degree() -> None:
    candidate = CandidateProfileExtraction(
        education=[EducationEntry(degree="BSc Computer Science", institution="TU Munich")]
    )
    job = JobProfileExtraction(
        title="AI Engineer",
        requirements=[JobRequirementClaim(text="BSc Computer Science or related", mandatory=False, category="education")],
    )
    result = score_match(candidate, job, cv_chunks=[], requirement_embeddings={})
    assert result.requirement_classifications[0].status == "met"


def test_language_requirement_matched() -> None:
    candidate = CandidateProfileExtraction(languages=[LanguageProficiency(language="German", level="C1")])
    job = JobProfileExtraction(
        title="AI Engineer",
        requirements=[JobRequirementClaim(text="German", mandatory=True, category="language")],
    )
    result = score_match(candidate, job, cv_chunks=[], requirement_embeddings={})
    assert result.requirement_classifications[0].status == "met"
    assert result.blockers == []


def test_seniority_exact_match_scores_full() -> None:
    candidate = CandidateProfileExtraction(seniority_target="senior")
    job = JobProfileExtraction(title="AI Engineer", seniority="senior")
    result = score_match(candidate, job, cv_chunks=[], requirement_embeddings={})
    assert result.score_seniority == 1.0


def test_seniority_large_gap_scores_low() -> None:
    candidate = CandidateProfileExtraction(seniority_target="intern")
    job = JobProfileExtraction(title="AI Engineer", seniority="principal")
    result = score_match(candidate, job, cv_chunks=[], requirement_embeddings={})
    assert result.score_seniority < 0.5


def test_location_match_scores_high() -> None:
    candidate = CandidateProfileExtraction(location_preference="Berlin", remote_preference="hybrid")
    job = JobProfileExtraction(title="AI Engineer", location="Berlin, Germany", remote="hybrid")
    result = score_match(candidate, job, cv_chunks=[], requirement_embeddings={})
    assert result.score_location_language > 0.8


def test_no_requirements_at_all_defaults_neutral() -> None:
    candidate = CandidateProfileExtraction()
    job = JobProfileExtraction(title="AI Engineer")
    result = score_match(candidate, job, cv_chunks=[], requirement_embeddings={})
    assert result.score_technical_skills == 0.5
    assert result.score_responsibilities == 0.5
    assert result.score_education == 0.5
    assert result.blockers == []
