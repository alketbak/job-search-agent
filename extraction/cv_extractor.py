"""CV -> CandidateProfile extraction and persistence."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from extraction.llm_json import extract_with_retry, load_prompt
from extraction.schemas import (
    CandidateProfileExtraction,
    EducationEntry,
    EmploymentType,
    EvidenceLevel,
    ExperienceEntry,
    ExtractionFailed,
    LanguageProficiency,
    RemotePreference,
    SkillCategory,
    SkillClaim,
)

if TYPE_CHECKING:
    from database.models import CandidateProfile, CandidateSkill

log = logging.getLogger(__name__)

PROMPT_VERSION = "cv_extraction_v1"
SYSTEM_PROMPT = (
    "You are a precise, conservative resume analyst. You only report what the "
    "CV text explicitly supports with evidence. You never invent skills, "
    "roles, dates, or qualifications not present in the text."
)


async def extract_candidate_profile(
    cv_text: str, pipeline_run_id: str | None = None
) -> CandidateProfileExtraction:
    """Extract a `CandidateProfileExtraction` from raw CV text.

    Raises `ExtractionFailed` (never anything else) if extraction fails
    after one retry: invalid LLM JSON gets one retry with the validation
    error, then is marked failed - no crash. Never invents
    data not present in `cv_text` - enforced by the prompt (required
    evidence_snippet per skill), not by code; this function trusts the
    model's output once it's schema-valid.
    """
    if not cv_text or not cv_text.strip():
        raise ExtractionFailed("empty_cv_text", PROMPT_VERSION)

    template = load_prompt("cv_extraction_v1.txt")
    prompt = template.format(input_text=cv_text)
    return await extract_with_retry(
        model_cls=CandidateProfileExtraction,
        prompt_version=PROMPT_VERSION,
        prompt=prompt,
        system=SYSTEM_PROMPT,
        pipeline_run_id=pipeline_run_id,
    )


async def persist_candidate_profile(
    session: AsyncSession,
    user_id: str,
    cv_id: str,
    extraction: CandidateProfileExtraction,
) -> CandidateProfile:
    """Write a `CandidateProfile` + its `CandidateSkill` rows. Caller commits."""
    from database.models import CandidateProfile, CandidateSkill

    profile = CandidateProfile(
        user_id=user_id,
        cv_id=cv_id,
        languages=[lang.model_dump() for lang in extraction.languages],
        experience_entries=[exp.model_dump() for exp in extraction.experience],
        education=[edu.model_dump() for edu in extraction.education],
        certifications=extraction.certifications,
        location_preference=extraction.location_preference,
        remote_preference=extraction.remote_preference,
        employment_type=extraction.employment_type,
        seniority_target=extraction.seniority_target,
        prompt_version=PROMPT_VERSION,
    )
    session.add(profile)
    await session.flush()

    for skill in extraction.skills:
        session.add(
            CandidateSkill(
                candidate_profile_id=profile.id,
                skill_name=skill.name,
                category=skill.category,
                evidence_level=skill.evidence_level,
                evidence_snippet=skill.evidence_snippet,
            )
        )
    return profile


def _profile_extraction_from_rows(
    profile: CandidateProfile, skills: Sequence[CandidateSkill]
) -> CandidateProfileExtraction:
    """Reconstruct a `CandidateProfileExtraction` from DB rows - used when
    an unchanged CV's profile is cached and we skip re-extraction.

    The DB columns are plain `str`/`str | None` (SQLAlchemy has no Literal
    column type); the casts below just restore the narrower type these
    values already had before `persist_candidate_profile` wrote them - they
    were validated against these same Literals by the `SkillClaim`/
    `CandidateProfileExtraction` that produced them, not re-validated here.
    """
    return CandidateProfileExtraction(
        skills=[
            SkillClaim(
                name=s.skill_name,
                category=cast(SkillCategory, s.category),
                evidence_level=cast(EvidenceLevel, s.evidence_level),
                evidence_snippet=s.evidence_snippet or "",
            )
            for s in skills
        ],
        languages=[LanguageProficiency(**lang) for lang in (profile.languages or [])],
        experience=[ExperienceEntry(**exp) for exp in (profile.experience_entries or [])],
        education=[EducationEntry(**edu) for edu in (profile.education or [])],
        certifications=profile.certifications or [],
        location_preference=profile.location_preference,
        remote_preference=cast(RemotePreference | None, profile.remote_preference),
        employment_type=cast(EmploymentType | None, profile.employment_type),
        seniority_target=profile.seniority_target,
    )


async def ensure_candidate_profile_from_file(
    user_id: str | None = None,
    pipeline_run_id: str | None = None,
) -> tuple[CandidateProfileExtraction, str] | None:
    """Load `settings.cv_path`, extract a `CandidateProfileExtraction` if
    needed, and return it together with its DB `CandidateProfile.id` - the
    bridge between the file-based CV (`CV_PATH`, also used for
    embeddings) and the structured profile `POST /cv` produces.
    The id is needed so `MatchResult.candidate_profile_id` can link back to
    it (agents/orchestrator.py's persist_node).

    Mirrors `agents/retrieval.py::ensure_cv_chunks`'s content-hash caching:
    an unchanged CV reuses the existing profile instead of re-running the
    (paid) extraction call every pipeline run.

    Returns `None` - never raises - if the CV file is missing or
    extraction fails after its retry. Callers (the orchestrator) must
    treat `None` as "no candidate profile available yet" and skip scoring
    for this run rather than crash.
    """
    from config import settings
    from database.db import get_session
    from database.models import CV, CandidateProfile, CandidateSkill
    from database.seed import ensure_default_user

    user_id = user_id or await ensure_default_user()
    cv_path = Path(settings.cv_path)
    if not cv_path.exists():
        log.warning("CV file not found at %s - no candidate profile available", cv_path)
        return None
    cv_text = cv_path.read_text(encoding="utf-8")
    fingerprint = hashlib.sha256(cv_text.encode("utf-8")).hexdigest()

    async with get_session() as session:
        existing_cv = (
            (await session.execute(select(CV).where(CV.user_id == user_id, CV.content_hash == fingerprint)))
            .scalars()
            .first()
        )
        if existing_cv:
            existing_profile = (
                (
                    await session.execute(
                        select(CandidateProfile)
                        .where(CandidateProfile.cv_id == existing_cv.id)
                        .order_by(desc(CandidateProfile.created_at))
                        .limit(1)
                    )
                )
                .scalars()
                .first()
            )
            if existing_profile:
                skills = (
                    await session.execute(
                        select(CandidateSkill).where(CandidateSkill.candidate_profile_id == existing_profile.id)
                    )
                ).scalars().all()
                log.info("CV unchanged (content hash matches) - reusing existing candidate profile")
                return _profile_extraction_from_rows(existing_profile, skills), existing_profile.id
            cv = existing_cv
        else:
            # File-based ingestion (CV_PATH) - see the identical note in
            # agents/retrieval.py::ensure_cv_chunks.
            cv = CV(
                user_id=user_id,
                raw_text=cv_text,
                content_hash=fingerprint,
                file_type="text",
                consent_given_at=datetime.utcnow(),
            )
            session.add(cv)
            await session.flush()

        try:
            extraction = await extract_candidate_profile(cv_text, pipeline_run_id=pipeline_run_id)
        except ExtractionFailed as e:
            log.warning("CV-from-file extraction failed: %s", e.reason)
            return None

        profile = await persist_candidate_profile(session, user_id, cv.id, extraction)
        return extraction, profile.id


async def load_uploaded_candidate_profile(
    user_id: str,
) -> tuple[CandidateProfileExtraction, str, str] | None:
    """Return `(profile, candidate_profile_id, cv_id)` for the newest CV
    `user_id` uploaded that has a successfully extracted profile, or None.

    This is what per-account matching scores against. `CV_PATH` is only the
    operator's CV for runs with no signed-in user (the scheduler); reading
    it for a signed-in account would score them against someone else's CV.
    A newer upload whose extraction failed or is still pending is skipped,
    so the previous working CV keeps being used until the new one is ready.
    """
    from database.db import get_session
    from database.models import CV, CandidateProfile, CandidateSkill

    async with get_session() as session:
        row = (
            await session.execute(
                select(CandidateProfile, CV.id)
                .join(CV, CV.id == CandidateProfile.cv_id)
                .where(CV.user_id == user_id)
                .order_by(desc(CV.uploaded_at), desc(CandidateProfile.created_at))
                .limit(1)
            )
        ).first()
        if row is None:
            return None
        profile, cv_id = row
        skills = (
            await session.execute(
                select(CandidateSkill).where(CandidateSkill.candidate_profile_id == profile.id)
            )
        ).scalars().all()
        return _profile_extraction_from_rows(profile, skills), profile.id, cv_id
