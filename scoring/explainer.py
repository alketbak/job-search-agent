"""LLM explanation of an already-computed deterministic score.

Writes prose ONLY from the structured `ScoreBreakdown` - never given raw
CV or job-ad text - so it cannot "explain" anything the deterministic
scorer didn't already classify: the explanation is written only from
the structured match result. No retry: this is free text, not
schema-validated JSON, so there's nothing to retry against - a failure
just falls back to a generic sentence.

`generate_motivation` also lives here (moved from the retired
`agents/analyst_agent.py`, which this module's `generate_explanation`
replaces) - it's the same "cheap one-shot LLM prose for the digest" family
of call, just not tied to a specific job's score.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from config import settings
from observability.tracing import record_llm_call
from scoring.schemas import ScoreBreakdown

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

log = logging.getLogger(__name__)

_client: AsyncAnthropic | None = None

SYSTEM_PROMPT = (
    "You are a career coach summarizing a job match for a candidate. You are "
    "given ONLY a structured score breakdown — not the original CV or job "
    "text — and must write your explanation strictly from these numbers and "
    "classifications. Never invent details not present in the data."
)

USER_TEMPLATE = """{breakdown_text}

Write a short explanation (3-5 sentences) covering:
1. Why this score, in plain language
2. Any blockers and what they mean
3. What would most improve the fit

Base this ONLY on the data above. Do not mention information not given here."""


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        from anthropic import AsyncAnthropic

        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set - required for explanations.")
        _client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


def _format_breakdown_for_prompt(breakdown: ScoreBreakdown, job_title: str, job_company: str | None) -> str:
    lines = [
        f"Job: {job_title}" + (f" at {job_company}" if job_company else ""),
        f"Final score: {breakdown.score_final:.2f} (0-1 scale)",
        "Component scores:",
        f"  technical_skills={breakdown.score_technical_skills:.2f}",
        f"  experience={breakdown.score_experience:.2f}",
        f"  responsibilities={breakdown.score_responsibilities:.2f}",
        f"  seniority={breakdown.score_seniority:.2f}",
        f"  location_language={breakdown.score_location_language:.2f}",
        f"  education={breakdown.score_education:.2f}",
    ]
    if breakdown.blockers:
        lines.append("Blockers (missing mandatory requirements):")
        lines.extend(f"  - {b}" for b in breakdown.blockers)
    met = [c for c in breakdown.requirement_classifications if c.status == "met"]
    gaps = [c for c in breakdown.requirement_classifications if c.status in ("missing", "partial")]
    if met:
        lines.append("Requirements met:")
        lines.extend(f"  - {c.text}" for c in met[:10])
    if gaps:
        lines.append("Requirements not fully met:")
        lines.extend(f"  - {c.text} ({c.status})" for c in gaps[:10])
    return "\n".join(lines)


def _fallback_explanation(breakdown: ScoreBreakdown) -> str:
    if breakdown.blockers:
        return (
            f"Score {breakdown.score_final:.2f} - capped due to missing mandatory "
            f"requirement(s): {', '.join(breakdown.blockers[:3])}."
        )
    return f"Score {breakdown.score_final:.2f} based on the deterministic match breakdown."


async def generate_explanation(
    breakdown: ScoreBreakdown,
    job_title: str,
    job_company: str | None = None,
    pipeline_run_id: str | None = None,
) -> str:
    """Write a short explanation from the structured score only. Never
    raises - falls back to a generic sentence on any LLM failure."""
    started_at = time.perf_counter()
    try:
        client = _get_client()
    except RuntimeError:
        return _fallback_explanation(breakdown)

    prompt = USER_TEMPLATE.format(
        breakdown_text=_format_breakdown_for_prompt(breakdown, job_title, job_company)
    )
    text: str | None = None
    input_tokens = output_tokens = None
    success = False
    error: str | None = None
    try:
        msg = await client.messages.create(
            model=settings.llm_model_smart,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        # `getattr`, not `b.text`, since content blocks are `TextBlock |
        # ToolUseBlock` - only TextBlock has `.text` - and tests mock blocks
        # as plain SimpleNamespace objects rather than real SDK classes.
        text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text")
        usage = getattr(msg, "usage", None)
        if usage is not None:
            input_tokens = getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)
        success = True
    except Exception as e:
        log.warning("Explanation generation failed (%s)", type(e).__name__)
        error = f"llm_call_error:{type(e).__name__}"
    finally:
        record_llm_call(
            name="explanation",
            model=settings.llm_model_smart,
            prompt_version="explanation_v1",
            prompt_text=prompt,
            completion_text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=time.perf_counter() - started_at,
            attempts=1,
            success=success,
            error=error,
            pipeline_run_id=pipeline_run_id,
        )
    return (text or "").strip() or _fallback_explanation(breakdown)


async def generate_motivation(top_jobs: list[dict[str, Any]], pipeline_run_id: str | None = None) -> str:
    """Short motivational line for the daily digest. Cheap one-shot call."""
    if not top_jobs:
        return "No strong matches today - keep refining your CV and try again tomorrow."
    started_at = time.perf_counter()
    try:
        client = _get_client()
    except RuntimeError:
        return "Stay focused - great opportunities reward consistency."

    titles = ", ".join(f"{j.get('title')} @ {j.get('company')}" for j in top_jobs[:3])
    prompt = (
        "Write one short motivational sentence (under 25 words) for a job seeker "
        f"whose top matches today are: {titles}. No preamble. Just the sentence."
    )
    text: str | None = None
    input_tokens = output_tokens = None
    success = False
    error: str | None = None
    try:
        msg = await client.messages.create(
            model=settings.llm_model_smart,
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        )
        # `getattr`, not `b.text`, since content blocks are `TextBlock |
        # ToolUseBlock` - only TextBlock has `.text` - and tests mock blocks
        # as plain SimpleNamespace objects rather than real SDK classes.
        text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text")
        usage = getattr(msg, "usage", None)
        if usage is not None:
            input_tokens = getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)
        success = True
    except Exception as e:
        log.warning("Motivation generation failed (%s)", type(e).__name__)
        error = f"llm_call_error:{type(e).__name__}"
    finally:
        record_llm_call(
            name="motivation",
            model=settings.llm_model_smart,
            prompt_version="motivation_v1",
            prompt_text=prompt,
            completion_text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=time.perf_counter() - started_at,
            attempts=1,
            success=success,
            error=error,
            pipeline_run_id=pipeline_run_id,
        )
    return (text or "").strip() or "Today's shortlist is strong - keep your momentum."
