"""Analyst agent — Claude reasons about job/candidate fit for each top job.

We call Claude once per job with a structured prompt and ask for a JSON object
(score, reasoning, skill_gaps, highlights, apply_recommendation). Claude's
native JSON output is not 100% reliable, so we also strip fences and parse
defensively.

Concurrency is capped by `ANALYST_CONCURRENCY` to avoid hammering rate limits.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from config import settings

log = logging.getLogger(__name__)

# Imported lazily — keeps test startup cheap when analyst calls are mocked.
_client = None


SYSTEM_PROMPT = (
    "You are an expert technical recruiter and career coach specializing in "
    "AI/ML engineering roles in Germany. Analyze job-candidate fit objectively "
    "based only on the CV and job description provided."
)

# We instruct Claude to emit exactly the JSON we want and nothing else.
# The schema is also validated client-side in `_parse_response`.
USER_TEMPLATE = """CV Summary:
{cv_summary}

Job:
Title: {title}
Company: {company}
Location: {location}
Description:
{description}

Respond with a single JSON object using exactly these fields and no others:
{{
  "score": <integer 1-10, where 10 is a perfect fit>,
  "reasoning": "<2-3 sentences explaining the score>",
  "skill_gaps": ["<skill>", ...],
  "highlights": ["<cv strength that matches this job>", ...],
  "apply_recommendation": <true or false>
}}

Return ONLY the JSON object. No markdown fences, no preamble, no trailing text."""


def _get_client():
    """Lazy Anthropic client init so missing API key doesn't break imports."""
    global _client
    if _client is None:
        from anthropic import AsyncAnthropic

        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set — required for the analyst agent."
            )
        _client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


def _load_cv_summary(max_chars: int = 500) -> str:
    """Return first `max_chars` of the CV — enough signal, small token cost."""
    try:
        text = Path(settings.cv_path).read_text(encoding="utf-8")
    except OSError as e:
        log.warning("Could not read CV for analyst prompt: %s", e)
        return ""
    text = " ".join(text.split())  # collapse whitespace
    return text[:max_chars]


def _parse_response(raw: str) -> dict[str, Any] | None:
    """Extract a JSON object from Claude's response. Tolerant of code fences."""
    if not raw:
        return None
    # Strip markdown fences if Claude decided to wrap anyway.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else raw.strip()
    # Isolate the first balanced JSON object — guards against trailing prose.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as e:
        log.warning("Analyst JSON parse failed: %s (raw first 200 chars: %r)", e, raw[:200])
        return None


def _coerce_fields(parsed: dict[str, Any]) -> dict[str, Any]:
    """Clamp/cast to the expected types; never trust LLM output blindly."""
    try:
        score = int(parsed.get("score") or 0)
    except (TypeError, ValueError):
        score = 0
    score = max(1, min(10, score))

    skill_gaps = parsed.get("skill_gaps") or []
    highlights = parsed.get("highlights") or []
    if not isinstance(skill_gaps, list):
        skill_gaps = []
    if not isinstance(highlights, list):
        highlights = []

    return {
        "score": score,
        "reasoning": str(parsed.get("reasoning") or ""),
        "skill_gaps": [str(s) for s in skill_gaps][:20],
        "highlights": [str(s) for s in highlights][:20],
        "apply_recommendation": bool(parsed.get("apply_recommendation", False)),
    }


async def _analyze_single(
    job: dict[str, Any], cv_summary: str, semaphore: asyncio.Semaphore
) -> dict[str, Any]:
    """Analyze one job. Never raises — on error, annotates `error` key."""
    async with semaphore:
        try:
            client = _get_client()
        except RuntimeError as e:
            log.error("%s", e)
            return {**job, "claude_score": 0, "claude_reasoning": str(e), "error": str(e)}

        user_prompt = USER_TEMPLATE.format(
            cv_summary=cv_summary,
            title=job.get("title", ""),
            company=job.get("company", ""),
            location=job.get("location", ""),
            description=(job.get("description", "") or "")[:3000],
        )
        try:
            msg = await client.messages.create(
                model=settings.claude_model,
                max_tokens=700,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            # The SDK returns a list of content blocks; join any text blocks.
            raw = "".join(
                block.text for block in msg.content if getattr(block, "type", "") == "text"
            )
        except Exception as e:
            log.error("Analyst Claude call failed for %s @ %s: %s", job.get("title"), job.get("company"), e)
            return {**job, "claude_score": 0, "claude_reasoning": f"LLM error: {e}", "error": str(e)}

    parsed = _parse_response(raw) or {}
    coerced = _coerce_fields(parsed) if parsed else {
        "score": 0,
        "reasoning": "Failed to parse analyst response.",
        "skill_gaps": [],
        "highlights": [],
        "apply_recommendation": False,
    }

    return {
        **job,
        "claude_score": coerced["score"],
        "claude_reasoning": coerced["reasoning"],
        "skill_gaps": coerced["skill_gaps"],
        "highlights": coerced["highlights"],
        "apply_recommendation": coerced["apply_recommendation"],
    }


async def analyze_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Analyze all jobs with bounded concurrency. Returns analyzed jobs list."""
    if not jobs:
        return []
    cv_summary = _load_cv_summary()
    semaphore = asyncio.Semaphore(max(1, settings.analyst_concurrency))
    log.info("Analyzing %s jobs (concurrency=%s)", len(jobs), settings.analyst_concurrency)
    results = await asyncio.gather(
        *(_analyze_single(j, cv_summary, semaphore) for j in jobs), return_exceptions=False
    )
    return list(results)


async def generate_motivation(top_jobs: list[dict[str, Any]]) -> str:
    """Short motivational line for the daily digest. Cheap one-shot call."""
    if not top_jobs:
        return "No strong matches today — keep refining your CV and try again tomorrow."
    try:
        client = _get_client()
    except RuntimeError:
        return "Stay focused — great opportunities reward consistency."

    titles = ", ".join(f"{j.get('title')} @ {j.get('company')}" for j in top_jobs[:3])
    prompt = (
        "Write one short motivational sentence (under 25 words) for a job seeker "
        f"whose top matches today are: {titles}. No preamble. Just the sentence."
    )
    try:
        msg = await client.messages.create(
            model=settings.claude_model,
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return text.strip() or "Today's shortlist is strong — keep your momentum."
    except Exception as e:
        log.warning("Motivation generation failed: %s", e)
        return "Today's shortlist is strong — keep your momentum."
