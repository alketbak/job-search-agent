"""Shared Claude JSON-extraction helper - one retry on invalid JSON, then
mark failed. Both extraction/cv_extractor.py and
extraction/job_extractor.py use this; same retry contract, different
prompt template and Pydantic model.

Every call (success or failure, after all retries) is traced via
observability/tracing.py - model, prompt_version,
tokens, latency, attempts, and outcome. Tracing no-ops if Langfuse isn't
configured.

Logging note: validation-error detail (which can echo back fragments of
the input text Pydantic rejected) is fed back to the model for the retry
but is NEVER logged or placed on the raised exception - only a short,
generic failure category is: no CV content in logs.
This applies to job-ad content too, out of the same caution, even though
the rule's literal text only names CVs. (Langfuse tracing is a separate
path with its own PII redaction - see observability/tracing.py's `_mask`.)
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, ValidationError

from config import settings
from extraction.schemas import ExtractionFailed
from observability.tracing import record_llm_call

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

log = logging.getLogger(__name__)

_client: AsyncAnthropic | None = None

T = TypeVar("T", bound=BaseModel)


def _get_client() -> AsyncAnthropic:
    """Lazy Anthropic client init so a missing API key doesn't break imports."""
    global _client
    if _client is None:
        from anthropic import AsyncAnthropic

        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set - required for extraction.")
        _client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


def load_prompt(filename: str) -> str:
    """Read a versioned prompt file from extraction/prompts/."""
    path = Path(__file__).parent / "prompts" / filename
    return path.read_text(encoding="utf-8")


def _parse_json_object(raw: str) -> dict:
    """Extract a JSON object from Claude's response. Tolerant of fences."""
    if not raw:
        raise ValueError("empty response")
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else raw.strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in response")
    return json.loads(candidate[start : end + 1])


async def extract_with_retry(
    *,
    model_cls: type[T],
    prompt_version: str,
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    max_tokens: int = 3000,
    pipeline_run_id: str | None = None,
) -> T:
    """Call Claude, parse+validate JSON into `model_cls`.

    One retry: on invalid JSON / schema mismatch, the validation error is
    fed back to the model so it can self-correct. If the retry also fails
    (or the LLM call itself fails twice), raises `ExtractionFailed` - never
    anything else, so callers have exactly one thing to catch and the
    pipeline never crashes on bad LLM output.
    """
    client = _get_client()
    llm_model = model or settings.llm_model_fast
    messages: list[dict] = [{"role": "user", "content": prompt}]
    failure_reason = "unknown_error"
    started_at = time.perf_counter()
    last_raw: str | None = None
    last_input_tokens: int | None = None
    last_output_tokens: int | None = None
    attempt = 0

    def _trace(success: bool, error: str | None = None) -> None:
        record_llm_call(
            name=f"extraction:{prompt_version}",
            model=llm_model,
            prompt_version=prompt_version,
            prompt_text=prompt,
            completion_text=last_raw,
            input_tokens=last_input_tokens,
            output_tokens=last_output_tokens,
            latency_seconds=time.perf_counter() - started_at,
            attempts=attempt,
            success=success,
            error=error,
            pipeline_run_id=pipeline_run_id,
        )

    for attempt in (1, 2):
        try:
            kwargs: dict = {"model": llm_model, "max_tokens": max_tokens, "messages": messages}
            if system:
                kwargs["system"] = system
            msg = await client.messages.create(**kwargs)
            raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            last_raw = raw
            usage = getattr(msg, "usage", None)
            if usage is not None:
                last_input_tokens = getattr(usage, "input_tokens", None)
                last_output_tokens = getattr(usage, "output_tokens", None)
        except Exception as e:
            failure_reason = f"llm_call_error:{type(e).__name__}"
            log.warning(
                "Extraction (%s) attempt %s: LLM call failed (%s)",
                prompt_version,
                attempt,
                type(e).__name__,
            )
            if attempt == 1:
                messages.append(
                    {
                        "role": "user",
                        "content": "The previous request failed. Please try again and return ONLY the JSON object.",
                    }
                )
                continue
            _trace(False, failure_reason)
            raise ExtractionFailed(failure_reason, prompt_version) from e

        try:
            parsed = _parse_json_object(raw)
            result = model_cls.model_validate(parsed)
            _trace(True)
            return result
        except (ValueError, json.JSONDecodeError) as e:
            failure_reason = "invalid_json"
            feedback = str(e)
        except ValidationError as e:
            failure_reason = "schema_validation_error"
            # Pydantic's ValidationError str() includes the rejected input
            # value - fine to feed back to the model, never to log.
            feedback = str(e)

        log.warning("Extraction (%s) attempt %s: %s", prompt_version, attempt, failure_reason)
        if attempt == 1:
            messages.append({"role": "assistant", "content": raw[:4000]})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That response was not valid JSON matching the required schema. "
                        f"Validation error: {feedback}\n"
                        "Return ONLY a corrected JSON object matching the schema exactly."
                    ),
                }
            )
            continue
        _trace(False, failure_reason)
        raise ExtractionFailed(failure_reason, prompt_version)

    _trace(False, failure_reason)
    raise ExtractionFailed(failure_reason, prompt_version)
