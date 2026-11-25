"""LLM call tracing via Langfuse.

Optional via env - if `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`
aren't both set, every function here no-ops and the pipeline runs
identically either way, same pattern as `memory/mcp_client.py`'s graceful
fallback when `npx` isn't available.

PII handling: Langfuse's own `mask` hook (not manual redaction scattered
through call sites) strips emails, phone numbers, and obvious street
addresses from every trace input/output/metadata value before it leaves
the process, by default. Name redaction is a best-effort line
heuristic (see `_NAME_LINE_RE`) - CVs don't follow one fixed format, so
this is not a guarantee, only a conservative floor; it will also redact
some non-name short capitalized lines (e.g. a job title alone on a line),
which is an acceptable over-redaction tradeoff for a safety mechanism.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from config import settings

log = logging.getLogger(__name__)

_client = None
_client_initialized = False

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# [ \t] (not \s) between digit groups - \s matches newlines, which let this
# greedily swallow a following line's leading digits (e.g. a house number
# on the next line) when it was \s-based. Caught in testing, not theory.
_PHONE_RE = re.compile(r"(?:\+\d{1,3}[ \t.-]?)?(?:\(?\d{2,4}\)?[ \t.-]?){2,5}\d{2,4}")
_ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+\w+(?:\s\w+){0,3}\s"
    r"(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Stra(?:ss|ß)e|Weg|Platz)\b",
    re.IGNORECASE,
)
# Each "word" is a capitalized token, optionally a bare initial ("A."/"A").
_NAME_WORD = r"[A-Z][a-zÀ-ÖØ-öø-ÿ'-]*\.?"
_NAME_LINE_RE = re.compile(rf"^(?:{_NAME_WORD}\s){{1,3}}{_NAME_WORD}$", re.MULTILINE)


def redact_pii(text: str) -> str:
    """Best-effort PII redaction. Emails and phone numbers are reliably
    caught by regex; addresses and names are heuristic and may miss
    unusual formats. Never raises on malformed input - returns it as-is."""
    if not text:
        return text
    text = _EMAIL_RE.sub("[redacted-email]", text)
    text = _PHONE_RE.sub("[redacted-phone]", text)
    text = _ADDRESS_RE.sub("[redacted-address]", text)
    text = _NAME_LINE_RE.sub("[redacted-name]", text)
    return text


def _mask(*, data: Any, **_kwargs: Any) -> Any:
    """Langfuse's `mask` hook - applied to every trace input/output/
    metadata value before export. Walks nested dict/list structures."""
    if isinstance(data, str):
        return redact_pii(data)
    if isinstance(data, dict):
        return {k: _mask(data=v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_mask(data=v) for v in data]
    return data


def get_client():
    """Lazy singleton. Returns `None` if Langfuse isn't configured or fails
    to initialize - callers must treat `None` as "tracing disabled," not
    as an error to propagate."""
    global _client, _client_initialized
    if _client_initialized:
        return _client
    _client_initialized = True
    if not settings.langfuse_enabled:
        return None
    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            mask=_mask,
        )
    except Exception as e:
        log.warning("Langfuse client init failed (%s) - tracing disabled", type(e).__name__)
        _client = None
    return _client


def _load_pricing() -> dict[str, dict[str, float]]:
    if not settings.anthropic_pricing_json:
        return {}
    try:
        return json.loads(settings.anthropic_pricing_json)
    except json.JSONDecodeError:
        log.warning("ANTHROPIC_PRICING_JSON is not valid JSON - cost tracking disabled")
        return {}


_PRICING = _load_pricing()


def estimate_cost(model: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    """Returns `None` - never a guess - if no pricing is configured for
    this model: an invented cost figure would be a fabricated result."""
    if input_tokens is None or output_tokens is None:
        return None
    rates = _PRICING.get(model)
    if not rates:
        return None
    input_rate = rates.get("input_per_mtok")
    output_rate = rates.get("output_per_mtok")
    if input_rate is None or output_rate is None:
        return None
    return (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate


def record_llm_call(
    *,
    name: str,
    model: str,
    prompt_version: str,
    prompt_text: str,
    completion_text: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    latency_seconds: float,
    attempts: int,
    success: bool,
    error: str | None = None,
    pipeline_run_id: str | None = None,
) -> None:
    """Record one completed LLM call (after all retries) - model,
    prompt_version, tokens, estimated cost, latency, errors, retries
   . No-op if Langfuse isn't configured. Never raises -
    observability must never break the pipeline it's observing.
    """
    client = get_client()
    if client is None:
        return
    try:
        usage_details: dict[str, int] = {}
        if input_tokens is not None:
            usage_details["input"] = input_tokens
        if output_tokens is not None:
            usage_details["output"] = output_tokens
        cost = estimate_cost(model, input_tokens, output_tokens)

        trace_context = (
            {"trace_id": client.create_trace_id(seed=pipeline_run_id)} if pipeline_run_id else None
        )
        with client.start_as_current_observation(
            trace_context=trace_context,
            name=name,
            as_type="generation",
            input=prompt_text,
            output=completion_text,
            model=model,
            usage_details=usage_details,
            cost_details={"total": cost} if cost is not None else {},
            metadata={
                "prompt_version": prompt_version,
                "attempts": attempts,
                "latency_seconds": round(latency_seconds, 3),
                "pipeline_run_id": pipeline_run_id,
            },
            level="DEFAULT" if success else "ERROR",
            status_message=error,
        ):
            pass
    except Exception as e:
        log.warning("Langfuse trace recording failed (%s) - continuing", type(e).__name__)


def flush() -> None:
    """Flush pending traces. Call at pipeline-run end / app shutdown so
    short-lived processes (tests, one-off CLI runs) don't lose traces."""
    client = get_client()
    if client is not None:
        try:
            client.flush()
        except Exception as e:
            log.warning("Langfuse flush failed (%s)", type(e).__name__)
