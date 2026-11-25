"""Tests for observability/tracing.py.

Covers PII redaction (including two regex bugs caught during manual
testing: phone-regex spanning newlines into a following address line, and
the name-line regex rejecting middle initials) and the graceful-no-op
contract when Langfuse isn't configured - the default in this test
environment, same as production without LANGFUSE_* env vars set.
"""

from __future__ import annotations

from unittest.mock import patch

from observability.tracing import _mask, estimate_cost, get_client, record_llm_call, redact_pii


def test_redact_pii_email() -> None:
    assert redact_pii("Contact: jane.doe@example.com") == "Contact: [redacted-email]"


def test_redact_pii_phone() -> None:
    assert "[redacted-phone]" in redact_pii("Phone: +49 171 234 5678")


def test_redact_pii_address() -> None:
    assert "[redacted-address]" in redact_pii("123 Main Street, Munich")


def test_redact_pii_name_line() -> None:
    assert redact_pii("Jane Doe") == "[redacted-name]"


def test_redact_pii_name_with_middle_initial() -> None:
    """Regression: the name regex originally required a lowercase letter
    after every capital, so "A." (a middle initial) broke the match."""
    assert redact_pii("John A. Smith") == "[redacted-name]"


def test_redact_pii_phone_does_not_swallow_following_line() -> None:
    """Regression: \\s in the phone regex matched newlines, so a phone
    number greedily consumed the house number on the next line."""
    text = "Phone: +49 171 234 5678\n123 Main Street, Munich"
    result = redact_pii(text)
    assert "[redacted-phone]" in result
    assert "[redacted-address]" in result
    # The address redaction must still see its leading "123" -- i.e. the
    # phone regex must not have eaten across the newline.
    assert "Main Street" not in result


def test_redact_pii_empty_and_none() -> None:
    assert redact_pii("") == ""
    assert redact_pii(None) is None  # type: ignore[arg-type]


def test_redact_pii_ordinary_sentence_survives() -> None:
    text = "Built REST APIs using Python and Django."
    assert redact_pii(text) == text


def test_mask_walks_nested_structures() -> None:
    data = {"a": "Contact: jane@x.com", "b": ["nested", {"c": "+49 171 234 5678"}], "d": 42}
    masked = _mask(data=data)
    assert masked["a"] == "Contact: [redacted-email]"
    assert masked["b"][1]["c"] == "[redacted-phone]"
    assert masked["d"] == 42


def test_get_client_returns_none_when_not_configured() -> None:
    assert get_client() is None


def test_record_llm_call_never_raises_when_disabled() -> None:
    record_llm_call(
        name="test",
        model="claude-sonnet-5",
        prompt_version="v1",
        prompt_text="x",
        completion_text="y",
        input_tokens=10,
        output_tokens=5,
        latency_seconds=1.0,
        attempts=1,
        success=True,
    )  # must not raise


def test_estimate_cost_without_pricing_configured_returns_none() -> None:
    assert estimate_cost("claude-sonnet-5", 100, 50) is None


def test_estimate_cost_with_missing_tokens_returns_none() -> None:
    assert estimate_cost("claude-sonnet-5", None, 50) is None


def test_estimate_cost_with_configured_pricing() -> None:
    with patch("observability.tracing._PRICING", {"claude-sonnet-5": {"input_per_mtok": 3.0, "output_per_mtok": 15.0}}):
        cost = estimate_cost("claude-sonnet-5", 1_000_000, 1_000_000)
        assert cost == 18.0
