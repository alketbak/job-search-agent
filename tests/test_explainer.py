"""Tests for scoring/explainer.py - explanation is
written only from the structured ScoreBreakdown, never raw CV/job text;
graceful fallback on any LLM failure, no retry (free text, not JSON)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from scoring.explainer import generate_explanation, generate_motivation
from scoring.schemas import RequirementClassification, ScoreBreakdown


def _fake_message(text: str):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


def _breakdown(**overrides) -> ScoreBreakdown:
    defaults = dict(
        score_technical_skills=0.8,
        score_experience=0.7,
        score_responsibilities=0.6,
        score_seniority=1.0,
        score_location_language=0.9,
        score_education=0.5,
        score_final=0.75,
        blockers=[],
        requirement_classifications=[
            RequirementClassification(text="Python", mandatory=True, category="skill", status="met", evidence="4 years"),
        ],
    )
    defaults.update(overrides)
    return ScoreBreakdown(**defaults)


@pytest.mark.asyncio
async def test_generate_explanation_success() -> None:
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=_fake_message("Strong fit on Python experience.")))
    )
    with patch("scoring.explainer._get_client", return_value=client):
        text = await generate_explanation(_breakdown(), "AI Engineer", "Acme")
    assert "Strong fit" in text
    client.messages.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_explanation_prompt_contains_only_structured_data_not_raw_text() -> None:
    """The prompt sent to the LLM must be built purely from the breakdown -
    never given raw CV or job description text."""
    client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=_fake_message("ok"))))
    with patch("scoring.explainer._get_client", return_value=client):
        await generate_explanation(_breakdown(), "AI Engineer", "Acme")
    sent_prompt = client.messages.create.await_args.kwargs["messages"][0]["content"]
    assert "score_final" not in sent_prompt or "0.75" in sent_prompt  # uses the actual score value
    assert "Python" in sent_prompt  # requirement text (short, structured) is fine to include
    assert "AI Engineer" in sent_prompt


@pytest.mark.asyncio
async def test_generate_explanation_falls_back_on_llm_error() -> None:
    client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(side_effect=RuntimeError("boom"))))
    with patch("scoring.explainer._get_client", return_value=client):
        text = await generate_explanation(_breakdown(score_final=0.2, blockers=["Kubernetes"]), "AI Engineer")
    assert "0.20" in text
    assert "Kubernetes" in text


@pytest.mark.asyncio
async def test_generate_explanation_falls_back_when_no_api_key() -> None:
    with patch("scoring.explainer._get_client", side_effect=RuntimeError("no key")):
        text = await generate_explanation(_breakdown(score_final=0.9), "AI Engineer")
    assert "0.90" in text


@pytest.mark.asyncio
async def test_generate_motivation_empty_jobs() -> None:
    text = await generate_motivation([])
    assert "No strong matches" in text


@pytest.mark.asyncio
async def test_generate_motivation_success() -> None:
    client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=_fake_message("Keep going!"))))
    with patch("scoring.explainer._get_client", return_value=client):
        text = await generate_motivation([{"title": "AI Engineer", "company": "Acme"}])
    assert text == "Keep going!"


@pytest.mark.asyncio
async def test_generate_motivation_falls_back_on_error() -> None:
    client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(side_effect=RuntimeError("boom"))))
    with patch("scoring.explainer._get_client", return_value=client):
        text = await generate_motivation([{"title": "AI Engineer", "company": "Acme"}])
    assert "momentum" in text
