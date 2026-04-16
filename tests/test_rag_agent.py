"""Tests for the RAG scorer. These load the real embedding model once, so
they're slower than the rest of the suite — keep them focused."""

from __future__ import annotations

import pytest

from agents.rag_agent import _chunk_text, score_jobs


def test_chunking_respects_overlap_and_size() -> None:
    text = " ".join(f"w{i}" for i in range(500))
    chunks = _chunk_text(text, chunk_tokens=100, overlap=20)
    # With 500 words, step=80, expected about 6-7 chunks.
    assert len(chunks) >= 5
    # Every chunk has at most chunk_tokens words.
    assert all(len(c.split()) <= 100 for c in chunks)
    # Consecutive chunks should share the overlap region.
    a = chunks[0].split()
    b = chunks[1].split()
    assert a[-20:] == b[:20]


@pytest.mark.asyncio
async def test_score_jobs_returns_floats_in_range() -> None:
    jobs = [
        {
            "title": "LLM Engineer",
            "company": "FooAI",
            "description": "Build production agent systems using LangGraph and Anthropic Claude.",
            "source": "mock",
            "location": "Munich",
            "url": "",
        },
        {
            "title": "Gardener",
            "company": "Green Thumb",
            "description": "Pruning roses and weeding beds.",
            "source": "mock",
            "location": "Munich",
            "url": "",
        },
    ]
    scored = await score_jobs(jobs)
    assert len(scored) == 2
    for j in scored:
        assert "raw_score" in j
        assert isinstance(j["raw_score"], float)
        assert -1.0 <= j["raw_score"] <= 1.0
    # The LLM job should score higher than the gardening job against the
    # placeholder AI/LLM engineer CV.
    llm = next(j for j in scored if j["title"] == "LLM Engineer")
    garden = next(j for j in scored if j["title"] == "Gardener")
    assert llm["raw_score"] > garden["raw_score"]


@pytest.mark.asyncio
async def test_score_jobs_empty_input() -> None:
    assert await score_jobs([]) == []
