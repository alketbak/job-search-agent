"""Tests for evaluation/label_jobs.py - plumbing only.

Every test here uses `tmp_path` for `labels_dir` and a scripted `input_fn`
standing in for a real terminal. None of this writes to the real
`evaluation/labels/` directory or represents real human judgment - the "never simulate
labels" rule is about the actual evaluation dataset,
not about unit-testing whether the CLI reads/writes JSONL correctly.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from evaluation.label_jobs import (
    load_existing_labels,
    prompt_relevance,
    run_labeling_session,
    write_label,
)


def test_write_and_load_label_roundtrip(tmp_path: Path) -> None:
    entry = {"job_id": "job1", "relevance": 2, "labeled_at": "2026-01-01T00:00:00+00:00"}
    write_label(tmp_path, entry)
    loaded = load_existing_labels(tmp_path)
    assert loaded == {"job1": entry}


def test_load_existing_labels_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "2026-01-01.jsonl"
    path.write_text('not json\n{"job_id": "job1", "relevance": 1}\n\n', encoding="utf-8")
    loaded = load_existing_labels(tmp_path)
    assert loaded == {"job1": {"job_id": "job1", "relevance": 1}}


def test_load_existing_labels_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert load_existing_labels(tmp_path / "does_not_exist") == {}


def test_load_existing_labels_later_file_wins_for_same_job(tmp_path: Path) -> None:
    (tmp_path / "2026-01-01.jsonl").write_text(
        json.dumps({"job_id": "job1", "relevance": 0}) + "\n", encoding="utf-8"
    )
    (tmp_path / "2026-01-02.jsonl").write_text(
        json.dumps({"job_id": "job1", "relevance": 2}) + "\n", encoding="utf-8"
    )
    loaded = load_existing_labels(tmp_path)
    assert loaded["job1"]["relevance"] == 2


def _scripted_input(responses: list[str]):
    it = iter(responses)

    def _fn(_prompt: str) -> str:
        return next(it)

    return _fn


def test_prompt_relevance_returns_int_for_valid_input() -> None:
    job = {"title": "AI Engineer", "company": "Acme", "description": "x"}
    result = prompt_relevance(job, "summary", input_fn=_scripted_input(["2"]))
    assert result == 2


def test_prompt_relevance_reprompts_on_invalid_input() -> None:
    job = {"title": "AI Engineer", "company": "Acme", "description": "x"}
    result = prompt_relevance(job, "summary", input_fn=_scripted_input(["banana", "1"]))
    assert result == 1


def test_prompt_relevance_skip_and_quit() -> None:
    job = {"title": "AI Engineer", "company": "Acme", "description": "x"}
    assert prompt_relevance(job, "summary", input_fn=_scripted_input(["s"])) == "skip"
    assert prompt_relevance(job, "summary", input_fn=_scripted_input(["q"])) == "quit"


def _fake_jobs(n: int) -> list[dict]:
    return [
        {"id": f"job{i}", "title": f"Job {i}", "company": "Acme", "description": "desc", "location": "Munich"}
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_run_labeling_session_writes_labels_for_scripted_input(tmp_path: Path) -> None:
    with (
        patch("evaluation.label_jobs._fetch_jobs", new=AsyncMock(return_value=_fake_jobs(2))),
        patch("evaluation.label_jobs._fetch_candidate_summary", new=AsyncMock(return_value=("summary", "profile1"))),
    ):
        written = await run_labeling_session(
            limit=2, relabel=False, labels_dir=tmp_path, input_fn=_scripted_input(["2", "0"])
        )
    assert written == 2
    loaded = load_existing_labels(tmp_path)
    assert loaded["job0"]["relevance"] == 2
    assert loaded["job1"]["relevance"] == 0
    assert loaded["job0"]["candidate_profile_id"] == "profile1"


@pytest.mark.asyncio
async def test_run_labeling_session_skip_and_quit(tmp_path: Path) -> None:
    with (
        patch("evaluation.label_jobs._fetch_jobs", new=AsyncMock(return_value=_fake_jobs(3))),
        patch("evaluation.label_jobs._fetch_candidate_summary", new=AsyncMock(return_value=("summary", None))),
    ):
        written = await run_labeling_session(
            limit=3, relabel=False, labels_dir=tmp_path, input_fn=_scripted_input(["s", "q"])
        )
    assert written == 0
    assert load_existing_labels(tmp_path) == {}


@pytest.mark.asyncio
async def test_run_labeling_session_skips_already_labeled_unless_relabel(tmp_path: Path) -> None:
    write_label(tmp_path, {"job_id": "job0", "relevance": 1, "labeled_at": "2026-01-01T00:00:00+00:00"})

    with (
        patch("evaluation.label_jobs._fetch_jobs", new=AsyncMock(return_value=_fake_jobs(2))),
        patch("evaluation.label_jobs._fetch_candidate_summary", new=AsyncMock(return_value=("summary", None))),
    ):
        # job0 already labeled -> only prompted for job1.
        written = await run_labeling_session(
            limit=5, relabel=False, labels_dir=tmp_path, input_fn=_scripted_input(["2"])
        )
    assert written == 1
    assert load_existing_labels(tmp_path)["job1"]["relevance"] == 2
    # Original job0 label untouched.
    assert load_existing_labels(tmp_path)["job0"]["relevance"] == 1


@pytest.mark.asyncio
async def test_run_labeling_session_respects_limit(tmp_path: Path) -> None:
    with (
        patch("evaluation.label_jobs._fetch_jobs", new=AsyncMock(return_value=_fake_jobs(5))),
        patch("evaluation.label_jobs._fetch_candidate_summary", new=AsyncMock(return_value=("summary", None))),
    ):
        written = await run_labeling_session(
            limit=2, relabel=False, labels_dir=tmp_path, input_fn=_scripted_input(["1", "1", "1", "1", "1"])
        )
    assert written == 2
