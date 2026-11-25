"""Tests for evaluation/run_eval.py.

`_render_report` is tested with synthetic `MethodResult`/relevance data -
this verifies the markdown-generation logic, not any real evaluation
claim; it never touches `evaluation/labels/`. The end-to-end test runs the
real harness against whatever labels exist right now (0, in this
environment) and checks the report says so - the required behavior for an
under-40-labels dataset, not a mock.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from database import init_db
from evaluation.run_eval import (
    MIN_MEANINGFUL_LABELS,
    MethodResult,
    _dataset_hash,
    _render_report,
    run_eval,
)
from extraction.schemas import CandidateProfileExtraction, SkillClaim


@pytest.fixture(autouse=True)
def _bootstrap_db() -> None:
    init_db()


def test_dataset_hash_is_deterministic() -> None:
    a = _dataset_hash({"job1": 2, "job2": 0})
    b = _dataset_hash({"job2": 0, "job1": 2})  # different insertion order
    assert a == b


def test_dataset_hash_changes_with_content() -> None:
    assert _dataset_hash({"job1": 2}) != _dataset_hash({"job1": 1})


def test_render_report_flags_insufficient_labels() -> None:
    relevance = {"job1": 2, "job2": 0}  # well under MIN_MEANINGFUL_LABELS
    results = [
        MethodResult(name="v1_similarity_minilm", ranking=["job1", "job2"], avg_latency_seconds=0.01, cost_per_job_usd=0.0),
    ]
    report = _render_report(
        relevance=relevance,
        jobs=[],
        results=results,
        ks=[5, 10],
        cv_extraction_metrics={"precision": None, "recall": None, "n": 0},
        job_extraction_metrics={"precision": None, "recall": None, "n": 0},
    )
    assert f"Not enough labels yet (2/{MIN_MEANINGFUL_LABELS})" in report
    assert "not measured" in report


def test_render_report_omits_warning_when_enough_labels() -> None:
    relevance = {f"job{i}": (i % 3) for i in range(MIN_MEANINGFUL_LABELS)}
    results = [MethodResult(name="v1_similarity_minilm", ranking=list(relevance.keys()), avg_latency_seconds=0.01, cost_per_job_usd=0.0)]
    report = _render_report(
        relevance=relevance,
        jobs=[],
        results=results,
        ks=[5, 10],
        cv_extraction_metrics={"precision": None, "recall": None, "n": 0},
        job_extraction_metrics={"precision": None, "recall": None, "n": 0},
    )
    assert "Not enough labels yet" not in report


def test_render_report_shows_skipped_method() -> None:
    results = [
        MethodResult(
            name="hybrid_llm_rerank",
            ranking=[],
            avg_latency_seconds=None,
            cost_per_job_usd=None,
            skipped_reason="not requested - pass --include-llm-rerank",
        )
    ]
    report = _render_report(
        relevance={},
        jobs=[],
        results=results,
        ks=[5, 10],
        cv_extraction_metrics={"precision": None, "recall": None, "n": 0},
        job_extraction_metrics={"precision": None, "recall": None, "n": 0},
    )
    assert "not requested" in report


def test_render_report_shows_extraction_metrics_when_present() -> None:
    report = _render_report(
        relevance={},
        jobs=[],
        results=[],
        ks=[5, 10],
        cv_extraction_metrics={"precision": 0.8, "recall": 0.6, "n": 3},
        job_extraction_metrics={"precision": None, "recall": None, "n": 0},
    )
    assert "precision=0.80, recall=0.60 (n=3)" in report
    assert "Job extraction: **0 hand-checked samples" in report


def _fake_candidate_extraction() -> CandidateProfileExtraction:
    return CandidateProfileExtraction(
        skills=[
            SkillClaim(
                name="Python",
                category="language",
                evidence_level="professional",
                evidence_snippet="4 years Python",
            )
        ],
        location_preference="Berlin",
    )


@pytest.mark.asyncio
async def test_run_eval_reports_insufficient_labels(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runs the actual harness against whatever's really in
    evaluation/labels/ right now. In this environment that's 0 labels -
    asserting that, not assuming it. The report goes to a temp
    dir so test runs don't add files to evaluation/reports/.

    CV extraction is mocked -
    `run_eval` needs *some* `CandidateProfileExtraction` to score jobs
    against regardless of `include_llm_rerank`, and whether that call would
    otherwise be free (cached profile from an earlier test in the same
    session DB) or hit the real API depends on test ordering, which isn't
    something this test's determinism should depend on.
    """
    from evaluation.label_jobs import DEFAULT_LABELS_DIR, load_existing_labels

    real_label_count = len(load_existing_labels(DEFAULT_LABELS_DIR))

    with patch(
        "extraction.cv_extractor.ensure_candidate_profile_from_file",
        new=AsyncMock(return_value=(_fake_candidate_extraction(), "fake-profile-id")),
    ):
        monkeypatch.setattr("evaluation.run_eval.REPORTS_DIR", tmp_path)
        path = await run_eval(include_llm_rerank=False)
    report = path.read_text(encoding="utf-8")

    assert f"Labeled jobs: {real_label_count}" in report
    if real_label_count < MIN_MEANINGFUL_LABELS:
        assert "Not enough labels yet" in report
    assert "not requested - pass --include-llm-rerank" in report
