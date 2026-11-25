"""Tests for evaluation/metrics.py.

Pure math - every input here is synthetic, hand-computed to verify the
formulas are implemented correctly. This is NOT evaluation data (no real
job/CV judgments are involved) - the design's "never simulate labels" rule
concerns real evaluation runs, not unit-testing arithmetic.
"""

from __future__ import annotations

import math

from evaluation.metrics import (
    dcg_at_k,
    extraction_precision_recall,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    set_precision_recall,
)


def test_precision_at_k_basic() -> None:
    relevance = {"a": 2, "b": 0, "c": 1, "d": 0, "e": 2}
    ranked = ["a", "b", "c", "d", "e"]
    assert precision_at_k(ranked, relevance, 3) == 2 / 3


def test_precision_at_k_empty_ranking() -> None:
    assert precision_at_k([], {"a": 1}, 5) == 0.0


def test_recall_at_k_basic() -> None:
    relevance = {"a": 2, "b": 0, "c": 1, "d": 0, "e": 2}
    ranked = ["a", "b", "c", "d", "e"]
    # 3 relevant total (a, c, e); top 3 contains a, c -> 2/3
    assert recall_at_k(ranked, relevance, 3) == 2 / 3


def test_recall_at_k_no_relevant_items() -> None:
    assert recall_at_k(["a", "b"], {"a": 0, "b": 0}, 5) == 0.0


def test_dcg_and_ndcg_hand_computed() -> None:
    relevance = {"a": 2, "b": 0, "c": 1, "d": 0, "e": 2}
    ranked = ["a", "b", "c", "d", "e"]
    dcg = dcg_at_k(ranked, relevance, 3)
    expected_dcg = 2 / math.log2(2) + 0 / math.log2(3) + 1 / math.log2(4)
    assert math.isclose(dcg, expected_dcg)

    ndcg = ndcg_at_k(ranked, relevance, 3)
    ideal = [2, 2, 1]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    assert math.isclose(ndcg, dcg / idcg)


def test_ndcg_perfect_ranking_is_one() -> None:
    relevance = {"a": 2, "b": 1, "c": 0}
    assert ndcg_at_k(["a", "b", "c"], relevance, 3) == 1.0


def test_ndcg_no_relevant_items_is_zero() -> None:
    assert ndcg_at_k(["a", "b"], {"a": 0, "b": 0}, 2) == 0.0


def test_set_precision_recall_partial_overlap() -> None:
    precision, recall = set_precision_recall({"python", "java"}, {"python", "go"})
    assert precision == 0.5
    assert recall == 0.5


def test_set_precision_recall_case_and_whitespace_insensitive() -> None:
    precision, recall = set_precision_recall({" Python ", "JAVA"}, {"python", "java"})
    assert precision == 1.0
    assert recall == 1.0


def test_set_precision_recall_empty_predicted_and_gold_is_vacuously_correct() -> None:
    assert set_precision_recall(set(), set()) == (1.0, 1.0)


def test_set_precision_recall_empty_predicted_nonempty_gold() -> None:
    precision, recall = set_precision_recall(set(), {"python"})
    assert precision == 0.0
    assert recall == 0.0


def test_extraction_precision_recall_empty_gold_reports_not_measured() -> None:
    result = extraction_precision_recall([], {})
    assert result == {"precision": None, "recall": None, "n": 0}


def test_extraction_precision_recall_aggregates_across_entries() -> None:
    gold = [
        {"id": "job1", "gold_skills": ["python", "sql"]},
        {"id": "job2", "gold_skills": ["java"]},
    ]
    extracted = {"job1": ["python"], "job2": ["java", "kotlin"]}
    result = extraction_precision_recall(gold, extracted)
    # job1: predicted={python}, gold={python,sql} -> precision=1.0, recall=0.5
    # job2: predicted={java,kotlin}, gold={java} -> precision=0.5, recall=1.0
    assert result["n"] == 2
    assert math.isclose(result["precision"], (1.0 + 0.5) / 2)
    assert math.isclose(result["recall"], (0.5 + 1.0) / 2)
