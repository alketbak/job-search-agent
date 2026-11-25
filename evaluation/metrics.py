"""Evaluation metrics: Precision@K, Recall@K, NDCG@K
for ranking quality, plus set-based precision/recall for structured
extraction accuracy against a hand-checked gold set.

All pure functions - no I/O, no LLM, deterministic. Relevance labels (0/1/2)
and gold-extraction sets must come from real human judgment
(`evaluation/label_jobs.py`, a hand-checked `evaluation/gold/*.jsonl`); this
module only does the arithmetic on whatever real data it's given.
"""

from __future__ import annotations

import math


def _relevant(relevance: dict[str, int], job_id: str, threshold: int) -> bool:
    return relevance.get(job_id, 0) >= threshold


def precision_at_k(
    ranked_ids: list[str], relevance: dict[str, int], k: int, relevant_threshold: int = 1
) -> float:
    """Fraction of the top K ranked items that are relevant (label >=
    `relevant_threshold`). 0.0 if `ranked_ids` is empty."""
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for jid in top_k if _relevant(relevance, jid, relevant_threshold))
    return hits / len(top_k)


def recall_at_k(
    ranked_ids: list[str], relevance: dict[str, int], k: int, relevant_threshold: int = 1
) -> float:
    """Fraction of ALL relevant items (across `relevance`, not just the
    ranked list) that appear in the top K. 0.0 if there are no relevant
    items to find."""
    total_relevant = sum(1 for v in relevance.values() if v >= relevant_threshold)
    if total_relevant == 0:
        return 0.0
    top_k = set(ranked_ids[:k])
    found = sum(1 for jid, v in relevance.items() if v >= relevant_threshold and jid in top_k)
    return found / total_relevant


def dcg_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    return sum(
        relevance.get(jid, 0) / math.log2(i + 1) for i, jid in enumerate(ranked_ids[:k], start=1)
    )


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    """Normalized DCG@K using the raw 0/1/2 relevance labels as gains.
    0.0 if the ideal ordering has zero gain (no relevant items at all)."""
    ideal_gains = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(gain / math.log2(i + 1) for i, gain in enumerate(ideal_gains, start=1))
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked_ids, relevance, k) / idcg


def set_precision_recall(predicted: set[str], gold: set[str]) -> tuple[float, float]:
    """Case/whitespace-insensitive set precision & recall - e.g. comparing
    extracted skill names against a hand-verified gold set for one CV/job.

    Precision is 1.0 (vacuously correct) if `predicted` is empty and so is
    `gold`; 0.0 if `predicted` is empty but `gold` is not. Recall mirrors
    this for an empty `gold` set (nothing to find -> 1.0).
    """
    predicted_norm = {p.strip().lower() for p in predicted if p.strip()}
    gold_norm = {g.strip().lower() for g in gold if g.strip()}

    if not predicted_norm:
        precision = 1.0 if not gold_norm else 0.0
    else:
        precision = len(predicted_norm & gold_norm) / len(predicted_norm)

    recall = 1.0 if not gold_norm else len(predicted_norm & gold_norm) / len(gold_norm)
    return precision, recall


def extraction_precision_recall(
    gold_entries: list[dict], extracted_by_id: dict[str, list[str]]
) -> dict[str, float | int | None]:
    """Aggregate (mean) precision/recall across a hand-checked gold set.

    `gold_entries`: `[{"id": ..., "gold_skills": [...]}, ...]`.
    `extracted_by_id`: `id -> list of extracted skill/requirement names`
    for that same id (from a real extraction run, not fabricated).

    Returns `{"precision": None, "recall": None, "n": 0}` if `gold_entries`
    is empty - callers must report this as "not measured," not as a 0.0
    score.
    """
    if not gold_entries:
        return {"precision": None, "recall": None, "n": 0}
    precisions, recalls = [], []
    for entry in gold_entries:
        predicted = set(extracted_by_id.get(entry["id"], []))
        gold = set(entry.get("gold_skills") or [])
        p, r = set_precision_recall(predicted, gold)
        precisions.append(p)
        recalls.append(r)
    return {
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "n": len(gold_entries),
    }
