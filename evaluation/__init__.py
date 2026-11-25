"""Evaluation + observability.

- `label_jobs.py` - interactive CLI, the ONLY source of ground-truth
  relevance labels: labels come from a real person judging their own
  fit, never generated, simulated or inferred. Nothing else in this package (or anywhere else in the codebase)
  may write to `evaluation/labels/`.
- `metrics.py` - Precision@K, Recall@K, NDCG@K, extraction precision/recall.
- `run_eval.py` - compares v1 MiniLM / pgvector multilingual-e5 /
  deterministic hybrid / hybrid+LLM rerank, against real labels, and
  writes a markdown report to `evaluation/reports/`.
"""
