"""Reproducible evaluation report.

Compares four similarity/scoring methods against real human relevance
labels (`evaluation/labels/*.jsonl` - the ONLY legitimate source; see
`label_jobs.py`'s docstring - labels are never generated, simulated or
inferred):

  1. v1 similarity - MiniLM embeddings, max cosine sim vs. CV chunks
     (what the v1 `agents/rag_agent.py` did).
  2. pgvector similarity - the current production embedding model
     (`settings.embedding_model`, default multilingual-e5-base).
  3. deterministic hybrid score (`scoring/hybrid_scorer.py`).
  4. hybrid + LLM rerank - OFF by default (`--include-llm-rerank` to run
     it). Incurs real LLM cost per job; it's marked "optional, if
     cheap," which this treats as "opt-in, not assumed."

If fewer than `MIN_MEANINGFUL_LABELS` labels exist, the report still runs
on whatever's there (even zero) and prominently says so - never silently
presents a near-empty comparison as if it were meaningful.

Only job IDs that have a real label are evaluated - metrics computed over
unlabeled jobs would just be measuring an assumed-irrelevant default, not
anything real.

Usage:
    python evaluation/run_eval.py [--include-llm-rerank] [--k 5,10]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Make the repo importable whether launched as `python evaluation/run_eval.py`
# or `python -m evaluation.run_eval` - same pattern as scripts/migrate_sqlite_to_postgres.py.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.label_jobs import DEFAULT_LABELS_DIR, load_existing_labels  # noqa: E402
from evaluation.metrics import extraction_precision_recall, ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402

MIN_MEANINGFUL_LABELS = 40
REPORTS_DIR = Path(__file__).parent / "reports"
GOLD_DIR = Path(__file__).parent / "gold"


@dataclass
class MethodResult:
    name: str
    ranking: list[str]  # job_id, best first
    avg_latency_seconds: float | None
    cost_per_job_usd: float | None
    skipped_reason: str | None = None


def _cosine(a: list[float], b: list[float]) -> float:
    import numpy as np

    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _job_text(job: dict[str, Any]) -> str:
    return f"{job.get('title','')} at {job.get('company','')}. {job.get('description','')}"


async def _load_labeled_jobs(job_ids: list[str]) -> list[dict[str, Any]]:
    from sqlalchemy import select

    from database.db import get_session
    from database.models import Job

    if not job_ids:
        return []
    async with get_session() as session:
        rows = (await session.execute(select(Job).where(Job.id.in_(job_ids)))).scalars().all()
    return [r.to_dict() for r in rows]


async def _load_job_requirements(job_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    from sqlalchemy import select

    from database.db import get_session
    from database.models import JobRequirement

    out: dict[str, list[dict[str, Any]]] = {jid: [] for jid in job_ids}
    if not job_ids:
        return out
    async with get_session() as session:
        rows = (
            await session.execute(select(JobRequirement).where(JobRequirement.job_id.in_(job_ids)))
        ).scalars().all()
    for r in rows:
        out.setdefault(r.job_id, []).append(
            {"text": r.text, "mandatory": r.mandatory, "category": r.requirement_type}
        )
    return out


async def _rank_by_similarity(
    jobs: list[dict[str, Any]], cv_chunk_texts: list[str], model_name: str, label: str
) -> MethodResult:
    from agents.retrieval import embed_texts

    if not jobs:
        return MethodResult(name=label, ranking=[], avg_latency_seconds=None, cost_per_job_usd=0.0)
    start = time.perf_counter()
    job_vecs = embed_texts([_job_text(j) for j in jobs], model_name=model_name)
    # CV chunks are only persisted with the production model's embeddings -
    # for a fair comparison, re-embed the chunk TEXT with each candidate
    # model rather than reusing the production vectors.
    chunk_vecs = embed_texts(cv_chunk_texts, model_name=model_name) if cv_chunk_texts else []
    scores: dict[str, float] = {}
    for job, jv in zip(jobs, job_vecs, strict=True):
        sims = [_cosine(jv, cv) for cv in chunk_vecs] or [0.0]
        scores[job["id"]] = max(sims)
    ranking = sorted(scores, key=lambda jid: scores[jid], reverse=True)
    elapsed = time.perf_counter() - start
    return MethodResult(name=label, ranking=ranking, avg_latency_seconds=elapsed / len(jobs), cost_per_job_usd=0.0)


async def _rank_by_hybrid(
    jobs: list[dict[str, Any]],
    candidate_extraction,
    cv_chunks: list[tuple[str, list[float]]],
    requirements_by_job: dict[str, list[dict[str, Any]]],
) -> MethodResult:
    from agents.retrieval import embed_texts
    from extraction.schemas import JobProfileExtraction, JobRequirementClaim
    from scoring.hybrid_scorer import score_match

    if not jobs:
        return MethodResult(
            name="deterministic_hybrid", ranking=[], avg_latency_seconds=None, cost_per_job_usd=0.0
        )
    start = time.perf_counter()
    scores: dict[str, float] = {}
    for job in jobs:
        reqs = [JobRequirementClaim(**r) for r in requirements_by_job.get(job["id"], [])]
        job_profile = JobProfileExtraction(title=job.get("title", ""), requirements=reqs)
        req_texts = sorted({r.text for r in reqs})
        req_vecs = embed_texts(req_texts) if req_texts else []
        requirement_embeddings = dict(zip(req_texts, req_vecs, strict=True))
        breakdown = score_match(candidate_extraction, job_profile, cv_chunks, requirement_embeddings)
        scores[job["id"]] = breakdown.score_final
    ranking = sorted(scores, key=lambda jid: scores[jid], reverse=True)
    elapsed = time.perf_counter() - start
    return MethodResult(
        name="deterministic_hybrid",
        ranking=ranking,
        avg_latency_seconds=elapsed / len(jobs),
        cost_per_job_usd=0.0,
    )


async def _rank_by_hybrid_llm_rerank(
    jobs: list[dict[str, Any]], candidate_summary: str, hybrid_scores: dict[str, float], pipeline_run_id: str
) -> MethodResult:
    """Opt-in only (`--include-llm-rerank`) - real LLM cost per job."""
    from config import settings
    from extraction.llm_json import _get_client
    from observability.tracing import estimate_cost, record_llm_call

    if not jobs:
        return MethodResult(
            name="hybrid_llm_rerank", ranking=[], avg_latency_seconds=None, cost_per_job_usd=None
        )
    try:
        client = _get_client()
    except RuntimeError:
        return MethodResult(
            name="hybrid_llm_rerank",
            ranking=sorted(hybrid_scores, key=lambda jid: hybrid_scores[jid], reverse=True),
            avg_latency_seconds=None,
            cost_per_job_usd=None,
            skipped_reason="ANTHROPIC_API_KEY not set",
        )

    start = time.perf_counter()
    reranked = dict(hybrid_scores)
    costs: list[float] = []
    for job in jobs:
        prompt = (
            f"Candidate summary: {candidate_summary}\n\n"
            f"Job: {job.get('title')} at {job.get('company')}\n{(job.get('description') or '')[:1000]}\n\n"
            f"Deterministic match score: {hybrid_scores.get(job['id'], 0):.2f} (0-1 scale)\n"
            "On a 0-100 scale, how well does this candidate fit this job? Respond with ONLY the number."
        )
        call_start = time.perf_counter()
        input_tokens = output_tokens = None
        try:
            msg = await client.messages.create(
                model=settings.llm_model_smart, max_tokens=10, messages=[{"role": "user", "content": prompt}]
            )
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
            reranked[job["id"]] = max(0.0, min(100.0, float(text))) / 100.0
            usage = getattr(msg, "usage", None)
            if usage is not None:
                input_tokens = getattr(usage, "input_tokens", None)
                output_tokens = getattr(usage, "output_tokens", None)
            success = True
        except Exception:
            success = False  # keep the hybrid score as this job's fallback
        record_llm_call(
            name="eval:hybrid_llm_rerank",
            model=settings.llm_model_smart,
            prompt_version="eval_rerank_v1",
            prompt_text=prompt,
            completion_text=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=time.perf_counter() - call_start,
            attempts=1,
            success=success,
            pipeline_run_id=pipeline_run_id,
        )
        cost = estimate_cost(settings.llm_model_smart, input_tokens, output_tokens)
        if cost is not None:
            costs.append(cost)

    ranking = sorted(reranked, key=lambda jid: reranked[jid], reverse=True)
    elapsed = time.perf_counter() - start
    return MethodResult(
        name="hybrid_llm_rerank",
        ranking=ranking,
        avg_latency_seconds=elapsed / len(jobs),
        cost_per_job_usd=(sum(costs) / len(costs)) if costs else None,
    )


def _dataset_hash(relevance: dict[str, int]) -> str:
    payload = json.dumps(sorted(relevance.items()), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


async def run_eval(include_llm_rerank: bool = False, ks: list[int] | None = None, pipeline_run_id: str | None = None) -> Path:
    from agents.retrieval import ensure_cv_chunks, get_cv_chunks
    from config import settings
    from database.seed import ensure_default_user
    from extraction.cv_extractor import ensure_candidate_profile_from_file

    ks = ks or [5, 10]
    user_id = await ensure_default_user()

    relevance = {jid: entry["relevance"] for jid, entry in load_existing_labels(DEFAULT_LABELS_DIR).items()}
    job_ids = list(relevance.keys())
    jobs = await _load_labeled_jobs(job_ids)
    requirements_by_job = await _load_job_requirements(job_ids)

    candidate_result = await ensure_candidate_profile_from_file(user_id=user_id)
    await ensure_cv_chunks(user_id)
    cv_chunks = await get_cv_chunks(user_id)
    cv_chunk_texts = [t for t, _ in cv_chunks]

    from extraction.schemas import CandidateProfileExtraction

    candidate_extraction = candidate_result[0] if candidate_result else CandidateProfileExtraction()

    v1_result = await _rank_by_similarity(
        jobs, cv_chunk_texts, "sentence-transformers/all-MiniLM-L6-v2", "v1_similarity_minilm"
    )
    pgvector_result = await _rank_by_similarity(
        jobs, cv_chunk_texts, settings.embedding_model, f"pgvector_similarity_{settings.embedding_model}"
    )
    hybrid_result = await _rank_by_hybrid(jobs, candidate_extraction, cv_chunks, requirements_by_job)

    results = [v1_result, pgvector_result, hybrid_result]
    if include_llm_rerank:
        from extraction.schemas import JobProfileExtraction, JobRequirementClaim
        from scoring.hybrid_scorer import score_match

        # Rebuild actual scores (not just rank order) for the rerank prompt.
        hybrid_scores = {}
        for job in jobs:
            reqs = requirements_by_job.get(job["id"], [])
            job_profile = JobProfileExtraction(
                title=job.get("title", ""), requirements=[JobRequirementClaim(**r) for r in reqs]
            )
            hybrid_scores[job["id"]] = score_match(candidate_extraction, job_profile, cv_chunks, {}).score_final
        candidate_summary = ", ".join(s.name for s in candidate_extraction.skills[:10])
        rerank_result = await _rank_by_hybrid_llm_rerank(jobs, candidate_summary, hybrid_scores, pipeline_run_id or "eval")
        results.append(rerank_result)
    else:
        results.append(
            MethodResult(
                name="hybrid_llm_rerank",
                ranking=[],
                avg_latency_seconds=None,
                cost_per_job_usd=None,
                skipped_reason="not requested - pass --include-llm-rerank",
            )
        )

    cv_gold = _load_gold(GOLD_DIR / "cv_extraction_gold.jsonl")
    job_gold = _load_gold(GOLD_DIR / "job_extraction_gold.jsonl")
    cv_extraction_metrics = extraction_precision_recall(cv_gold, {})
    job_extraction_metrics = extraction_precision_recall(job_gold, {})

    report = _render_report(
        relevance=relevance,
        jobs=jobs,
        results=results,
        ks=ks,
        cv_extraction_metrics=cv_extraction_metrics,
        job_extraction_metrics=job_extraction_metrics,
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    dataset_hash = _dataset_hash(relevance)
    path = REPORTS_DIR / f"{date_str}_{dataset_hash}.md"
    path.write_text(report, encoding="utf-8")
    return path


def _load_gold(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _render_report(
    *,
    relevance: dict[str, int],
    jobs: list[dict[str, Any]],
    results: list[MethodResult],
    ks: list[int],
    cv_extraction_metrics: dict[str, Any],
    job_extraction_metrics: dict[str, Any],
) -> str:
    from config import settings

    n_labels = len(relevance)
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    dataset_hash = _dataset_hash(relevance)

    lines = [
        "# JobRadar Evaluation Report",
        "",
        f"- Date: {date_str}",
        f"- Dataset hash: `{dataset_hash}`",
        f"- Labeled jobs: {n_labels} (of {len(jobs)} resolved to a DB row)",
        f"- Embedding model (production): `{settings.embedding_model}`",
        "- Score weights: "
        f"technical_skills={settings.score_weight_technical_skills}, "
        f"experience={settings.score_weight_experience}, "
        f"responsibilities={settings.score_weight_responsibilities}, "
        f"seniority={settings.score_weight_seniority}, "
        f"location_language={settings.score_weight_location_language}, "
        f"education={settings.score_weight_education}",
        f"- Blocker cap: {settings.score_blocker_cap}, semantic match threshold: {settings.semantic_match_threshold}",
        "",
    ]

    if n_labels < MIN_MEANINGFUL_LABELS:
        lines += [
            f"## ⚠️ Not enough labels yet ({n_labels}/{MIN_MEANINGFUL_LABELS})",
            "",
            f"With fewer than {MIN_MEANINGFUL_LABELS} labeled jobs, "
            "these numbers are **not statistically meaningful** - they describe "
            "the harness working correctly, not real ranking quality. Run "
            "`python evaluation/label_jobs.py` to add real labels (a real person "
            "judging their own fit - never simulated).",
            "",
        ]

    lines += ["## Ranking methods compared", ""]
    header = ["Method"] + [f"P@{k}" for k in ks] + [f"R@{k}" for k in ks] + [f"NDCG@{ks[-1]}", "Avg latency/job", "Cost/job"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for result in results:
        if result.skipped_reason:
            row = [result.name] + ["skipped"] * (len(header) - 2) + [result.skipped_reason]
            lines.append("| " + " | ".join(row) + " |")
            continue
        row = [result.name]
        for k in ks:
            row.append(f"{precision_at_k(result.ranking, relevance, k):.2f}")
        for k in ks:
            row.append(f"{recall_at_k(result.ranking, relevance, k):.2f}")
        row.append(f"{ndcg_at_k(result.ranking, relevance, ks[-1]):.3f}")
        row.append(f"{result.avg_latency_seconds:.4f}s" if result.avg_latency_seconds is not None else "n/a")
        row.append(f"${result.cost_per_job_usd:.5f}" if result.cost_per_job_usd is not None else "unavailable")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Extraction precision/recall (hand-checked gold set)", ""]
    for label, metrics in (("CV extraction", cv_extraction_metrics), ("Job extraction", job_extraction_metrics)):
        if metrics["n"] == 0:
            lines.append(
                f"- {label}: **0 hand-checked samples - not measured.** "
                f"Add entries to `evaluation/gold/` to enable this."
            )
        else:
            lines.append(
                f"- {label}: precision={metrics['precision']:.2f}, recall={metrics['recall']:.2f} "
                f"(n={metrics['n']})"
            )

    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "python evaluation/run_eval.py",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-llm-rerank", action="store_true")
    parser.add_argument("--k", default="5,10", help="Comma-separated K values, e.g. 5,10")
    args = parser.parse_args()
    ks = [int(x) for x in args.k.split(",")]
    path = asyncio.run(run_eval(include_llm_rerank=args.include_llm_rerank, ks=ks))
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()
