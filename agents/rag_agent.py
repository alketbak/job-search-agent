"""RAG agent — embeds the CV once, scores every job by cosine similarity.

Design notes:
- We use sentence-transformers (all-MiniLM-L6-v2) for free, local embeddings.
  384-dim vectors, fast on CPU, good enough for this scoring task.
- We build a FAISS index once per CV and persist it as `cv/cv.faiss` + a
  sidecar `.meta.json` with chunk texts and the CV file's mtime+hash. On each
  run we check whether the CV has changed; if not, we load the cached index.
- FAISS IndexFlatIP with L2-normalized vectors == cosine similarity. We use
  inner product on normalized vectors because it's the simplest exact search
  and scales fine for CV-sized inputs (tens of chunks).
- Score per job = MAX similarity across all CV chunks. Taking the max (rather
  than mean) rewards jobs that strongly match *any* part of the candidate's
  background — which is what a human reviewer would do.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config import settings

log = logging.getLogger(__name__)

# Heavy imports (sentence-transformers, faiss) are deferred to first use so
# importing this module stays cheap in tests that don't need the model.
_model = None
_index = None
_chunks: list[str] = []


@dataclass
class ScoredJob:
    """Thin wrapper used only for internal clarity — we return plain dicts."""

    job: dict[str, Any]
    raw_score: float


def _cv_fingerprint(text: str) -> str:
    """Stable hash of the CV content. Rebuilds index when the CV changes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk_text(text: str, chunk_tokens: int, overlap: int) -> list[str]:
    """Overlapping word-based chunking.

    We use words as a proxy for tokens — it's simple, deterministic, and close
    enough for MiniLM's ~256-token context. A proper tokenizer would be more
    precise but adds a dependency and startup cost for no meaningful gain here.
    """
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, chunk_tokens - overlap)
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_tokens]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + chunk_tokens >= len(words):
            break
    return chunks


def _load_model():
    """Lazy-load the embedding model. Heavy — ~80MB download on first use."""
    global _model
    if _model is None:
        # Imported lazily — the import itself is non-trivial.
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model: %s", settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def _embed(texts: list[str]) -> np.ndarray:
    """Embed texts and L2-normalize so inner product == cosine similarity."""
    model = _load_model()
    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    vecs = vecs.astype(np.float32)
    # Normalize in-place. Add epsilon to avoid divide-by-zero on empty strings.
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    return vecs / norms


def _build_or_load_index() -> tuple[Any, list[str]]:
    """Return (faiss_index, chunks). Build it if missing or CV changed."""
    global _index, _chunks
    if _index is not None and _chunks:
        return _index, _chunks

    # Imported lazily so the module is importable on systems without FAISS.
    import faiss  # type: ignore[import-untyped]

    cv_path = Path(settings.cv_path)
    if not cv_path.exists():
        raise FileNotFoundError(
            f"CV file not found at {cv_path}. Set CV_PATH or place a CV there."
        )
    cv_text = cv_path.read_text(encoding="utf-8")
    fingerprint = _cv_fingerprint(cv_text)

    index_path = Path(settings.faiss_index_path)
    meta_path = index_path.with_suffix(".faiss.meta.json")

    if index_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("fingerprint") == fingerprint:
                log.info("Loading FAISS index from %s", index_path)
                idx = faiss.read_index(str(index_path))
                _index, _chunks = idx, meta.get("chunks", [])
                return _index, _chunks
            log.info("CV changed — rebuilding FAISS index")
        except (OSError, json.JSONDecodeError) as e:
            log.warning("Failed to read cached FAISS index (%s); rebuilding", e)

    chunks = _chunk_text(cv_text, settings.cv_chunk_tokens, settings.cv_chunk_overlap)
    if not chunks:
        raise ValueError("CV produced no chunks — is the file empty?")
    log.info("Embedding %s CV chunks", len(chunks))
    vecs = _embed(chunks)
    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx, str(index_path))
    meta_path.write_text(
        json.dumps({"fingerprint": fingerprint, "chunks": chunks}, ensure_ascii=False),
        encoding="utf-8",
    )

    _index, _chunks = idx, chunks
    return _index, _chunks


def _score_jobs_sync(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Synchronous scoring — called from a thread by `score_jobs`."""
    if not jobs:
        return []
    try:
        idx, _ = _build_or_load_index()
    except Exception as e:
        log.error("RAG index build failed: %s", e)
        # Fail soft: return jobs with neutral score rather than killing the pipeline.
        return [{**j, "raw_score": 0.0} for j in jobs]

    texts = [
        f"{j.get('title','')} at {j.get('company','')}. {j.get('description','')}"
        for j in jobs
    ]
    vecs = _embed(texts)
    # Search against all CV chunks; take max similarity per job.
    k = idx.ntotal
    scores, _ = idx.search(vecs, k)
    max_scores = scores.max(axis=1)
    # FAISS inner product on normalized vectors is in [-1, 1]; clamp defensively.
    max_scores = np.clip(max_scores, -1.0, 1.0)

    scored: list[dict[str, Any]] = []
    for job, score in zip(jobs, max_scores):
        scored.append({**job, "raw_score": float(score)})
    scored.sort(key=lambda j: j["raw_score"], reverse=True)
    return scored


async def score_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Public entrypoint — score each job by max cosine similarity vs CV chunks.

    Returns jobs sorted by `raw_score` descending. On any internal failure,
    returns jobs unscored with `raw_score=0.0` rather than raising.
    """
    # sentence-transformers.encode is CPU-bound (and somewhat blocking), so we
    # run it in a thread to keep the event loop responsive.
    return await asyncio.to_thread(_score_jobs_sync, jobs)


async def warmup() -> None:
    """Preload model + index. Call on startup to avoid first-request latency."""
    try:
        await asyncio.to_thread(_build_or_load_index)
        log.info("RAG agent warmed up")
    except Exception as e:
        log.warning("RAG warmup skipped: %s", e)
