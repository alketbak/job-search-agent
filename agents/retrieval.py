"""pgvector-backed retrieval - replaces the FAISS-based CV<->job scoring
that used to live in `agents/rag_agent.py` (pgvector only, no second
production backend; see docs/ARCHITECTURE.md).

Keeps the same chunking strategy and the same "max cosine similarity across
CV chunks" scoring metric as the old agent - only the storage/search layer
changed, from a local FAISS index file to Postgres + pgvector. CV chunks and
job embeddings are now persisted (`cv_chunks`, `job_embeddings`) instead of
living only in an in-process FAISS index, which also lets
`agents/dedup.py`'s embedding-similarity stage query past job embeddings.

The default embedding model is `intfloat/multilingual-e5-base`,
an E5 model - E5 models require a "query: "/"passage: " prefix on input
text to score correctly; `_prefix()` handles that and is a no-op for
non-E5 models (e.g. MiniLM, kept selectable via `EMBEDDING_MODEL` for the
evaluation comparison).
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database.db import get_session
from database.models import CV, CVChunk
from database.seed import ensure_default_user

log = logging.getLogger(__name__)

# Keyed by model name rather than a single global - the eval harness
# (evaluation/run_eval.py) loads MiniLM and multilingual-e5 side by side to
# compare them, which a single-slot cache couldn't do.
_models: dict[str, Any] = {}


def _load_model(model_name: str | None = None):
    """Lazy-load an embedding model, cached by name. Heavy - large download
    on first use per model."""
    model_name = model_name or settings.embedding_model
    if model_name not in _models:
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model: %s", model_name)
        _models[model_name] = SentenceTransformer(model_name)
    return _models[model_name]


def _is_e5_model(model_name: str) -> bool:
    return "e5" in model_name.lower()


def _prefix(texts: list[str], kind: str, model_name: str) -> list[str]:
    """E5 models need a "query: "/"passage: " prefix to embed correctly.
    No-op for other embedding families (e.g. sentence-transformers MiniLM)."""
    if not _is_e5_model(model_name):
        return texts
    tag = "query: " if kind == "query" else "passage: "
    return [f"{tag}{t}" for t in texts]


def embed(texts: list[str], kind: str = "passage", model_name: str | None = None) -> np.ndarray:
    """Embed texts and L2-normalize so cosine similarity == inner product.

    `model_name` defaults to `settings.embedding_model` (the production
    default); pass it explicitly to use a different model for one call -
    e.g. the eval harness embedding with MiniLM for the v1-similarity
    comparison while the rest of the app still uses multilingual-e5.
    """
    model_name = model_name or settings.embedding_model
    if not texts:
        # Dimension doesn't matter for a 0-row array in practice; avoid
        # loading the model just to ask its dimension for an empty call.
        return np.zeros((0, settings.embedding_dim), dtype=np.float32)
    model = _load_model(model_name)
    prefixed = _prefix(texts, kind, model_name)
    vecs = model.encode(prefixed, convert_to_numpy=True, show_progress_bar=False)
    vecs = vecs.astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    return vecs / norms


def _chunk_text(text: str, chunk_tokens: int, overlap: int) -> list[str]:
    """Overlapping word-based chunking - unchanged from the old rag_agent.

    We use words as a proxy for tokens - simple, deterministic, and close
    enough for the embedding model's context window. A proper tokenizer
    would be more precise but adds a dependency and startup cost for no
    meaningful gain here.
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


def _cv_fingerprint(text: str) -> str:
    """Stable hash of CV content - rebuilds chunks/embeddings when it changes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def ensure_cv_chunks(user_id: str | None = None) -> None:
    """Load `settings.cv_path`, chunk + embed it, and upsert into Postgres.

    This is the pgvector replacement for the old FAISS cache file
    (`cv/cv.faiss` + sidecar `.meta.json`) - same invalidation strategy
    (content-hash fingerprint), different storage.
    """
    user_id = user_id or await ensure_default_user()
    cv_path = Path(settings.cv_path)
    if not cv_path.exists():
        raise FileNotFoundError(
            f"CV file not found at {cv_path}. Set CV_PATH or place a CV there."
        )
    cv_text = cv_path.read_text(encoding="utf-8")
    fingerprint = _cv_fingerprint(cv_text)

    async with get_session() as session:
        existing = (
            (
                await session.execute(
                    select(CV).where(CV.user_id == user_id, CV.content_hash == fingerprint)
                )
            )
            .scalars()
            .first()
        )
        if existing:
            has_chunks = (
                (
                    await session.execute(
                        select(CVChunk).where(CVChunk.cv_id == existing.id).limit(1)
                    )
                )
                .scalars()
                .first()
            )
            if has_chunks:
                log.info("CV unchanged (content hash matches) - reusing existing chunks")
                return
            cv = existing
        else:
            # File-based ingestion (CV_PATH), not the upload endpoint -
            # there's no HTTP request to collect explicit consent from.
            # Having placed the file at CV_PATH themselves is the operator's
            # consent; api/router.py::upload_cv is the path that requires
            # an explicit `consent=true` from an end user.
            cv = CV(
                user_id=user_id,
                raw_text=cv_text,
                content_hash=fingerprint,
                file_type="text",
                consent_given_at=datetime.utcnow(),
            )
            session.add(cv)
            await session.flush()

        _store_chunks(session, cv, user_id, cv_text)


def _store_chunks(session: AsyncSession, cv: CV, user_id: str, cv_text: str) -> None:
    chunks = _chunk_text(cv_text, settings.cv_chunk_tokens, settings.cv_chunk_overlap)
    if not chunks:
        raise ValueError("CV produced no chunks - is it empty?")
    vecs = embed(chunks, kind="passage")
    log.info("Embedding %s CV chunks for user %s", len(chunks), user_id)
    for idx, (text, vec) in enumerate(zip(chunks, vecs, strict=True)):
        session.add(
            CVChunk(
                cv_id=cv.id,
                user_id=user_id,
                chunk_index=idx,
                chunk_text=text,
                embedding_model=settings.embedding_model,
                embedding=vec.tolist(),
            )
        )


async def ensure_chunks_for_cv(cv_id: str) -> None:
    """Chunk + embed an uploaded CV (`POST /cv`) if it has no chunks yet.

    Uploads only run LLM extraction in the worker; embedding happens here,
    lazily, the first time the CV is used for matching."""
    async with get_session() as session:
        cv = await session.get(CV, cv_id)
        if cv is None:
            raise ValueError(f"CV {cv_id} no longer exists")
        has_chunks = (
            await session.execute(select(CVChunk.id).where(CVChunk.cv_id == cv_id).limit(1))
        ).first()
        if has_chunks:
            return
        _store_chunks(session, cv, cv.user_id, cv.raw_text)


async def _max_similarity_to_cv(
    session: AsyncSession, user_id: str, job_vec: np.ndarray
) -> float:
    """The pgvector replacement for FAISS's `idx.search` - distance to the
    single closest CV chunk, i.e. the same "max similarity across chunks"
    metric the old agent used (closest chunk == highest similarity)."""
    stmt = (
        select(CVChunk.embedding.cosine_distance(job_vec.tolist()).label("distance"))
        .where(CVChunk.user_id == user_id)
        .order_by("distance")
        .limit(1)
    )
    row = (await session.execute(stmt)).first()
    if row is None or row[0] is None:
        return 0.0
    distance = float(row[0])
    # pgvector cosine_distance is in [0, 2]; similarity = 1 - distance, clamp defensively.
    return max(-1.0, min(1.0, 1.0 - distance))


async def score_jobs(
    jobs: list[dict[str, Any]], user_id: str | None = None
) -> list[dict[str, Any]]:
    """Score each job by max cosine similarity vs CV chunks, sorted descending.

    Same public contract as the old `agents.rag_agent.score_jobs`. Each
    returned dict also carries a `_embedding` key (a plain `list[float]`,
    NOT a numpy array - this state flows through LangGraph, whose
    `MemorySaver` checkpointer msgpack-serializes every node's output, and
    `ndarray` isn't msgpack-serializable) so
    `agents/orchestrator.py::persist_node` can reuse it for
    `agents/dedup.py`'s embedding-similarity stage and for persisting
    `job_embeddings` rows, without re-embedding. Callers that persist jobs
    to the DB must not pass `_embedding` through to the ORM constructor.

    On any internal failure, returns jobs unscored with `raw_score=0.0`
    rather than raising - same fail-soft behavior as before.
    """
    if not jobs:
        return []
    user_id = user_id or await ensure_default_user()
    try:
        await ensure_cv_chunks(user_id)
    except Exception as e:
        log.error("CV chunk preparation failed: %s", e)
        return [{**j, "raw_score": 0.0} for j in jobs]

    texts = [
        f"{j.get('title','')} at {j.get('company','')}. {j.get('description','')}"
        for j in jobs
    ]
    vecs = embed(texts, kind="passage")

    scored: list[dict[str, Any]] = []
    async with get_session() as session:
        for job, vec in zip(jobs, vecs, strict=True):
            try:
                sim = await _max_similarity_to_cv(session, user_id, vec)
            except Exception as e:
                log.warning("pgvector similarity query failed (%s) - scoring job 0.0", e)
                sim = 0.0
            scored.append({**job, "raw_score": sim, "_embedding": vec.tolist()})
    scored.sort(key=lambda j: j["raw_score"], reverse=True)
    return scored


def embed_texts(texts: list[str], kind: str = "passage", model_name: str | None = None) -> list[list[float]]:
    """Thin wrapper around `embed()` returning plain `list[float]` vectors -
    state-safe for LangGraph (see `score_jobs`'s docstring on why `ndarray`
    can't cross node boundaries). Used by the pipeline to embed job
    requirement texts for `scoring/hybrid_scorer.py`, and by the eval
    harness to embed with a specific model for comparison."""
    return [vec.tolist() for vec in embed(texts, kind=kind, model_name=model_name)]


async def get_cv_chunks(user_id: str, cv_id: str | None = None) -> list[tuple[str, list[float]]]:
    """Return (chunk_text, embedding) pairs for `scoring/hybrid_scorer.py`.

    Caller should have already called `ensure_cv_chunks()` /
    `ensure_chunks_for_cv()` this run - this function only reads. Pass
    `cv_id` to read one CV's chunks; otherwise every chunk the user has.
    """
    stmt = select(CVChunk).where(CVChunk.user_id == user_id)
    if cv_id is not None:
        stmt = stmt.where(CVChunk.cv_id == cv_id)
    async with get_session() as session:
        rows = (await session.execute(stmt)).scalars().all()
    return [(row.chunk_text, [float(x) for x in row.embedding]) for row in rows if row.embedding is not None]


async def warmup() -> None:
    """Preload the embedding model + CV chunks. Call on startup."""
    try:
        await ensure_cv_chunks()
        log.info("Retrieval warmed up (pgvector)")
    except Exception as e:
        log.warning("Retrieval warmup skipped: %s", e)
