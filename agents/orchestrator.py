"""LangGraph orchestrator — wires all agents into a state machine.

Graph shape (linear, with an error-tolerant policy on every node):

    scrape -> rag -> analyst -> memory -> delivery -> persist -> END

Each node:
- accepts the full state, mutates a subset, returns the updated state
- catches its own exceptions and appends to `state["errors"]` rather than crashing
- is async so LangGraph's async runtime can interleave them

We compile with an in-memory checkpointer so `/runs` can retrieve run history.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents.analyst_agent import analyze_jobs
from agents.delivery_agent import deliver
from agents.rag_agent import score_jobs
from agents.scraper_agent import scrape
from config import settings
from database import Job, JobStatus, get_session
from database.models import PipelineRun
from memory.knowledge_graph import store_job_run, store_skill_gap

log = logging.getLogger(__name__)


class JobRadarState(TypedDict, total=False):
    """State shared across all orchestrator nodes.

    Using `total=False` means every key is optional — LangGraph merges per-node
    partial updates rather than requiring a full state every time.
    """

    query: str
    location: str
    run_id: str
    started_at: str
    raw_jobs: list[dict[str, Any]]
    scored_jobs: list[dict[str, Any]]
    analyzed_jobs: list[dict[str, Any]]
    skill_gaps: list[str]
    delivery_result: dict[str, Any]
    errors: list[str]


def _log_error(state: JobRadarState, node: str, err: Exception) -> list[str]:
    """Append a structured error marker to state.errors, return updated list."""
    msg = f"{node}: {type(err).__name__}: {err}"
    log.exception("Pipeline node %s failed: %s", node, err)
    return list(state.get("errors") or []) + [msg]


# ---------------- nodes ----------------

async def scrape_node(state: JobRadarState) -> dict[str, Any]:
    query = state.get("query") or settings.default_query
    location = state.get("location") or settings.default_location
    try:
        jobs = await scrape(query=query, location=location)
        log.info("scrape_node: %s jobs", len(jobs))
        return {"raw_jobs": jobs}
    except Exception as e:
        return {"raw_jobs": [], "errors": _log_error(state, "scrape", e)}


async def rag_node(state: JobRadarState) -> dict[str, Any]:
    try:
        scored = await score_jobs(state.get("raw_jobs") or [])
        # Keep only the best N for the expensive analyst step.
        top = scored[: settings.top_n_for_analyst]
        log.info(
            "rag_node: scored %s jobs, forwarding top %s to analyst",
            len(scored),
            len(top),
        )
        return {"scored_jobs": top}
    except Exception as e:
        return {"scored_jobs": [], "errors": _log_error(state, "rag", e)}


async def analyst_node(state: JobRadarState) -> dict[str, Any]:
    try:
        analyzed = await analyze_jobs(state.get("scored_jobs") or [])
        gaps: list[str] = []
        for j in analyzed:
            gaps.extend(j.get("skill_gaps") or [])
        log.info("analyst_node: analyzed %s jobs, %s gap mentions", len(analyzed), len(gaps))
        return {"analyzed_jobs": analyzed, "skill_gaps": gaps}
    except Exception as e:
        return {"analyzed_jobs": [], "skill_gaps": [], "errors": _log_error(state, "analyst", e)}


async def memory_node(state: JobRadarState) -> dict[str, Any]:
    try:
        run_id = state.get("run_id") or str(uuid.uuid4())
        await store_job_run(
            run_id=run_id,
            jobs_found=len(state.get("raw_jobs") or []),
            top_jobs=state.get("analyzed_jobs") or [],
        )
        # Aggregate gap frequencies and upsert into the graph.
        from collections import Counter

        c: Counter[str] = Counter(state.get("skill_gaps") or [])
        for skill, freq in c.most_common():
            await store_skill_gap(skill, freq)
        log.info("memory_node: wrote run %s to knowledge graph", run_id)
        return {"run_id": run_id}
    except Exception as e:
        return {"errors": _log_error(state, "memory", e)}


async def delivery_node(state: JobRadarState) -> dict[str, Any]:
    try:
        result = await deliver(
            scraped_count=len(state.get("raw_jobs") or []),
            scored_count=len(state.get("scored_jobs") or []),
            analyzed_jobs=state.get("analyzed_jobs") or [],
            run_id=state.get("run_id") or "unknown",
        )
        return {"delivery_result": result}
    except Exception as e:
        return {"delivery_result": {}, "errors": _log_error(state, "delivery", e)}


async def persist_node(state: JobRadarState) -> dict[str, Any]:
    """Write analyzed jobs + run metadata to SQLite."""
    try:
        analyzed = state.get("analyzed_jobs") or []
        run_id = state.get("run_id") or str(uuid.uuid4())
        async with get_session() as session:
            # Record the run.
            run = PipelineRun(
                id=run_id,
                started_at=datetime.fromisoformat(state["started_at"])
                if state.get("started_at")
                else datetime.utcnow(),
                finished_at=datetime.utcnow(),
                query=state.get("query"),
                location=state.get("location"),
                jobs_found=len(state.get("raw_jobs") or []),
                jobs_scored=len(state.get("scored_jobs") or []),
                jobs_analyzed=len(analyzed),
                errors=state.get("errors") or [],
            )
            session.add(run)

            # Record each job as a new row. We don't dedupe against the DB
            # here — the pipeline has already deduped within this run, and
            # cross-run dedup is handled by the UI layer.
            for j in analyzed:
                session.add(
                    Job(
                        title=j.get("title", ""),
                        company=j.get("company", ""),
                        location=j.get("location", ""),
                        description=j.get("description", ""),
                        url=j.get("url", ""),
                        source=j.get("source", "unknown"),
                        raw_score=j.get("raw_score"),
                        claude_score=j.get("claude_score"),
                        claude_reasoning=j.get("claude_reasoning"),
                        skill_gaps=j.get("skill_gaps") or [],
                        highlights=j.get("highlights") or [],
                        apply_recommendation=bool(j.get("apply_recommendation")),
                        status=JobStatus.NEW,
                        run_id=run_id,
                    )
                )
        log.info("persist_node: saved %s jobs and run %s", len(analyzed), run_id)
        return {}
    except Exception as e:
        return {"errors": _log_error(state, "persist", e)}


# ---------------- graph assembly ----------------

def _build_graph():
    graph = StateGraph(JobRadarState)
    graph.add_node("scrape", scrape_node)
    graph.add_node("rag", rag_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("memory", memory_node)
    graph.add_node("delivery", delivery_node)
    graph.add_node("persist", persist_node)

    graph.set_entry_point("scrape")
    graph.add_edge("scrape", "rag")
    graph.add_edge("rag", "analyst")
    graph.add_edge("analyst", "memory")
    graph.add_edge("memory", "delivery")
    graph.add_edge("delivery", "persist")
    graph.add_edge("persist", END)

    # Checkpointer lets us replay / inspect past runs from /runs.
    return graph.compile(checkpointer=MemorySaver())


# Compile once at import time — the graph is stateless between runs.
_compiled = _build_graph()


async def run_pipeline(
    query: str | None = None, location: str | None = None
) -> JobRadarState:
    """Run the full pipeline once. Returns the final state dict."""
    run_id = str(uuid.uuid4())
    initial: JobRadarState = {
        "query": query or settings.default_query,
        "location": location or settings.default_location,
        "run_id": run_id,
        "started_at": datetime.utcnow().isoformat(),
        "raw_jobs": [],
        "scored_jobs": [],
        "analyzed_jobs": [],
        "skill_gaps": [],
        "errors": [],
    }
    log.info(
        "Starting pipeline run %s (query=%s location=%s)",
        run_id,
        initial["query"],
        initial["location"],
    )
    final = await _compiled.ainvoke(
        initial,
        # LangGraph requires a thread_id to checkpoint — we use run_id so every
        # run is its own thread / trace.
        config={"configurable": {"thread_id": run_id}},
    )
    log.info(
        "Pipeline %s done — found=%s analyzed=%s errors=%s",
        run_id,
        len(final.get("raw_jobs") or []),
        len(final.get("analyzed_jobs") or []),
        len(final.get("errors") or []),
    )
    return final  # type: ignore[return-value]
