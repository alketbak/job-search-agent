"""High-level helpers over the MCP memory client.

This module defines the *semantic model* we write into the knowledge graph:
- Entities: Company, Role, Skill, Application, SkillGap, Location, PipelineRun
- Relations:
    REQUIRES        Role -> Skill
    LOCATED_IN      Role -> Location
    POSTED_BY       Role -> Company
    APPLIED_TO      Application -> Role
    HAS_GAP         Role -> SkillGap
    INTERVIEWED_AT  Application -> Company
    DISCOVERED_IN   Role -> PipelineRun

Keeping the schema here (and not in mcp_client) means the client stays a pure
transport layer and we can evolve the semantic model without touching RPC code.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from memory.mcp_client import get_memory_client

log = logging.getLogger(__name__)


def _safe_name(prefix: str, value: str) -> str:
    """MCP entity names should be stable and safe. Strip whitespace, prefix by type."""
    cleaned = re.sub(r"\s+", " ", (value or "").strip())
    # Avoid empty names — use a placeholder rather than crashing.
    cleaned = cleaned or "unknown"
    return f"{prefix}::{cleaned}"


async def store_job_run(
    run_id: str, jobs_found: int, top_jobs: list[dict[str, Any]]
) -> None:
    """Persist a pipeline run: run entity + per-job entities + relations.

    `top_jobs` are the analyzed jobs (usually top 20). We create:
    - 1 PipelineRun entity
    - 1 Company entity per unique company
    - 1 Role entity per job
    - 1 Skill entity per skill mentioned in skill_gaps + highlights
    - Relations linking them
    """
    client = await get_memory_client()

    run_name = _safe_name("Run", run_id)
    now = datetime.utcnow().isoformat()

    entities: list[dict[str, Any]] = [
        {
            "name": run_name,
            "entityType": "PipelineRun",
            "observations": [
                f"Started at {now}",
                f"Jobs found: {jobs_found}",
                f"Top jobs analyzed: {len(top_jobs)}",
            ],
        }
    ]
    relations: list[dict[str, Any]] = []
    # Dedupe entity creations within this batch — the MCP memory server rejects
    # duplicates by name; we also want fewer round-trips.
    seen: set[str] = {run_name}

    def _add_entity(name: str, entity_type: str, observations: list[str]) -> None:
        if name in seen:
            return
        seen.add(name)
        entities.append(
            {"name": name, "entityType": entity_type, "observations": observations}
        )

    for job in top_jobs:
        title = job.get("title", "Untitled")
        company = job.get("company", "Unknown Company")
        location = job.get("location", "") or ""
        role_name = _safe_name("Role", f"{title} @ {company}")
        company_name = _safe_name("Company", company)

        role_obs = [
            f"Title: {title}",
            f"Company: {company}",
            f"Source: {job.get('source', 'unknown')}",
            f"URL: {job.get('url', '')}",
            f"Raw RAG score: {job.get('raw_score', 0):.3f}",
            f"Claude score: {job.get('claude_score', 0)}/10",
        ]
        if reasoning := job.get("claude_reasoning"):
            role_obs.append(f"Reasoning: {reasoning}")
        _add_entity(role_name, "Role", role_obs)

        _add_entity(
            company_name,
            "Company",
            [f"Name: {company}", f"First seen: {now}"],
        )

        relations.append(
            {"from": role_name, "to": company_name, "relationType": "POSTED_BY"}
        )
        relations.append(
            {"from": role_name, "to": run_name, "relationType": "DISCOVERED_IN"}
        )

        if location:
            loc_name = _safe_name("Location", location)
            _add_entity(loc_name, "Location", [f"Name: {location}"])
            relations.append(
                {"from": role_name, "to": loc_name, "relationType": "LOCATED_IN"}
            )

        # Skills the CV matches — these light up as strengths in the graph.
        for skill in job.get("highlights") or []:
            s_name = _safe_name("Skill", skill)
            _add_entity(s_name, "Skill", [f"Name: {skill}", "Direction: highlight"])
            relations.append(
                {"from": role_name, "to": s_name, "relationType": "MATCHES"}
            )

        # Skill gaps — what to study next. These become SkillGap entities too.
        for skill in job.get("skill_gaps") or []:
            gap_name = _safe_name("SkillGap", skill)
            _add_entity(
                gap_name,
                "SkillGap",
                [f"Skill: {skill}", f"Last seen: {now}"],
            )
            relations.append(
                {"from": role_name, "to": gap_name, "relationType": "HAS_GAP"}
            )

    # Batch-create to minimize RPC round-trips.
    if entities:
        await client.create_entities(entities)
    if relations:
        await client.create_relations(relations)
    log.info(
        "Knowledge graph: wrote %s entities, %s relations for run %s",
        len(entities),
        len(relations),
        run_id,
    )


async def store_application(
    company: str, role: str, date: str, status: str, notes: str = ""
) -> None:
    """Record an application event into the graph."""
    client = await get_memory_client()

    app_name = _safe_name("Application", f"{role} @ {company} ({date})")
    role_name = _safe_name("Role", f"{role} @ {company}")
    company_name = _safe_name("Company", company)

    obs = [f"Applied: {date}", f"Status: {status}"]
    if notes:
        obs.append(f"Notes: {notes}")

    await client.create_entities(
        [
            {"name": app_name, "entityType": "Application", "observations": obs},
            {
                "name": company_name,
                "entityType": "Company",
                "observations": [f"Name: {company}"],
            },
        ]
    )
    relations = [
        {"from": app_name, "to": role_name, "relationType": "APPLIED_TO"},
        {"from": app_name, "to": company_name, "relationType": "INTERVIEWED_AT"},
    ]
    await client.create_relations(relations)


async def store_skill_gap(skill: str, frequency: int) -> None:
    """Create or refresh a SkillGap entity with aggregated frequency."""
    client = await get_memory_client()
    name = _safe_name("SkillGap", skill)
    obs = [
        f"Skill: {skill}",
        f"Frequency: {frequency}",
        f"Last seen: {datetime.utcnow().isoformat()}",
    ]
    # create_entities is a no-op on duplicates in the real server; our fallback
    # merges observations. Either way we're safe to call it repeatedly.
    await client.create_entities(
        [{"name": name, "entityType": "SkillGap", "observations": obs}]
    )


async def get_full_graph() -> dict[str, Any]:
    """Return the whole graph in a D3-friendly shape: {nodes, links}."""
    client = await get_memory_client()
    raw = await client.read_graph()
    entities = raw.get("entities") or []
    relations = raw.get("relations") or []

    nodes = [
        {
            "id": e["name"],
            "name": e["name"].split("::", 1)[-1],
            "type": e.get("entityType", "Unknown"),
            "observations": e.get("observations", []),
        }
        for e in entities
    ]
    links = [
        {"source": r["from"], "target": r["to"], "type": r.get("relationType", "RELATED")}
        for r in relations
    ]
    return {"nodes": nodes, "links": links}


async def get_company_history(company_name: str) -> dict[str, Any]:
    """Return all entities/relations mentioning a given company."""
    client = await get_memory_client()
    return await client.search_nodes(company_name)
