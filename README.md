# JobRadar

![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-1C3A53)
![Anthropic](https://img.shields.io/badge/Claude-Sonnet%204-D97757)
![FAISS](https://img.shields.io/badge/FAISS-1.9-005F9E)
![D3.js](https://img.shields.io/badge/D3.js-v7-F9A03C?logo=d3.js&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

**JobRadar** is a production-grade job search intelligence system that every day at 6 AM:
scrapes job APIs, scores each posting against your CV via FAISS RAG, has Claude
reason about fit, stores the results in an MCP knowledge graph, emails you a
markdown digest, and visualizes the whole memory as an interactive D3.js graph.

---

## Architecture

```
                                 ┌────────────────────────────────────┐
                                 │          FastAPI application       │
                                 │  GET / → D3 dashboard              │
                                 │  POST /run, GET /jobs, /graph, …   │
                                 └──────────────┬─────────────────────┘
                                                │
                                                ▼
                                 ┌────────────────────────────────────┐
  ┌──────────┐   ┌──────────┐    │        LangGraph Orchestrator      │
  │APScheduler│──▶│run_pipeline │──▶│  scrape → rag → analyst → memory   │
  │ daily 6am │   │             │    │           → delivery → persist     │
  └──────────┘   └──────────┘    └──────────────┬─────────────────────┘
                                                │
                 ┌─────────────┬───────────────┼───────────────┬─────────────┐
                 ▼             ▼               ▼               ▼             ▼
           ┌──────────┐  ┌──────────┐    ┌──────────┐    ┌──────────┐  ┌──────────┐
           │ Scraper  │  │   RAG    │    │ Analyst  │    │  Memory  │  │ Delivery │
           │  agent   │  │  agent   │    │  agent   │    │  (MCP KG)│  │  agent   │
           └────┬─────┘  └────┬─────┘    └────┬─────┘    └────┬─────┘  └────┬─────┘
                │             │               │               │             │
         ┌──────┴──────┐  ┌───┴────┐    ┌─────┴─────┐   ┌────┴─────┐  ┌────┴────┐
         │ Arbeitnow   │  │ FAISS  │    │ Claude    │   │ MCP      │  │ SMTP +  │
         │ Adzuna      │  │ cv.faiss    │ Sonnet 4  │   │ memory   │  │ markdown│
         │ Bundesagntr │  │ MiniLM │    │           │   │ subproc  │  │ digest  │
         │ Mock        │  └────────┘    └───────────┘   └──────────┘  └─────────┘
         └─────────────┘
                                                │
                                                ▼
                                     ┌────────────────────┐
                                     │   SQLite jobradar  │
                                     │    Jobs · Apps ·   │
                                     │  SkillGaps · Runs  │
                                     └────────────────────┘
```

---

## Quick start (Docker)

```bash
# 1. Clone & enter
git clone <this repo> job-search-agent && cd job-search-agent

# 2. Copy env and set ANTHROPIC_API_KEY.
#    Default USE_MOCK_DATA=true works out of the box — no other keys needed.
cp .env.example .env
$EDITOR .env                # set ANTHROPIC_API_KEY=sk-ant-…

# 3. Boot
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000) for the dashboard, and
[http://localhost:8000/docs](http://localhost:8000/docs) for the OpenAPI UI.

Trigger a pipeline run manually:

```bash
curl -X POST http://localhost:8000/run \
  -H 'content-type: application/json' \
  -d '{"query": "AI Engineer", "location": "Munich"}'
```

Then visit `/` and watch the knowledge graph populate.

---

## Manual setup (no Docker)

Requires Python 3.11+ and Node.js 20+ (for the MCP memory server via `npx`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY
python main.py
```

The first launch downloads the `all-MiniLM-L6-v2` model (~80 MB), then builds
the FAISS index from your CV.

---

## Use your own CV

1. Replace `cv/cv.txt` with your CV in **plain text** (paste, no formatting).
2. Delete the stale cached index: `rm -f cv/cv.faiss cv/cv.faiss.meta.json`.
3. Restart — the index rebuilds automatically on next launch. (The system also
   fingerprints the CV content and invalidates the cache when it changes.)

---

## API endpoints

| Method | Path                  | Purpose                                            |
|-------:|-----------------------|----------------------------------------------------|
| GET    | `/`                   | D3.js knowledge-graph dashboard                    |
| GET    | `/health`             | Liveness probe                                     |
| POST   | `/run`                | Trigger a pipeline run now                         |
| GET    | `/jobs`               | List jobs (filters: `?status=&limit=&offset=`)     |
| GET    | `/jobs/{id}`          | Get one job's full detail                          |
| PATCH  | `/jobs/{id}/status`   | Update application status (new/applied/…)          |
| GET    | `/applications`       | List recorded applications                         |
| GET    | `/graph`              | Knowledge graph JSON (fuel for the D3 dashboard)   |
| GET    | `/skill-gaps`         | Aggregated skill gaps with frequency               |
| GET    | `/runs`               | Past pipeline runs and their stats                 |
| GET    | `/docs`               | OpenAPI / Swagger UI                               |

---

## How the knowledge graph works

JobRadar persists semantic memory via the official MCP
`@modelcontextprotocol/server-memory` server, spawned as a subprocess and
talked to over stdio JSON-RPC. Every pipeline run writes:

- **Entities** — `Company`, `Role`, `Skill`, `Application`, `SkillGap`,
  `Location`, `PipelineRun`
- **Relations** — `POSTED_BY`, `LOCATED_IN`, `MATCHES`, `HAS_GAP`,
  `APPLIED_TO`, `INTERVIEWED_AT`, `DISCOVERED_IN`

The D3 dashboard at `/` fetches `/graph` and renders a force-directed layout —
colors by entity type, node size by connectivity, hover for observations,
click to highlight a neighborhood, with search + per-type filters.

If Node.js / npx isn't available, the MCP client transparently falls back to
an in-memory graph so the app keeps working (you just lose cross-run
persistence).

---

## Enabling live job APIs

**Arbeitnow** (Germany-focused) — free, no auth, always on when
`USE_MOCK_DATA=false`.

**Adzuna** — free tier gives 250 req/day.

1. Register at [developer.adzuna.com](https://developer.adzuna.com/).
2. Set `ADZUNA_APP_ID` and `ADZUNA_APP_KEY` in `.env`.
3. Set `USE_MOCK_DATA=false`.

**Bundesagentur für Arbeit** — the official German government jobs API.
Enabled automatically when `USE_MOCK_DATA=false` via a public client-id header.

---

## Adding LinkedIn jobs manually

LinkedIn has no free public API and aggressively blocks scrapers. Two supported
workarounds:

1. **Paste-in CSV** — drop a CSV with columns `title,company,location,description,url`
   into `data/linkedin_paste.csv`; a tiny ingest script reads it as a
   synthetic source.
2. **Playwright fallback** — Playwright is already in `requirements.txt`; add a
   `sources/linkedin_playwright.py` using your authenticated session cookie.
   Use responsibly and within LinkedIn's ToS.

---

## Tests

```bash
pytest -q
```

Covers: source contracts, RAG scoring, full orchestrator pipeline (with
mocked Claude), and all API endpoints.

---

## Roadmap

- Replace in-process MCP memory with Neo4j for production durability
  (compose service already stubbed out)
- Playwright-based LinkedIn ingestor
- Per-company historical dashboards (already supported by `/graph` data)
- Application-status Kanban UI
- Webhook on `/run` completion (Slack / Discord)
- Fine-tuned CV-role matcher using application outcomes as feedback signal
- Multi-user support (auth + tenant-scoped memory)

---

## License

MIT.
