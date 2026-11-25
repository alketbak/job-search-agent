# JobRadar

![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-1C3A53)
![PostgreSQL](https://img.shields.io/badge/Postgres-pgvector-4169E1?logo=postgresql&logoColor=white)
![Claude](https://img.shields.io/badge/LLM-Claude-D97757)
![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)

**JobRadar** is a job matching system that ranks job postings against a CV
with a **deterministic, explainable score**. The LLM extracts structured data
and explains each result, but it never decides the score.

It collects jobs from German job APIs, removes duplicates, turns CVs and job
ads into typed profiles, and scores every requirement against evidence from
the CV. A missing mandatory requirement becomes an explicit **blocker**
instead of disappearing into a high average. You can also add jobs you
found yourself: they stay private to your account and are scored against
your own CV. On top of that: application tracking, LLM observability, an
evaluation harness, and multi-user auth with GDPR basics.

---

## Why it's built this way

| Problem with "ask the LLM for a score" | What JobRadar does |
|---|---|
| Scores aren't reproducible or testable | Weighted hybrid score computed in code; the same inputs always give the same result |
| A good average hides a hard disqualifier | Missing mandatory requirements are **blockers** that cap the score |
| "Fit: 7/10" without evidence | Each requirement is classified `met / partial / missing / unverifiable`, with the CV snippet that supports it |
| "Knows Python" counts the same everywhere | Skills are weighted by evidence: professional > academic > personal project > only mentioned |
| Job ads can contain prompt injection | Job text is treated as untrusted content in prompts (tested with an adversarial posting) |
| Quality claims without measurement | Precision@K / Recall@K / NDCG@K harness based on real human labels only |

Full rationale: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Architecture

```
                 ┌──────────────────────────────────────────────┐
                 │                FastAPI (api)                  │
                 │  JWT auth · user-scoped endpoints · rate      │
                 │  limits · per-user LLM quota · D3 dashboard   │
                 └──────┬──────────────────────────────┬────────┘
                        │ enqueue                      │
                        ▼                              │
     ┌────────────────────────────┐   ┌───────────────┴─────────┐
     │  worker (arq + Redis)       │   │  scheduler               │
     │  CV extraction ·            │   │  (separate process,      │
     │  per-user matching runs     │   │   daily pipeline run)    │
     └─────────────┬──────────────┘   └───────────┬─────────────┘
                   └──────────────┬────────────────┘
                                  ▼
                 ┌──────────────────────────────────────────────┐
                 │            LangGraph pipeline                 │
                 │  ingest → dedupe → extract → retrieve         │
                 │        → score → explain → persist → notify   │
                 └──────────────────────────────────────────────┘
                                  │
                                  ▼
          PostgreSQL + pgvector (Alembic) · Langfuse (optional, PII-redacted)
```

| Stage | What happens |
|---|---|
| **Ingest** | Arbeitnow, Adzuna, Bundesagentur für Arbeit, or mock data |
| **Dedupe** | Canonical URL → normalized (title, company, location) → embedding similarity |
| **Extract** | CV → `CandidateProfile`, job ad → `JobProfile` (Pydantic). Versioned prompts, one retry on invalid JSON, job extraction cached by content hash |
| **Retrieve** | `multilingual-e5-base` embeddings in pgvector (many postings are German) |
| **Score** | Technical skills 0.35 · experience 0.25 · responsibilities 0.15 · seniority 0.10 · location/language 0.10 · education 0.05, all configurable |
| **Explain** | The smart-tier LLM explains the top N matches **using only the structured match result** |
| **Persist / notify** | Postgres, markdown digest (optional email), MCP knowledge graph |

---

## Quick start (Docker)

```bash
git clone <this repo> job-search-agent && cd job-search-agent
cp .env.example .env     # set ANTHROPIC_API_KEY and AUTH_SECRET;
                         # USE_MOCK_DATA=true needs no job-API keys
docker compose up -d --build
```

This starts six services: `postgres`, `redis`, a one-shot `migrate`, `api`,
`scheduler` and `worker`. The OpenAPI UI is at
[localhost:8000/docs](http://localhost:8000/docs).

```bash
# register and log in
curl -X POST localhost:8000/auth/register -H 'content-type: application/json' \
  -d '{"email": "me@example.com", "password": "a-long-password"}'
TOKEN=$(curl -s -X POST localhost:8000/auth/jwt/login \
  -d 'username=me@example.com&password=a-long-password' | jq -r .access_token)

# upload a CV (extraction runs in the background worker)
curl -X POST localhost:8000/cv -H "Authorization: Bearer $TOKEN" \
  -F file=@/path/to/your-cv.pdf -F consent=true

# start a matching run, then poll it
curl -X POST localhost:8000/run -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{"query": "AI Engineer", "location": "Munich"}'
curl localhost:8000/runs/<run_id> -H "Authorization: Bearer $TOKEN"
curl localhost:8000/match-results -H "Authorization: Bearer $TOKEN"

# add a job you found yourself: private to you, tracked and scored in the background
curl -X POST localhost:8000/jobs -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"title": "ML Engineer", "company": "Acme", "url": "https://example.com/jobs/1"}'
```

**Public demo:** with `DEMO_MODE=true`, the app always uses mock jobs and a
placeholder CV, and CV upload is disabled. A real CV can't end up in a public
deployment by accident.

Then open [localhost:8000](http://localhost:8000) for the dashboard: sign in
or create an account, then browse the knowledge graph or the Kanban board of
your applications (status, interviews, feedback, score breakdown). From the
Applications tab you can upload your CV, add jobs by hand, and (re)score any
job against your current CV.

Local development without Docker, tests and implementation notes:
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). Deployment (Railway/Neon):
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

---

## API

| Method | Path | Purpose |
|---:|---|---|
| POST | `/auth/register`, `/auth/jwt/login` | Create an account, get a JWT |
| POST | `/cv` | Upload a CV (PDF/text, explicit consent), extracted in the background |
| GET | `/cv/{id}`, `/me/cv` | Structured profile (skills with evidence levels, experience, education); your latest CV and its status |
| POST | `/run` | Queue a matching run scored against your own CV (rate-limited; needs an uploaded CV) |
| GET | `/runs`, `/runs/{id}` | Run history and status |
| GET | `/match-results` | Score breakdown, blockers, requirement evidence, explanation |
| GET | `/jobs`, `/jobs/{id}` | Collected jobs (shared) plus the ones you added (private) |
| POST | `/jobs` | Add a job yourself: private to your account, tracked and scored against your CV |
| POST | `/jobs/{id}/score` | (Re)score one job against your current CV |
| GET/POST/PATCH | `/applications`, `/applications/{id}` | Application tracking with a 9-status lifecycle |
| GET | `/applications/kanban` | Applications grouped by status |
| GET/POST | `/applications/{id}/interviews`, `/applications/{id}/feedback` | Interview stages and outcomes |
| GET | `/me/usage` | Today's LLM quota usage |
| GET | `/me/export` | GDPR export of everything the account owns |
| DELETE | `/me` | Account deletion, with per-table counts of what was deleted |
| GET | `/graph`, `/skill-gaps` | Knowledge graph and aggregated skill gaps |
| GET | `/health` | Liveness check |

Application lifecycle: `interested → saved → applied → screening → interview
→ technical_assessment → offer / rejected / withdrawn`. Outcomes are kept for
your own records and are **never** used as relevance labels, because
rejections have many causes that have nothing to do with fit. A static test
enforces this.

---

## Evaluation

`evaluation/run_eval.py` compares four ranking methods on the same labeled
set and writes a reproducible report (config, dataset hash, date) to
`evaluation/reports/`:

1. v1 baseline: MiniLM cosine similarity
2. pgvector similarity with `multilingual-e5-base`
3. the deterministic hybrid score
4. the hybrid score plus LLM rerank (optional)

Metrics: Precision@5/10, Recall@5/10, NDCG@10, latency and cost per job, plus
extraction precision/recall against a hand-checked gold set.

**Current status: no results to report yet.** Labels come only from a real
person judging their own fit (`python evaluation/label_jobs.py`); they are
never generated or simulated. Reports mark themselves as not meaningful
below 40 labeled jobs, and this README will only show numbers that come from
an actual run.

---

## Engineering

- **Tests:** 199 pytest tests. LLM calls are always mocked, and CI runs with
  an empty API key so an unmocked call fails immediately. Postgres- and
  Redis-dependent tests skip cleanly when those services aren't available.
- **CI:** GitHub Actions runs ruff, mypy on the core modules, pytest against
  a Postgres service container, and a Docker build.
- **Security and privacy:** JWT auth, per-user data isolation (tested),
  IP-based rate limiting, a per-user daily LLM quota, PII redaction in
  Langfuse traces, no CV content in logs, private hand-added jobs, and a
  multi-stage non-root Docker image whose build context excludes secrets
  and CVs.
- **Observability:** Langfuse traces for every LLM call (model, prompt
  version, tokens, latency, retries). Costs are only estimated when real
  prices are configured; they are never hardcoded.

---

## Known limitations

- The MCP knowledge graph is shared across users rather than partitioned
  per user. Postgres is the source of truth.
- Scores aren't recomputed automatically after you upload a new CV; rescore
  a job from its detail panel (or `POST /jobs/{id}/score`).
- No labeled evaluation data yet (see above).

More in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#known-issues).

## Roadmap

- Per-user knowledge graph derived from Postgres
- Labeled evaluation set and first published metrics
- Automatic data-retention enforcement
