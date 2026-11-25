# Development

Local setup, conventions, and implementation notes that aren't obvious from
reading the code. Architecture and design decisions:
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env              # USE_MOCK_DATA=true needs no job-API keys
docker compose up -d postgres redis
alembic upgrade head
```

Run the pieces:

```bash
python main.py                      # API (+ in-process scheduler unless RUN_SCHEDULER=false)
arq agents.worker.WorkerSettings    # background worker: POST /cv and POST /run need it
docker compose up -d --build        # full stack: postgres, redis, migrate, api, scheduler, worker
```

Checks:

```bash
pytest -q
ruff check .
mypy config.py database scoring extraction
```

## Conventions

- Python 3.11, `from __future__ import annotations`, type hints everywhere,
  Google-style docstrings on modules and public functions.
- `config.py` is the only place environment variables are read.
- Pydantic models at every API and LLM boundary; structured logging; never
  log secrets or CV content.
- LLM calls are always mocked in tests. CI runs with `ANTHROPIC_API_KEY=""`
  on purpose so any unmocked call fails fast instead of reaching the network.

## Testing notes

- **SQLite by default, Postgres where needed.** Most tests run on SQLite for
  speed. Vector-query tests need a dedicated `jobradar_test` Postgres
  database (setup in `tests/pg_helpers.py`'s docstring) and **skip** rather
  than fail without it. A green run doesn't prove they ran; check the
  "skipped" count.
- **Redis-backed tests** (quota, rate limiting) also skip without Redis.
  `tests/conftest.py` disables rate limiting by default because
  `TestClient` reports every request from the same IP, and Redis counters
  persist across test runs. `tests/test_rate_limit.py` re-enables it
  explicitly.
- **redis-asyncio clients bind to the event loop they were first used in.**
  pytest-asyncio gives each test a fresh loop, so module-level cached
  clients have to be reset between tests (see `tests/test_quota.py`).
- **Mock `ensure_candidate_profile_from_file` explicitly** when a test
  needs a candidate profile. It's imported inside the calling function, so
  patch `extraction.cv_extractor.ensure_candidate_profile_from_file`, not
  the caller's module attribute. Don't rely on another test having cached a
  profile in the shared DB.
- `api/__init__.py` re-exports `router`, which shadows the `api.router`
  submodule. To reach module-level state (e.g. monkeypatching `settings`),
  use `sys.modules["api.router"]`.

## Implementation notes

### Pipeline

- **LangGraph state must stay msgpack-serializable.** The checkpointer
  serializes every node's output. Embeddings in state are `list[float]`,
  never `numpy.ndarray`; Pydantic objects go in as `.model_dump()` dicts.
- **One evolving `jobs` list.** Each node enriches the same list in place
  (`job_profile`, `score_breakdown`, `explanation`, …). New stages should
  do the same rather than add a parallel state key.
- **`user_id` is threaded through state.** Nodes that need "whose CV is
  this" read `state.get("user_id")` and fall back to
  `ensure_default_user()` only when it's absent (the scheduler's
  unauthenticated daily run). Calling `ensure_default_user()`
  unconditionally would score a user's run against the wrong profile.
- **A signed-in account is scored against its own uploaded CV, never
  `CV_PATH`.** With a `user_id`, `extract_node` loads the newest CV that
  account uploaded with a successfully extracted profile
  (`load_uploaded_candidate_profile`), and `retrieve_node` embeds that CV
  lazily (`ensure_chunks_for_cv`). `CV_PATH` is only the operator's CV for
  runs without a user. An account without a processed CV gets no scores
  (an empty-profile score would look real but mean nothing), and
  `POST /run` refuses it with 409.
- **Single-job scoring** (`score_job_for_user`, worker task
  `score_job_job`) runs extract -> retrieve -> score -> explain -> persist
  for one existing job. Used for hand-added jobs and the "Score against my
  CV" button (`POST /jobs/{id}/score`).
- **`PipelineRun` rows are upserted.** The worker pre-creates a `pending`
  row so `GET /runs/{run_id}` is pollable immediately; `persist_node`
  fetches-or-creates by id. The scheduler doesn't pre-create one, so the
  insert path is still needed.
- **Deduped jobs skip re-extraction**, which works because the scalar
  `JobProfile` fields (seniority, remote, salary, …) are also columns on
  `Job`. A new scalar field on `JobProfileExtraction` needs a matching
  `Job` column, or deduped jobs silently lose it.
- The LangGraph `MemorySaver` checkpointer is wired up but nothing reads
  it; `/runs` reads the `pipeline_runs` table.

### Dashboard

- `dashboard/index.html` is a single file (vanilla JS + D3, no build step).
- Every API call goes through `JobRadarAuth.apiFetch`, which attaches the
  JWT bearer token and shows the sign-in overlay on a 401. Use it for any
  new call instead of a bare `fetch`.
- The token is kept in `sessionStorage`: it survives a reload but not
  closing the tab, which limits the lifetime of a token leaked via XSS.

### Database

- **Vector columns only work on Postgres.** Models use
  `Vector(...).with_variant(JSON(), "sqlite")`, so schema creation works on
  SQLite, but vector operators raise at query time. Vector-querying code
  that must stay SQLite-safe wraps the query in a try/except and degrades
  (see `agents/dedup.py`).
- **Autogenerate always proposes dropping the HNSW indexes**
  (`ix_cv_chunks_embedding_hnsw`, `ix_job_embeddings_embedding_hnsw`)
  because they were created with raw SQL. Remove those lines from every
  newly generated migration before applying it.
- **Two status enums on purpose.** `Job.status` (legacy, drives `/jobs`)
  and `Application.status` (the real 9-state lifecycle) are independent;
  `PATCH /jobs/{id}/status` does not touch applications.
- `jobs`, `job_requirements`, `job_embeddings`, `skill_gaps` are global by
  design; everything personal is scoped to the current user.
- **Except hand-added jobs.** `Job.owner_user_id` is set for jobs a user
  added (`POST /jobs`) and NULL for scraped ones. Filter job reads with
  `job_visible_to(user_id)` / the router's `_get_visible_job` (404 for
  someone else's private job). Dedup only considers shared jobs plus the
  caller's own, so the pipeline can never merge a scraped job into a
  private one. Private jobs stay out of the global knowledge graph and the
  labeling CLI, and are exported/deleted with the account.
- **Match results are per user.** Any query for "the score of this job"
  must filter `MatchResult.user_id` - a breakdown contains evidence from
  that user's CV.

### LLM, cost and privacy

- Two model tiers: `LLM_MODEL_FAST` for extraction, `LLM_MODEL_SMART` for
  explanations.
- **Prompt-injection framing is per prompt.** `extraction/job_extractor.py`
  treats the job ad as untrusted content. Any new prompt that includes raw
  job text needs the same framing.
- **Pricing is deliberately unconfigured.** `estimate_cost` returns `None`
  unless `ANTHROPIC_PRICING_JSON` provides real prices.
- **Langfuse redaction happens in one place**, the client's `mask` hook in
  `observability/tracing.py`. Name redaction is a line heuristic that
  deliberately over-redacts short capitalized lines.
- **The per-user LLM quota covers CV extraction and single-job scoring**
  (one unit per scoring request, for the extraction + explanation it
  triggers). Extraction of shared scraped jobs isn't charged: it's cached
  and benefits every user. Explanations inside a full matching run aren't
  charged either: there's no clean per-request check point there.
- `DEMO_MODE=true` overrides `USE_MOCK_DATA` and `CV_PATH` unconditionally,
  even when they're set explicitly. That's intentional: a misconfigured
  `.env` must not put a real CV into a public demo.

- **`MCP_MEMORY_FILE` is made absolute by the client.** The memory server
  resolves a relative path against its own install directory in the npx
  cache, so every write used to fail with ENOENT and the graph stayed
  empty. Tool failures (`isError` results) now raise `MCPToolError`
  instead of being returned as data.
- Don't re-add `from api.router import router` to `api/__init__.py`: it
  rebinds `api.router` to the router object, so `import api.router` and
  `patch("api.router.x")` silently reach the wrong thing.
- `scripts/merge_duplicate_jobs.py` merges shared jobs with the same
  canonical URL or content hash (dry run by default, `--apply` to merge).

### Infrastructure

- **Rate limiting uses the `limits` library directly** via
  `api/rate_limit.py`'s `rate_limiter(limit, action)` dependency.
  `slowapi`'s decorator breaks FastAPI's forward-ref resolution under
  `from __future__ import annotations`.
- **CPU-only torch in Docker.** On `linux/aarch64`, `sentence-transformers`
  pulls a CUDA torch wheel (~1.5 GB) by default. The `Dockerfile` installs
  CPU torch first; keep that ordering.
- **Scheduler placement depends on how you run it.** Docker Compose runs it
  as its own service (`scheduler_worker.py`, API has
  `RUN_SCHEDULER=false`); bare `python main.py` runs it in-process.

### Evaluation

- `evaluation/labels/` is empty on purpose. Labels come only from a person
  judging their own fit via `evaluation/label_jobs.py`. Reports are not a
  meaningful quality signal until they show at least 40 labeled jobs.
- `evaluation/` must never import `Application` or `Feedback`: outcomes are
  not relevance labels. A static test enforces this.

## Known issues

- The MCP knowledge graph isn't partitioned per user: every authenticated
  user sees the same graph built from all users' runs.
- Every process (API, worker, scheduler) runs its own MCP memory server
  against the same `data/memory.json`. The server re-reads the file on each
  call, so they see each other's writes, but two writes at the same moment
  can lose one of them (last writer wins).
- No tests yet for `sources/adzuna.py`, `sources/bundesagentur.py`,
  `memory/knowledge_graph.py`, `agents/delivery_agent.py`;
  `memory/mcp_client.py` only has unit tests (no real server spawned).
