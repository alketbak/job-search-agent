# Deployment

A deployment guide for Railway (optionally with Neon Postgres). The local
Docker Compose stack has been built and health-checked; the cloud steps below have not
yet been run against a real Railway/Neon account.

JobRadar is three processes sharing one Postgres + pgvector database:
`api` (FastAPI), `scheduler` (the daily pipeline run, `scheduler_worker.py`),
and a one-shot `migrate` step (`alembic upgrade head`). `docker-compose.yml`
already expresses this shape locally - a cloud deploy is the same three
things pointed at a managed Postgres instead of the local container.

## 1. Choose a Postgres provider

Either works; pick based on what you already have an account with.

- **Neon** - serverless Postgres, generous free tier, needs the `vector`
  extension enabled once (`CREATE EXTENSION IF NOT EXISTS vector;` - Neon
  supports pgvector out of the box, just run the statement).
- **Railway Postgres** - a Postgres plugin inside the same Railway project
  as the app services. Simpler (one provider, one bill) but ties the DB's
  lifecycle to the Railway project.

Either way you need the resulting connection string in
`postgresql+psycopg://...` form (see `config.py`'s `database_url` -
`psycopg` v3 is the one driver used for both the sync Alembic engine and
the async runtime engine, so no URL rewriting is needed beyond that).

## 2. Railway services

Create one Railway project with three services, all built from this repo's
`Dockerfile` (same image for all three - they differ only by `command`,
exactly like `docker-compose.yml`'s `api`/`scheduler`/`migrate`):

| Service | Command | Notes |
|---|---|---|
| `migrate` | `alembic upgrade head` | Run once per deploy, before `api`/`scheduler` start. Railway's "run as a pre-deploy step" / one-off command feature fits this; a `restart: never` one-shot service also works, matching `docker-compose.yml`'s `migrate`. |
| `api` | `python main.py` | Set `RUN_SCHEDULER=false` - the separate `scheduler` service owns the daily run, same split as locally. Expose the service's public port (Railway sets `PORT`; see note below). |
| `scheduler` | `python scheduler_worker.py` | No public port needed. Needs the same `DATABASE_URL`/`CV_PATH`/`MCP_MEMORY_FILE` as `api`. |

**Port note**: `main.py` currently binds uvicorn to the fixed port 8000
(matching `docker-compose.yml`'s `EXPOSE 8000`/`ports: ["8000:8000"]`).
Railway injects a `PORT` env var and expects the app to bind to it. If
deploying to Railway specifically, either set a fixed `PORT=8000` in the
service's env vars (simplest, no code change) or change `main.py` to read
`PORT` from the environment - the latter is a real code change, not just
config, so make it deliberately rather than as a deploy-day surprise.

**Volumes**: `docker-compose.yml` mounts `./cv`, `./output`, `./data` so the
CV file, digests, and the MCP memory JSON survive a container restart.
Railway services are ephemeral by default - attach a Railway volume to
`api` and `scheduler` (both need the same CV/data volume if `RUN_SCHEDULER`
ever gets flipped between them) or the CV/memory-graph state resets on
every redeploy. For a `DEMO_MODE=true` deployment this doesn't matter (no
real CV, no scheduler needed) - see §4.

## 3. Required environment variables

Set these on both `api` and `scheduler` (the `migrate` one-off only needs
`DATABASE_URL`). Full reference with defaults: `.env.example`.

| Variable | Required | Notes |
|---|---|---|
| `DATABASE_URL` | yes | `postgresql+psycopg://...` from your Neon/Railway Postgres instance. |
| `ANTHROPIC_API_KEY` | yes, unless `DEMO_MODE=true` disables the paths that need it | Real key with billing enabled - `POST /run` and `POST /cv` spend real money per call (protected by auth, rate limits and the per-user quota). |
| `DEMO_MODE` | recommended for any public URL | See §4. |
| `RUN_SCHEDULER` | yes | `false` on `api`, unset (defaults `true`) on `scheduler`. |
| `MCP_MEMORY_FILE` | yes if you want the knowledge graph to persist | Point at the mounted volume path, e.g. `/app/data/memory.json`. |
| `CV_PATH` | yes unless `DEMO_MODE=true` | Point at the mounted volume path for a real CV; `DEMO_MODE` overrides this to `cv/cv.example.txt` regardless. |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | no | Leave blank to disable tracing entirely - `observability/tracing.py` no-ops without both keys set. |
| `EMAIL_RECIPIENT` / `EMAIL_SENDER` / `EMAIL_PASSWORD` | no | Leave blank and the daily digest is written to `output/` only, no SMTP send attempted. |

Everything else in `.env.example` has a workable default and doesn't need
setting for a first deploy.

## 4. Public demo mode

A public demo runs with mock jobs and a placeholder CV only; real CVs and
applications never go into a public deployment. Set `DEMO_MODE=true` on any service reachable from a public
URL. This (see `config.py`'s `__post_init__` and `api/router.py`'s
`upload_cv`):

- Forces `USE_MOCK_DATA=true` regardless of what `USE_MOCK_DATA` is set to.
- Forces `CV_PATH` to `cv/cv.example.txt` regardless of what `CV_PATH` is
  set to.
- Makes `POST /cv` return `403` - no new CV can be uploaded into a public
  deployment.

With `DEMO_MODE=true`, `ANTHROPIC_API_KEY` is still needed for the
scoring explanation / CV extraction calls the pipeline makes against the
placeholder example CV and mock job data - those are real (small, bounded)
API calls, just never against real user data.

## 5. Migrations on deploy

Run `alembic upgrade head` with `DATABASE_URL` pointed at the production
database *before* `api`/`scheduler` start on a new image - exactly the
`migrate` service's job in `docker-compose.yml`. On Railway, either:

- a one-off "run command" step wired into the deploy pipeline, or
- a pre-deploy hook if the project plan supports one, or
- manually triggering the `migrate` service's single run after each deploy
  and confirming it exits 0 before promoting `api`/`scheduler`.

Never run `alembic upgrade head` from a developer machine against the
production `DATABASE_URL` as a matter of habit - treat it as a deploy-time
step with its own logs, not an ad hoc local command.

## 6. First-deploy checklist

1. Provision Postgres (Neon or Railway), enable the `vector` extension.
2. Set env vars from §3 on `migrate`, `api`, `scheduler`.
3. Run `migrate`, confirm it exits 0 (check logs for the applied revision
   IDs, not just "no error").
4. Deploy `api` and `scheduler`.
5. Hit `GET /health` on the public `api` URL - expect
   `{"status": "ok", "version": "1.0.0"}` (see `api/schemas.py`'s
   `HealthResponse`; verified locally against the Docker image).
6. If `DEMO_MODE=true`, confirm `POST /cv` returns `403` and `/jobs`
   returns mock data, not an empty or real list.
7. Watch `scheduler`'s logs for the heartbeat-file write
   (`SCHEDULER_HEARTBEAT_FILE`, every 60s) - if it stops, the health check
   `docker-compose.yml` defines for it (heartbeat younger than 120s) is the
   same signal Railway's health check should use if configured.
