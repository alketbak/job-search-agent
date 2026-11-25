# Privacy / data handling

Describes what this app actually stores, for how long, and what the
account-management endpoints do and don't cover. This is technical
documentation of GDPR basics, not a legal privacy policy for a production
deployment.

## What's stored, and why

| Data | Where | Why |
|---|---|---|
| Email + hashed password | `users` table | Account identity (fastapi-users). |
| CV text + consent timestamp | `cvs` table | What you uploaded, and proof you consented to it (`consent_given_at`). |
| Structured CV extraction (skills, experience, education, ...) | `candidate_profiles`, `candidate_skills` | What `POST /cv` parsed from your CV. |
| CV text embeddings | `cv_chunks` | Used for matching jobs to your CV (pgvector similarity). |
| Applications you're tracking, interviews, feedback | `applications`, `interviews`, `feedback` | Your own job-search tracking. |
| Match scores + LLM explanations | `match_results` | Your CV scored against a job. |
| LLM call traces (model, tokens, cost, latency - CV/job text redacted by default) | Langfuse (external, optional) | Observability - only if `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are set. |

Jobs themselves (`jobs`, `job_requirements`, `job_embeddings`) are **not**
personal data - they're job postings, shared across every account
(extraction is cached per job, not per user).

## Consent

`POST /cv` requires `consent=true` as an explicit form field - there is no
default, and omitting it is a validation error, not an implicit "no."
Consent is timestamped (`CV.consent_given_at`), not just a boolean, so
"when" is answerable later, e.g. from `GET /me/export`.

CVs loaded from `CV_PATH` (the file-based path used by the daily scheduled
pipeline run, predating the upload endpoint) don't go through this consent
flow - there's no HTTP request to collect it from. Having placed a file at
`CV_PATH` is the operator's own consent to their own deployment processing
their own file.

## Data export - `GET /me/export`

Returns everything the authenticated account owns: your user record, CVs,
candidate profiles, applications, interviews, feedback, and match results
— each one the *actual stored row* (via each model's `to_dict()`), not a
UI-trimmed summary. This does not include Langfuse traces (see below).

## Account deletion - `DELETE /me`

Deletes, explicitly and table-by-table (not relying solely on the
database's `ON DELETE CASCADE`, since SQLite - used for fast tests -
doesn't enforce foreign keys by default and a GDPR deletion silently not
happening in one environment is worth actively preventing):

- CVs and their embeddings (`cv_chunks`)
- Candidate profiles and skills
- Applications, interviews, feedback
- Match results
- The user account itself

Returns a count of what was deleted per table, so the response is
verifiable, not just a bare "ok."

**What this does NOT delete: Langfuse traces.** If `LANGFUSE_PUBLIC_KEY`/
`LANGFUSE_SECRET_KEY` were ever configured (optional), every LLM
call was traced to Langfuse's cloud service - model, tokens, cost,
latency, with CV/job text redacted by `observability/tracing.py`'s `mask`
hook, but still an external record tied (via `pipeline_run_id`) to
activity that happened under this account. `observability/tracing.py`
has no delete-by-trace-id integration with Langfuse's API - building one
hasn't been built yet. **This is a known, documented gap, not a
silent one**: if you've enabled Langfuse and need those traces gone too,
that's currently a manual step in Langfuse's own dashboard, not something
`DELETE /me` does for you.

`PipelineRun` rows are also not deleted - `PipelineRun.user_id` is
nullable and set to `NULL` (not deleted) when its owning user is removed,
consistent with pipeline runs being operational/audit data about the
shared job pipeline rather than personal data.

## Retention

`DATA_RETENTION_DAYS` (default 365, see `.env.example`) is a **policy
value**, not an automatic purge job - automatic deletion is a decision
for whoever operates a real deployment, not something to run silently by
default. In practice,
retention today means: data is kept until you call `DELETE /me` yourself.
If a real deployment needs automatic expiry after `DATA_RETENTION_DAYS`,
that's a deliberate follow-up decision for whoever runs it, not something
silently running in the background right now.

## What a public demo (`DEMO_MODE=true`) changes

See `docs/DEPLOYMENT.md` §4. In demo mode, `POST /cv` is disabled
entirely (403) and the pipeline only ever runs against the placeholder
`cv/cv.example.txt` - there is no real CV to protect, because none can be
uploaded.
