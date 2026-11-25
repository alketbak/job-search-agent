"""Central configuration loaded from environment variables.

All configurable values live here so no other module needs to touch os.environ.
Keep this the single source of truth - if you add a new env var, wire it in this
file and document it in `.env.example`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root. This is idempotent; calling it multiple times
# is safe because python-dotenv only sets values that are not already in env.
load_dotenv()


def _bool(value: str | None, default: bool = False) -> bool:
    """Parse a string into a boolean. Accepts common truthy values."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable settings snapshot. Instantiate once via `settings`."""

    # --- LLM ---
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    # Legacy single-tier setting from when the LLM decided the score. Nothing
    # reads it anymore; kept defined so an existing CLAUDE_MODEL in .env
    # doesn't error. Use LLM_MODEL_FAST / LLM_MODEL_SMART instead.
    claude_model: str = field(
        default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    )
    # Two-tier model config - configurable via env, never hardcoded
    # elsewhere. Defaults are current Anthropic model IDs; override via env
    # if your account has different model access. extraction/ uses the fast
    # tier, scoring/explainer.py the smart tier.
    llm_model_fast: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL_FAST", "claude-haiku-4-5-20251001")
    )
    llm_model_smart: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL_SMART", "claude-sonnet-5")
    )

    # --- Public demo ---
    # When true: forces mock job data and the fictional example CV
    # regardless of USE_MOCK_DATA/CV_PATH, and POST /cv returns 403 (see
    # api/router.py::upload_cv). A real CV or real applications must never
    # be reachable from a public deployment - this is a hard override, not
    # a default, so a misconfigured .env can't accidentally expose one.
    demo_mode: bool = field(default_factory=lambda: _bool(os.getenv("DEMO_MODE"), False))

    # --- Job sources ---
    adzuna_app_id: str = field(default_factory=lambda: os.getenv("ADZUNA_APP_ID", ""))
    adzuna_app_key: str = field(default_factory=lambda: os.getenv("ADZUNA_APP_KEY", ""))
    use_mock_data: bool = field(default_factory=lambda: _bool(os.getenv("USE_MOCK_DATA"), True))
    default_query: str = field(
        default_factory=lambda: os.getenv("DEFAULT_QUERY", "AI Engineer")
    )
    default_location: str = field(
        default_factory=lambda: os.getenv("DEFAULT_LOCATION", "Munich")
    )

    # --- CV / RAG ---
    cv_path: str = field(default_factory=lambda: os.getenv("CV_PATH", "cv/cv.txt"))
    # Upload validation for POST /cv.
    cv_max_upload_mb: int = field(default_factory=lambda: _int(os.getenv("CV_MAX_UPLOAD_MB"), 5))
    # Unused since FAISS was replaced by pgvector
    # (agents/retrieval.py) - kept defined so a stray FAISS_INDEX_PATH in an
    # existing .env doesn't error, not because anything still reads it.
    faiss_index_path: str = field(
        default_factory=lambda: os.getenv("FAISS_INDEX_PATH", "cv/cv.faiss")
    )
    # multilingual-e5-base is the default (docs/ARCHITECTURE.md)
    # (many postings are German); MiniLM stays selectable for the evaluation
    # comparison. E5 models expect "query: "/"passage: " prefixes - see
    # agents/retrieval.py.
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    )
    embedding_dim: int = field(default_factory=lambda: _int(os.getenv("EMBEDDING_DIM"), 768))
    cv_chunk_tokens: int = field(default_factory=lambda: _int(os.getenv("CV_CHUNK_TOKENS"), 200))
    cv_chunk_overlap: int = field(default_factory=lambda: _int(os.getenv("CV_CHUNK_OVERLAP"), 40))

    # --- Job dedup ---
    # Cosine similarity above this threshold, after canonical-URL and
    # normalized-triple matches fail, is treated as the same job.
    job_dedup_embedding_threshold: float = field(
        default_factory=lambda: _float(os.getenv("JOB_DEDUP_EMBEDDING_THRESHOLD"), 0.93)
    )

    # --- Scoring ---
    # Weights for the deterministic hybrid score. Must
    # sum to 1.0 - checked in __post_init__, not enforced at the type level
    # since each is independently overridable via env for tuning.
    score_weight_technical_skills: float = field(
        default_factory=lambda: _float(os.getenv("SCORE_WEIGHT_TECHNICAL_SKILLS"), 0.35)
    )
    score_weight_experience: float = field(
        default_factory=lambda: _float(os.getenv("SCORE_WEIGHT_EXPERIENCE"), 0.25)
    )
    score_weight_responsibilities: float = field(
        default_factory=lambda: _float(os.getenv("SCORE_WEIGHT_RESPONSIBILITIES"), 0.15)
    )
    score_weight_seniority: float = field(
        default_factory=lambda: _float(os.getenv("SCORE_WEIGHT_SENIORITY"), 0.10)
    )
    score_weight_location_language: float = field(
        default_factory=lambda: _float(os.getenv("SCORE_WEIGHT_LOCATION_LANGUAGE"), 0.10)
    )
    score_weight_education: float = field(
        default_factory=lambda: _float(os.getenv("SCORE_WEIGHT_EDUCATION"), 0.05)
    )
    # Final score is capped at this value if any mandatory requirement is
    # classified "missing" - a blocker caps the score, never hidden in a
    # high weighted average.
    score_blocker_cap: float = field(
        default_factory=lambda: _float(os.getenv("SCORE_BLOCKER_CAP"), 0.3)
    )
    # Cosine similarity above this counts as a semantic (not exact-name)
    # requirement match, inside the skills/responsibilities components.
    semantic_match_threshold: float = field(
        default_factory=lambda: _float(os.getenv("SEMANTIC_MATCH_THRESHOLD"), 0.55)
    )
    # Only the top N scored jobs get an LLM explanation (cost control).
    explanation_top_n: int = field(
        default_factory=lambda: _int(os.getenv("EXPLANATION_TOP_N"), 5)
    )
    # Deterministic apply_recommendation rule: score_final >= this AND no
    # blockers. Replaces the old holistic-LLM apply_recommendation boolean.
    score_apply_recommendation_threshold: float = field(
        default_factory=lambda: _float(os.getenv("SCORE_APPLY_RECOMMENDATION_THRESHOLD"), 0.6)
    )

    # --- Email ---
    email_sender: str = field(default_factory=lambda: os.getenv("EMAIL_SENDER", ""))
    email_recipient: str = field(default_factory=lambda: os.getenv("EMAIL_RECIPIENT", ""))
    email_password: str = field(default_factory=lambda: os.getenv("EMAIL_PASSWORD", ""))
    smtp_host: str = field(default_factory=lambda: os.getenv("SMTP_HOST", "smtp.gmail.com"))
    smtp_port: int = field(default_factory=lambda: _int(os.getenv("SMTP_PORT"), 587))

    # --- Database ---
    # Postgres+pgvector is the default store (docs/ARCHITECTURE.md §4). SQLite stays supported - `tests/conftest.py` overrides
    # this to a temp sqlite file for fast, dependency-free unit tests - but
    # anything involving vector columns (job_embeddings, cv_chunks) requires
    # a real Postgres and is skipped on sqlite (see tests/test_retrieval.py).
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "postgresql+psycopg://jobradar:jobradar@localhost:5432/jobradar"
        )
    )
    # async URL derived from the sync one. SQLAlchemy's `psycopg` (v3) driver
    # name is shared by both the sync Alembic engine and the async runtime
    # engine - same connection string works for both, no URL rewrite needed.
    # sqlite still needs the `+aiosqlite` rewrite for the async engine.
    async_database_url: str = field(init=False)

    # --- Scheduler ---
    schedule_hour: int = field(default_factory=lambda: _int(os.getenv("SCHEDULE_HOUR"), 6))
    schedule_minute: int = field(default_factory=lambda: _int(os.getenv("SCHEDULE_MINUTE"), 0))
    # The scheduler can run as its own process (scheduler_worker.py),
    # separate from the API. Default true so `python main.py`
    # alone (the documented no-Docker manual setup) still runs the daily
    # digest; docker-compose's `api` service sets this false and the
    # separate `scheduler` service (scheduler_worker.py) owns it instead.
    run_scheduler: bool = field(default_factory=lambda: _bool(os.getenv("RUN_SCHEDULER"), True))
    scheduler_heartbeat_file: str = field(
        default_factory=lambda: os.getenv("SCHEDULER_HEARTBEAT_FILE", "data/scheduler_heartbeat")
    )

    # --- Pipeline tuning ---
    max_jobs_per_source: int = field(
        default_factory=lambda: _int(os.getenv("MAX_JOBS_PER_SOURCE"), 25)
    )
    # Unused - deterministic scoring is cheap
    # enough to run on every scraped job (no LLM involved), so there's no
    # need to pre-filter before scoring. `explanation_top_n` now controls
    # how many jobs get the one expensive step (the LLM explanation).
    # Kept defined so a stray TOP_N_FOR_ANALYST in an existing .env doesn't
    # error, same pattern as `faiss_index_path`.
    top_n_for_analyst: int = field(
        default_factory=lambda: _int(os.getenv("TOP_N_FOR_ANALYST"), 20)
    )
    # Bounds concurrent LLM explanation calls
    # (scoring/explainer.py) rather than the old analyst agent's calls.
    # Env var name kept for backward compatibility.
    analyst_concurrency: int = field(
        default_factory=lambda: _int(os.getenv("ANALYST_CONCURRENCY"), 5)
    )

    # --- MCP memory server ---
    mcp_memory_cmd: str = field(
        default_factory=lambda: os.getenv(
            "MCP_MEMORY_CMD", "npx -y @modelcontextprotocol/server-memory"
        )
    )
    mcp_memory_file: str = field(
        default_factory=lambda: os.getenv("MCP_MEMORY_FILE", "data/memory.json")
    )

    # --- Observability ---
    # Langfuse cloud, optional - enabled only if both
    # keys are set; otherwise observability/tracing.py no-ops, same pattern
    # as the MCP memory client's graceful fallback above.
    langfuse_public_key: str = field(default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", ""))
    langfuse_secret_key: str = field(default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY", ""))
    langfuse_host: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )
    langfuse_enabled: bool = field(init=False)
    # Optional JSON string: {"<model_id>": {"input_per_mtok": <usd>, "output_per_mtok": <usd>}}.
    # Left unset by default - we don't hardcode Anthropic pricing we aren't
    # certain is current - a wrong cost figure is a fabricated result. Without this, cost per
    # job is reported as unavailable rather than guessed.
    anthropic_pricing_json: str = field(default_factory=lambda: os.getenv("ANTHROPIC_PRICING_JSON", ""))

    # --- Auth ---
    # Signs/verifies JWTs (fastapi_users.authentication.JWTStrategy). No
    # default - an empty secret would mean every deployment that forgets
    # to set this signs tokens with the same well-known value, so
    # `auth/backend.py` raises at first use rather than silently accepting
    # one (same "required, raise at point of use" pattern as
    # `anthropic_api_key` in extraction/llm_json.py).
    auth_secret: str = field(default_factory=lambda: os.getenv("AUTH_SECRET", ""))
    auth_token_lifetime_seconds: int = field(
        default_factory=lambda: _int(os.getenv("AUTH_TOKEN_LIFETIME_SECONDS"), 3600)
    )

    # --- Redis ---
    # Backs both the arq job queue (CV extraction, per-user matching -
    # agents/worker.py) and slowapi's rate limiter (api/rate_limit.py) -
    # one piece of infra for both, not two.
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))

    # --- Per-user LLM quota ---
    # A request-count cap, not a dollar cap - `anthropic_pricing_json`
    # below is deliberately often unset, so a dollar-denominated quota
    # would either be fabricated or silently inactive. Counting real LLM
    # calls per user per day works regardless of whether pricing is known.
    llm_daily_quota_per_user: int = field(
        default_factory=lambda: _int(os.getenv("LLM_DAILY_QUOTA_PER_USER"), 50)
    )

    # --- Rate limiting ---
    # `api/rate_limit.py`'s IP-keyed counters live in real Redis, which
    # persists across separate pytest invocations (unlike the per-test
    # temp SQLite file) - and FastAPI's TestClient reports every request
    # as coming from the same fixed "testclient" host, so without this
    # escape hatch every test file sharing that one IP would slowly burn
    # down a single shared budget across the whole suite (and across
    # re-runs within the same rate-limit window), eventually turning
    # unrelated tests flaky. tests/conftest.py disables this by default;
    # tests/test_rate_limit.py re-enables it explicitly to test the real
    # boundary behavior.
    rate_limiting_enabled: bool = field(
        default_factory=lambda: _bool(os.getenv("RATE_LIMITING_ENABLED"), True)
    )

    # --- GDPR ---
    # Policy value surfaced in docs/PRIVACY.md and `GET /me/export`, not an
    # automatic purge job: there's no need yet for a retention-enforcement
    # cron, and building one speculatively isn't worth it.
    data_retention_days: int = field(
        default_factory=lambda: _int(os.getenv("DATA_RETENTION_DAYS"), 365)
    )

    # --- Logging / misc ---
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    output_dir: str = field(default_factory=lambda: os.getenv("OUTPUT_DIR", "output"))

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "langfuse_enabled", bool(self.langfuse_public_key and self.langfuse_secret_key)
        )

        # DEMO_MODE overrides USE_MOCK_DATA/CV_PATH unconditionally - a
        # public demo must never serve real job data or a real CV even if
        # the rest of .env was copied from a non-demo deployment.
        if self.demo_mode:
            object.__setattr__(self, "use_mock_data", True)
            object.__setattr__(self, "cv_path", "cv/cv.example.txt")

        # Derive the async-driver URL from the sync one. We write via
        # object.__setattr__ because the dataclass is frozen.
        sync_url = self.database_url
        if sync_url.startswith("sqlite:///") and "+aiosqlite" not in sync_url:
            async_url = sync_url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
        elif sync_url.startswith("postgresql://"):
            # Bare "postgresql://" has no driver - both the sync (Alembic)
            # and async (runtime) engines use the same psycopg driver name,
            # so this one rewrite covers both.
            async_url = sync_url.replace("postgresql://", "postgresql+psycopg://", 1)
        else:
            async_url = sync_url
        object.__setattr__(self, "async_database_url", async_url)

        weight_sum = (
            self.score_weight_technical_skills
            + self.score_weight_experience
            + self.score_weight_responsibilities
            + self.score_weight_seniority
            + self.score_weight_location_language
            + self.score_weight_education
        )
        if abs(weight_sum - 1.0) > 1e-6:
            logging.getLogger("jobradar.config").warning(
                "Score weights sum to %.4f, not 1.0 - scores will be scaled "
                "accordingly rather than failing startup.",
                weight_sum,
            )


settings = Settings()


def configure_logging() -> None:
    """Configure root logger. Safe to call multiple times."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Quiet down noisy third-party loggers so our pipeline logs stand out.
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def ensure_dirs() -> None:
    """Ensure runtime directories exist. Called on startup."""
    for d in (settings.output_dir, "cv", "data"):
        Path(d).mkdir(parents=True, exist_ok=True)
