"""Central configuration loaded from environment variables.

All configurable values live here so no other module needs to touch os.environ.
Keep this the single source of truth — if you add a new env var, wire it in this
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


@dataclass(frozen=True)
class Settings:
    """Immutable settings snapshot. Instantiate once via `settings`."""

    # --- LLM ---
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    # The spec pins this exact model ID. Do not auto-upgrade without user approval.
    claude_model: str = field(
        default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    )

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
    faiss_index_path: str = field(
        default_factory=lambda: os.getenv("FAISS_INDEX_PATH", "cv/cv.faiss")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    cv_chunk_tokens: int = field(default_factory=lambda: _int(os.getenv("CV_CHUNK_TOKENS"), 200))
    cv_chunk_overlap: int = field(default_factory=lambda: _int(os.getenv("CV_CHUNK_OVERLAP"), 40))

    # --- Email ---
    email_sender: str = field(default_factory=lambda: os.getenv("EMAIL_SENDER", ""))
    email_recipient: str = field(default_factory=lambda: os.getenv("EMAIL_RECIPIENT", ""))
    email_password: str = field(default_factory=lambda: os.getenv("EMAIL_PASSWORD", ""))
    smtp_host: str = field(default_factory=lambda: os.getenv("SMTP_HOST", "smtp.gmail.com"))
    smtp_port: int = field(default_factory=lambda: _int(os.getenv("SMTP_PORT"), 587))

    # --- Database ---
    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///jobradar.db")
    )
    # async URL derived from the sync one; SQLAlchemy needs `sqlite+aiosqlite://`.
    # We keep both so the sync engine (migrations, init) and async engine can coexist.
    async_database_url: str = field(init=False)

    # --- Scheduler ---
    schedule_hour: int = field(default_factory=lambda: _int(os.getenv("SCHEDULE_HOUR"), 6))
    schedule_minute: int = field(default_factory=lambda: _int(os.getenv("SCHEDULE_MINUTE"), 0))

    # --- Pipeline tuning ---
    max_jobs_per_source: int = field(
        default_factory=lambda: _int(os.getenv("MAX_JOBS_PER_SOURCE"), 25)
    )
    top_n_for_analyst: int = field(
        default_factory=lambda: _int(os.getenv("TOP_N_FOR_ANALYST"), 20)
    )
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

    # --- Logging / misc ---
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    output_dir: str = field(default_factory=lambda: os.getenv("OUTPUT_DIR", "output"))

    def __post_init__(self) -> None:
        # Derive aiosqlite URL for async operations. We write via object.__setattr__
        # because the dataclass is frozen.
        sync_url = self.database_url
        if sync_url.startswith("sqlite:///") and "+aiosqlite" not in sync_url:
            async_url = sync_url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
        else:
            async_url = sync_url
        object.__setattr__(self, "async_database_url", async_url)


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
