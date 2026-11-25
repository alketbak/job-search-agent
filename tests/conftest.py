"""Shared test fixtures.

We force `USE_MOCK_DATA=true` and point the DB at an in-memory SQLite file
before any app code imports. This keeps tests isolated and fast.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Test env - set BEFORE importing config. This is the only place we touch
# os.environ directly; everywhere else the code reads through config.settings.
_TMP = tempfile.mkdtemp(prefix="jobradar_test_")
os.environ.setdefault("USE_MOCK_DATA", "true")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP}/test.db")
os.environ.setdefault("CV_PATH", "cv/cv.txt")
os.environ.setdefault("FAISS_INDEX_PATH", f"{_TMP}/cv.faiss")
os.environ.setdefault("OUTPUT_DIR", f"{_TMP}/output")
os.environ.setdefault("MCP_MEMORY_FILE", f"{_TMP}/memory.json")
os.environ.setdefault("LOG_LEVEL", "WARNING")
# A fresh checkout's `.env.example` leaves AUTH_SECRET
# blank on purpose (no safe default - see config.py) - tests need *some*
# value to construct a UserManager at all, same reasoning as the fake
# ANTHROPIC_API_KEY above; this one is equally unusable for anything real.
os.environ.setdefault("AUTH_SECRET", "test-secret-do-not-use-in-production")
# Real Redis-backed IP rate limiting (api/rate_limit.py) would otherwise
# apply across the whole suite under one shared counter - FastAPI's
# TestClient reports every request as coming from the same "testclient"
# host, and that counter lives in real Redis, which persists across
# separate pytest invocations too. See config.py's `rate_limiting_enabled`
# docstring; tests/test_rate_limit.py re-enables it explicitly to test the
# real boundary behavior.
os.environ.setdefault("RATE_LIMITING_ENABLED", "false")

# Make the repo importable whether pytest is launched from root or tests/.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
