"""API tests for POST /cv and GET /cv/{id}. Extraction itself runs as a
background job - see tests/test_worker.py for that; it's mocked out of
these tests entirely.

Claude extraction is mocked - these test the upload/validation/persistence
plumbing, not extraction quality (that's tests/test_extraction.py).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agents.worker import extract_cv_job
from api.router import router
from database import init_db
from extraction.schemas import CandidateProfileExtraction, ExtractionFailed, SkillClaim
from tests.auth_helpers import override_auth


@pytest.fixture(scope="module")
def client() -> TestClient:
    init_db()
    app = FastAPI()
    app.include_router(router)
    override_auth(app)
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _mock_arq_pool():
    """`POST /cv` enqueues `extract_cv_job` via agents.worker.get_arq_pool -
    these tests exercise the HTTP/DB layer, not a real arq worker (that's
    tests/test_worker.py), so the pool itself is a mock whose
    `enqueue_job` calls are inspected directly."""
    pool = AsyncMock()
    with patch("api.router.get_arq_pool", new=AsyncMock(return_value=pool)):
        yield pool


def _valid_extraction() -> CandidateProfileExtraction:
    return CandidateProfileExtraction(
        skills=[
            SkillClaim(
                name="Python",
                category="language",
                evidence_level="professional",
                evidence_snippet="4 years Python",
            )
        ],
        location_preference="Berlin",
    )


def test_upload_cv_enqueues_extraction_and_returns_pending(
    client: TestClient, _mock_arq_pool: AsyncMock
) -> None:
    r = client.post(
        "/cv",
        files={"file": ("cv.txt", b"I have 4 years of Python experience.", "text/plain")},
        data={"consent": "true"},
    )
    assert r.status_code == 202
    body = r.json()
    assert body["extraction_status"] == "pending"
    assert body["profile"] is None
    assert body["consent_given_at"] is not None
    # Not asserting a specific version number - the shared sqlite test DB
    # (tests/conftest.py) persists across the whole pytest run, so other
    # test modules' CV rows for the same default user may have run first.
    assert isinstance(body["version"], int) and body["version"] >= 1

    cv_id = body["id"]
    _mock_arq_pool.enqueue_job.assert_awaited_once()
    call = _mock_arq_pool.enqueue_job.await_args
    assert call.args[0] == "extract_cv_job"
    assert call.args[1] == cv_id
    assert call.kwargs["_job_id"] == cv_id  # idempotency key


def test_upload_cv_rejects_missing_consent(client: TestClient, _mock_arq_pool: AsyncMock) -> None:
    """GDPR: no consent, no upload."""
    r = client.post(
        "/cv",
        files={"file": ("cv.txt", b"I have 4 years of Python experience.", "text/plain")},
        data={"consent": "false"},
    )
    assert r.status_code == 400
    assert "Consent is required" in r.json()["detail"]
    _mock_arq_pool.enqueue_job.assert_not_awaited()


def test_upload_cv_rejects_oversized_file(client: TestClient, _mock_arq_pool: AsyncMock) -> None:
    from config import settings

    oversized = b"x" * (settings.cv_max_upload_mb * 1024 * 1024 + 1)
    r = client.post("/cv", files={"file": ("cv.txt", oversized, "text/plain")}, data={"consent": "true"})
    assert r.status_code == 400
    assert "exceeds" in r.json()["detail"]
    _mock_arq_pool.enqueue_job.assert_not_awaited()


def test_upload_cv_rejects_unsupported_type(client: TestClient, _mock_arq_pool: AsyncMock) -> None:
    r = client.post(
        "/cv",
        files={"file": ("cv.docx", b"some bytes", "application/octet-stream")},
        data={"consent": "true"},
    )
    assert r.status_code == 400
    assert "Unsupported file type" in r.json()["detail"]


def test_upload_cv_rejects_empty_file(client: TestClient, _mock_arq_pool: AsyncMock) -> None:
    r = client.post("/cv", files={"file": ("cv.txt", b"", "text/plain")}, data={"consent": "true"})
    assert r.status_code == 400


def test_get_cv_not_found(client: TestClient) -> None:
    r = client.get("/cv/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_get_cv_after_worker_completes_extraction_roundtrips(
    client: TestClient, _mock_arq_pool: AsyncMock
) -> None:
    """Upload (pending) -> run the real worker job function directly
    (same thing a real arq worker would call) -> GET reflects success."""
    upload = client.post(
        "/cv",
        files={"file": ("cv.txt", b"4 years Python experience.", "text/plain")},
        data={"consent": "true"},
    )
    cv_id = upload.json()["id"]
    assert upload.json()["extraction_status"] == "pending"

    user_id = _mock_arq_pool.enqueue_job.await_args.args[2]
    with patch(
        "agents.worker.extract_candidate_profile", new=AsyncMock(return_value=_valid_extraction())
    ):
        result = await extract_cv_job({}, cv_id, user_id)
    assert result["status"] == "success"

    r = client.get(f"/cv/{cv_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == cv_id
    assert body["extraction_status"] == "success"
    assert body["profile"]["skills"][0]["name"] == "Python"


@pytest.mark.asyncio
async def test_extract_cv_job_marks_failed_on_extraction_failure(
    client: TestClient, _mock_arq_pool: AsyncMock
) -> None:
    upload = client.post(
        "/cv",
        files={"file": ("cv.txt", b"Some CV text that fails to parse.", "text/plain")},
        data={"consent": "true"},
    )
    cv_id = upload.json()["id"]
    user_id = _mock_arq_pool.enqueue_job.await_args.args[2]

    with patch(
        "agents.worker.extract_candidate_profile",
        new=AsyncMock(side_effect=ExtractionFailed("invalid_json", "cv_extraction_v1")),
    ):
        result = await extract_cv_job({}, cv_id, user_id)
    assert result["status"] == "failed"
    assert result["reason"] == "invalid_json"

    r = client.get(f"/cv/{cv_id}")
    body = r.json()
    assert body["extraction_status"] == "failed"
    assert body["extraction_failure_reason"] == "invalid_json"
    assert body["profile"] is None
