"""API tests for application tracking.

Covers the status lifecycle, timestamp tracking, duplicate guard,
interview/feedback sub-resources, and the Kanban grouping shape the
dashboard consumes.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.router import router
from database import init_db
from tests.auth_helpers import override_auth


@pytest.fixture(scope="module")
def client() -> TestClient:
    init_db()
    app = FastAPI()
    app.include_router(router)
    override_auth(app)
    with TestClient(app) as c:
        yield c


def _create_job(client: TestClient, title: str = "AI Engineer") -> str:
    """Insert a pipeline-style job directly via the DB (`POST /jobs` would
    also create an application, which these tests do themselves). Same approach test_dedup.py
    uses. `TestClient` fixtures here are sync, so run the async insert in
    its own loop rather than threading async through the whole file."""
    import asyncio

    from database.db import get_session
    from database.models import Job

    async def _insert() -> str:
        async with get_session() as session:
            job = Job(title=title, company="Acme", location="Munich", source="mock")
            session.add(job)
            await session.flush()
            return job.id

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_insert())
    finally:
        loop.close()


def test_create_application_defaults_to_interested(client: TestClient) -> None:
    job_id = _create_job(client, "Job A")
    r = client.post("/applications", json={"job_id": job_id})
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "interested"
    assert body["applied_at"] is None
    assert body["status_changed_at"] is not None


def test_create_application_with_explicit_applied_status_sets_applied_at(client: TestClient) -> None:
    job_id = _create_job(client, "Job B")
    r = client.post("/applications", json={"job_id": job_id, "status": "applied"})
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "applied"
    assert body["applied_at"] is not None


def test_create_application_duplicate_returns_409(client: TestClient) -> None:
    job_id = _create_job(client, "Job C")
    first = client.post("/applications", json={"job_id": job_id})
    assert first.status_code == 201
    second = client.post("/applications", json={"job_id": job_id})
    assert second.status_code == 409
    assert first.json()["id"] in second.json()["detail"]


def test_create_application_missing_job_404(client: TestClient) -> None:
    r = client.post("/applications", json={"job_id": "00000000-0000-0000-0000-000000000000"})
    assert r.status_code == 404


def test_create_application_invalid_status_400(client: TestClient) -> None:
    job_id = _create_job(client, "Job D")
    r = client.post("/applications", json={"job_id": job_id, "status": "nonsense"})
    assert r.status_code == 422  # Pydantic Literal rejects it before the handler runs


def test_patch_status_transition_sets_timestamp_and_applied_at_once(client: TestClient) -> None:
    job_id = _create_job(client, "Job E")
    app_id = client.post("/applications", json={"job_id": job_id}).json()["id"]

    r1 = client.patch(f"/applications/{app_id}", json={"status": "applied"})
    assert r1.status_code == 200
    first_applied_at = r1.json()["applied_at"]
    assert first_applied_at is not None

    # Advancing further must not reset applied_at.
    r2 = client.patch(f"/applications/{app_id}", json={"status": "screening"})
    assert r2.status_code == 200
    assert r2.json()["applied_at"] == first_applied_at
    assert r2.json()["status"] == "screening"


def test_patch_invalid_status_400(client: TestClient) -> None:
    job_id = _create_job(client, "Job F")
    app_id = client.post("/applications", json={"job_id": job_id}).json()["id"]
    r = client.patch(f"/applications/{app_id}", json={"status": "bogus"})
    assert r.status_code == 422


def test_patch_notes_and_salary_without_status_change(client: TestClient) -> None:
    job_id = _create_job(client, "Job G")
    app_id = client.post("/applications", json={"job_id": job_id}).json()["id"]
    r = client.patch(f"/applications/{app_id}", json={"notes": "great culture fit", "salary_offered": 85000})
    assert r.status_code == 200
    body = r.json()
    assert body["notes"] == "great culture fit"
    assert body["salary_offered"] == 85000
    assert body["status"] == "interested"  # untouched


def test_patch_missing_application_404(client: TestClient) -> None:
    r = client.patch("/applications/00000000-0000-0000-0000-000000000000", json={"status": "applied"})
    assert r.status_code == 404


def test_get_application_detail_includes_job_and_empty_subresources(client: TestClient) -> None:
    job_id = _create_job(client, "Job H")
    app_id = client.post("/applications", json={"job_id": job_id}).json()["id"]
    r = client.get(f"/applications/{app_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["job"]["title"] == "Job H"
    assert body["interviews"] == []
    assert body["feedback_entries"] == []


def test_get_application_detail_404(client: TestClient) -> None:
    assert client.get("/applications/00000000-0000-0000-0000-000000000000").status_code == 404


def test_interview_create_and_list(client: TestClient) -> None:
    job_id = _create_job(client, "Job I")
    app_id = client.post("/applications", json={"job_id": job_id}).json()["id"]

    r = client.post(f"/applications/{app_id}/interviews", json={"stage": "onsite", "notes": "panel of 4"})
    assert r.status_code == 201
    assert r.json()["stage"] == "onsite"

    listing = client.get(f"/applications/{app_id}/interviews")
    assert listing.status_code == 200
    assert len(listing.json()) == 1


def test_interview_create_missing_application_404(client: TestClient) -> None:
    r = client.post(
        "/applications/00000000-0000-0000-0000-000000000000/interviews", json={"stage": "onsite"}
    )
    assert r.status_code == 404


def test_feedback_create_and_list(client: TestClient) -> None:
    job_id = _create_job(client, "Job J")
    app_id = client.post("/applications", json={"job_id": job_id}).json()["id"]

    r = client.post(f"/applications/{app_id}/feedback", json={"outcome": "rejected", "notes": "went with internal candidate"})
    assert r.status_code == 201
    assert r.json()["outcome"] == "rejected"

    listing = client.get(f"/applications/{app_id}/feedback")
    assert listing.status_code == 200
    assert len(listing.json()) == 1


def test_list_applications_filters_by_status(client: TestClient) -> None:
    job_id = _create_job(client, "Job K")
    client.post("/applications", json={"job_id": job_id, "status": "offer"})
    r = client.get("/applications", params={"status": "offer"})
    assert r.status_code == 200
    assert all(a["status"] == "offer" for a in r.json())
    assert len(r.json()) >= 1


def test_list_applications_invalid_status_400(client: TestClient) -> None:
    r = client.get("/applications", params={"status": "nonsense"})
    assert r.status_code == 400


def test_kanban_groups_by_all_nine_statuses(client: TestClient) -> None:
    job_id = _create_job(client, "Job L")
    client.post("/applications", json={"job_id": job_id, "status": "withdrawn"})
    r = client.get("/applications/kanban")
    assert r.status_code == 200
    columns = r.json()["columns"]
    expected = {
        "interested", "saved", "applied", "screening", "interview",
        "technical_assessment", "offer", "rejected", "withdrawn",
    }
    assert set(columns.keys()) == expected
    assert any(card["job_title"] == "Job L" for card in columns["withdrawn"])


def test_outcomes_never_feed_into_evaluation_module() -> None:
    """Outcomes are stored as feedback signals but must never become
    automatic relevance labels. Static
    check that evaluation/ never imports the Feedback/Application models -
    the only legitimate label source is evaluation/label_jobs.py's human
    CLI, nothing derived from application outcomes.
    """
    import pathlib

    evaluation_dir = pathlib.Path(__file__).resolve().parents[1] / "evaluation"
    for path in evaluation_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "Feedback" not in text, f"{path} must not reference the Feedback model"
        assert "import Application" not in text and "Application," not in text, (
            f"{path} must not import the Application model"
        )
