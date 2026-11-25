"""Tests for DEMO_MODE.

DEMO_MODE must force mock job data + the fictional example CV regardless of
USE_MOCK_DATA/CV_PATH, and POST /cv must be unreachable - a public demo must
never serve real job data, a real CV, or accept a new one.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.router import router
from config import Settings
from database import init_db
from tests.auth_helpers import override_auth


def test_demo_mode_forces_mock_data_and_example_cv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("USE_MOCK_DATA", "false")
    monkeypatch.setenv("CV_PATH", "cv/cv.txt")

    demo_settings = Settings()

    assert demo_settings.demo_mode is True
    assert demo_settings.use_mock_data is True
    assert demo_settings.cv_path == "cv/cv.example.txt"


def test_demo_mode_off_leaves_settings_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.setenv("USE_MOCK_DATA", "false")
    monkeypatch.setenv("CV_PATH", "cv/cv.txt")

    plain_settings = Settings()

    assert plain_settings.demo_mode is False
    assert plain_settings.use_mock_data is False
    assert plain_settings.cv_path == "cv/cv.txt"


@pytest.fixture(scope="module")
def client() -> TestClient:
    init_db()
    app = FastAPI()
    app.include_router(router)
    override_auth(app)
    with TestClient(app) as c:
        yield c


def test_post_cv_rejected_in_demo_mode(client: TestClient) -> None:
    import api.router as router_module

    object.__setattr__(router_module.settings, "demo_mode", True)
    try:
        response = client.post(
            "/cv",
            files={"file": ("cv.txt", b"Some CV content", "text/plain")},
            data={"consent": "true"},
        )
    finally:
        object.__setattr__(router_module.settings, "demo_mode", False)

    assert response.status_code == 403
