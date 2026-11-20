from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sentiment.api.main import create_app
from sentiment.api.settings import Settings


def test_healthz(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readyz(client: TestClient) -> None:
    resp = client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json()["model_version"] == "test-v1"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a wonderful and brilliant film, superb cast", "positive"),
        ("a dull, boring and awful mess", "negative"),
    ],
)
def test_predict_label(client: TestClient, text: str, expected: str) -> None:
    resp = client.post("/predict", json={"text": text})
    assert resp.status_code == 200
    assert resp.json()["label"] == expected


@pytest.mark.parametrize("text", ["wonderful", "awful", "a film", "it was a movie with actors"])
def test_confidence_matches_label(client: TestClient, text: str) -> None:
    """Regression test: confidence is the probability of the *returned* label."""
    body = client.post("/predict", json={"text": text}).json()
    p_pos = body["positive_probability"]
    assert 0.5 <= body["confidence"] <= 1.0
    if body["label"] == "positive":
        assert p_pos >= 0.5
        assert body["confidence"] == pytest.approx(p_pos, abs=1e-4)
    else:
        assert p_pos < 0.5
        assert body["confidence"] == pytest.approx(1 - p_pos, abs=1e-4)
    assert body["low_confidence"] == (body["confidence"] < 0.65)


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "   "}, {}, {"text": 123}])
def test_predict_rejects_invalid_input(client: TestClient, payload: dict[str, Any]) -> None:
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_too_long(client: TestClient) -> None:
    resp = client.post("/predict", json={"text": "a" * 501})
    assert resp.status_code == 422
    assert "500" in resp.json()["detail"]


def test_batch(client: TestClient) -> None:
    resp = client.post("/predict/batch", json={"texts": ["wonderful film", "awful film", "great cast"]})
    assert resp.status_code == 200
    body = resp.json()
    assert [p["label"] for p in body["predictions"]] == ["positive", "negative", "positive"]
    assert body["model_version"] == "test-v1"


def test_batch_limits(client: TestClient) -> None:
    assert client.post("/predict/batch", json={"texts": ["good"] * 11}).status_code == 422
    assert client.post("/predict/batch", json={"texts": []}).status_code == 422
    assert client.post("/predict/batch", json={"texts": ["good", " "]}).status_code == 422
    assert client.post("/predict/batch", json={"texts": ["good", "b" * 501]}).status_code == 422


def test_info(client: TestClient) -> None:
    body = client.get("/info").json()
    assert body["model_type"] == "logreg"
    assert body["decision_threshold"] == 0.5
    assert body["validation_f1"] == 0.9
    assert body["drift_reference_loaded"] is True


def test_request_id_round_trip(client: TestClient) -> None:
    assert client.get("/healthz", headers={"X-Request-ID": "abc123"}).headers["X-Request-ID"] == "abc123"
    assert len(client.get("/healthz").headers["X-Request-ID"]) == 32


def test_metrics_exposed(client: TestClient) -> None:
    client.post("/predict", json={"text": "wonderful"})
    client.get("/does-not-exist")
    text = client.get("/metrics").text
    for name in (
        "sentiment_predictions_total",
        "sentiment_http_requests_total",
        "sentiment_inference_duration_seconds",
        "sentiment_confidence_score",
        "sentiment_oov_ratio",
        "sentiment_drift_score_psi",
        "sentiment_model_loaded 1.0",
        'route="/predict"',
        'route="unmatched"',
    ):
        assert name in text, name


def test_model_version_override(settings: Settings) -> None:
    custom = settings.model_copy(update={"model_version": "sha-deadbeef"})
    with TestClient(create_app(custom)) as c:
        assert c.post("/predict", json={"text": "great"}).json()["model_version"] == "sha-deadbeef"


def test_works_without_reference_profile(settings: Settings, tmp_path: Path) -> None:
    custom = settings.model_copy(update={"reference_profile_path": tmp_path / "none.json"})
    with TestClient(create_app(custom)) as c:
        assert c.post("/predict", json={"text": "great"}).status_code == 200
        assert c.get("/info").json()["drift_reference_loaded"] is False


def test_not_ready_without_model(tmp_path: Path) -> None:
    settings = Settings(model_path=tmp_path / "missing.pkl")
    with TestClient(create_app(settings)) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/readyz").status_code == 503
        assert c.post("/predict", json={"text": "great"}).status_code == 503
