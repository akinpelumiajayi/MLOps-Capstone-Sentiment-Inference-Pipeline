"""
API integration tests.

Run:  pytest tests/test_api.py -v
"""
import os
import joblib
import tempfile
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ── build a minimal trained model before importing the app ───────────────────
@pytest.fixture(scope="session", autouse=True)
def trained_model(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("models")
    model_path = tmp / "model.pkl"

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf",   LogisticRegression(max_iter=200)),
    ])
    texts  = ["great product", "terrible product", "love it", "hate it",
              "excellent", "awful", "good", "bad", "perfect", "broken"]
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    pipeline.fit(texts, labels)
    joblib.dump(pipeline, model_path)

    os.environ["MODEL_PATH"] = str(model_path)
    return model_path


@pytest.fixture(scope="session")
def client(trained_model):
    from app.main import app
    return TestClient(app)


# ── tests ─────────────────────────────────────────────────────────────────────
def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_positive(client):
    resp = client.post("/predict", json={"text": "This model is excellent and very accurate"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in ("positive", "negative")   # valid label
    assert 0.0 <= body["confidence"] <= 1.0
    assert "model_version" in body


def test_predict_negative(client):
    resp = client.post("/predict", json={"text": "Terrible results, completely broken"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in ("positive", "negative")
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_empty_text(client):
    resp = client.post("/predict", json={"text": "   "})
    assert resp.status_code == 422


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "sentiment_requests_total" in resp.text
