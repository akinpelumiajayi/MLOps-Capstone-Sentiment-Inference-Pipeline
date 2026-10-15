import os
import time
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# ── app ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Sentiment API", version="2.0.0")

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
THRESHOLD  = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))
MODEL_VER  = os.getenv("MODEL_VERSION", "v1")


def load_model():
    path = Path(MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run `python training/train.py` or `dvc repro` first."
        )
    return joblib.load(path)


model = load_model()

# ── prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "sentiment_requests_total",
    "Total prediction requests",
    ["label"],
)
REQUEST_LATENCY = Histogram(
    "sentiment_request_duration_seconds",
    "Prediction latency",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
MODEL_CONFIDENCE = Histogram(
    "sentiment_confidence_score",
    "Distribution of model confidence scores",
    buckets=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
)
ACTIVE_REQUESTS = Gauge(
    "sentiment_active_requests",
    "Requests currently being processed",
)

# ── schemas ───────────────────────────────────────────────────────────────────
class Review(BaseModel):
    text: str


class Prediction(BaseModel):
    label: str
    confidence: float
    model_version: str


# ── routes ────────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=Prediction)
def predict(review: Review):
    if not review.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    ACTIVE_REQUESTS.inc()
    start = time.perf_counter()
    try:
        proba      = model.predict_proba([review.text])[0]
        confidence = float(max(proba))
        label      = "positive" if proba[1] >= THRESHOLD else "negative"

        REQUEST_COUNT.labels(label=label).inc()
        REQUEST_LATENCY.observe(time.perf_counter() - start)
        MODEL_CONFIDENCE.observe(confidence)

        return Prediction(
            label=label,
            confidence=round(confidence, 3),
            model_version=MODEL_VER,
        )
    finally:
        ACTIVE_REQUESTS.dec()


@app.get("/healthz")
def health():
    return {"status": "ok", "model_version": MODEL_VER}


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
