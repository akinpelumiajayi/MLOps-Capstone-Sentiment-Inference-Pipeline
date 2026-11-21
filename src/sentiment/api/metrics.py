"""Prometheus metrics. Defined once at import time on the default registry."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info

HTTP_REQUESTS = Counter(
    "sentiment_http_requests_total",
    "HTTP requests by route, method and status code",
    ["method", "route", "status"],
)
HTTP_LATENCY = Histogram(
    "sentiment_http_request_duration_seconds",
    "End-to-end HTTP request latency",
    ["route"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
IN_FLIGHT = Gauge("sentiment_active_requests", "Requests currently being processed")

PREDICTIONS = Counter("sentiment_predictions_total", "Predictions by label", ["label"])
LOW_CONFIDENCE = Counter("sentiment_low_confidence_total", "Predictions below the low-confidence threshold")
INFERENCE_LATENCY = Histogram(
    "sentiment_inference_duration_seconds",
    "Model inference time per call (single or batch)",
    ["endpoint"],
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)
CONFIDENCE = Histogram(
    "sentiment_confidence_score",
    "Confidence of the returned label",
    buckets=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0),
)
TEXT_LENGTH = Histogram(
    "sentiment_text_length_chars",
    "Input length in characters",
    buckets=(10, 25, 50, 100, 200, 400, 800, 1600, 3200, 5000),
)
OOV_RATIO = Histogram(
    "sentiment_oov_ratio",
    "Share of input tokens unseen during training",
    buckets=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0),
)

MODEL_LOADED = Gauge("sentiment_model_loaded", "1 if a model is loaded and serving")
MODEL_INFO = Info("sentiment_model", "Metadata of the model being served")

DRIFT_SCORE_PSI = Gauge("sentiment_drift_score_psi", "PSI of live P(positive) vs. reference")
DRIFT_LENGTH_PSI = Gauge("sentiment_drift_length_psi", "PSI of live input length vs. reference")
DRIFT_POSITIVE_RATE = Gauge("sentiment_drift_positive_rate", "Positive rate over the rolling window")
DRIFT_REFERENCE_POSITIVE_RATE = Gauge("sentiment_drift_reference_positive_rate", "Positive rate on the test set")
DRIFT_WINDOW = Gauge("sentiment_drift_window_size", "Predictions currently in the drift window")
