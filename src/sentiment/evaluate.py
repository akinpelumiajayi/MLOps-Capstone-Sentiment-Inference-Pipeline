"""Evaluation stage: held-out test metrics, latency benchmark, drift reference.

Writes:

* ``reports/metrics.json``           - DVC metrics (read by the quality gate)
* ``reports/plots/confusion.json``   - DVC confusion-matrix plot data
* ``models/reference_profile.json``  - baseline for live drift monitoring
* ``models/registry.json``           - MLflow registry name/version (if registered)

Usage::

    python -m sentiment.evaluate
"""

from __future__ import annotations

import logging
import time
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from sentiment import tracking
from sentiment.config import LABELS, Paths, load_params, read_json, write_json
from sentiment.data import load_split
from sentiment.drift import build_reference_profile
from sentiment.model import positive_proba

log = logging.getLogger(__name__)


def classification_metrics(y_true: list[int], scores: list[float], threshold: float) -> dict[str, float]:
    y_pred = [int(s >= threshold) for s in scores]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "log_loss": float(log_loss(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
    }


def benchmark_latency(model: Pipeline, texts: list[str], n: int) -> dict[str, float]:
    """Single-request in-process latency (ms), i.e. the model's share of API latency."""
    sample = (texts * (n // max(len(texts), 1) + 1))[:n]
    for t in sample[:20]:  # warm-up
        model.predict_proba([t])
    timings = []
    for t in sample:
        start = time.perf_counter()
        model.predict_proba([t])
        timings.append((time.perf_counter() - start) * 1000)
    arr = np.asarray(timings)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def evaluate(params: dict[str, Any], paths: Paths) -> dict[str, Any]:
    threshold = float(params.get("serve", {}).get("decision_threshold", 0.5))
    model: Pipeline = joblib.load(paths.model)
    metadata = read_json(paths.metadata)
    texts, labels = load_split(paths, "test")

    scores = positive_proba(model, texts)
    metrics: dict[str, Any] = {
        "test": classification_metrics(labels, scores, threshold),
        "latency": benchmark_latency(model, texts, int(params.get("evaluate", {}).get("latency_samples", 500))),
        "model": {
            "type": metadata["model_type"],
            "size_mb": round(paths.model.stat().st_size / 1e6, 3),
            "n_test_rows": len(texts),
        },
    }
    write_json(paths.metrics, metrics)

    confusion = [
        {"actual": LABELS[y], "predicted": LABELS[int(s >= threshold)]} for y, s in zip(labels, scores, strict=True)
    ]
    write_json(paths.reports_dir / "plots" / "confusion.json", confusion)
    write_json(paths.reference_profile, build_reference_profile(model, texts, scores, threshold))

    with tracking.run(params.get("mlflow", {}).get("experiment", "sentiment"), run_id=metadata.get("mlflow_run_id")):
        tracking.log_metrics(metrics["test"], prefix="test_")
        tracking.log_metrics(metrics["latency"], prefix="latency_")
        tracking.log_artifact(str(paths.metrics))
        tracking.log_artifact(str(paths.reference_profile))
        registered = params.get("mlflow", {}).get("registered_model")
        version = tracking.log_model(model, registered, texts[:3])
        write_json(paths.registry, {"registered_model": registered if version else None, "version": version})

    t = metrics["test"]
    log.info("Test  acc=%.4f  f1=%.4f  auc=%.4f", t["accuracy"], t["f1"], t["roc_auc"])
    log.info("Latency p50=%.2fms  p95=%.2fms", metrics["latency"]["p50_ms"], metrics["latency"]["p95_ms"])
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths = Paths()
    evaluate(load_params(paths.params), paths)


if __name__ == "__main__":
    main()
