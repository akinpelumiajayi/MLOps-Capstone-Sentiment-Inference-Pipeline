"""Drift detection.

Two pieces share the same maths:

* ``build_reference_profile`` runs at evaluation time and stores what "normal"
  traffic looks like (score distribution, positive rate, text length,
  out-of-vocabulary rate) next to the model.
* ``DriftMonitor`` runs inside the API, keeps a rolling window of live
  predictions and compares it to that reference with the Population
  Stability Index (PSI). Prometheus scrapes the result and alerts on it.

A batch CLI is also provided for offline checks::

    python -m sentiment.drift --input new_reviews.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

EPS = 1e-4
SCORE_BINS = [i / 10 for i in range(11)]
LENGTH_BINS = [0, 25, 50, 75, 100, 150, 200, 300, 500, 1000, 1e9]


def histogram(values: Sequence[float] | np.ndarray, bins: Sequence[float]) -> list[float]:
    """Normalised histogram (proportions summing to 1)."""
    counts, _ = np.histogram(np.asarray(values, dtype=float), bins=bins)
    total = counts.sum()
    if total == 0:
        return [0.0] * (len(bins) - 1)
    return [float(c) / float(total) for c in counts]


def psi(expected: list[float], actual: list[float]) -> float:
    """Population Stability Index. <0.1 stable, 0.1-0.2 moderate, >0.2 significant."""
    total = 0.0
    for e, a in zip(expected, actual, strict=True):
        e, a = max(e, EPS), max(a, EPS)
        total += (a - e) * math.log(a / e)
    return float(total)


def oov_ratio(model: Pipeline, text: str) -> float:
    """Share of word tokens the TF-IDF vocabulary has never seen."""
    vec = model.named_steps["tfidf"]
    tokens = vec.build_tokenizer()(vec.build_preprocessor()(text))
    if not tokens:
        return 1.0
    vocab = vec.vocabulary_
    return sum(1 for t in tokens if t not in vocab) / len(tokens)


def build_reference_profile(
    model: Pipeline,
    texts: list[str],
    scores: list[float],
    decision_threshold: float,
) -> dict[str, Any]:
    lengths = [len(t) for t in texts]
    oov = [oov_ratio(model, t) for t in texts]
    return {
        "n": len(texts),
        "score_bins": SCORE_BINS,
        "score_hist": histogram(scores, SCORE_BINS),
        "length_bins": LENGTH_BINS,
        "length_hist": histogram(lengths, LENGTH_BINS),
        "positive_rate": float(np.mean([s >= decision_threshold for s in scores])),
        "mean_length": float(np.mean(lengths)),
        "mean_oov_ratio": float(np.mean(oov)),
    }


def compare(
    reference: dict[str, Any], scores: Sequence[float], lengths: Sequence[int], threshold: float
) -> dict[str, float]:
    return {
        "score_psi": psi(reference["score_hist"], histogram(scores, reference["score_bins"])),
        "length_psi": psi(reference["length_hist"], histogram(lengths, reference["length_bins"])),
        "positive_rate": float(np.mean([s >= threshold for s in scores])) if scores else 0.0,
        "reference_positive_rate": float(reference["positive_rate"]),
    }


class DriftMonitor:
    """Thread-safe rolling window of recent predictions."""

    def __init__(self, reference: dict[str, Any] | None, window: int, threshold: float) -> None:
        self.reference = reference
        self.threshold = threshold
        self._scores: deque[float] = deque(maxlen=window)
        self._lengths: deque[int] = deque(maxlen=window)
        self._lock = threading.Lock()

    def add(self, score: float, length: int) -> None:
        with self._lock:
            self._scores.append(score)
            self._lengths.append(length)

    def __len__(self) -> int:
        return len(self._scores)

    def snapshot(self) -> dict[str, float] | None:
        if self.reference is None or not self._scores:
            return None
        with self._lock:
            scores, lengths = list(self._scores), list(self._lengths)
        return compare(self.reference, scores, lengths, self.threshold)


def main(argv: list[str] | None = None) -> int:
    import joblib
    import pandas as pd

    from sentiment.model import positive_proba

    parser = argparse.ArgumentParser(description="Offline drift check for a batch of texts.")
    parser.add_argument("--input", required=True, help="CSV with a `text` column")
    parser.add_argument("--model", default="models/model.pkl")
    parser.add_argument("--reference", default="models/reference_profile.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-psi", type=float, default=0.2)
    args = parser.parse_args(argv)

    model = joblib.load(args.model)
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    texts = pd.read_csv(args.input, keep_default_na=False)["text"].astype(str).tolist()
    scores = positive_proba(model, texts)
    report = compare(reference, scores, [len(t) for t in texts], args.threshold)
    report["mean_oov_ratio"] = float(np.mean([oov_ratio(model, t) for t in texts]))
    report["reference_mean_oov_ratio"] = reference["mean_oov_ratio"]
    report["drift_detected"] = report["score_psi"] > args.max_psi or report["length_psi"] > args.max_psi
    print(json.dumps(report, indent=2))
    return 1 if report["drift_detected"] else 0


if __name__ == "__main__":
    sys.exit(main())
