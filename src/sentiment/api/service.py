"""Model loading and prediction logic, kept free of HTTP concerns."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from sentiment.api import metrics as m
from sentiment.api.schemas import Prediction
from sentiment.api.settings import Settings
from sentiment.drift import DriftMonitor, oov_ratio
from sentiment.model import positive_proba

log = logging.getLogger(__name__)


@dataclass
class ModelService:
    model: Pipeline
    metadata: dict[str, Any]
    settings: Settings
    reference: dict[str, Any] | None = None
    monitor: DriftMonitor = field(init=False)
    _since_update: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        self.monitor = DriftMonitor(self.reference, self.settings.drift_window_size, self.settings.decision_threshold)
        if self.reference:
            m.DRIFT_REFERENCE_POSITIVE_RATE.set(self.reference["positive_rate"])

    @property
    def version(self) -> str:
        return self.settings.model_version or str(self.metadata.get("version", "unknown"))

    @classmethod
    def load(cls, settings: Settings) -> ModelService:
        if not settings.model_path.exists():
            raise FileNotFoundError(f"Model not found at {settings.model_path}. Run `dvc repro` or `make train` first.")
        model = joblib.load(settings.model_path)
        metadata: dict[str, Any] = {}
        if settings.metadata_path.exists():
            metadata = json.loads(settings.metadata_path.read_text(encoding="utf-8"))
        reference = None
        if settings.reference_profile_path.exists():
            reference = json.loads(settings.reference_profile_path.read_text(encoding="utf-8"))
        else:
            log.warning("No drift reference profile at %s; drift metrics disabled", settings.reference_profile_path)
        return cls(model=model, metadata=metadata, settings=settings, reference=reference)

    def predict(self, texts: list[str], endpoint: str) -> list[Prediction]:
        start = time.perf_counter()
        scores = positive_proba(self.model, texts)
        m.INFERENCE_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)

        out: list[Prediction] = []
        for text, p_pos in zip(texts, scores, strict=True):
            positive = p_pos >= self.settings.decision_threshold
            confidence = p_pos if positive else 1.0 - p_pos
            low = confidence < self.settings.low_confidence_threshold
            label = "positive" if positive else "negative"

            m.PREDICTIONS.labels(label=label).inc()
            m.CONFIDENCE.observe(confidence)
            m.TEXT_LENGTH.observe(len(text))
            m.OOV_RATIO.observe(oov_ratio(self.model, text))
            if low:
                m.LOW_CONFIDENCE.inc()
            self.monitor.add(p_pos, len(text))

            out.append(
                Prediction(
                    label=label,
                    confidence=round(confidence, 4),
                    positive_probability=round(p_pos, 4),
                    low_confidence=low,
                    model_version=self.version,
                )
            )
        self._maybe_update_drift(len(texts))
        return out

    def _maybe_update_drift(self, n: int) -> None:
        with self._lock:
            self._since_update += n
            if self._since_update < self.settings.drift_update_every:
                return
            self._since_update = 0
        snap = self.monitor.snapshot()
        m.DRIFT_WINDOW.set(len(self.monitor))
        if snap:
            m.DRIFT_SCORE_PSI.set(snap["score_psi"])
            m.DRIFT_LENGTH_PSI.set(snap["length_psi"])
            m.DRIFT_POSITIVE_RATE.set(snap["positive_rate"])
