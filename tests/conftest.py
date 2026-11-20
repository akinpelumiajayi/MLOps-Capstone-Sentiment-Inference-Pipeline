"""Shared fixtures. Everything is hermetic: no network, no MLflow server."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

os.environ["MLFLOW_ENABLED"] = "false"

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from sentiment.api.main import create_app
from sentiment.api.settings import Settings
from sentiment.config import Paths, write_json
from sentiment.drift import build_reference_profile
from sentiment.model import build_pipeline, positive_proba

POSITIVE = [
    "a wonderful, moving and beautifully acted film",
    "brilliant script and a great cast",
    "charming, funny and genuinely delightful",
    "an excellent, gripping thriller",
    "the best movie of the year, superb",
    "warm, witty and wonderful to watch",
    "a great performance in a lovely story",
    "smart, touching and beautifully made",
]
NEGATIVE = [
    "a dull, boring and lifeless mess",
    "terrible script and awful acting",
    "tedious, predictable and painfully slow",
    "the worst film I have seen, dreadful",
    "a bland, forgettable waste of time",
    "clumsy, stupid and badly edited",
    "awful dialogue and a boring plot",
    "lazy, dull and utterly pointless",
]


def tiny_split(n_repeat: int = 1) -> pd.DataFrame:
    rows = [(t, 1) for t in POSITIVE] + [(t, 0) for t in NEGATIVE]
    return pd.DataFrame(rows * n_repeat, columns=["text", "label"])


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("models")
    texts = POSITIVE + NEGATIVE
    labels = [1] * len(POSITIVE) + [0] * len(NEGATIVE)
    model = build_pipeline("logreg", {"ngram_max": 2, "min_df": 1}, C=10.0)
    model.fit(texts, labels)
    joblib.dump(model, d / "model.pkl")
    write_json(d / "metadata.json", {"version": "test-v1", "model_type": "logreg", "validation": {"f1": 0.9}})
    write_json(d / "reference_profile.json", build_reference_profile(model, texts, positive_proba(model, texts), 0.5))
    return d


@pytest.fixture(scope="session")
def settings(model_dir: Path) -> Settings:
    return Settings(
        model_path=model_dir / "model.pkl",
        metadata_path=model_dir / "metadata.json",
        reference_profile_path=model_dir / "reference_profile.json",
        max_text_length=500,
        max_batch_size=10,
        drift_update_every=1,
    )


@pytest.fixture(scope="session")
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:  # the context manager runs the lifespan
        yield c


@pytest.fixture
def project(tmp_path: Path) -> Paths:
    """A throwaway project tree with processed splits, for pipeline tests."""
    paths = Paths(root=tmp_path)
    paths.processed_dir.mkdir(parents=True)
    for split, n in (("train", 3), ("validation", 1), ("test", 1)):
        df = tiny_split(n)
        df["text"] = [f"{t} ({split} {i})" for i, t in enumerate(df["text"])]  # unique across splits
        df.to_csv(paths.processed(split), index=False)
    return paths


@pytest.fixture
def params() -> dict[str, Any]:
    return {
        "data": {"dataset": "test/dataset", "revision": "abc", "min_text_chars": 3, "max_text_chars": 500},
        "validation": {"min_rows": {"train": 1, "validation": 1, "test": 1}, "max_majority_class_share": 0.6},
        "train": {"model": "logreg", "C_grid": [1.0, 10.0], "ngram_max": 2, "min_df": 1, "random_seed": 0},
        "evaluate": {"latency_samples": 20},
        "compare": {"models": {"logreg": [1.0], "complement_nb": [0.5]}},
        "gate": {"min_f1": 0.5, "min_accuracy": 0.5, "max_f1_regression": 0.01, "max_p95_latency_ms": 1000},
        "serve": {"decision_threshold": 0.5},
        "mlflow": {"experiment": "test", "registered_model": None},
    }
