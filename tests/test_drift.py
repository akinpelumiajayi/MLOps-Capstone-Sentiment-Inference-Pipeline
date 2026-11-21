from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from sentiment.config import write_json
from sentiment.drift import DriftMonitor, build_reference_profile, histogram, main, oov_ratio, psi
from sentiment.model import build_pipeline, positive_proba


def test_histogram_normalised() -> None:
    assert histogram([0.05, 0.15, 0.15, 0.95], [0, 0.1, 0.2, 1.0]) == [0.25, 0.5, 0.25]
    assert histogram([], [0, 1]) == [0.0]


def test_psi_identical_is_small_and_shift_is_large() -> None:
    rng = np.random.default_rng(0)
    bins = [i / 10 for i in range(11)]
    ref = histogram(rng.uniform(0, 1, 5000), bins)
    assert psi(ref, histogram(rng.uniform(0, 1, 5000), bins)) < 0.02
    assert psi(ref, histogram(rng.uniform(0.8, 1, 5000), bins)) > 1.0


@pytest.fixture
def tiny_model() -> Pipeline:
    model = build_pipeline("logreg", {}, C=1.0)
    model.fit(["good film", "great cast", "bad film", "awful plot"], [1, 1, 0, 0])
    return model


def test_oov_ratio(tiny_model: Pipeline) -> None:
    assert oov_ratio(tiny_model, "good film") == 0.0
    assert oov_ratio(tiny_model, "good zebra") == 0.5
    assert oov_ratio(tiny_model, "!!!") == 1.0


def test_monitor(tiny_model: Pipeline) -> None:
    texts = ["good film", "bad film"] * 50
    ref = build_reference_profile(tiny_model, texts, positive_proba(tiny_model, texts), 0.5)
    mon = DriftMonitor(ref, window=10, threshold=0.5)
    assert mon.snapshot() is None
    for _ in range(25):
        mon.add(0.99, 9)
    assert len(mon) == 10
    snap = mon.snapshot()
    assert snap is not None
    assert snap["positive_rate"] == 1.0
    assert snap["score_psi"] > 0.2
    assert DriftMonitor(None, 10, 0.5).snapshot() is None


def test_cli(tmp_path: Path, tiny_model: Pipeline) -> None:
    texts = ["good film", "bad film"] * 50
    joblib.dump(tiny_model, tmp_path / "model.pkl")
    profile = build_reference_profile(tiny_model, texts, positive_proba(tiny_model, texts), 0.5)
    write_json(tmp_path / "ref.json", profile)
    pd.DataFrame({"text": texts}).to_csv(tmp_path / "same.csv", index=False)
    pd.DataFrame({"text": ["great cast " * 30] * 100}).to_csv(tmp_path / "shifted.csv", index=False)

    common = ["--model", str(tmp_path / "model.pkl"), "--reference", str(tmp_path / "ref.json")]
    assert main(["--input", str(tmp_path / "same.csv"), *common]) == 0
    assert main(["--input", str(tmp_path / "shifted.csv"), *common]) == 1
