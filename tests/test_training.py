from __future__ import annotations

import json
from typing import Any

import joblib
import pytest

from sentiment.compare import compare
from sentiment.config import Paths, read_json
from sentiment.evaluate import evaluate
from sentiment.model import MODEL_TYPES, build_pipeline, positive_proba
from sentiment.train import model_version, train


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_every_model_type_gives_probabilities(model_type: str) -> None:
    # repeated so LinearSVC's 3-fold calibration has enough samples per fold
    texts = ["good", "great", "fine", "nice", "bad", "awful", "poor", "dull"] * 4
    labels = [1, 1, 1, 1, 0, 0, 0, 0] * 4
    model = build_pipeline(model_type, {}, C=1.0)
    model.fit(texts, labels)
    probs = positive_proba(model, ["good", "awful"])
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert probs[0] > probs[1]


def test_unknown_model_type() -> None:
    with pytest.raises(ValueError, match="Unknown model type"):
        build_pipeline("xgboost", {})


def test_model_version(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("GIT_SHA", "GITHUB_SHA", "GIT_COMMIT"):
        monkeypatch.delenv(var, raising=False)
    assert model_version("logreg") == "logreg-local"
    monkeypatch.setenv("GITHUB_SHA", "0123456789abcdef")
    assert model_version("logreg") == "logreg-0123456"


def test_train_then_evaluate(project: Paths, params: dict[str, Any]) -> None:
    metadata = train(params, project)
    assert project.model.exists()
    assert metadata["best_C"] in params["train"]["C_grid"]
    assert read_json(project.metadata)["model_type"] == "logreg"

    metrics = evaluate(params, project)
    assert set(metrics) == {"test", "latency", "model"}
    assert metrics["test"]["accuracy"] > 0.5
    profile = read_json(project.reference_profile)
    assert abs(sum(profile["score_hist"]) - 1.0) < 1e-6
    confusion = json.loads((project.reports_dir / "plots" / "confusion.json").read_text())
    assert len(confusion) == metrics["model"]["n_test_rows"]
    assert joblib.load(project.model).predict(["wonderful"])[0] in (0, 1)
    assert read_json(project.registry) == {"registered_model": None, "version": None}  # MLflow disabled


def test_compare_writes_report(project: Paths, params: dict[str, Any]) -> None:
    rows = compare(params, project)
    assert {r["model"] for r in rows} == {"logreg", "complement_nb"}
    assert (project.reports_dir / "model_comparison.md").read_text().startswith("| Model |")
