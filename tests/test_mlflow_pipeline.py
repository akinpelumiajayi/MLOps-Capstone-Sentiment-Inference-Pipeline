"""Integration tests: real MLflow (local SQLite registry) and the no-DVC runner."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from sentiment import pipeline
from sentiment.config import SPLITS, Paths, read_json
from sentiment.evaluate import evaluate
from sentiment.gate import main as gate_main
from sentiment.train import train


@pytest.fixture
def mlflow_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_ENABLED", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


def test_train_evaluate_register_and_promote(
    project: Paths, params: dict[str, Any], mlflow_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mlflow import MlflowClient

    params["mlflow"]["registered_model"] = "sentiment-test"
    metadata = train(params, project)
    assert metadata["mlflow_run_id"]

    evaluate(params, project)
    meta = read_json(project.metadata)
    assert read_json(project.registry) == {"registered_model": "sentiment-test", "version": "1"}
    run = MlflowClient().get_run(meta["mlflow_run_id"])
    assert "test_f1" in run.data.metrics
    assert "val_f1" in run.data.metrics

    # The gate CLI reads params.yaml from the working directory.
    monkeypatch.chdir(project.root)
    (project.root / "params.yaml").write_text(_yaml(params))
    assert gate_main(["--promote"]) == 0
    assert str(MlflowClient().get_model_version_by_alias("sentiment-test", "champion").version) == "1"


def _yaml(data: dict[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(data)


def test_pipeline_runner(project: Paths, params: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = project.root
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(_yaml(params))
    calls: list[str] = []
    frames = {split: pd.read_csv(project.processed(split)) for split in SPLITS}
    shutil.rmtree(project.processed_dir)

    def fake_download(p: dict[str, Any], paths: Paths) -> None:
        calls.append("download")
        paths.raw_dir.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            frames[split].to_parquet(paths.raw(split))

    monkeypatch.setattr(pipeline, "download", fake_download)

    pipeline.main([])
    assert calls == ["download"]
    assert (tmp_path / "models" / "model.pkl").exists()
    assert (tmp_path / "reports" / "metrics.json").exists()

    pipeline.main(["--skip-if-exists"])  # second run is a no-op
    assert calls == ["download"]
