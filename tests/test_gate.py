from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sentiment.config import write_json
from sentiment.gate import main, run_gate, to_markdown

CFG = {"min_f1": 0.75, "min_accuracy": 0.75, "max_f1_regression": 0.01, "max_p95_latency_ms": 25}


def metrics(f1: float = 0.8, acc: float = 0.8, p95: float = 1.0) -> dict[str, Any]:
    return {"test": {"f1": f1, "accuracy": acc}, "latency": {"p95_ms": p95}}


def test_pass() -> None:
    assert run_gate(metrics(), CFG)["passed"]


@pytest.mark.parametrize("kwargs", [{"f1": 0.7}, {"acc": 0.6}, {"p95": 30.0}])
def test_absolute_thresholds(kwargs: dict[str, float]) -> None:
    assert not run_gate(metrics(**kwargs), CFG)["passed"]


def test_regression_vs_baseline() -> None:
    assert run_gate(metrics(f1=0.795), CFG, baseline=metrics(f1=0.80))["passed"]
    result = run_gate(metrics(f1=0.78), CFG, baseline=metrics(f1=0.80))
    assert not result["passed"]
    assert "FAILED" in to_markdown(result)


def test_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(
        "gate: {min_f1: 0.75, min_accuracy: 0.75, max_f1_regression: 0.01, max_p95_latency_ms: 25}\n"
    )
    write_json(tmp_path / "reports" / "metrics.json", metrics(f1=0.78))
    write_json(tmp_path / "baseline.json", metrics(f1=0.80))
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    assert main([]) == 0
    assert main(["--baseline", "baseline.json"]) == 1
    assert "quality gate" in summary.read_text()
