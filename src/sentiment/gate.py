"""Quality gate: block a model from shipping unless it clears absolute
thresholds **and** does not regress against the current production baseline.

The baseline is the ``reports/metrics.json`` committed on ``main`` (CI copies
it aside before re-running the pipeline).

Usage::

    python -m sentiment.gate --baseline baseline_metrics.json [--promote]

Exit code 0 = pass, 1 = fail.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from sentiment import tracking
from sentiment.config import Paths, load_params, read_json, write_json


def run_gate(
    metrics: dict[str, Any],
    cfg: dict[str, Any],
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    test, latency = metrics["test"], metrics["latency"]
    checks: list[dict[str, Any]] = [
        {"name": "f1 >= min_f1", "value": test["f1"], "limit": cfg["min_f1"], "passed": test["f1"] >= cfg["min_f1"]},
        {
            "name": "accuracy >= min_accuracy",
            "value": test["accuracy"],
            "limit": cfg["min_accuracy"],
            "passed": test["accuracy"] >= cfg["min_accuracy"],
        },
        {
            "name": "p95 latency <= max_p95_latency_ms",
            "value": latency["p95_ms"],
            "limit": cfg["max_p95_latency_ms"],
            "passed": latency["p95_ms"] <= cfg["max_p95_latency_ms"],
        },
    ]
    if baseline is not None:
        floor = baseline["test"]["f1"] - cfg["max_f1_regression"]
        checks.append(
            {
                "name": "f1 >= baseline_f1 - max_f1_regression",
                "value": test["f1"],
                "limit": round(floor, 4),
                "passed": test["f1"] >= floor,
            }
        )
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def to_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"### Model quality gate: {'PASSED' if result['passed'] else 'FAILED'}",
        "",
        "| Check | Value | Limit | Result |",
        "|---|---|---|---|",
    ]
    for c in result["checks"]:
        lines.append(f"| {c['name']} | {c['value']:.4f} | {c['limit']} | {'pass' if c['passed'] else 'FAIL'} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, help="metrics.json of the current production model")
    parser.add_argument("--promote", action="store_true", help="on pass, set the MLflow 'champion' alias")
    args = parser.parse_args(argv)

    paths = Paths()
    params = load_params(paths.params)
    baseline = read_json(args.baseline) if args.baseline and args.baseline.exists() else None
    result = run_gate(read_json(paths.metrics), params["gate"], baseline)
    result["baseline_used"] = baseline is not None
    write_json(paths.reports_dir / "gate.json", result)

    md = to_markdown(result)
    print(md)
    if summary := os.getenv("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as f:
            f.write(md)

    if result["passed"] and args.promote:
        reg = read_json(paths.registry) if paths.registry.exists() else {}
        if reg.get("registered_model") and reg.get("version"):
            tracking.set_alias(reg["registered_model"], "champion", reg["version"])
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
