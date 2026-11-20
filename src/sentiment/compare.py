"""Model comparison: run the same select-on-validation / score-on-test protocol
for every candidate algorithm and write a side-by-side report.

Usage::

    python -m sentiment.compare
"""

from __future__ import annotations

import logging
import pickle
from typing import Any

from sentiment import tracking
from sentiment.config import Paths, load_params, write_json
from sentiment.data import load_split
from sentiment.evaluate import benchmark_latency, classification_metrics
from sentiment.model import positive_proba
from sentiment.train import select_and_fit

log = logging.getLogger(__name__)


def compare(params: dict[str, Any], paths: Paths) -> list[dict[str, Any]]:
    cfg = params["compare"]
    base = {**params["train"]}
    experiment = params.get("mlflow", {}).get("experiment", "sentiment") + "-comparison"
    threshold = float(params.get("serve", {}).get("decision_threshold", 0.5))
    train_xy, val_xy = load_split(paths, "train"), load_split(paths, "validation")
    test_x, test_y = load_split(paths, "test")

    rows: list[dict[str, Any]] = []
    for model_type, grid in cfg["models"].items():
        with tracking.run(experiment, run_name=model_type):
            model, best_c, _ = select_and_fit(model_type, base, grid, train_xy, val_xy)
            metrics = classification_metrics(test_y, positive_proba(model, test_x), threshold)
            latency = benchmark_latency(model, test_x, 300)
            row = {
                "model": model_type,
                "best_C": best_c,
                **{f"test_{k}": round(v, 4) for k, v in metrics.items()},
                "p95_latency_ms": round(latency["p95_ms"], 3),
                "size_mb": round(len(pickle.dumps(model)) / 1e6, 2),
            }
            tracking.log_params({"model": model_type, "best_C": best_c})
            tracking.log_metrics({k: v for k, v in row.items() if isinstance(v, float)})
            rows.append(row)
            log.info("%s: %s", model_type, row)

    rows.sort(key=lambda r: r["test_f1"], reverse=True)
    write_json(paths.reports_dir / "model_comparison.json", rows)
    header = (
        "| Model | Best C/alpha | Test F1 | Test accuracy | ROC AUC | Log loss | Brier "
        "| p95 latency (ms) | Pickled size (MB) |"
    )
    lines = [header, "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['best_C']} | {r['test_f1']:.4f} | {r['test_accuracy']:.4f} | "
            f"{r['test_roc_auc']:.4f} | {r['test_log_loss']:.4f} | {r['test_brier']:.4f} | "
            f"{r['p95_latency_ms']:.2f} | {r['size_mb']} |"
        )
    (paths.reports_dir / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths = Paths()
    compare(load_params(paths.params), paths)


if __name__ == "__main__":
    main()
