"""Training stage.

1. Grid-search the regularisation strength on the **validation** split.
2. Refit the best configuration on train + validation.
3. Save the pipeline and its metadata; log everything to MLflow.

The test split is never touched here; see ``sentiment.evaluate``.

Usage::

    python -m sentiment.train
"""

from __future__ import annotations

import logging
import os
import platform
from datetime import UTC, datetime
from typing import Any

import joblib
import sklearn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from sentiment import __version__, tracking
from sentiment.config import Paths, load_params, read_json, write_json
from sentiment.data import load_split
from sentiment.model import build_pipeline

log = logging.getLogger(__name__)


def select_and_fit(
    model_type: str,
    params: dict[str, Any],
    grid: list[float],
    train: tuple[list[str], list[int]],
    val: tuple[list[str], list[int]],
    experiment: str | None = None,
) -> tuple[Pipeline, float, list[dict[str, float]]]:
    """Pick ``C`` on validation F1, then refit on train+val. Returns (model, C, grid results)."""
    results: list[dict[str, float]] = []
    for C in grid:
        candidate = build_pipeline(model_type, params, C=C)
        candidate.fit(*train)
        pred = candidate.predict(val[0])
        row = {
            "C": float(C),
            "val_f1": float(f1_score(val[1], pred)),
            "val_accuracy": float(accuracy_score(val[1], pred)),
        }
        results.append(row)
        log.info("%s C=%s  val_f1=%.4f  val_acc=%.4f", model_type, C, row["val_f1"], row["val_accuracy"])
        if experiment:
            with tracking.run(experiment, run_name=f"{model_type}-C{C}", nested=True):
                tracking.log_params({"model": model_type, "C": C})
                tracking.log_metrics({k: v for k, v in row.items() if k != "C"})

    best = max(results, key=lambda r: (r["val_f1"], -r["C"]))
    model = build_pipeline(model_type, params, C=best["C"])
    model.fit(train[0] + val[0], train[1] + val[1])
    return model, best["C"], results


def model_version(model_type: str) -> str:
    sha = os.getenv("GIT_SHA") or os.getenv("GITHUB_SHA") or os.getenv("GIT_COMMIT")
    return f"{model_type}-{sha[:7]}" if sha else f"{model_type}-local"


def train(params: dict[str, Any], paths: Paths) -> dict[str, Any]:
    cfg = params["train"]
    mlflow_cfg = params.get("mlflow", {})
    experiment = mlflow_cfg.get("experiment", "sentiment")
    model_type = cfg["model"]

    train_xy = load_split(paths, "train")
    val_xy = load_split(paths, "validation")

    with tracking.run(experiment, run_name=f"train-{model_type}") as run:
        tracking.log_params({**cfg, "C_grid": cfg["C_grid"], "dataset_revision": params["data"]["revision"]})
        model, best_c, grid = select_and_fit(model_type, cfg, cfg["C_grid"], train_xy, val_xy, experiment)
        best = next(r for r in grid if r["C"] == best_c)
        tracking.log_metrics({"best_C": best_c, "val_f1": best["val_f1"], "val_accuracy": best["val_accuracy"]})

        paths.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, paths.model, compress=3)

        manifest_path = paths.raw_dir / "manifest.json"
        metadata = {
            "version": model_version(model_type),
            "package_version": __version__,
            "model_type": model_type,
            "best_C": best_c,
            "grid_search": grid,
            "validation": {"f1": best["val_f1"], "accuracy": best["val_accuracy"]},
            "n_train_rows": len(train_xy[0]) + len(val_xy[0]),
            "vocabulary_size": len(model.named_steps["tfidf"].vocabulary_),
            "dataset": read_json(manifest_path) if manifest_path.exists() else None,
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
            "mlflow_run_id": run.info.run_id if run else None,
        }
        write_json(paths.metadata, metadata)
        log.info("Saved %s (C=%s, val_f1=%.4f) to %s", model_type, best_c, best["val_f1"], paths.model)
    return metadata


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths = Paths()
    train(load_params(paths.params), paths)


if __name__ == "__main__":
    main()
