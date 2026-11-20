"""Thin MLflow wrapper.

Tracking is on by default and writes to ``MLFLOW_TRACKING_URI`` (falling back
to MLflow's local default, ``sqlite:///mlflow.db``). Set ``MLFLOW_ENABLED=false``
to turn it off, e.g. in unit tests. When disabled, every call is a no-op so
the training code never branches on whether MLflow is available.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator
from typing import Any

log = logging.getLogger(__name__)


def enabled() -> bool:
    return os.getenv("MLFLOW_ENABLED", "true").lower() not in {"0", "false", "no"}


@contextlib.contextmanager
def run(
    experiment: str,
    run_name: str | None = None,
    run_id: str | None = None,
    nested: bool = False,
) -> Iterator[Any]:
    """Start (or resume, if ``run_id`` is given) an MLflow run."""
    if not enabled():
        yield None
        return
    import mlflow

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, run_id=run_id, nested=nested) as active:
        yield active


def log_params(params: dict[str, Any]) -> None:
    if enabled():
        import mlflow

        mlflow.log_params({k: str(v) for k, v in params.items()})


def log_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    if enabled():
        import mlflow

        mlflow.log_metrics({f"{prefix}{k}": float(v) for k, v in metrics.items()})


def log_artifact(path: str) -> None:
    if enabled():
        import mlflow

        mlflow.log_artifact(path)


def log_model(model: Any, registered_model_name: str | None, input_example: list[str]) -> str | None:
    """Log the sklearn pipeline and register it. Returns the registry version."""
    if not enabled():
        return None
    from importlib.metadata import version as pkg_version

    import mlflow.sklearn
    import numpy as np

    # Explicit requirements: MLflow's automatic inference re-imports the model
    # in a subprocess, which costs minutes for no extra information here.
    requirements = [f"{pkg}=={pkg_version(pkg)}" for pkg in ("scikit-learn", "numpy", "scipy", "joblib")]
    info = mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name=registered_model_name,
        # A 1-D string array (not a list) so MLflow infers a string-tensor
        # signature that the TF-IDF pipeline can consume directly.
        input_example=np.asarray(input_example, dtype=object),
        pip_requirements=requirements,
        # MLflow 3 defaults to skops, which is quadratic in the number of
        # arrays and takes minutes on a 100k-term TF-IDF vocabulary. The model
        # is a trusted, self-produced artifact (the API loads it with joblib
        # anyway), so pickle-based serialisation is the pragmatic choice.
        serialization_format="cloudpickle",
    )
    version = getattr(info, "registered_model_version", None)
    return str(version) if version is not None else None


def set_alias(registered_model_name: str, alias: str, version: str) -> bool:
    """Point a registry alias (e.g. ``champion``) at a model version."""
    if not enabled():
        return False
    try:
        from mlflow import MlflowClient

        MlflowClient().set_registered_model_alias(registered_model_name, alias, version)
        return True
    except Exception as exc:  # registry may be unreachable; never fail the gate on it
        log.warning("Could not set MLflow alias %s -> v%s: %s", alias, version, exc)
        return False
