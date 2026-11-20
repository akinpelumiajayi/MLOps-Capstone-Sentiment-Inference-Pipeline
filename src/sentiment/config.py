"""Project paths and pipeline parameters (``params.yaml``)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SPLITS = ("train", "validation", "test")
LABELS = ("negative", "positive")


@dataclass(frozen=True)
class Paths:
    """Filesystem layout of the pipeline, rooted at the repository directory."""

    root: Path = Path()

    @property
    def params(self) -> Path:
        return self.root / "params.yaml"

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def model(self) -> Path:
        return self.models_dir / "model.pkl"

    @property
    def metadata(self) -> Path:
        return self.models_dir / "metadata.json"

    @property
    def reference_profile(self) -> Path:
        return self.models_dir / "reference_profile.json"

    @property
    def registry(self) -> Path:
        return self.models_dir / "registry.json"

    @property
    def metrics(self) -> Path:
        return self.reports_dir / "metrics.json"

    def processed(self, split: str) -> Path:
        return self.processed_dir / f"{split}.csv"

    def raw(self, split: str) -> Path:
        return self.raw_dir / f"{split}.parquet"


def load_params(path: Path | str = "params.yaml") -> dict[str, Any]:
    """Load the full parameter file, returning an empty dict if it is missing."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain a mapping at the top level")
    return data


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
