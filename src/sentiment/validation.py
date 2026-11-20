"""Data-quality checks run before any model is trained.

Each check returns a structured result so the full report can be stored as a
DVC metric (``reports/data_validation.json``) and inspected in CI.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

REQUIRED_COLUMNS = ("text", "label")


def _check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def validate_splits(splits: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> dict[str, Any]:
    min_rows: dict[str, int] = cfg.get("min_rows", {})
    max_majority = float(cfg.get("max_majority_class_share", 0.6))
    checks: list[dict[str, Any]] = []

    for split, df in splits.items():
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        checks.append(_check(f"{split}.schema", not missing, {"missing_columns": missing}))
        if missing:
            continue

        checks.append(
            _check(
                f"{split}.min_rows",
                len(df) >= min_rows.get(split, 1),
                {"rows": len(df), "min": min_rows.get(split, 1)},
            )
        )
        nulls = int(df["text"].isna().sum() + df["label"].isna().sum())
        empties = int((df["text"].astype(str).str.strip() == "").sum())
        checks.append(_check(f"{split}.no_null_or_empty", nulls + empties == 0, {"nulls": nulls, "empty": empties}))

        labels = sorted(int(x) for x in df["label"].unique())
        checks.append(_check(f"{split}.label_domain", set(labels) <= {0, 1}, {"labels": labels}))

        share = float(df["label"].value_counts(normalize=True).max()) if len(df) else 1.0
        checks.append(
            _check(
                f"{split}.class_balance",
                share <= max_majority,
                {"majority_share": round(share, 4), "max": max_majority},
            )
        )
        checks.append(_check(f"{split}.no_duplicates", not df["text"].duplicated().any(), {}))

    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if "text" in splits[a] and "text" in splits[b]:
                overlap = len(set(splits[a]["text"]) & set(splits[b]["text"]))
                checks.append(_check(f"no_overlap.{a}-{b}", overlap == 0, {"overlap": overlap}))

    return {
        "passed": all(c["passed"] for c in checks),
        "rows": {s: len(df) for s, df in splits.items()},
        "checks": checks,
    }
