from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from sentiment.config import Paths
from sentiment.data import clean_split, clean_text, load_split, prepare, remove_leakage
from sentiment.validation import validate_splits


def test_clean_text_normalises_whitespace() -> None:
    assert clean_text("  a \n\t good   film ") == "a good film"


def test_clean_split_drops_short_long_and_duplicates() -> None:
    df = pd.DataFrame(
        {
            "text": ["good film", "good  film", "ok", "x" * 50, "bad film", "same", "same"],
            "label": [1, 1, 1, 1, 0, 1, 0],
        }
    )
    out = clean_split(df, min_chars=3, max_chars=20)
    # "ok" is too short, "x"*50 too long, "good  film" is a duplicate after
    # cleaning, and "same" carries conflicting labels so both copies go.
    assert sorted(out["text"]) == ["bad film", "good film"]


def test_remove_leakage() -> None:
    splits = {
        "train": pd.DataFrame({"text": ["a", "b"], "label": [1, 0]}),
        "validation": pd.DataFrame({"text": ["b", "c"], "label": [0, 1]}),
        "test": pd.DataFrame({"text": ["a", "c", "d"], "label": [1, 1, 0]}),
    }
    removed = remove_leakage(splits)
    assert removed == {"validation": 1, "test": 2}
    assert splits["test"]["text"].tolist() == ["d"]


def test_validation_detects_problems() -> None:
    splits = {
        "train": pd.DataFrame({"text": ["a", "b", "c"], "label": [1, 1, 1]}),
        "test": pd.DataFrame({"text": ["a", "z"], "label": [1, 2]}),
    }
    report = validate_splits(splits, {"min_rows": {"train": 5}, "max_majority_class_share": 0.6})
    failed = {c["name"] for c in report["checks"] if not c["passed"]}
    assert not report["passed"]
    assert {"train.min_rows", "train.class_balance", "test.label_domain", "no_overlap.train-test"} <= failed


def test_validation_missing_column() -> None:
    report = validate_splits({"train": pd.DataFrame({"review": ["a"]})}, {})
    assert not report["passed"]
    assert report["checks"][0]["detail"] == {"missing_columns": ["text", "label"]}


def _write_raw(paths: Paths, texts: dict[str, list[str]], labels: dict[str, list[int]]) -> None:
    paths.raw_dir.mkdir(parents=True)
    for split, tx in texts.items():
        pd.DataFrame({"text": tx, "label": labels[split]}).to_parquet(paths.raw(split))


def test_prepare_end_to_end(tmp_path: Path, params: dict[str, Any]) -> None:
    paths = Paths(root=tmp_path)
    _write_raw(
        paths,
        {
            "train": ["great film", "awful film", "lovely cast", "boring plot"],
            "validation": ["superb acting", "dull script", "great film"],  # the last one leaks from train
            "test": ["fine story", "weak ending"],
        },
        {"train": [1, 0, 1, 0], "validation": [1, 0, 1], "test": [1, 0]},
    )
    report = prepare(params, paths)
    assert report["passed"]
    assert report["leakage_removed"]["validation"] == 1
    x, y = load_split(paths, "validation")
    assert x == ["superb acting", "dull script"]
    assert y == [1, 0]


def test_prepare_raises_on_failed_validation(tmp_path: Path, params: dict[str, Any]) -> None:
    paths = Paths(root=tmp_path)
    splits = ("train", "validation", "test")
    _write_raw(paths, {s: [f"only positives {s}"] for s in splits}, {s: [1] for s in splits})
    with pytest.raises(ValueError, match="class_balance"):
        prepare(params, paths)
