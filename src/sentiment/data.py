"""Data stages: download the raw dataset, then clean, de-leak and validate it.

Dataset: Rotten Tomatoes movie reviews (Pang & Lee, 2005), ~10.6k sentences
with balanced binary sentiment labels, fetched from the Hugging Face Hub at a
pinned revision so every run sees the exact same bytes.

Usage::

    python -m sentiment.data download
    python -m sentiment.data prepare
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from sentiment.config import SPLITS, Paths, load_params, write_json
from sentiment.validation import validate_splits

log = logging.getLogger(__name__)

HF_URL = "https://huggingface.co/datasets/{dataset}/resolve/{revision}/{split}.parquet"
_WS = re.compile(r"\s+")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(params: dict[str, Any], paths: Paths) -> dict[str, Any]:
    """Download each split as parquet and write a manifest with checksums."""
    cfg = params["data"]
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "dataset": cfg["dataset"],
        "revision": cfg["revision"],
        "files": {},
    }
    for split in SPLITS:
        url = HF_URL.format(dataset=cfg["dataset"], revision=cfg["revision"], split=split)
        dest = paths.raw(split)
        log.info("Downloading %s -> %s", url, dest)
        if not url.startswith("https://"):
            raise ValueError(f"Refusing non-HTTPS download URL: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "sentiment-pipeline/3"})  # noqa: S310
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - scheme checked above
            dest.write_bytes(resp.read())
        manifest["files"][split] = {"url": url, "sha256": sha256(dest)}
    write_json(paths.raw_dir / "manifest.json", manifest)
    return manifest


def clean_text(text: str) -> str:
    return _WS.sub(" ", str(text)).strip()


def clean_split(df: pd.DataFrame, min_chars: int, max_chars: int) -> pd.DataFrame:
    """Normalise whitespace, drop empty/too-short/too-long rows and duplicates."""
    out = df[["text", "label"]].copy()
    out["text"] = out["text"].astype(str).map(clean_text)
    out["label"] = out["label"].astype(int)
    lengths = out["text"].str.len()
    out = out[(lengths >= min_chars) & (lengths <= max_chars)]
    # Exact duplicate texts: keep one; if the same text carries conflicting
    # labels it is ambiguous, so drop every copy.
    conflicting = out.groupby("text")["label"].nunique()
    out = out[~out["text"].isin(conflicting[conflicting > 1].index)]
    out = out.drop_duplicates(subset="text")
    return out.reset_index(drop=True)


def remove_leakage(splits: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Drop validation/test rows whose text also occurs in an earlier split."""
    removed: dict[str, int] = {}
    seen = set(splits["train"]["text"])
    for split in ("validation", "test"):
        df = splits[split]
        mask = df["text"].isin(seen)
        removed[split] = int(mask.sum())
        splits[split] = df[~mask].reset_index(drop=True)
        seen |= set(splits[split]["text"])
    return removed


def prepare(params: dict[str, Any], paths: Paths) -> dict[str, Any]:
    """Clean every split, remove cross-split leakage, validate and save CSVs."""
    cfg = params["data"]
    splits: dict[str, pd.DataFrame] = {}
    raw_counts: dict[str, int] = {}
    for split in SPLITS:
        df = pd.read_parquet(paths.raw(split))
        raw_counts[split] = len(df)
        splits[split] = clean_split(df, cfg["min_text_chars"], cfg["max_text_chars"])

    leakage_removed = remove_leakage(splits)
    report = validate_splits(splits, params.get("validation", {}))
    report["raw_rows"] = raw_counts
    report["leakage_removed"] = leakage_removed
    write_json(paths.reports_dir / "data_validation.json", report)
    if not report["passed"]:
        failed = [c["name"] for c in report["checks"] if not c["passed"]]
        raise ValueError(f"Data validation failed: {failed}")

    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    for split, df in splits.items():
        df.to_csv(paths.processed(split), index=False)
        log.info("Wrote %s rows to %s", len(df), paths.processed(split))
    return report


def load_split(paths: Paths, split: str) -> tuple[list[str], list[int]]:
    df = pd.read_csv(paths.processed(split), keep_default_na=False)
    return df["text"].astype(str).tolist(), df["label"].astype(int).tolist()


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["download", "prepare"])
    args = parser.parse_args(argv)
    paths = Paths()
    params = load_params(paths.params)
    if args.stage == "download":
        download(params, paths)
    else:
        prepare(params, paths)


if __name__ == "__main__":
    main()
