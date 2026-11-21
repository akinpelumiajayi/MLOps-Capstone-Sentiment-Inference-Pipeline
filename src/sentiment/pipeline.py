"""Run the whole training pipeline without DVC (used by the Docker trainer).

``dvc repro`` is the canonical entry point; this module exists so the
container image does not need git or DVC installed.

Usage::

    python -m sentiment.pipeline [--skip-if-exists]
"""

from __future__ import annotations

import argparse
import logging

from sentiment.config import SPLITS, Paths, load_params
from sentiment.data import download, prepare
from sentiment.evaluate import evaluate
from sentiment.train import train

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-if-exists", action="store_true", help="do nothing if a model is already present")
    args = parser.parse_args(argv)

    paths = Paths()
    if args.skip_if_exists and paths.model.exists() and paths.reference_profile.exists():
        log.info("Model already present at %s; skipping training", paths.model)
        return
    params = load_params(paths.params)
    if not all(paths.raw(s).exists() for s in SPLITS):
        download(params, paths)
    prepare(params, paths)
    train(params, paths)
    evaluate(params, paths)


if __name__ == "__main__":
    main()
