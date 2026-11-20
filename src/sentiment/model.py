"""Model definitions. Every candidate is a single sklearn ``Pipeline`` that maps
raw text to calibrated class probabilities, so the API never needs to know
which algorithm is behind it."""

from __future__ import annotations

from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

MODEL_TYPES = ("logreg", "linear_svc", "complement_nb")


def build_vectorizer(params: dict[str, Any]) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, int(params.get("ngram_max", 2))),
        max_features=params.get("max_features"),
        min_df=int(params.get("min_df", 1)),
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )


def build_pipeline(model_type: str, params: dict[str, Any], C: float = 1.0) -> Pipeline:
    """Build an untrained text-classification pipeline.

    ``C`` is the regularisation strength for the linear models and the
    smoothing ``alpha`` for Naive Bayes, so one grid search loop works for all.
    """
    seed = int(params.get("random_seed", 42))
    clf: Any
    if model_type == "logreg":
        clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    elif model_type == "linear_svc":
        # LinearSVC has no predict_proba; sigmoid calibration provides it.
        clf = CalibratedClassifierCV(LinearSVC(C=C, random_state=seed), method="sigmoid", cv=3)
    elif model_type == "complement_nb":
        clf = ComplementNB(alpha=C)
    else:
        raise ValueError(f"Unknown model type {model_type!r}; expected one of {MODEL_TYPES}")
    return Pipeline([("tfidf", build_vectorizer(params)), ("clf", clf)])


def positive_proba(model: Pipeline, texts: list[str]) -> list[float]:
    """Probability of the positive class for each text."""
    classes = list(model.classes_)
    idx = classes.index(1)
    return [float(p[idx]) for p in model.predict_proba(texts)]
