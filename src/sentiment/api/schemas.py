"""Request / response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Upper bounds are enforced against Settings in the route handlers; these
# static limits protect the JSON parser from absurd payloads.
HARD_MAX_TEXT = 100_000
HARD_MAX_BATCH = 10_000


class Review(BaseModel):
    text: str = Field(..., min_length=1, max_length=HARD_MAX_TEXT, examples=["A gripping, beautifully shot film."])

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=HARD_MAX_BATCH)

    @field_validator("texts")
    @classmethod
    def none_blank(cls, v: list[str]) -> list[str]:
        if any(not t.strip() for t in v):
            raise ValueError("texts must not contain blank strings")
        return v


class Prediction(BaseModel):
    label: Literal["positive", "negative"]
    confidence: float = Field(..., description="Probability of the returned label (0.5-1.0)")
    positive_probability: float = Field(..., description="P(positive), useful for custom thresholds")
    low_confidence: bool = Field(..., description="True when confidence is below LOW_CONFIDENCE_THRESHOLD")
    model_version: str


class BatchPrediction(BaseModel):
    predictions: list[Prediction]
    model_version: str


class ModelInfo(BaseModel):
    model_version: str
    model_type: str | None
    trained_at: str | None
    decision_threshold: float
    low_confidence_threshold: float
    max_text_length: int
    max_batch_size: int
    validation_f1: float | None
    drift_reference_loaded: bool
