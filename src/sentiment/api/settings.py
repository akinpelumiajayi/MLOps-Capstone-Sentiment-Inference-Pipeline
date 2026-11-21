"""Runtime configuration, read from environment variables (12-factor)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False, protected_namespaces=())

    model_path: Path = Path("models/model.pkl")
    metadata_path: Path = Path("models/metadata.json")
    reference_profile_path: Path = Path("models/reference_profile.json")
    # Overrides the version stored in metadata.json (CI sets it to the image tag).
    model_version: str | None = None

    decision_threshold: float = Field(0.5, ge=0.0, le=1.0)
    low_confidence_threshold: float = Field(0.65, ge=0.5, le=1.0)

    max_text_length: int = Field(5000, gt=0)
    max_batch_size: int = Field(100, gt=0)

    drift_window_size: int = Field(500, gt=0)
    drift_update_every: int = Field(25, gt=0)

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
