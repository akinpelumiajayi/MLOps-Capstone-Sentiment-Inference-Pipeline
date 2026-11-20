# syntax=docker/dockerfile:1.7
# ---- API image ---------------------------------------------------------------
# Base image pinned by digest for reproducibility; Dependabot bumps it.
ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

FROM ${PYTHON_IMAGE} AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- base: code and dependencies, no model ------------------------------------
# Used by docker compose, which mounts ./models produced by the trainer service.
FROM ${PYTHON_IMAGE} AS base
LABEL org.opencontainers.image.title="sentiment-api" \
      org.opencontainers.image.description="Sentiment inference API (FastAPI + scikit-learn)" \
      org.opencontainers.image.licenses="MIT"

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model.pkl \
    METADATA_PATH=/app/models/metadata.json \
    REFERENCE_PROFILE_PATH=/app/models/reference_profile.json

RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --no-create-home app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app src/ ./src/

USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=2).status != 200)"]

# One worker per container: scale out with replicas (HPA), not processes.
CMD ["uvicorn", "sentiment.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--timeout-graceful-shutdown", "20"]

# ---- runtime: the shippable image, model baked in -----------------------------
# Immutable: one image tag == one code version + one model version.
FROM base AS runtime
COPY --chown=app:app models/model.pkl models/metadata.json models/reference_profile.json ./models/
