"""FastAPI application.

Run locally::

    uvicorn sentiment.api.main:app --reload
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from sentiment import __version__
from sentiment.api import metrics as m
from sentiment.api.log_config import configure_logging, request_id_var
from sentiment.api.schemas import BatchPrediction, BatchRequest, ModelInfo, Prediction, Review
from sentiment.api.service import ModelService
from sentiment.api.settings import Settings, get_settings

log = logging.getLogger("sentiment.api")

UNINSTRUMENTED = {"/metrics", "/healthz", "/readyz"}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level)
        try:
            service = ModelService.load(settings)
        except Exception:
            # Stay up so liveness passes and the reason is visible in logs and
            # /readyz; Kubernetes keeps the pod out of the Service meanwhile.
            log.exception("Model failed to load")
            app.state.service = None
            m.MODEL_LOADED.set(0)
        else:
            app.state.service = service
            m.MODEL_LOADED.set(1)
            m.MODEL_INFO.info(
                {
                    "version": service.version,
                    "type": str(service.metadata.get("model_type", "unknown")),
                    "trained_at": str(service.metadata.get("trained_at", "unknown")),
                }
            )
            log.info("Model loaded", extra={"model_version": service.version})
        yield
        m.MODEL_LOADED.set(0)

    app = FastAPI(
        title="Sentiment Inference API",
        version=__version__,
        description="Binary sentiment classification for short English reviews.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def observe(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = request_id_var.set(rid)
        start = time.perf_counter()
        status = 500
        m.IN_FLIGHT.inc()
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            m.IN_FLIGHT.dec()
            elapsed = time.perf_counter() - start
            route = getattr(request.scope.get("route"), "path", "unmatched")
            if route not in UNINSTRUMENTED:
                m.HTTP_REQUESTS.labels(request.method, route, str(status)).inc()
                m.HTTP_LATENCY.labels(route).observe(elapsed)
                log.info(
                    "request",
                    extra={
                        "method": request.method,
                        "route": route,
                        "status": status,
                        "duration_ms": round(elapsed * 1000, 2),
                    },
                )
            request_id_var.reset(token)

    def service_or_503(request: Request) -> ModelService:
        service: ModelService | None = request.app.state.service
        if service is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return service

    def check_length(texts: list[str]) -> None:
        too_long = [i for i, t in enumerate(texts) if len(t) > settings.max_text_length]
        if too_long:
            raise HTTPException(
                status_code=422,
                detail=f"text exceeds {settings.max_text_length} characters at index {too_long[:5]}",
            )

    @app.post("/predict", response_model=Prediction, tags=["inference"])
    def predict(review: Review, request: Request) -> Prediction:
        service = service_or_503(request)
        check_length([review.text])
        return service.predict([review.text], endpoint="single")[0]

    @app.post("/predict/batch", response_model=BatchPrediction, tags=["inference"])
    def predict_batch(batch: BatchRequest, request: Request) -> BatchPrediction:
        service = service_or_503(request)
        if len(batch.texts) > settings.max_batch_size:
            raise HTTPException(status_code=422, detail=f"batch size exceeds {settings.max_batch_size}")
        check_length(batch.texts)
        return BatchPrediction(
            predictions=service.predict(batch.texts, endpoint="batch"), model_version=service.version
        )

    @app.get("/info", response_model=ModelInfo, tags=["ops"])
    def info(request: Request) -> ModelInfo:
        service = service_or_503(request)
        md = service.metadata
        return ModelInfo(
            model_version=service.version,
            model_type=md.get("model_type"),
            trained_at=md.get("trained_at"),
            decision_threshold=settings.decision_threshold,
            low_confidence_threshold=settings.low_confidence_threshold,
            max_text_length=settings.max_text_length,
            max_batch_size=settings.max_batch_size,
            validation_f1=(md.get("validation") or {}).get("f1"),
            drift_reference_loaded=service.reference is not None,
        )

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict[str, str]:
        """Liveness: the process is up. Never depends on the model."""
        return {"status": "ok"}

    @app.get("/readyz", tags=["ops"])
    def readyz(request: Request) -> dict[str, str]:
        """Readiness: a model is loaded and can serve traffic."""
        service = service_or_503(request)
        return {"status": "ready", "model_version": service.version}

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
