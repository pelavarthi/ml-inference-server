"""FastAPI entry point: health, model list, predict, metrics, async queue."""

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.backend import load_backend
from app.batching import Batcher
from app.metrics import Metrics
from app.queue import AsyncPredictQueue, QueueFullError
from app.registry import ModelRegistry
from app.schemas import (
    JobResponse,
    PredictRequest,
    PredictResponse,
    PreprocessRequest,
    PreprocessResponse,
)
from app.streaming import ConcurrencyLimiter, stream_prediction
from app.versioning import Experiment, Router

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def create_app(
    registry=None, max_concurrent_streams=10, text_backend=None, experiments=None
):
    registry = registry or ModelRegistry()
    if not registry.list_models() and MODELS_DIR.exists():
        registry.load_directory(MODELS_DIR)
    text_backend = text_backend or load_backend()
    router = Router(registry, experiments)
    if experiments is None:
        # A model with more than one version gets an even A/B split
        # between its versions, so a new version can be compared
        # against the old one before it is promoted to default.
        for info in registry.list_models():
            name = info["name"]
            versions = registry.versions(name)
            if len(versions) > 1 and f"{name}-ab" not in router.experiments:
                router.register_experiment(
                    Experiment(
                        f"{name}-ab",
                        name,
                        [(version, 1) for version in versions],
                    )
                )

    metrics = Metrics()
    queue = AsyncPredictQueue(registry)
    batchers = {}
    stream_limiter = ConcurrencyLimiter(max_concurrent_streams)

    def batcher_for(model):
        if model.name not in batchers:
            batchers[model.name] = Batcher(model.predict_batch)
        return batchers[model.name]

    def batching_stats():
        total = {"batches": 0, "requests": 0, "largest_batch": 0}
        for batcher in batchers.values():
            stats = batcher.stats
            total["batches"] += stats["batches"]
            total["requests"] += stats["requests"]
            total["largest_batch"] = max(total["largest_batch"], stats["largest_batch"])
        return total

    @asynccontextmanager
    async def lifespan(app):
        queue.ensure_started()
        yield
        await queue.stop()
        for batcher in batchers.values():
            await batcher.stop()

    app = FastAPI(title="ml-inference-server", lifespan=lifespan)

    def job_response(job):
        return JobResponse(
            job_id=job["job_id"],
            model=job["model"],
            status=job["status"],
            prediction=job["prediction"],
            error=job["error"],
        )

    @app.get("/health")
    def health():
        metrics.record_request("GET", "/health", 200)
        return {"status": "ok", "models": len(registry.list_models())}

    @app.get("/health/detail")
    def health_detail():
        detail = metrics.snapshot()
        detail["status"] = "ok"
        detail["models"] = len(registry.list_models())
        detail["queue_depth"] = queue.depth
        detail["preprocessing_backend"] = text_backend.name
        metrics.record_request("GET", "/health/detail", 200)
        return detail

    @app.get("/metrics")
    def get_metrics():
        snapshot = metrics.snapshot()
        snapshot["models"] = len(registry.list_models())
        snapshot["queue_depth"] = queue.depth
        snapshot["batching"] = batching_stats()
        snapshot["preprocessing_backend"] = text_backend.name
        metrics.record_request("GET", "/metrics", 200)
        return snapshot

    @app.get("/models")
    def list_models():
        metrics.record_request("GET", "/models", 200)
        return {"models": registry.list_models()}

    @app.get("/models/{name}/versions")
    def get_model_versions(name: str):
        try:
            versions = registry.versions(name)
        except KeyError:
            metrics.record_request("GET", "/models/{name}/versions", 404)
            raise HTTPException(status_code=404, detail="unknown model")
        metrics.record_request("GET", "/models/{name}/versions", 200)
        return {
            "model": name,
            "versions": versions,
            "default": registry.default_version(name),
        }

    @app.get("/experiments")
    def list_experiments():
        metrics.record_request("GET", "/experiments", 200)
        return {
            "experiments": [
                experiment.info()
                for experiment in router.experiments.values()
            ]
        }

    @app.get("/models/{name}")
    def get_model(name: str):
        try:
            model = registry.get(name).info()
        except KeyError:
            metrics.record_request("GET", "/models/{name}", 404)
            raise HTTPException(status_code=404, detail="unknown model")
        metrics.record_request("GET", "/models/{name}", 200)
        return model

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest):
        try:
            model, experiment = router.resolve(
                request.model,
                version=request.version,
                experiment=request.experiment,
                subject=request.subject,
            )
        except KeyError:
            metrics.record_request("POST", "/predict", 404)
            metrics.record_prediction(False)
            raise HTTPException(status_code=404, detail="unknown model, version, or experiment")
        except ValueError as exc:
            metrics.record_request("POST", "/predict", 400)
            metrics.record_prediction(False)
            raise HTTPException(status_code=400, detail=str(exc))
        try:
            prediction = model.predict(request.features)
        except ValueError as exc:
            metrics.record_request("POST", "/predict", 400)
            metrics.record_prediction(False, model.name, model.version)
            raise HTTPException(status_code=400, detail=str(exc))
        metrics.record_request("POST", "/predict", 200)
        metrics.record_prediction(True, model.name, model.version)
        return PredictResponse(
            model=model.name,
            version=model.version,
            prediction=prediction,
            experiment=experiment,
        )

    @app.post("/predict/batched", response_model=PredictResponse)
    async def predict_batched(request: PredictRequest):
        try:
            model = registry.get(request.model)
        except KeyError:
            metrics.record_request("POST", "/predict/batched", 404)
            metrics.record_prediction(False)
            raise HTTPException(status_code=404, detail="unknown model")
        batcher = batcher_for(model)
        batcher.ensure_started()
        try:
            prediction = await batcher.submit(request.features)
        except ValueError as exc:
            metrics.record_request("POST", "/predict/batched", 400)
            metrics.record_prediction(False)
            raise HTTPException(status_code=400, detail=str(exc))
        metrics.record_request("POST", "/predict/batched", 200)
        metrics.record_prediction(True)
        return PredictResponse(
            model=model.name, version=model.version, prediction=prediction
        )

    @app.post("/preprocess", response_model=PreprocessResponse)
    def preprocess(request: PreprocessRequest):
        # A vocabulary supplied with the request is learned on a fresh
        # backend of the same kind, so one caller's vocabulary never
        # leaks into the shared backend other requests use.
        backend = text_backend
        if request.vocabulary is not None:
            backend = load_backend(prefer=text_backend.name)
            backend.build_vocab(request.vocabulary)
        batch = backend.prepare_batch(request.texts, max_length=request.max_length)
        metrics.record_request("POST", "/preprocess", 200)
        metrics.record_preprocessing(backend.name)
        return PreprocessResponse(
            ids=batch.ids,
            attention_mask=batch.attention_mask,
            lengths=batch.lengths,
            backend=backend.name,
        )

    @app.post("/predict/async", response_model=JobResponse, status_code=202)
    async def predict_async(request: PredictRequest):
        try:
            registry.get(request.model)
        except KeyError:
            metrics.record_request("POST", "/predict/async", 404)
            raise HTTPException(status_code=404, detail="unknown model")
        queue.ensure_started()
        try:
            job = queue.submit(request.model, request.features)
        except QueueFullError:
            metrics.record_request("POST", "/predict/async", 429)
            raise HTTPException(status_code=429, detail="prediction queue is full")
        metrics.record_request("POST", "/predict/async", 202)
        return job_response(job)

    @app.get("/predict/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: str):
        job = queue.get(job_id)
        if job is None:
            metrics.record_request("GET", "/predict/jobs/{job_id}", 404)
            raise HTTPException(status_code=404, detail="unknown job")
        metrics.record_request("GET", "/predict/jobs/{job_id}", 200)
        return job_response(job)

    @app.post("/predict/stream")
    async def predict_stream(
        request: Request,
        body: PredictRequest,
        timeout_seconds: float = Query(default=30.0, gt=0),
        chunk_size: int = Query(default=4, ge=1),
    ):
        try:
            model = registry.get(body.model)
        except KeyError:
            metrics.record_request("POST", "/predict/stream", 404)
            raise HTTPException(status_code=404, detail="unknown model")
        if len(body.features) != len(model.weights):
            metrics.record_request("POST", "/predict/stream", 400)
            raise HTTPException(
                status_code=400,
                detail=f"expected {len(model.weights)} features, "
                f"got {len(body.features)}",
            )
        if not stream_limiter.try_acquire():
            metrics.record_request("POST", "/predict/stream", 429)
            metrics.record_stream_rejected()
            raise HTTPException(
                status_code=429, detail="too many concurrent streams"
            )
        metrics.record_request("POST", "/predict/stream", 200)

        async def event_stream():
            try:
                async for event in stream_prediction(
                    model,
                    body.features,
                    chunk_size=chunk_size,
                    timeout_seconds=timeout_seconds,
                ):
                    if await request.is_disconnected():
                        break
                    yield f"data: {json.dumps(event)}\n\n"
            finally:
                stream_limiter.release()

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


app = create_app()

