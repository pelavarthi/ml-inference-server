"""FastAPI entry point: health, model list, predict, metrics, async queue."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.batching import Batcher
from app.metrics import Metrics
from app.queue import AsyncPredictQueue, QueueFullError
from app.registry import ModelRegistry
from app.schemas import JobResponse, PredictRequest, PredictResponse

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def create_app(registry=None):
    registry = registry or ModelRegistry()
    if not registry.list_models() and MODELS_DIR.exists():
        registry.load_directory(MODELS_DIR)

    metrics = Metrics()
    queue = AsyncPredictQueue(registry)
    batchers = {}

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
        metrics.record_request("GET", "/health/detail", 200)
        return detail

    @app.get("/metrics")
    def get_metrics():
        snapshot = metrics.snapshot()
        snapshot["models"] = len(registry.list_models())
        snapshot["queue_depth"] = queue.depth
        snapshot["batching"] = batching_stats()
        metrics.record_request("GET", "/metrics", 200)
        return snapshot

    @app.get("/models")
    def list_models():
        metrics.record_request("GET", "/models", 200)
        return {"models": registry.list_models()}

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
            model = registry.get(request.model)
        except KeyError:
            metrics.record_request("POST", "/predict", 404)
            metrics.record_prediction(False)
            raise HTTPException(status_code=404, detail="unknown model")
        try:
            prediction = model.predict(request.features)
        except ValueError as exc:
            metrics.record_request("POST", "/predict", 400)
            metrics.record_prediction(False)
            raise HTTPException(status_code=400, detail=str(exc))
        metrics.record_request("POST", "/predict", 200)
        metrics.record_prediction(True)
        return PredictResponse(
            model=model.name, version=model.version, prediction=prediction
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

    return app


app = create_app()

