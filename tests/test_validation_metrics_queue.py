import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.queue import AsyncPredictQueue, QueueFullError
from app.registry import Model, ModelRegistry


def make_client():
    registry = ModelRegistry()
    registry.register(Model("tiny-linear", "0.1.0", [0.5, 0.25, -1.0], bias=0.1))
    return TestClient(create_app(registry))


def test_rejects_blank_model_and_nonfinite_features():
    client = make_client()
    assert client.post("/predict", json={"model": "   ", "features": [1.0]}).status_code == 422
    # Checked at the schema level: a 422 error body echoing a NaN input
    # cannot itself be JSON-encoded, so the HTTP path is not a clean
    # way to assert this case. The validator is the same code the
    # endpoint runs.
    from pydantic import ValidationError

    from app.schemas import PredictRequest

    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            PredictRequest(model="tiny-linear", features=[bad])
    assert client.post(
        "/predict", json={"model": "tiny-linear", "features": [1.0] * 1025}
    ).status_code == 422


def test_health_detail_and_metrics_count_predictions():
    client = make_client()
    detail = client.get("/health/detail").json()
    assert detail["status"] == "ok"
    assert detail["models"] == 1
    assert detail["queue_depth"] == 0
    assert "uptime_seconds" in detail

    ok = client.post(
        "/predict", json={"model": "tiny-linear", "features": [2.0, 4.0, 1.0]}
    )
    assert ok.json()["prediction"] == 1.1
    client.post("/predict", json={"model": "nope", "features": [1.0]})

    metrics = client.get("/metrics").json()
    assert metrics["predictions"]["total"] == 2
    assert metrics["predictions"]["failed"] == 1
    assert metrics["requests"]["POST /predict 200"] == 1
    assert metrics["requests"]["POST /predict 404"] == 1


def poll_job(client, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/predict/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.05)
    return job


def test_async_job_completes_and_failed_job_reports_error():
    client = make_client()
    submitted = client.post(
        "/predict/async", json={"model": "tiny-linear", "features": [2.0, 4.0, 1.0]}
    )
    assert submitted.status_code == 202
    job = poll_job(client, submitted.json()["job_id"])
    assert job["status"] == "done"
    assert job["prediction"] == 1.1

    bad = client.post(
        "/predict/async", json={"model": "tiny-linear", "features": [1.0]}
    )
    job = poll_job(client, bad.json()["job_id"])
    assert job["status"] == "failed"
    assert "expected 3 features" in job["error"]


def test_async_unknown_model_and_unknown_job_are_404():
    client = make_client()
    assert client.post(
        "/predict/async", json={"model": "nope", "features": [1.0]}
    ).status_code == 404
    assert client.get("/predict/jobs/doesnotexist").status_code == 404


def test_bounded_queue_rejects_when_full():
    async def run():
        registry = ModelRegistry()
        registry.register(Model("m", "0.1.0", [1.0]))
        queue = AsyncPredictQueue(registry, maxsize=1)
        queue.submit("m", [1.0])
        with pytest.raises(QueueFullError):
            queue.submit("m", [2.0])

    asyncio.run(run())
