import asyncio

from fastapi.testclient import TestClient

from app.batching import Batcher
from app.main import create_app
from app.registry import Model, ModelRegistry


def make_client():
    registry = ModelRegistry()
    registry.register(Model("tiny-linear", "0.1.0", [0.5, 0.25, -1.0], bias=0.1))
    return TestClient(create_app(registry))


def test_concurrent_requests_share_one_batch():
    calls = []

    def predict_batch(batch):
        calls.append(len(batch))
        return [sum(features) for features in batch]

    async def run():
        batcher = Batcher(predict_batch, max_batch_size=8, max_wait_seconds=0.05)
        batcher.ensure_started()
        try:
            return batcher, await asyncio.gather(
                *(batcher.submit([1.0, 2.0]) for _ in range(5))
            )
        finally:
            await batcher.stop()

    batcher, results = asyncio.run(run())
    assert results == [3.0] * 5
    assert calls == [5]
    assert batcher.stats == {"batches": 1, "requests": 5, "largest_batch": 5}


def test_batch_never_exceeds_max_size():
    calls = []

    def predict_batch(batch):
        calls.append(len(batch))
        return [0.0] * len(batch)

    async def run():
        batcher = Batcher(predict_batch, max_batch_size=4, max_wait_seconds=0.05)
        batcher.ensure_started()
        try:
            await asyncio.gather(*(batcher.submit([1.0]) for _ in range(10)))
        finally:
            await batcher.stop()

    asyncio.run(run())
    assert max(calls) == 4
    assert sum(calls) == 10


def test_lone_request_does_not_wait_for_a_full_batch():
    async def run():
        batcher = Batcher(
            lambda batch: [7.0] * len(batch), max_wait_seconds=0.01
        )
        batcher.ensure_started()
        try:
            return await asyncio.wait_for(batcher.submit([1.0]), timeout=2.0)
        finally:
            await batcher.stop()

    assert asyncio.run(run()) == 7.0


def test_a_failed_batch_fails_every_request_in_it():
    def predict_batch(batch):
        raise ValueError("bad features")

    async def run():
        batcher = Batcher(predict_batch)
        batcher.ensure_started()
        try:
            await asyncio.gather(batcher.submit([1.0]), batcher.submit([2.0]))
        except ValueError as exc:
            return str(exc)
        finally:
            await batcher.stop()
        raise AssertionError("expected ValueError")

    assert asyncio.run(run()) == "bad features"


def test_batched_endpoint_returns_prediction_and_reports_batching():
    client = make_client()
    response = client.post(
        "/predict/batched",
        json={"model": "tiny-linear", "features": [2.0, 4.0, 1.0]},
    )
    assert response.status_code == 200
    assert response.json()["prediction"] == 1.1
    assert client.post(
        "/predict/batched", json={"model": "nope", "features": [1.0]}
    ).status_code == 404
    assert client.post(
        "/predict/batched", json={"model": "tiny-linear", "features": [1.0]}
    ).status_code == 400
    batching = client.get("/metrics").json()["batching"]
    assert batching["batches"] >= 1
    assert batching["requests"] >= 1
