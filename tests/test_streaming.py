import asyncio
import json

from fastapi.testclient import TestClient

from app.main import create_app
from app.registry import Model, ModelRegistry
from app.streaming import ConcurrencyLimiter, stream_prediction


def make_client(**kwargs):
    registry = ModelRegistry()
    registry.register(Model("tiny-linear", "0.1.0", [0.5, 0.25, -1.0], bias=0.1))
    return TestClient(create_app(registry, **kwargs))


def parse_events(text):
    return [
        json.loads(line[6:])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


def test_stream_prediction_matches_model_predict():
    async def run():
        model = Model("tiny-linear", "0.1.0", [0.5, 0.25, -1.0], bias=0.1)
        events = []
        async for event in stream_prediction(
            model, [2.0, 4.0, 1.0], chunk_size=2
        ):
            events.append(event)
        return events

    events = asyncio.run(run())
    assert events[0] == {"chunk": 0, "partial": 2.0, "done": False}
    assert events[1] == {"chunk": 1, "partial": 1.0, "done": False}
    assert events[2] == {"prediction": 1.1, "done": True}


def test_stream_times_out_instead_of_hanging():
    async def run():
        model = Model("slow", "0.1.0", [1.0] * 8)
        events = []
        async for event in stream_prediction(
            model, [1.0] * 8, chunk_size=2,
            chunk_delay=0.05, timeout_seconds=0.12,
        ):
            events.append(event)
        return events

    events = asyncio.run(run())
    assert events[-1] == {"error": "stream timed out", "done": True}
    assert all("prediction" not in event for event in events)


def test_limiter_rejects_when_full_and_frees_on_release():
    limiter = ConcurrencyLimiter(2)
    assert limiter.try_acquire()
    assert limiter.try_acquire()
    assert not limiter.try_acquire()
    assert limiter.rejected == 1
    limiter.release()
    assert limiter.try_acquire()


def test_stream_endpoint_sends_partials_then_final():
    client = make_client()
    response = client.post(
        "/predict/stream?chunk_size=1",
        json={"model": "tiny-linear", "features": [2.0, 4.0, 1.0]},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    events = parse_events(response.text)
    assert [event["partial"] for event in events[:-1]] == [1.0, 2.0, 1.0]
    assert events[-1] == {"prediction": 1.1, "done": True}


def test_stream_endpoint_errors():
    client = make_client()
    assert client.post(
        "/predict/stream", json={"model": "nope", "features": [1.0]}
    ).status_code == 404
    assert client.post(
        "/predict/stream", json={"model": "tiny-linear", "features": [1.0]}
    ).status_code == 400
    assert client.post(
        "/predict/stream", json={"model": "tiny-linear", "features": []}
    ).status_code == 422
    assert client.post(
        "/predict/stream?timeout_seconds=0",
        json={"model": "tiny-linear", "features": [2.0, 4.0, 1.0]},
    ).status_code == 422


def test_stream_endpoint_rejects_when_no_slots():
    client = make_client(max_concurrent_streams=0)
    response = client.post(
        "/predict/stream",
        json={"model": "tiny-linear", "features": [2.0, 4.0, 1.0]},
    )
    assert response.status_code == 429
    assert client.get("/metrics").json()["streaming"]["rejected"] >= 1
