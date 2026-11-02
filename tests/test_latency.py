from fastapi.testclient import TestClient

from app.latency import LatencyTracker, percentile
from app.main import create_app
from app.registry import Model, ModelRegistry


def make_client():
    registry = ModelRegistry()
    registry.register(Model("tiny-linear", "0.1.0", [0.5, 0.25, -1.0], bias=0.1))
    return TestClient(create_app(registry))


def test_percentile_uses_nearest_rank():
    samples = list(range(1, 101))
    assert percentile(samples, 50) == 50
    assert percentile(samples, 95) == 95
    assert percentile(samples, 99) == 99
    assert percentile([], 95) is None


def test_tracker_summary_and_slow_count():
    tracker = LatencyTracker()
    for value in [10.0, 20.0, 30.0, 40.0, 2000.0]:
        tracker.record("POST /predict", value)
    summary = tracker.route_summary("POST /predict")
    assert summary["count"] == 5
    assert summary["p50_ms"] == 30.0
    assert summary["p90_ms"] == 2000.0
    assert summary["min_ms"] == 10.0
    assert summary["max_ms"] == 2000.0
    assert summary["slow_requests"] == 1
    assert tracker.route_summary("GET /nope") is None


def test_tracker_keeps_only_newest_samples():
    tracker = LatencyTracker(max_samples=3)
    for value in [1.0, 2.0, 3.0, 4.0]:
        tracker.record("GET /health", value)
    assert tracker.route_summary("GET /health")["count"] == 3
    assert tracker.route_summary("GET /health")["min_ms"] == 2.0


def test_latency_endpoint_groups_by_route_template():
    client = make_client()
    client.post("/predict", json={"model": "tiny-linear", "features": [2.0, 4.0, 1.0]})
    client.post("/predict", json={"model": "tiny-linear", "features": [1.0, 1.0, 1.0]})
    client.get("/models/tiny-linear")
    client.get("/models/other")
    data = client.get("/metrics/latency").json()
    assert data["routes"]["POST /predict"]["count"] == 2
    # Both model names collapse into the one route template.
    assert data["routes"]["GET /models/{name}"]["count"] == 2
    assert data["overall"]["count"] >= 4
    assert data["overall"]["p95_ms"] >= data["overall"]["p50_ms"]
    assert data["overall"]["p99_ms"] >= data["overall"]["p95_ms"]


def test_latency_endpoint_single_route_and_unknown():
    client = make_client()
    client.get("/health")
    single = client.get("/metrics/latency", params={"route": "GET /health"})
    assert single.status_code == 200
    assert single.json()["latency"]["count"] >= 1
    assert client.get(
        "/metrics/latency", params={"route": "GET /nope"}
    ).status_code == 404


def test_metrics_includes_latency():
    client = make_client()
    client.get("/health")
    data = client.get("/metrics").json()
    assert data["latency"]["routes"]["GET /health"]["count"] >= 1


def test_streaming_still_streams_with_middleware():
    client = make_client()
    response = client.post(
        "/predict/stream?chunk_size=1",
        json={"model": "tiny-linear", "features": [2.0, 4.0, 1.0]},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert response.text.count("data: ") == 4
