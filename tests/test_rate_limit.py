import asyncio

import httpx
from fastapi.testclient import TestClient

from app.main import create_app
from app.rate_limit import RateLimiter
from app.registry import Model, ModelRegistry
from benchmarks.load_test import run_load_test, summarise


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make_client(**kwargs):
    registry = ModelRegistry()
    registry.register(Model("tiny-linear", "0.1.0", [0.5, 0.25, -1.0], bias=0.1))
    return TestClient(create_app(registry, **kwargs))


def test_bucket_allows_burst_then_blocks():
    limiter = RateLimiter(limit_per_minute=60, burst=2)
    assert limiter.allow("alice")[0] is True
    assert limiter.allow("alice")[0] is True
    allowed, retry_after = limiter.allow("alice")
    assert allowed is False
    assert retry_after > 0


def test_bucket_refills_over_time():
    clock = FakeClock()
    limiter = RateLimiter(limit_per_minute=60, burst=2, clock=clock)
    limiter.allow("alice")
    limiter.allow("alice")
    assert limiter.allow("alice")[0] is False
    clock.now += 1.5  # 60 a minute refills one token a second
    assert limiter.allow("alice")[0] is True
    assert limiter.allow("alice")[0] is False


def test_keys_have_separate_budgets():
    limiter = RateLimiter(limit_per_minute=60, burst=1)
    assert limiter.allow("alice")[0] is True
    assert limiter.allow("alice")[0] is False
    assert limiter.allow("bob")[0] is True


def test_retry_after_is_at_least_one_second():
    limiter = RateLimiter(limit_per_minute=60, burst=1)
    assert limiter.retry_after_seconds(0.2) == 1
    assert limiter.retry_after_seconds(2.4) == 3


def test_stats_and_reset():
    limiter = RateLimiter(limit_per_minute=60, burst=1)
    limiter.allow("alice")
    limiter.allow("alice")
    stats = limiter.stats()
    assert stats["clients_tracked"] == 1
    assert stats["limited"] == 1
    assert stats["limit_per_minute"] == 60
    assert stats["burst"] == 1
    limiter.reset()
    assert limiter.stats() == {
        "clients_tracked": 0,
        "limited": 0,
        "limit_per_minute": 60,
        "burst": 1,
    }


def test_predict_is_limited_with_retry_after_header():
    client = make_client(rate_limit_per_minute=60, rate_limit_burst=2)
    body = {"model": "tiny-linear", "features": [2.0, 4.0, 1.0]}
    headers = {"X-Client-Id": "alice"}
    assert client.post("/predict", json=body, headers=headers).status_code == 200
    assert client.post("/predict", json=body, headers=headers).status_code == 200
    limited = client.post("/predict", json=body, headers=headers)
    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "1"
    assert "rate limit" in limited.json()["detail"]


def test_other_clients_and_health_are_not_limited():
    client = make_client(rate_limit_per_minute=60, rate_limit_burst=1)
    body = {"model": "tiny-linear", "features": [2.0, 4.0, 1.0]}
    assert client.post("/predict", json=body, headers={"X-Client-Id": "alice"}).status_code == 200
    assert client.post("/predict", json=body, headers={"X-Client-Id": "alice"}).status_code == 429
    # A different client id has its own budget.
    assert client.post("/predict", json=body, headers={"X-Client-Id": "bob"}).status_code == 200
    # Health and metrics are never limited.
    assert client.get("/health").status_code == 200
    assert client.get("/metrics").status_code == 200


def test_metrics_reports_rate_limiting():
    client = make_client(rate_limit_per_minute=60, rate_limit_burst=1)
    body = {"model": "tiny-linear", "features": [2.0, 4.0, 1.0]}
    headers = {"X-Client-Id": "alice"}
    client.post("/predict", json=body, headers=headers)
    client.post("/predict", json=body, headers=headers)
    data = client.get("/metrics").json()
    assert data["rate_limit"]["limited"] == 1
    assert data["rate_limit"]["limit_per_minute"] == 60
    assert data["rate_limit"]["burst"] == 1


def test_summarise_counts_statuses_and_percentiles():
    results = [(10.0, 200), (20.0, 200), (30.0, 429), (40.0, 500)]
    summary = summarise(results, wall_seconds=2.0)
    assert summary["total"] == 4
    assert summary["ok"] == 2
    assert summary["rate_limited"] == 1
    assert summary["errors"] == 1
    assert summary["requests_per_second"] == 2.0
    assert summary["mean_ms"] == 25.0
    assert summary["p50_ms"] == 20.0
    assert summary["p95_ms"] == 40.0


def test_load_test_reports_limiting_against_live_app():
    registry = ModelRegistry()
    registry.register(Model("tiny-linear", "0.1.0", [0.5, 0.25, -1.0], bias=0.1))
    app = create_app(registry, rate_limit_per_minute=60, rate_limit_burst=5)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await run_load_test(client, total=20, concurrency=5)

    summary = asyncio.run(run())
    # Burst of 5 lets 5 through; the rest hit the limiter, not errors.
    assert summary["total"] == 20
    assert summary["ok"] == 5
    assert summary["rate_limited"] == 15
    assert summary["errors"] == 0
    assert summary["p95_ms"] >= summary["p50_ms"]
