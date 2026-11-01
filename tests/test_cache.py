import time

from fastapi.testclient import TestClient

from app.cache import PredictionCache
from app.main import create_app
from app.registry import Model, ModelRegistry


def test_cache_hit_miss_and_key_isolation():
    cache = PredictionCache(max_size=10, ttl_seconds=60)
    assert cache.get("tiny-linear", "0.1.0", [2.0, 4.0, 1.0]) is None
    cache.put("tiny-linear", "0.1.0", [2.0, 4.0, 1.0], 1.1)
    assert cache.get("tiny-linear", "0.1.0", [2.0, 4.0, 1.0]) == 1.1
    # A different version or different features is a different entry.
    assert cache.get("tiny-linear", "0.2.0", [2.0, 4.0, 1.0]) is None
    assert cache.get("tiny-linear", "0.1.0", [2.0, 4.0, 2.0]) is None
    stats = cache.stats()
    assert stats["hits"] == 1 and stats["misses"] == 3
    assert stats["size"] == 1


def test_cache_evicts_least_recently_used():
    cache = PredictionCache(max_size=2, ttl_seconds=60)
    cache.put("m", "0.1.0", [1.0], 1.0)
    cache.put("m", "0.1.0", [2.0], 2.0)
    cache.get("m", "0.1.0", [1.0])  # [1.0] is now the most recent
    cache.put("m", "0.1.0", [3.0], 3.0)  # evicts [2.0], not [1.0]
    assert cache.get("m", "0.1.0", [1.0]) == 1.0
    assert cache.get("m", "0.1.0", [2.0]) is None
    assert cache.stats()["evictions"] == 1


def test_cache_entry_expires_after_ttl():
    cache = PredictionCache(max_size=10, ttl_seconds=0.05)
    cache.put("m", "0.1.0", [1.0], 1.0)
    assert cache.get("m", "0.1.0", [1.0]) == 1.0
    time.sleep(0.06)
    assert cache.get("m", "0.1.0", [1.0]) is None
    assert cache.stats()["size"] == 0


def test_cache_clear_returns_count():
    cache = PredictionCache(max_size=10, ttl_seconds=60)
    cache.put("m", "0.1.0", [1.0], 1.0)
    cache.put("m", "0.1.0", [2.0], 2.0)
    assert cache.clear() == 2
    assert cache.stats()["size"] == 0


def make_client():
    registry = ModelRegistry()
    registry.register(Model("tiny-linear", "0.1.0", [0.5, 0.25, -1.0], bias=0.1))
    return TestClient(create_app(registry))


def test_predict_second_call_is_cached():
    client = make_client()
    body = {"model": "tiny-linear", "features": [2.0, 4.0, 1.0]}
    first = client.post("/predict", json=body)
    assert first.json()["cached"] is False
    second = client.post("/predict", json=body)
    assert second.json()["cached"] is True
    assert second.json()["prediction"] == first.json()["prediction"]
    stats = client.get("/cache/stats").json()
    assert stats["hits"] == 1 and stats["misses"] == 1
    assert stats["size"] == 1
    assert client.get("/metrics").json()["cache"]["hits"] == 1


def test_predict_cache_clear_endpoint():
    client = make_client()
    body = {"model": "tiny-linear", "features": [2.0, 4.0, 1.0]}
    client.post("/predict", json=body)
    assert client.post("/predict", json=body).json()["cached"] is True
    cleared = client.delete("/cache")
    assert cleared.json()["cleared"] == 1
    assert client.post("/predict", json=body).json()["cached"] is False
