from fastapi.testclient import TestClient

from app.main import create_app
from app.registry import Model, ModelRegistry


def make_client():
    registry = ModelRegistry()
    registry.register(Model("tiny-linear", "0.1.0", [0.5, 0.25, -1.0], bias=0.1))
    return TestClient(create_app(registry))


def test_health_and_model_list():
    client = make_client()
    assert client.get("/health").json() == {"status": "ok", "models": 1}
    assert client.get("/models").json()["models"][0]["name"] == "tiny-linear"


def test_predict_and_errors():
    client = make_client()
    ok = client.post(
        "/predict", json={"model": "tiny-linear", "features": [2.0, 4.0, 1.0]}
    )
    assert ok.status_code == 200
    assert ok.json()["prediction"] == 1.1  # 1.0 + 1.0 - 1.0 + 0.1
    assert client.post("/predict", json={"model": "nope", "features": [1.0]}).status_code == 404
    assert client.post(
        "/predict", json={"model": "tiny-linear", "features": [1.0]}
    ).status_code == 400
    assert client.post(
        "/predict", json={"model": "tiny-linear", "features": []}
    ).status_code == 422
