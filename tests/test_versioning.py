import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.registry import Model, ModelRegistry
from app.versioning import Experiment, Router


def make_registry():
    registry = ModelRegistry()
    registry.register(Model("tiny-linear", "0.1.0", [0.5, 0.25, -1.0], bias=0.1))
    registry.register(Model("tiny-linear", "0.2.0", [0.6, 0.3, -0.9], bias=0.2))
    return registry


def test_registry_keeps_both_versions_and_defaults_to_lowest():
    registry = make_registry()
    assert registry.versions("tiny-linear") == ["0.1.0", "0.2.0"]
    assert registry.get("tiny-linear").version == "0.1.0"
    assert registry.get("tiny-linear", "0.2.0").predict([2.0, 4.0, 1.0]) == 1.7
    with pytest.raises(KeyError):
        registry.get("tiny-linear", "9.9.9")
    with pytest.raises(ValueError):
        registry.register(Model("tiny-linear", "0.1.0", [1.0]))


def test_registry_set_default_promotes_a_version():
    registry = make_registry()
    registry.set_default("tiny-linear", "0.2.0")
    assert registry.get("tiny-linear").version == "0.2.0"
    with pytest.raises(KeyError):
        registry.set_default("tiny-linear", "9.9.9")


def test_experiment_assignment_is_stable_and_splits():
    experiment = Experiment("ab", "tiny-linear", [("0.1.0", 1), ("0.2.0", 1)])
    assert experiment.assign("user-7") == experiment.assign("user-7")
    seen = {experiment.assign(f"user-{i}") for i in range(200)}
    assert seen == {"0.1.0", "0.2.0"}
    with pytest.raises(ValueError):
        Experiment("bad", "tiny-linear", [("0.1.0", 0), ("0.2.0", 1)])
    with pytest.raises(ValueError):
        Experiment("empty", "tiny-linear", [])


def test_router_precedence_and_errors():
    router = Router(
        make_registry(),
        [Experiment("ab", "tiny-linear", [("0.1.0", 1), ("0.2.0", 1)])],
    )
    model, experiment = router.resolve("tiny-linear", version="0.2.0",
                                       experiment="ab", subject="user-1")
    assert model.version == "0.2.0" and experiment is None
    model, experiment = router.resolve("tiny-linear", experiment="ab",
                                       subject="user-1")
    assert experiment == "ab"
    with pytest.raises(ValueError):
        router.resolve("tiny-linear", experiment="ab")
    with pytest.raises(KeyError):
        router.resolve("tiny-linear", experiment="nope", subject="user-1")
    with pytest.raises(ValueError):
        router.resolve("other-model", experiment="ab", subject="user-1")


def make_client():
    router_registry = make_registry()
    return TestClient(create_app(router_registry))


def test_versions_endpoint_and_versioned_predict():
    client = make_client()
    versions = client.get("/models/tiny-linear/versions")
    assert versions.status_code == 200
    assert versions.json() == {
        "model": "tiny-linear",
        "versions": ["0.1.0", "0.2.0"],
        "default": "0.1.0",
    }
    assert client.get("/models/nope/versions").status_code == 404
    default = client.post("/predict", json={"model": "tiny-linear",
                                            "features": [2.0, 4.0, 1.0]})
    assert default.json()["version"] == "0.1.0"
    assert default.json()["prediction"] == 1.1
    pinned = client.post("/predict", json={"model": "tiny-linear",
                                           "version": "0.2.0",
                                           "features": [2.0, 4.0, 1.0]})
    assert pinned.json()["version"] == "0.2.0"
    assert pinned.json()["prediction"] == 1.7
    assert client.post("/predict", json={"model": "tiny-linear",
                                         "version": "9.9.9",
                                         "features": [1.0, 1.0, 1.0]}).status_code == 404


def test_experiment_predict_is_stable_per_subject():
    client = make_client()
    experiments = client.get("/experiments")
    assert experiments.json()["experiments"][0]["name"] == "tiny-linear-ab"
    first = client.post("/predict", json={"model": "tiny-linear",
                                          "experiment": "tiny-linear-ab",
                                          "subject": "user-7",
                                          "features": [2.0, 4.0, 1.0]})
    second = client.post("/predict", json={"model": "tiny-linear",
                                           "experiment": "tiny-linear-ab",
                                           "subject": "user-7",
                                           "features": [2.0, 4.0, 1.0]})
    assert first.json()["version"] == second.json()["version"]
    assert first.json()["experiment"] == "tiny-linear-ab"
    assert client.post("/predict", json={"model": "tiny-linear",
                                         "experiment": "tiny-linear-ab",
                                         "features": [1.0, 1.0, 1.0]}).status_code == 400
    assert client.post("/predict", json={"model": "tiny-linear",
                                         "experiment": "nope",
                                         "subject": "user-7",
                                         "features": [1.0, 1.0, 1.0]}).status_code == 404
    metrics = client.get("/metrics").json()
    served = first.json()["version"]
    assert metrics["predictions"]["by_version"][f"tiny-linear@{served}"] >= 2
