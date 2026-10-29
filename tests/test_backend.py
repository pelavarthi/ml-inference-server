import importlib.util
import sys

import pytest
from fastapi.testclient import TestClient

from app.backend import load_backend
from app.main import create_app
from app.registry import Model, ModelRegistry
from app.tokenizer import PAD_ID, Tokenizer

RUST_AVAILABLE = importlib.util.find_spec("ml_rust") is not None

CORPUS = ["hello world", "hello there"]


def trained_python_backend():
    backend = load_backend(prefer="python")
    backend.build_vocab(CORPUS)
    return backend


def make_client(backend=None):
    registry = ModelRegistry()
    registry.register(Model("tiny-linear", "0.1.0", [0.5, 0.25, -1.0], bias=0.1))
    return TestClient(create_app(registry, text_backend=backend))


def test_python_backend_prepare_batch_matches_preprocessor():
    batch = trained_python_backend().prepare_batch(["hello", "hello world"])
    assert batch.ids == [[2, PAD_ID], [2, 3]]
    assert batch.attention_mask == [[1, 0], [1, 1]]
    assert batch.lengths == [1, 2]


def test_auto_backend_falls_back_when_extension_import_fails(monkeypatch):
    monkeypatch.setitem(sys.modules, "ml_rust", None)
    monkeypatch.setattr("app.backend.rust_available", lambda: False)
    backend = load_backend()
    assert backend.name == "python"
    batch = backend.prepare_batch(["hello"])
    assert batch.ids == [[1]]


def test_explicit_rust_backend_raises_when_extension_missing():
    if RUST_AVAILABLE:
        pytest.skip("rust extension is built")
    with pytest.raises(ImportError):
        load_backend(prefer="rust")


@pytest.mark.skipif(not RUST_AVAILABLE, reason="rust extension not built")
def test_auto_backend_prefers_rust_when_built():
    import ml_rust

    backend = load_backend()
    assert backend.name == "rust"
    assert isinstance(backend._tokenizer, ml_rust.Tokenizer)
    backend.build_vocab(CORPUS)
    batch = backend.prepare_batch(["hello", "hello world"])
    assert batch.ids == [[2, PAD_ID], [2, 3]]
    assert batch.lengths == [1, 2]


@pytest.mark.skipif(not RUST_AVAILABLE, reason="rust extension not built")
def test_rust_and_python_backends_agree():
    rust = load_backend(prefer="rust")
    rust.build_vocab(CORPUS)
    python = trained_python_backend()
    texts = ["hello", "hello world", "hello nobody"]
    assert rust.prepare_batch(texts).ids == python.prepare_batch(texts).ids


def test_preprocess_endpoint_uses_shared_backend():
    client = make_client(trained_python_backend())
    response = client.post("/preprocess", json={"texts": ["hello world"]})
    assert response.status_code == 200
    assert response.json() == {
        "ids": [[2, 3]],
        "attention_mask": [[1, 1]],
        "lengths": [2],
        "backend": "python",
    }


def test_preprocess_endpoint_learns_request_vocabulary():
    client = make_client(load_backend(prefer="python"))
    response = client.post(
        "/preprocess",
        json={"texts": ["hello world"], "vocabulary": CORPUS},
    )
    assert response.json()["ids"] == [[2, 3]]
    # The shared backend was not polluted by the request vocabulary.
    again = client.post("/preprocess", json={"texts": ["hello"]})
    assert again.json()["ids"] == [[1]]


def test_preprocess_endpoint_handles_empty_batch_and_bad_length():
    client = make_client(trained_python_backend())
    empty = client.post("/preprocess", json={"texts": []})
    assert empty.json()["ids"] == []
    assert client.post(
        "/preprocess", json={"texts": ["hello"], "max_length": 0}
    ).status_code == 422


def test_health_and_metrics_report_backend_and_counts():
    client = make_client(trained_python_backend())
    client.post("/preprocess", json={"texts": ["hello"]})
    assert client.get("/health/detail").json()["preprocessing_backend"] == "python"
    metrics = client.get("/metrics").json()
    assert metrics["preprocessing_backend"] == "python"
    assert metrics["preprocessing"] == {"python": 1}
