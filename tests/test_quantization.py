import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.quantization import QuantizedModel, quantize_weights
from app.registry import Model, ModelRegistry


def test_roundtrip_error_is_bounded_by_half_a_step():
    weights = [0.5, 0.25, -1.0, 0.75]
    quantized = quantize_weights(weights, bits=8)
    assert quantized.max_weight_error(weights) <= quantized.scale / 2
    assert quantized.dequantize() == pytest.approx(weights, abs=quantized.scale)


def test_fewer_bits_means_larger_error_and_smaller_size():
    weights = [0.5, 0.25, -1.0, 0.75, -0.33, 0.9]
    coarse = quantize_weights(weights, bits=4)
    fine = quantize_weights(weights, bits=8)
    assert coarse.max_weight_error(weights) >= fine.max_weight_error(weights)
    assert coarse.quantized_size_bytes() == 3  # 6 weights * 4 bits
    assert fine.quantized_size_bytes() == 6
    assert fine.original_size_bytes() == 48  # 6 weights * 8 bytes


def test_constant_weights_roundtrip_exactly():
    quantized = quantize_weights([0.7, 0.7, 0.7], bits=8)
    assert quantized.dequantize() == [0.7, 0.7, 0.7]
    assert quantized.max_weight_error([0.7, 0.7, 0.7]) == 0.0


def test_unsupported_bits_rejected():
    with pytest.raises(ValueError, match="unsupported quantization bits"):
        quantize_weights([1.0, 2.0], bits=2)


def test_quantized_model_prediction_is_close():
    model = Model("tiny-linear", "0.1.0", [0.5, 0.25, -1.0], bias=0.1)
    quantized = QuantizedModel(model, bits=8)
    assert quantized.predict([2.0, 4.0, 1.0]) == pytest.approx(
        model.predict([2.0, 4.0, 1.0]), abs=0.02
    )


def make_client():
    registry = ModelRegistry()
    registry.register(Model("tiny-linear", "0.1.0", [0.5, 0.25, -1.0], bias=0.1))
    return TestClient(create_app(registry))


def test_predict_quantized_is_close_and_flagged():
    client = make_client()
    full = client.post(
        "/predict", json={"model": "tiny-linear", "features": [2.0, 4.0, 1.0]}
    ).json()
    quantized = client.post(
        "/predict",
        json={
            "model": "tiny-linear",
            "features": [2.0, 4.0, 1.0],
            "quantization": 8,
        },
    ).json()
    assert quantized["quantized"] is True
    assert quantized["quantization_bits"] == 8
    assert quantized["cached"] is False
    assert quantized["prediction"] == pytest.approx(full["prediction"], abs=0.02)


def test_quantized_and_full_predictions_cache_separately():
    client = make_client()
    body = {"model": "tiny-linear", "features": [2.0, 4.0, 1.0]}
    client.post("/predict", json=body)
    first_quantized = client.post("/predict", json={**body, "quantization": 8})
    assert first_quantized.json()["cached"] is False
    second_quantized = client.post("/predict", json={**body, "quantization": 8})
    assert second_quantized.json()["cached"] is True
    assert (
        second_quantized.json()["prediction"]
        == first_quantized.json()["prediction"]
    )


def test_predict_unsupported_bits_is_rejected():
    client = make_client()
    response = client.post(
        "/predict",
        json={
            "model": "tiny-linear",
            "features": [2.0, 4.0, 1.0],
            "quantization": 2,
        },
    )
    assert response.status_code == 422


def test_quantization_summary_endpoint():
    client = make_client()
    response = client.get("/models/tiny-linear/quantization?bits=8")
    assert response.status_code == 200
    body = response.json()
    assert body["bits"] == 8
    assert body["quantized_size_bytes"] == 3
    assert body["original_size_bytes"] == 24
    assert body["codes"] == [255, 212, 0]
    assert body["max_weight_error"] <= body["scale"] / 2
    assert client.get("/models/nope/quantization").status_code == 404
    assert client.get("/models/tiny-linear/quantization?bits=2").status_code == 400
