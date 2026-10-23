import pytest

from app.registry import Model, ModelRegistry


def test_predict_is_dot_product_plus_bias():
    model = Model("m", "0.1.0", [2.0, 3.0], bias=1.0)
    assert model.predict([1.0, 2.0]) == 9.0


def test_predict_rejects_wrong_feature_count():
    model = Model("m", "0.1.0", [1.0, 1.0])
    with pytest.raises(ValueError):
        model.predict([1.0])


def test_registry_loads_and_lists(tmp_path):
    path = tmp_path / "m.json"
    path.write_text('{"name": "m", "version": "0.1.0", "weights": [1.0], "bias": 0.5}')
    registry = ModelRegistry()
    registry.load_from_file(path)
    assert registry.get("m").predict([2.0]) == 2.5
    assert registry.list_models() == [
        {"name": "m", "version": "0.1.0", "num_features": 1}
    ]


def test_registry_rejects_duplicates_and_unknown():
    registry = ModelRegistry()
    registry.register(Model("m", "0.1.0", [1.0]))
    with pytest.raises(ValueError):
        registry.register(Model("m", "0.1.0", [1.0]))
    with pytest.raises(KeyError):
        registry.get("nope")
