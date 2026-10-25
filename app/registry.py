"""A small registry that holds loaded models by name.

Models are loaded from JSON files so adding a new one does not mean
changing the server code: drop a file in, register it, serve it.
"""

import json
from pathlib import Path


class Model:
    """A tiny linear model: prediction is weights . features + bias."""

    def __init__(self, name, version, weights, bias=0.0):
        self.name = name
        self.version = version
        self.weights = list(weights)
        self.bias = bias

    def predict(self, features):
        if len(features) != len(self.weights):
            raise ValueError(
                f"expected {len(self.weights)} features, got {len(features)}"
            )
        return sum(w * x for w, x in zip(self.weights, features)) + self.bias

    def predict_batch(self, batch):
        return [self.predict(features) for features in batch]

    def info(self):
        return {
            "name": self.name,
            "version": self.version,
            "num_features": len(self.weights),
        }


class ModelRegistry:
    def __init__(self):
        self._models = {}

    def register(self, model):
        if model.name in self._models:
            raise ValueError(f"model already registered: {model.name}")
        self._models[model.name] = model

    def load_from_file(self, path):
        data = json.loads(Path(path).read_text())
        model = Model(
            name=data["name"],
            version=data.get("version", "0.1.0"),
            weights=data["weights"],
            bias=data.get("bias", 0.0),
        )
        self.register(model)
        return model

    def load_directory(self, directory):
        loaded = []
        for path in sorted(Path(directory).glob("*.json")):
            loaded.append(self.load_from_file(path))
        return loaded

    def get(self, name):
        try:
            return self._models[name]
        except KeyError:
            raise KeyError(f"unknown model: {name}") from None

    def list_models(self):
        return [model.info() for model in self._models.values()]
