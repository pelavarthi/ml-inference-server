"""A small registry that holds loaded models by name and version.

Models are loaded from JSON files so adding a new one does not mean
changing the server code: drop a file in, register it, serve it.

A model name can now have several versions. Registering a new version
never replaces an old one, so a request can pin a version, and the
default for a name is its lowest version unless a different one is
promoted with set_default.
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
        self._defaults = {}

    def register(self, model):
        key = (model.name, model.version)
        if key in self._models:
            raise ValueError(
                f"model already registered: {model.name} {model.version}"
            )
        self._models[key] = model

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

    def versions(self, name):
        found = sorted(
            version for (model_name, version) in self._models
            if model_name == name
        )
        if not found:
            raise KeyError(f"unknown model: {name}")
        return found

    def default_version(self, name):
        if name in self._defaults:
            return self._defaults[name]
        return self.versions(name)[0]

    def set_default(self, name, version):
        self.get(name, version)
        self._defaults[name] = version

    def get(self, name, version=None):
        if version is None:
            version = self.default_version(name)
        try:
            return self._models[(name, version)]
        except KeyError:
            raise KeyError(f"unknown model: {name} {version}") from None

    def list_models(self):
        return [self._models[key].info() for key in sorted(self._models)]
