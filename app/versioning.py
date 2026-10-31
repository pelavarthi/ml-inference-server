"""Model versioning and A/B routing.

An explicit version always wins. Otherwise, an experiment splits
traffic between versions of one model by hashing the subject (a user
id, a session id, anything stable for one caller). Hashing, rather
than random choice, means the same subject always lands on the same
variant without storing any assignment, so results stay comparable
across requests.
"""

import hashlib


class Experiment:
    def __init__(self, name, model, variants):
        if not variants:
            raise ValueError("experiment needs at least one variant")
        if any(weight <= 0 for _, weight in variants):
            raise ValueError("variant weights must be positive")
        self.name = name
        self.model = model
        self.variants = list(variants)
        self.total_weight = sum(weight for _, weight in variants)

    def assign(self, subject):
        digest = hashlib.sha256(f"{self.name}:{subject}".encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") / 2**64 * self.total_weight
        running = 0.0
        for version, weight in self.variants:
            running += weight
            if bucket < running:
                return version
        return self.variants[-1][0]

    def info(self):
        return {
            "name": self.name,
            "model": self.model,
            "variants": [
                {"version": version, "weight": weight}
                for version, weight in self.variants
            ],
        }


class Router:
    """Resolves a predict request to one concrete model version."""

    def __init__(self, registry, experiments=None):
        self.registry = registry
        self.experiments = {}
        for experiment in experiments or []:
            self.register_experiment(experiment)

    def register_experiment(self, experiment):
        if experiment.name in self.experiments:
            raise ValueError(f"experiment already registered: {experiment.name}")
        self.experiments[experiment.name] = experiment

    def resolve(self, model, version=None, experiment=None, subject=None):
        if version is not None:
            return self.registry.get(model, version), None
        if experiment is not None:
            try:
                chosen = self.experiments[experiment]
            except KeyError:
                raise KeyError(f"unknown experiment: {experiment}") from None
            if chosen.model != model:
                raise ValueError(
                    f"experiment {experiment} is for model {chosen.model}, "
                    f"not {model}"
                )
            if not subject or not subject.strip():
                raise ValueError(
                    "subject is required for an experiment so the same "
                    "caller always gets the same variant"
                )
            version = chosen.assign(subject.strip())
            return self.registry.get(model, version), experiment
        return self.registry.get(model), None
