"""Deployment config: the Dockerfile, compose file, and ignore file agree.

These tests do not build an image or need a Docker daemon. They check
the config the image would be built from: the base image, what gets
copied in, the port, the healthcheck, and the compose service that
runs it. A mismatch here is the usual reason a container starts and
then answers nothing, so it is worth catching before building.
"""

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from app.main import create_app
from app.registry import Model, ModelRegistry

ROOT = Path(__file__).resolve().parent.parent


def read(name):
    return (ROOT / name).read_text()


def dockerfile_lines():
    # Join backslash continuations first, so the multi-line
    # HEALTHCHECK reads as one instruction instead of a stray CMD.
    joined = []
    current = ""
    for raw in read("Dockerfile").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            current += line[:-1] + " "
        else:
            current += line
            joined.append(current)
            current = ""
    if current:
        joined.append(current)
    return joined


def dockerfile_value(instruction):
    for line in dockerfile_lines():
        if line.startswith(instruction):
            return line[len(instruction):].strip()
    raise AssertionError(f"Dockerfile has no {instruction}")


def compose():
    return yaml.safe_load(read("docker-compose.yml"))


def test_dockerfile_base_and_copies():
    lines = dockerfile_lines()
    assert lines[0] == "FROM python:3.12-slim"
    assert "WORKDIR /app" in lines
    assert "COPY requirements.txt ." in lines
    assert "RUN pip install --no-cache-dir -r requirements.txt" in lines
    assert "COPY app/ ./app/" in lines
    assert "COPY models/ ./models/" in lines
    # Tests and the Rust sources are not needed to serve, so they stay
    # out of the image instead of bloating it.
    assert not any(line.startswith("COPY tests") for line in lines)
    assert not any(line.startswith("COPY rust") for line in lines)


def test_dockerfile_port_healthcheck_and_command():
    assert dockerfile_value("EXPOSE") == "8000"
    healthcheck = " ".join(dockerfile_lines())
    assert "HEALTHCHECK" in healthcheck
    assert "http://localhost:8000/health" in healthcheck
    command = dockerfile_value("CMD")
    assert "uvicorn" in command
    assert "app.main:app" in command
    assert "--host" in command and "0.0.0.0" in command
    assert "--port" in command and "8000" in command


def test_dockerignore_keeps_caches_and_build_dirs_out():
    ignored = read(".dockerignore").splitlines()
    assert ".git" in ignored
    assert "__pycache__/" in ignored
    assert "*.pyc" in ignored
    assert ".pytest_cache/" in ignored
    assert "rust/target/" in ignored


def test_compose_service_matches_dockerfile():
    api = compose()["services"]["api"]
    assert api["build"] == "."
    assert "8000:8000" in api["ports"]
    assert api["restart"] == "unless-stopped"
    healthcheck = " ".join(api["healthcheck"]["test"])
    assert "http://localhost:8000/health" in healthcheck
    # Models are mounted read-only, so swapping a model file does not
    # need a rebuild, and the container cannot rewrite the models.
    assert "./models:/app/models:ro" in api["volumes"]


def test_requirements_cover_what_the_image_runs():
    requirements = read("requirements.txt")
    assert "fastapi" in requirements
    assert "uvicorn" in requirements


def test_app_the_image_runs_answers_health():
    registry = ModelRegistry()
    registry.register(Model("tiny-linear", "0.1.0", [0.5, 0.25, -1.0], bias=0.1))
    client = TestClient(create_app(registry))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "models": 1}
