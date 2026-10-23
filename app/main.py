"""FastAPI entry point: health, model list, and predict."""

from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.registry import ModelRegistry
from app.schemas import PredictRequest, PredictResponse

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def create_app(registry=None):
    registry = registry or ModelRegistry()
    if not registry.list_models() and MODELS_DIR.exists():
        registry.load_directory(MODELS_DIR)

    app = FastAPI(title="ml-inference-server")

    @app.get("/health")
    def health():
        return {"status": "ok", "models": len(registry.list_models())}

    @app.get("/models")
    def list_models():
        return {"models": registry.list_models()}

    @app.get("/models/{name}")
    def get_model(name: str):
        try:
            return registry.get(name).info()
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown model")

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest):
        try:
            model = registry.get(request.model)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown model")
        try:
            prediction = model.predict(request.features)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return PredictResponse(
            model=model.name, version=model.version, prediction=prediction
        )

    return app


app = create_app()
