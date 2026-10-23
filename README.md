# ml-inference-server

ML inference server in Python + Rust: FastAPI serving, batching, streaming, Rust acceleration.

## Run

```
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then `GET /health`, `GET /models`, and `POST /predict` with
`{"model": "tiny-linear", "features": [2.0, 4.0, 1.0]}`.

Models live in `models/` as JSON files and are loaded by the registry at
startup. Tests: `pytest`.
