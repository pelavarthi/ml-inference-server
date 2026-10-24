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

## Validation, metrics, async predictions

Requests are validated before any model runs: the model name must not be
blank, features must be finite numbers, and at most 1024 features are
accepted. Bad input gets a 422, an unknown model a 404, and a wrong
feature count a 400.

`GET /health/detail` adds uptime, queue depth, and prediction counts to
the basic health check, and `GET /metrics` reports request counts by
method, path, and status plus prediction totals.

For slow models, `POST /predict/async` returns a job id right away (202)
instead of holding the request open. Poll `GET /predict/jobs/{job_id}`
for `pending`, `running`, `done`, or `failed`. The queue holds at most
100 jobs and answers 429 when full.
