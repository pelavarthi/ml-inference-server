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

## Rust extension (tokenizer)

`rust/` holds a PyO3 extension module, `ml_rust`, with a word tokenizer:
words and numbers stay whole, punctuation becomes its own token, and a
small vocabulary maps tokens to ids (0 is padding, 1 is unknown). It has
the same tokenize, encode, and decode behavior as the reference
implementation in `app/tokenizer.py`, so the two can be compared directly.

Build it with maturin from the `rust/` directory (`maturin develop`), and
run its unit tests with `cargo test`. The server does not call it yet;
wiring it into preprocessing, with a fallback when it is missing, is a
later step.
