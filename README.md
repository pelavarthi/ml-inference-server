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
run its unit tests with `cargo test`. The server calls it through
`app/backend.py`: when the extension is built, preprocessing runs in
Rust; when it is missing, the server starts anyway and uses the pure
Python tokenizer instead. `POST /preprocess` turns texts into a padded
batch and reports which backend served the request, and
`GET /health/detail` and `GET /metrics` report the active backend too.
A request can also supply its own `vocabulary` texts, which are learned
on a fresh backend of the same kind for that request only.

## Preprocessing and benchmarking

`app/preprocessing.py` turns raw texts into a padded batch: token ids
padded with 0 to the longest sequence in the batch, an attention mask
(1 for real tokens, 0 for padding), and each sequence's true length. It
accepts either tokenizer, and the Rust side has a matching
`Tokenizer.prepare_batch` that does the whole batch in one call instead
of one call per text.

`benchmarks/preprocess_bench.py` times all three paths (Python, Rust
per text, Rust per batch) over 500 texts and prints the speedup. The
Rust rows are skipped when the extension is not built.
