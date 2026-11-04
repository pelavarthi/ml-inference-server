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

## Model versions and A/B tests

The registry is keyed by name and version, so one model can have
several versions at once and a new version never overwrites the old
one. `GET /models/{name}/versions` lists them, and the default for a
name is its lowest version unless another is promoted with
`set_default`.

`POST /predict` accepts an optional `version` to pin one, or an
`experiment` plus a `subject` (a user id, a session id) to join an A/B
test. The subject is hashed, so the same caller always gets the same
variant with nothing stored. An explicit version wins over an
experiment. A model with more than one version automatically gets an
even A/B experiment named `{model}-ab`; `GET /experiments` lists
them. The shipped `tiny-linear` has versions 0.1.0 and 0.2.0, so
`tiny-linear-ab` splits between them. `GET /metrics` counts
predictions per version, which is how you compare variants.

## Prediction caching

`POST /predict` caches results in memory: the key is the model name,
version, and features, so asking the same question twice runs the
model once and reuses the answer the second time (the response says
`"cached": true`). Entries expire after 5 minutes and the cache holds
at most 1024 entries, dropping the least recently used first, so a
stale or unbounded cache cannot grow forever. `GET /cache/stats`
reports hits, misses, evictions, and size, `DELETE /cache` clears it,
and `GET /metrics` includes the same cache stats.

## Quantization

`POST /predict` accepts an optional `quantization` of 4, 8, or 16
bits. The model's weights are rounded to that many bits per weight
before predicting, so an 8-bit model stores each weight in 1 byte
instead of 8, at the cost of a small rounding error (at most half a
grid step per weight). The response says `"quantized": true` and
reports the bit width, and quantized answers are cached separately
from full-precision ones, so the two never get mixed up.
`GET /models/{name}/quantization?bits=8` shows the codes, the scale,
the size saved, and the largest weight error for a model.

## Latency percentiles and observability

Averages hide the slow requests, so every request is timed by
middleware in `app/latency.py` and grouped by route template (both
`/models/tiny-linear` and `/models/other` count as
`GET /models/{name}`). `GET /metrics/latency` reports, per route and
overall, the count, mean, min, max, p50, p90, p95, and p99 in
milliseconds, plus how many requests took over 1000 ms. The same
summary is included in `GET /metrics`. Add `?route=POST /predict` to
`GET /metrics/latency` for one route only. Only the newest 1000
samples per route are kept, so tracking cannot grow without bound.
The middleware is plain ASGI, so streaming responses still stream
instead of being buffered. Note the latency of the request asking
for the numbers is not in its own answer yet: it is recorded only
once that response has been sent.

## Docker and deployment

The server ships as a container so it runs the same on any machine
with Docker, with no local Python setup to match. The image is built
from `python:3.12-slim`, installs `requirements.txt`, and copies in
`app/` and `models/` only. A healthcheck polls `GET /health` every
10 seconds, so a container that stopped answering gets flagged
instead of sitting there looking healthy.

```
docker compose up --build
```

That starts the API on http://localhost:8000, where `GET /health`
should answer `{"status": "ok"}`. Or build and run it directly:

```
docker build -t ml-inference-server .
docker run -p 8000:8000 ml-inference-server
```

Compose mounts `./models` read-only into the container, so a model
file can be swapped without rebuilding the image; without compose,
the copy baked into the image is used instead. The container
restarts on its own unless stopped. `.dockerignore` keeps `.git`,
Python caches, and `rust/target/` out of the build context, so
builds stay small.

The image does not include the Rust extension, so preprocessing in
the container runs on the Python backend. That is the normal
fallback from the integration work: the server still starts and
serves, just without the Rust speedup.

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
