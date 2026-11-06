# Benchmark report

Measured 2026-11-06 on Linux-7.0.0-43-generic-x86_64-with-glibc2.39, Python 3.12.3, 4 CPUs, in 0.3 seconds. The model is tiny-linear, a 3-weight linear model, so these numbers describe the serving path, not a large model.

## Serving throughput and latency

200 POST /predict requests, 10 at a time, with the rate limit opened up so the numbers show the server, not the limiter:

| metric | value |
| --- | --- |
| throughput | 820.5 req/s |
| mean latency | 8.614 ms |
| p50 latency | 7.761 ms |
| p95 latency | 14.037 ms |
| p99 latency | 34.784 ms |
| ok | 200 / 200 |
| errors | 0 |

## Rate limiting under a burst

Same endpoint with the default limit (60 requests a minute, burst of 10), 120 requests, 10 at a time:

| metric | value |
| --- | --- |
| ok (200) | 10 |
| rate limited (429) | 110 |
| errors | 0 |
| throughput | 2273.5 req/s |

The 429s are the limiter working as designed: a burst bigger than 10 is answered with 429 and a Retry-After header instead of queueing without bound.

## Text preprocessing

Preparing a batch of 500 texts (vocabulary of 56 tokens), pure Python:

| metric | value |
| --- | --- |
| mean | 3.56 ms/batch (7.12 us/text) |
| best | 3.42 ms/batch (6.84 us/text) |

The Rust extension was not built on this machine, so there are no Rust rows here; see benchmarks/preprocess_bench.py for the comparison when it is built.

## What these numbers do not show

- The load test runs in-process through an ASGI transport, not over a real network, so it leaves out TCP and HTTP connection costs; a live-server run will be slower.
- tiny-linear is a toy model: a real model would add its own compute time on top of everything measured here.
- This machine is shared, so rerunning the report on another machine or at another time will give different numbers. Rerun `python benchmarks/final_report.py` to refresh them.

## How to reproduce

```
python benchmarks/final_report.py   # regenerates this file
python benchmarks/load_test.py --requests 120 --concurrency 10
python benchmarks/preprocess_bench.py
```
