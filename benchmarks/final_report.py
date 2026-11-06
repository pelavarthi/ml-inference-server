"""Final benchmark report: measure the server and write it down.

Runs the load test in-process against the real app (once with the
rate limit opened up, once with the defaults, so the report shows
both raw serving speed and the limiter working), times Python text
preprocessing, and writes docs/BENCHMARK_REPORT.md with the measured
numbers.

Run from the repository root:

    python benchmarks/final_report.py

The numbers come from this machine at this moment: rerun it to
refresh the report after a change, instead of editing the numbers
by hand.
"""

import asyncio
import importlib.util
import os
import platform
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from app.benchmark import benchmark, per_item  # noqa: E402
from app.main import create_app  # noqa: E402
from app.preprocessing import Preprocessor  # noqa: E402
from app.registry import Model, ModelRegistry  # noqa: E402
from app.tokenizer import Tokenizer  # noqa: E402
from benchmarks.load_test import run_load_test  # noqa: E402
from benchmarks.preprocess_bench import (  # noqa: E402
    NUM_TEXTS,
    REPEAT,
    SENTENCES,
    build_corpus,
)

REPORT_PATH = Path(__file__).resolve().parent.parent / "docs" / "BENCHMARK_REPORT.md"

LOAD_REQUESTS = 200
LOAD_CONCURRENCY = 10
LIMITED_REQUESTS = 120
LIMITED_CONCURRENCY = 10


def make_registry():
    registry = ModelRegistry()
    registry.register(Model("tiny-linear", "0.1.0", [0.5, 0.25, -1.0], bias=0.1))
    return registry


def run_scenario(rate_limit_per_minute, rate_limit_burst, total, concurrency):
    """Load test one rate-limit setting and return its summary."""
    app = create_app(
        make_registry(),
        rate_limit_per_minute=rate_limit_per_minute,
        rate_limit_burst=rate_limit_burst,
    )

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await run_load_test(
                client, total=total, concurrency=concurrency
            )

    return asyncio.run(run())


def measure_preprocessing():
    """Time Python batch prep over the 500-text corpus, in us per text."""
    texts = build_corpus()
    tokenizer = Tokenizer()
    tokenizer.build_vocab(texts)
    prep = Preprocessor(tokenizer)
    result = benchmark(lambda: prep.prepare_batch(texts), repeat=REPEAT)
    scaled = per_item(result, len(texts))
    return {
        "texts": len(texts),
        "vocab_size": tokenizer.vocab_size,
        "mean_ms_per_batch": round(result["mean_seconds"] * 1000, 2),
        "best_ms_per_batch": round(result["best_seconds"] * 1000, 2),
        "mean_us_per_text": round(scaled["mean_us"], 2),
        "best_us_per_text": round(scaled["best_us"], 2),
    }


def collect_results():
    """Measure everything the report needs and return one dict."""
    started = time.perf_counter()
    unlimited = run_scenario(
        rate_limit_per_minute=100_000, rate_limit_burst=1000,
        total=LOAD_REQUESTS, concurrency=LOAD_CONCURRENCY,
    )
    limited = run_scenario(
        rate_limit_per_minute=60, rate_limit_burst=10,
        total=LIMITED_REQUESTS, concurrency=LIMITED_CONCURRENCY,
    )
    preprocessing = measure_preprocessing()
    return {
        "date": date.today().isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "rust_available": importlib.util.find_spec("ml_rust") is not None,
        "load_requests": LOAD_REQUESTS,
        "load_concurrency": LOAD_CONCURRENCY,
        "limited_requests": LIMITED_REQUESTS,
        "limited_concurrency": LIMITED_CONCURRENCY,
        "unlimited": unlimited,
        "limited": limited,
        "preprocessing": preprocessing,
        "collect_seconds": round(time.perf_counter() - started, 1),
    }


def render_markdown(results):
    """Turn measured results into the report text."""
    unlimited = results["unlimited"]
    limited = results["limited"]
    prep = results["preprocessing"]
    rust_note = (
        "The Rust extension was built, so the Rust rows come from this run."
        if results["rust_available"]
        else "The Rust extension was not built on this machine, so there "
        "are no Rust rows here; see benchmarks/preprocess_bench.py for "
        "the comparison when it is built."
    )
    lines = [
        "# Benchmark report",
        "",
        f"Measured {results['date']} on {results['platform']}, "
        f"Python {results['python']}, {results['cpus']} CPUs, "
        f"in {results['collect_seconds']} seconds. The model is "
        "tiny-linear, a 3-weight linear model, so these numbers describe "
        "the serving path, not a large model.",
        "",
        "## Serving throughput and latency",
        "",
        f"{results['load_requests']} POST /predict requests, "
        f"{results['load_concurrency']} at a time, with the rate limit "
        "opened up so the numbers show the server, not the limiter:",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| throughput | {unlimited['requests_per_second']} req/s |",
        f"| mean latency | {unlimited['mean_ms']} ms |",
        f"| p50 latency | {unlimited['p50_ms']} ms |",
        f"| p95 latency | {unlimited['p95_ms']} ms |",
        f"| p99 latency | {unlimited['p99_ms']} ms |",
        f"| ok | {unlimited['ok']} / {unlimited['total']} |",
        f"| errors | {unlimited['errors']} |",
        "",
        "## Rate limiting under a burst",
        "",
        f"Same endpoint with the default limit (60 requests a minute, "
        f"burst of 10), {results['limited_requests']} requests, "
        f"{results['limited_concurrency']} at a time:",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| ok (200) | {limited['ok']} |",
        f"| rate limited (429) | {limited['rate_limited']} |",
        f"| errors | {limited['errors']} |",
        f"| throughput | {limited['requests_per_second']} req/s |",
        "",
        "The 429s are the limiter working as designed: a burst bigger "
        "than 10 is answered with 429 and a Retry-After header instead "
        "of queueing without bound.",
        "",
        "## Text preprocessing",
        "",
        f"Preparing a batch of {prep['texts']} texts "
        f"(vocabulary of {prep['vocab_size']} tokens), pure Python:",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| mean | {prep['mean_ms_per_batch']} ms/batch ({prep['mean_us_per_text']} us/text) |",
        f"| best | {prep['best_ms_per_batch']} ms/batch ({prep['best_us_per_text']} us/text) |",
        "",
        rust_note,
        "",
        "## What these numbers do not show",
        "",
        "- The load test runs in-process through an ASGI transport, not "
        "over a real network, so it leaves out TCP and HTTP connection "
        "costs; a live-server run will be slower.",
        "- tiny-linear is a toy model: a real model would add its own "
        "compute time on top of everything measured here.",
        "- This machine is shared, so rerunning the report on another "
        "machine or at another time will give different numbers. "
        "Rerun `python benchmarks/final_report.py` to refresh them.",
        "",
        "## How to reproduce",
        "",
        "```",
        "python benchmarks/final_report.py   # regenerates this file",
        "python benchmarks/load_test.py --requests 120 --concurrency 10",
        "python benchmarks/preprocess_bench.py",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main():
    results = collect_results()
    text = render_markdown(results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {REPORT_PATH}")
    print(f"unlimited: {results['unlimited']['requests_per_second']} req/s, "
          f"p95 {results['unlimited']['p95_ms']} ms")
    print(f"limited: {results['limited']['ok']} ok, "
          f"{results['limited']['rate_limited']} rate limited")
    print(f"preprocessing: {results['preprocessing']['best_us_per_text']} us/text (best)")


if __name__ == "__main__":
    main()
