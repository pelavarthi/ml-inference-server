"""Load test: fire concurrent predictions at the server and report.

Hammers POST /predict with a bounded number of concurrent clients and
prints how many requests succeeded, how many hit the rate limit (429),
how many failed outright, the throughput, and the latency percentiles.
The 429 count is a feature, not a failure: with the default limit of
60 requests a minute, a run bigger than the burst shows the limiter
working.

Run against a live server from the repository root:

    python benchmarks/load_test.py --requests 120 --concurrency 10

Or point it elsewhere:

    python benchmarks/load_test.py --url http://localhost:8000 --requests 500
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from app.latency import percentile  # noqa: E402

DEFAULT_URL = "http://localhost:8000"
DEFAULT_MODEL = "tiny-linear"
DEFAULT_FEATURES = [2.0, 4.0, 1.0]
CLIENT_ID = "load-test"


def summarise(results, wall_seconds):
    """Turn (duration_ms, status) pairs into one report dict."""
    durations = sorted(duration for duration, _ in results)
    by_status = {}
    for _, status in results:
        by_status[status] = by_status.get(status, 0) + 1
    total = len(results)
    return {
        "total": total,
        "ok": by_status.get(200, 0),
        "rate_limited": by_status.get(429, 0),
        "errors": total - by_status.get(200, 0) - by_status.get(429, 0),
        "requests_per_second": round(total / wall_seconds, 1) if wall_seconds else 0,
        "mean_ms": round(sum(durations) / total, 3) if total else 0,
        "p50_ms": round(percentile(durations, 50), 3) if total else 0,
        "p95_ms": round(percentile(durations, 95), 3) if total else 0,
        "p99_ms": round(percentile(durations, 99), 3) if total else 0,
    }


async def run_load_test(client, model=DEFAULT_MODEL,
                        features=DEFAULT_FEATURES,
                        total=120, concurrency=10):
    """Run total predictions through client and return the summary."""
    semaphore = asyncio.Semaphore(concurrency)

    async def one():
        async with semaphore:
            started = time.perf_counter()
            try:
                response = await client.post(
                    "/predict",
                    json={"model": model, "features": list(features)},
                    headers={"X-Client-Id": CLIENT_ID},
                )
                status = response.status_code
            except httpx.HTTPError:
                status = 0
            return (time.perf_counter() - started) * 1000, status

    started = time.perf_counter()
    results = await asyncio.gather(*(one() for _ in range(total)))
    return summarise(results, time.perf_counter() - started)


def report(summary):
    print(f"total requests:     {summary['total']}")
    print(f"ok (200):           {summary['ok']}")
    print(f"rate limited (429): {summary['rate_limited']}")
    print(f"errors:             {summary['errors']}")
    print(f"throughput:         {summary['requests_per_second']} req/s")
    print(f"mean latency:       {summary['mean_ms']} ms")
    print(f"p50 latency:        {summary['p50_ms']} ms")
    print(f"p95 latency:        {summary['p95_ms']} ms")
    print(f"p99 latency:        {summary['p99_ms']} ms")


def main():
    parser = argparse.ArgumentParser(description="Load test the inference server.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--requests", type=int, default=120)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()
    print(f"hitting {args.url} with {args.requests} requests, "
          f"{args.concurrency} at a time")
    # trust_env=False: the target is a local server, so no proxy from
    # the environment should ever get in the way of the test traffic.
    # The client is opened and closed in the same event loop, so its
    # connection pool is never touched from a different loop.
    async def run():
        async with httpx.AsyncClient(
            base_url=args.url, timeout=30.0, trust_env=False
        ) as client:
            return await run_load_test(
                client, model=args.model, total=args.requests,
                concurrency=args.concurrency,
            )
    report(asyncio.run(run()))


if __name__ == "__main__":
    main()
