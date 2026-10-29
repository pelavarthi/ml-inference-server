"""Small timing helper for comparing implementations.

Runs a callable repeatedly and reports the average and best time. The
best time is usually the honest number for a short function: it is the
run least disturbed by whatever else the machine was doing.
"""

import time


def benchmark(fn, repeat=5):
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    runs = []
    for _ in range(repeat):
        started = time.perf_counter()
        fn()
        runs.append(time.perf_counter() - started)
    return {
        "runs": runs,
        "mean_seconds": sum(runs) / len(runs),
        "best_seconds": min(runs),
    }


def per_item(result, count):
    """Scale a benchmark result to a per-item time in microseconds."""
    if count < 1:
        raise ValueError("count must be at least 1")
    return {
        "mean_us": result["mean_seconds"] / count * 1_000_000,
        "best_us": result["best_seconds"] / count * 1_000_000,
    }
