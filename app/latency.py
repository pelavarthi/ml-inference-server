"""Latency tracking: per-route durations, percentiles, slow requests.

Averages hide the requests users actually notice, so this keeps the
raw durations (bounded, newest 1000 per route) and summarises them as
percentiles instead. p95 is the latency 95 percent of requests beat;
if p95 climbs while the average stays flat, a small slice of requests
got much slower, which is usually the first sign of trouble.
"""

import math
import threading
from collections import deque

DEFAULT_MAX_SAMPLES = 1000
SLOW_THRESHOLD_MS = 1000.0


def percentile(sorted_samples, pct):
    """Nearest-rank percentile of an already sorted list, in ms."""
    if not sorted_samples:
        return None
    rank = math.ceil(pct / 100 * len(sorted_samples))
    return sorted_samples[max(0, min(rank - 1, len(sorted_samples) - 1))]


class LatencyTracker:
    def __init__(self, max_samples=DEFAULT_MAX_SAMPLES, slow_threshold_ms=SLOW_THRESHOLD_MS):
        self._lock = threading.Lock()
        self._max_samples = max_samples
        self._slow_threshold_ms = slow_threshold_ms
        self._samples = {}

    def record(self, route, duration_ms, status=None):
        with self._lock:
            samples = self._samples.setdefault(route, deque(maxlen=self._max_samples))
            samples.append(duration_ms)

    def _summarise(self, samples):
        ordered = sorted(samples)
        slow = sum(1 for value in ordered if value > self._slow_threshold_ms)
        return {
            "count": len(ordered),
            "mean_ms": round(sum(ordered) / len(ordered), 3),
            "min_ms": round(ordered[0], 3),
            "max_ms": round(ordered[-1], 3),
            "p50_ms": round(percentile(ordered, 50), 3),
            "p90_ms": round(percentile(ordered, 90), 3),
            "p95_ms": round(percentile(ordered, 95), 3),
            "p99_ms": round(percentile(ordered, 99), 3),
            "slow_requests": slow,
            "slow_threshold_ms": self._slow_threshold_ms,
        }

    def route_summary(self, route):
        with self._lock:
            samples = self._samples.get(route)
            if not samples:
                return None
            return self._summarise(list(samples))

    def summary(self):
        with self._lock:
            routes = {
                route: self._summarise(list(samples))
                for route, samples in self._samples.items()
            }
        all_samples = []
        for samples in self._samples.values():
            all_samples.extend(samples)
        overall = self._summarise(all_samples) if all_samples else None
        return {"overall": overall, "routes": routes}

    def reset(self):
        with self._lock:
            self._samples.clear()
