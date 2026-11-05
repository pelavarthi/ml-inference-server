"""Per-client rate limiting with a token bucket.

A client gets a steady allowance (for example 60 requests a minute)
plus a burst it can spend at once. Each request costs one token; the
bucket refills continuously, so a client that pauses briefly gets its
budget back instead of waiting for a whole window to roll over. That
is friendlier than a fixed window, where a client can be blocked for
almost a full minute after one unlucky second.

Limits are tracked per client, keyed on the X-Client-Id header when
one is sent and the client IP otherwise, so one greedy client cannot
spend another client's budget.
"""

import math
import threading
import time

DEFAULT_LIMIT_PER_MINUTE = 60
DEFAULT_BURST = 10
MAX_TRACKED_CLIENTS = 1000
IDLE_PRUNE_SECONDS = 600.0


class RateLimiter:
    def __init__(self, limit_per_minute=DEFAULT_LIMIT_PER_MINUTE,
                 burst=DEFAULT_BURST, clock=None):
        self._rate = limit_per_minute / 60.0  # tokens refilled per second
        self._capacity = burst
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        # key -> [tokens, last_seen]
        self._buckets = {}
        self._limited = 0

    def allow(self, key):
        """Spend one token for key. Returns (allowed, retry_after).

        retry_after is 0 when allowed, otherwise the seconds until the
        next token arrives, so the caller can set a Retry-After header.
        """
        now = self._clock()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = [float(self._capacity), now]
                self._buckets[key] = bucket
            tokens, last_seen = bucket
            tokens = min(
                self._capacity, tokens + (now - last_seen) * self._rate
            )
            if tokens >= 1.0:
                bucket[0] = tokens - 1.0
                bucket[1] = now
                return True, 0.0
            bucket[0] = tokens
            bucket[1] = now
            self._limited += 1
            retry_after = (1.0 - tokens) / self._rate
            self._prune(now)
            return False, retry_after

    def _prune(self, now):
        # Only sweep when the table is big, so a normal request never
        # pays for scanning every bucket.
        if len(self._buckets) <= MAX_TRACKED_CLIENTS:
            return
        stale = [
            key for key, (_, last_seen) in self._buckets.items()
            if now - last_seen > IDLE_PRUNE_SECONDS
        ]
        for key in stale:
            del self._buckets[key]

    def retry_after_seconds(self, retry_after):
        """Whole seconds for a Retry-After header, at least 1."""
        return max(1, math.ceil(retry_after))

    def stats(self):
        with self._lock:
            return {
                "clients_tracked": len(self._buckets),
                "limited": self._limited,
                "limit_per_minute": int(self._rate * 60),
                "burst": self._capacity,
            }

    def reset(self):
        with self._lock:
            self._buckets.clear()
            self._limited = 0
