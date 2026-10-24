"""In-memory counters for the server: requests, predictions, uptime.

Kept deliberately small: a dict of counters behind a lock, no external
dependency, so /metrics works even when nothing else is configured.
"""

import threading
import time


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._requests = {}
        self._predictions = 0
        self._prediction_failures = 0

    def record_request(self, method, path, status):
        key = f"{method} {path} {status}"
        with self._lock:
            self._requests[key] = self._requests.get(key, 0) + 1

    def record_prediction(self, ok):
        with self._lock:
            self._predictions += 1
            if not ok:
                self._prediction_failures += 1

    def snapshot(self):
        with self._lock:
            return {
                "uptime_seconds": round(time.monotonic() - self._started, 3),
                "requests": dict(self._requests),
                "predictions": {
                    "total": self._predictions,
                    "failed": self._prediction_failures,
                },
            }
