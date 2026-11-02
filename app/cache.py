"""A small in-memory cache for predictions.

The same request asked twice should not run the model twice: identical
inputs to the same model version always give the same prediction, so
the first answer can be reused. Entries expire after a TTL so a model
that changes underneath the server is not served forever, and when the
cache is full the least recently used entry is dropped first.
"""

import threading
import time
from collections import OrderedDict


class PredictionCache:
    def __init__(self, max_size=1024, ttl_seconds=300):
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._entries = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _expired(self, stored_at):
        return time.monotonic() - stored_at > self.ttl_seconds

    def get(self, model, version, features, quantization=None):
        # A quantized prediction is a different answer from the
        # full-precision one, so the bit width is part of the key.
        key = (model, version, tuple(features), quantization)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            prediction, stored_at = entry
            if self._expired(stored_at):
                del self._entries[key]
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return prediction

    def put(self, model, version, features, prediction, quantization=None):
        key = (model, version, tuple(features), quantization)
        with self._lock:
            if key in self._entries:
                del self._entries[key]
            self._entries[key] = (prediction, time.monotonic())
            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)
                self._evictions += 1

    def clear(self):
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            return count

    def stats(self):
        with self._lock:
            return {
                "size": len(self._entries),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
            }
