"""Streaming predictions, per-request timeouts, and backpressure.

A normal prediction waits for the whole result before answering. A
streaming prediction sends partial results as the model works through
the features in chunks, so the client sees the first numbers immediately
and can stop early if it already has enough.

Two guards keep a slow stream from hurting the server. A timeout ends
the stream with an error event once the deadline passes, and a
concurrency limiter caps how many streams run at once: when every slot
is busy, a new stream is rejected with 429 instead of joining an
unbounded queue.
"""

import asyncio
import time


class ConcurrencyLimiter:
    """Counts in-flight streams and fails fast once the cap is hit.

    The count is only changed in code that runs without awaiting in
    between, so on one event loop the check-and-increment cannot race.
    """

    def __init__(self, max_concurrent):
        self._max = max_concurrent
        self._in_flight = 0
        self._rejected = 0

    def try_acquire(self):
        if self._in_flight >= self._max:
            self._rejected += 1
            return False
        self._in_flight += 1
        return True

    def release(self):
        self._in_flight = max(0, self._in_flight - 1)

    @property
    def in_flight(self):
        return self._in_flight

    @property
    def rejected(self):
        return self._rejected


async def stream_prediction(model, features, chunk_size=4, chunk_delay=0.0,
                            timeout_seconds=30.0):
    """Yield partial predictions chunk by chunk, then the final result.

    Each chunk event carries the running weighted sum so far; the final
    event adds the bias, matching what model.predict returns. If the
    deadline passes mid-stream, the last event is an error instead of a
    prediction, so the client can tell the stream ended early.
    """
    features = list(features)
    deadline = time.monotonic() + timeout_seconds
    partial = 0.0
    for start in range(0, len(features), chunk_size):
        if time.monotonic() > deadline:
            yield {"error": "stream timed out", "done": True}
            return
        if chunk_delay:
            await asyncio.sleep(chunk_delay)
            if time.monotonic() > deadline:
                yield {"error": "stream timed out", "done": True}
                return
        else:
            await asyncio.sleep(0)
        chunk = features[start:start + chunk_size]
        weights = model.weights[start:start + chunk_size]
        partial += sum(w * x for w, x in zip(weights, chunk))
        yield {"chunk": start // chunk_size, "partial": partial, "done": False}
    yield {"prediction": partial + model.bias, "done": True}
