"""Dynamic batching: group requests that arrive close together into one call.

A model call has fixed overhead, so running ten requests one at a time
costs ten overheads. The batcher waits a short window for more requests
to arrive, then hands the whole group to the model in a single call and
returns each result to the request it came from.

Two knobs bound the wait: the batch runs as soon as it reaches
max_batch_size, or when max_wait_seconds passes after the first request,
so a lone request never waits long for company that never comes.
"""

import asyncio
import time


class Batcher:
    def __init__(self, predict_batch, max_batch_size=8, max_wait_seconds=0.01):
        self._predict_batch = predict_batch
        self._max_batch_size = max_batch_size
        self._max_wait_seconds = max_wait_seconds
        self._queue = None
        self._loop = None
        self._worker = None
        self._batches = 0
        self._requests = 0
        self._largest_batch = 0

    def ensure_started(self):
        # An asyncio.Queue is bound to the loop that first uses it, so a
        # batcher that outlives its loop (a server restart, a fresh test
        # client loop) needs a fresh queue for the new loop.
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._loop = loop
            self._queue = asyncio.Queue()
            self._worker = None
        if self._worker is None or self._worker.done():
            self._worker = loop.create_task(self._run())

    async def stop(self):
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    async def submit(self, features):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._queue.put((list(features), future))
        return await future

    @property
    def stats(self):
        return {
            "batches": self._batches,
            "requests": self._requests,
            "largest_batch": self._largest_batch,
        }

    async def _run(self):
        while True:
            first_features, first_future = await self._queue.get()
            batch = [(first_features, first_future)]
            deadline = time.monotonic() + self._max_wait_seconds
            while len(batch) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), remaining)
                except asyncio.TimeoutError:
                    break
                batch.append(item)
            self._batches += 1
            self._requests += len(batch)
            self._largest_batch = max(self._largest_batch, len(batch))
            try:
                results = self._predict_batch([features for features, _ in batch])
            except Exception as exc:
                for _, future in batch:
                    if not future.done():
                        future.set_exception(exc)
                continue
            for (_, future), result in zip(batch, results):
                if not future.done():
                    future.set_result(result)
