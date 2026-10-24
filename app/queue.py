"""A bounded async queue for predictions.

Callers get a job id back immediately and poll for the result, instead of
holding a request open while a slow model runs. The queue is bounded so a
flood of requests fails fast with 429 instead of growing memory forever.
"""

import asyncio
import uuid


class QueueFullError(Exception):
    pass


class AsyncPredictQueue:
    def __init__(self, registry, maxsize=100):
        self._registry = registry
        self._queue = asyncio.Queue(maxsize=maxsize)
        self._jobs = {}
        self._worker = None

    def ensure_started(self):
        if self._worker is None or self._worker.done():
            self._worker = asyncio.get_running_loop().create_task(self._run())

    async def stop(self):
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    def submit(self, model_name, features):
        job_id = uuid.uuid4().hex[:12]
        job = {"job_id": job_id, "model": model_name, "status": "pending",
               "features": list(features), "prediction": None, "error": None}
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull:
            raise QueueFullError("prediction queue is full")
        self._jobs[job_id] = job
        return job

    def get(self, job_id):
        return self._jobs.get(job_id)

    @property
    def depth(self):
        return self._queue.qsize()

    async def _run(self):
        while True:
            job = await self._queue.get()
            job["status"] = "running"
            try:
                model = self._registry.get(job["model"])
                job["prediction"] = model.predict(job["features"])
                job["status"] = "done"
            except Exception as exc:
                job["error"] = str(exc)
                job["status"] = "failed"
            finally:
                self._queue.task_done()
