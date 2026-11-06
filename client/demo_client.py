"""Demo client: exercise every endpoint once and print what happened.

Walks a running server through health, models, a normal prediction,
a quantized prediction, a batched prediction, an async job, a
streaming prediction, preprocessing, and the latency summary, so one
command shows the whole server working end to end.

Run against a live server from the repository root:

    python client/demo_client.py

Or point it elsewhere:

    python client/demo_client.py --url http://localhost:8000 --model tiny-linear

The client only needs httpx. Any object with get/post/stream methods
works, so tests pass a fastapi TestClient instead of a live server.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

DEFAULT_URL = "http://localhost:8000"
DEFAULT_MODEL = "tiny-linear"
DEFAULT_FEATURES = [2.0, 4.0, 1.0]
DEFAULT_TEXTS = [
    "Hello, world! How are you doing today?",
    "Machine learning models need clean, well prepared input data.",
]
CLIENT_ID = "demo-client"
POLL_ATTEMPTS = 50
POLL_INTERVAL = 0.1


def step(name, ok, detail=""):
    """One demo step: its name, whether it worked, and a short result."""
    return {"name": name, "ok": bool(ok), "detail": str(detail)}


def run_demo(client, model=DEFAULT_MODEL, features=DEFAULT_FEATURES,
             texts=DEFAULT_TEXTS):
    """Run every endpoint through client and return the step results.

    A failed step records ok=False instead of raising, so the demo
    reports which endpoint broke rather than stopping at the first one.
    """
    steps = []
    headers = {"X-Client-Id": CLIENT_ID}
    body = {"model": model, "features": list(features)}

    try:
        response = client.get("/health")
        ok = response.status_code == 200 and response.json().get("status") == "ok"
        steps.append(step("health", ok, response.json() if ok else response.text))
    except httpx.HTTPError as exc:
        steps.append(step("health", False, exc))

    try:
        response = client.get("/models")
        names = [item["name"] for item in response.json().get("models", [])]
        steps.append(step("models", response.status_code == 200, ", ".join(names)))
    except httpx.HTTPError as exc:
        steps.append(step("models", False, exc))

    try:
        response = client.post("/predict", json=body, headers=headers)
        data = response.json()
        ok = response.status_code == 200
        detail = f"prediction {data.get('prediction')}" if ok else data
        steps.append(step("predict", ok, detail))
    except httpx.HTTPError as exc:
        steps.append(step("predict", False, exc))

    try:
        response = client.post(
            "/predict", json={**body, "quantization": 8}, headers=headers
        )
        data = response.json()
        ok = response.status_code == 200
        detail = f"8-bit prediction {data.get('prediction')}" if ok else data
        steps.append(step("predict quantized", ok, detail))
    except httpx.HTTPError as exc:
        steps.append(step("predict quantized", False, exc))

    try:
        response = client.post("/predict/batched", json=body, headers=headers)
        data = response.json()
        ok = response.status_code == 200
        detail = f"prediction {data.get('prediction')}" if ok else data
        steps.append(step("predict batched", ok, detail))
    except httpx.HTTPError as exc:
        steps.append(step("predict batched", False, exc))

    try:
        response = client.post("/predict/async", json=body, headers=headers)
        if response.status_code != 202:
            steps.append(step("predict async", False, response.text))
        else:
            job_id = response.json()["job_id"]
            prediction = None
            for _ in range(POLL_ATTEMPTS):
                job = client.get(f"/predict/jobs/{job_id}").json()
                if job["status"] == "done":
                    prediction = job["prediction"]
                    break
                if job["status"] == "failed":
                    break
                time.sleep(POLL_INTERVAL)
            ok = prediction is not None
            detail = f"job {job_id}, prediction {prediction}" if ok else f"job {job_id} never finished"
            steps.append(step("predict async", ok, detail))
    except httpx.HTTPError as exc:
        steps.append(step("predict async", False, exc))

    try:
        events = []
        with client.stream(
            "POST", "/predict/stream", json=body, headers=headers
        ) as response:
            if response.status_code != 200:
                steps.append(step("predict stream", False, response.read().decode()))
            else:
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        events.append(json.loads(line[6:]))
                final = next((e for e in events if e.get("done")), None)
                ok = final is not None and "prediction" in final
                detail = (
                    f"{len(events)} events, prediction {final['prediction']}"
                    if ok else f"events: {events}"
                )
                steps.append(step("predict stream", ok, detail))
    except httpx.HTTPError as exc:
        steps.append(step("predict stream", False, exc))

    try:
        response = client.post(
            "/preprocess", json={"texts": list(texts)}, headers=headers
        )
        data = response.json()
        ok = response.status_code == 200
        detail = f"backend {data.get('backend')}, {len(data.get('ids', []))} texts" if ok else data
        steps.append(step("preprocess", ok, detail))
    except httpx.HTTPError as exc:
        steps.append(step("preprocess", False, exc))

    try:
        response = client.get("/metrics/latency", params={"route": "POST /predict"})
        data = response.json()
        latency = data.get("latency", {})
        ok = response.status_code == 200
        detail = (
            f"p50 {latency.get('p50_ms')} ms, p95 {latency.get('p95_ms')} ms "
            f"over {latency.get('count')} requests"
            if ok else data
        )
        steps.append(step("latency", ok, detail))
    except httpx.HTTPError as exc:
        steps.append(step("latency", False, exc))

    return steps


def print_demo(steps):
    for item in steps:
        mark = "ok  " if item["ok"] else "FAIL"
        print(f"[{mark}] {item['name']}: {item['detail']}")
    failed = sum(1 for item in steps if not item["ok"])
    print(f"{len(steps) - failed}/{len(steps)} steps passed")


def main():
    parser = argparse.ArgumentParser(description="Demo the inference server.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(f"demo against {args.url}, model {args.model}")
    # trust_env=False: the target is a local server, so no proxy from
    # the environment should ever get in the way of the demo traffic.
    with httpx.Client(base_url=args.url, timeout=30.0, trust_env=False) as client:
        steps = run_demo(client, model=args.model)
    print_demo(steps)
    return 0 if all(item["ok"] for item in steps) else 1


if __name__ == "__main__":
    raise SystemExit(main())
