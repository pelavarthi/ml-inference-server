import httpx
from fastapi.testclient import TestClient

from app.main import create_app
from app.registry import Model, ModelRegistry
from benchmarks.final_report import collect_results, render_markdown
from client.demo_client import print_demo, run_demo


def make_client(**kwargs):
    registry = ModelRegistry()
    registry.register(Model("tiny-linear", "0.1.0", [0.5, 0.25, -1.0], bias=0.1))
    return TestClient(create_app(registry, **kwargs))


def step_by_name(steps, name):
    return next(item for item in steps if item["name"] == name)


def test_demo_all_steps_pass():
    steps = run_demo(make_client())
    assert len(steps) == 9
    assert all(item["ok"] for item in steps), steps
    assert "1.1" in step_by_name(steps, "predict")["detail"]
    assert "1.1" in step_by_name(steps, "predict batched")["detail"]
    assert "1.1" in step_by_name(steps, "predict async")["detail"]
    assert "1.1" in step_by_name(steps, "predict stream")["detail"]


def test_demo_reports_latency_from_real_requests():
    steps = run_demo(make_client())
    detail = step_by_name(steps, "latency")["detail"]
    assert "p50" in detail and "p95" in detail
    assert "over" in detail  # the count of timed requests is reported


def test_demo_unknown_model_reports_failure_without_raising():
    steps = run_demo(make_client(), model="no-such-model")
    assert step_by_name(steps, "health")["ok"] is True
    assert step_by_name(steps, "models")["ok"] is True
    assert step_by_name(steps, "predict")["ok"] is False
    assert step_by_name(steps, "predict stream")["ok"] is False
    # A failed step never stops the demo: later steps still ran.
    assert step_by_name(steps, "preprocess")["ok"] is True


def test_demo_works_against_live_server():
    """The demo client as main() builds it, against a real uvicorn server."""
    import threading
    import time

    import uvicorn

    registry = ModelRegistry()
    registry.register(Model("tiny-linear", "0.1.0", [0.5, 0.25, -1.0], bias=0.1))
    config = uvicorn.Config(
        create_app(registry), host="127.0.0.1", port=0, log_level="error"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        assert server.started
        port = server.servers[0].sockets[0].getsockname()[1]
        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}", timeout=30.0, trust_env=False
        ) as client:
            steps = run_demo(client)
        assert all(item["ok"] for item in steps), steps
        assert "1.1" in step_by_name(steps, "predict async")["detail"]
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_print_demo_counts_passes(capsys):
    steps = run_demo(make_client())
    print_demo(steps)
    out = capsys.readouterr().out
    assert "9/9 steps passed" in out
    assert "[ok  ] predict: prediction 1.1" in out


def sample_results():
    return {
        "date": "2026-11-06",
        "python": "3.12.3",
        "platform": "Linux-test",
        "cpus": 4,
        "rust_available": False,
        "load_requests": 200,
        "load_concurrency": 10,
        "limited_requests": 120,
        "limited_concurrency": 10,
        "unlimited": {
            "total": 200, "ok": 200, "rate_limited": 0, "errors": 0,
            "requests_per_second": 385.0, "mean_ms": 20.1,
            "p50_ms": 22.0, "p95_ms": 30.0, "p99_ms": 45.0,
        },
        "limited": {
            "total": 120, "ok": 10, "rate_limited": 110, "errors": 0,
            "requests_per_second": 300.0, "mean_ms": 15.0,
            "p50_ms": 14.0, "p95_ms": 25.0, "p99_ms": 40.0,
        },
        "preprocessing": {
            "texts": 500, "vocab_size": 42,
            "mean_ms_per_batch": 12.5, "best_ms_per_batch": 11.0,
            "mean_us_per_text": 25.0, "best_us_per_text": 22.0,
        },
        "collect_seconds": 3.2,
    }


def test_render_markdown_includes_measured_numbers():
    text = render_markdown(sample_results())
    assert "# Benchmark report" in text
    assert "385.0 req/s" in text
    assert "p95 latency | 30.0 ms" in text
    assert "rate limited (429) | 110" in text
    assert "22.0 us/text" in text
    assert "not built on this machine" in text
    assert "TODO" not in text


def test_render_markdown_notes_rust_when_built():
    results = sample_results()
    results["rust_available"] = True
    text = render_markdown(results)
    assert "was built" in text
    assert "not built" not in text


def test_collect_results_measures_real_numbers(monkeypatch):
    import benchmarks.final_report as report

    monkeypatch.setattr(report, "LOAD_REQUESTS", 20)
    monkeypatch.setattr(report, "LIMITED_REQUESTS", 20)
    results = collect_results()
    assert results["unlimited"]["total"] == 20
    assert results["unlimited"]["ok"] == 20
    assert results["unlimited"]["errors"] == 0
    assert results["unlimited"]["p95_ms"] >= results["unlimited"]["p50_ms"]
    assert results["limited"]["ok"] == 10
    assert results["limited"]["rate_limited"] == 10
    assert results["preprocessing"]["texts"] == 500
    assert results["preprocessing"]["best_us_per_text"] > 0
    assert results["rust_available"] is False
