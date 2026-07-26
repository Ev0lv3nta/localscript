from app.core.benchmarks import run_quality_benchmark
from tests.support_backends import DeterministicTestBackend


def test_quality_benchmark_uses_injected_backend_without_live_ollama(monkeypatch):
    def fail_if_live_backend_requested(*args, **kwargs):
        raise AssertionError("unit benchmark path must not construct a live OllamaBackend")

    def fake_dataset_report(dataset_path, profile=None, backend=None):
        return {
            "dataset": str(dataset_path),
            "backend_type": "fake_backend",
            "total": 1,
            "passed": 1,
            "failed": 0,
            "ok": True,
            "failures": [],
        }

    monkeypatch.setattr("app.core.benchmarks.OllamaBackend", fail_if_live_backend_requested)
    monkeypatch.setattr("app.core.benchmarks.run_dataset_benchmark", fake_dataset_report)
    monkeypatch.setattr("app.core.benchmarks.run_rich_dataset_benchmark", fake_dataset_report)

    report = run_quality_benchmark(backend=DeterministicTestBackend(), mode="dev")

    assert report["backend_type"] == "fake_backend"
    assert report["ok"] is True
