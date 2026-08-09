from app.core.benchmarks import run_quality_benchmark
from tests.support_backends import DeterministicTestBackend


def test_quality_benchmark_reports_backend_type_and_aggregates(monkeypatch):
    def fake_dataset_report(dataset_path, profile=None, backend=None):
        name = str(dataset_path).split("/")[-1].replace(".jsonl", "")
        totals = {
            "v1": 12,
            "public_gold": 8,
            "stress_eval": 64,
            "showcase_eval": 16,
            "model_backed_eval": 12,
            "composition_eval": 4,
            "regression_eval": 3,
            "multilingual_eval": 4,
            "adversarial_eval": 4,
            "large_context_eval": 3,
        }
        total = totals.get(name, 1)
        return {
            "dataset": name,
            "backend_type": "fake_backend",
            "total": total,
            "passed": total,
            "failed": 0,
            "ok": True,
            "failures": [],
        }

    def fake_rich_report(dataset_path, profile=None, backend=None):
        name = str(dataset_path).split("/")[-1].replace(".jsonl", "")
        totals = {"ambiguity_eval": 12, "clarification_eval": 10}
        total = totals.get(name, 1)
        return {
            "dataset": name,
            "backend_type": "fake_backend",
            "total": total,
            "passed": total,
            "failed": 0,
            "ok": True,
            "failures": [],
        }

    monkeypatch.setattr("app.core.benchmarks.run_dataset_benchmark", fake_dataset_report)
    monkeypatch.setattr("app.core.benchmarks.run_rich_dataset_benchmark", fake_rich_report)

    report = run_quality_benchmark(backend=DeterministicTestBackend(), mode="dev")

    assert report["backend_type"] == "fake_backend"
    assert report["public_v1"]["passed"] == 12
    assert report["public_gold"]["passed"] == 8
    assert report["stress_eval"]["passed"] == 64
    assert report["showcase_eval"]["passed"] == 16
    assert report["model_backed_eval"]["passed"] == 12
    assert report["ambiguity_eval"]["passed"] == 12
    assert report["clarification_eval"]["passed"] == 10
    assert report["adversarial_ok"] is True
    assert report["ok"] is True
