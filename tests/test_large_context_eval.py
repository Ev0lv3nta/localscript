from app.core.benchmarks import run_dataset_benchmark
from tests.support_backends import DeterministicTestBackend


def test_large_context_eval_unit_runner_is_explicitly_fake_backend():
    report = run_dataset_benchmark("datasets/large_context_eval.jsonl", backend=DeterministicTestBackend())

    assert report["ok"] is True
    assert report["passed"] == 3
    assert report["backend_type"] == "fake_backend"
