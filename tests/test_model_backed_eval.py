from app.core.benchmarks import run_dataset_benchmark
from app.core.public_eval import load_cases
from app.core.resources import materialized_resource
from app.generation.extractor import TaskExtractor
from tests.support_backends import DeterministicTestBackend

DATASET_PATH = "evals/regression/model_backed_eval.jsonl"

def test_model_backed_eval_dataset_stays_off_shortcut_path():
    extractor = TaskExtractor()
    failures = []
    with materialized_resource(DATASET_PATH) as dataset_path:
        for case in load_cases(dataset_path):
            task_spec = extractor.extract(case["prompt"], case.get("context"))
            if task_spec.safety_fallback:
                failures.append({"id": case["id"], "reason": "unexpected_safety_fallback"})

    assert failures == []


def test_model_backed_eval_runner_reports_fake_backend_explicitly():
    report = run_dataset_benchmark(DATASET_PATH, backend=DeterministicTestBackend())

    assert report["backend_type"] == "fake_backend"
    assert report["ok"] is True
    assert report["passed"] == 12
