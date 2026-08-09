from app.core.benchmarks import (
    _aggregate_metrics,
    run_dataset_benchmark,
    run_repeated_dataset_benchmark,
)
from tests.support_backends import DeterministicTestBackend


def test_dataset_benchmark_emits_case_and_aggregate_metrics():
    report = run_dataset_benchmark(
        "evals/regression/model_backed_eval.jsonl",
        backend=DeterministicTestBackend(),
    )

    assert report["schema_version"] == 2
    assert len(report["case_results"]) == report["total"] == 12
    assert report["dataset_sha256"]
    assert report["metrics"]["verified_completion_rate"] == 1.0
    assert report["metrics"]["invalid_success_rate"] == 0.0
    assert report["metrics"]["backend_calls_total"] > 0
    assert report["metrics"]["model_call_latency_ms"]["p50"] >= 0
    assert report["metrics"]["latency_ms"]["overall_p95"] >= 0
    assert all(
        item["milestone_intervals_ms"]
        for item in report["case_results"]
    )


def test_aggregate_metrics_exposes_invalid_success_and_repair_rescue():
    case_results = [
        {
            "passed": True,
            "status": "completed",
            "strategy": "ollama_chain",
            "repair_rounds": 1,
            "degraded_mode": False,
            "duration_ms": 100.0,
            "backend_calls": 3,
            "model_call_durations_ms": [20.0, 30.0, 40.0],
            "model_duration_ms": 90.0,
            "milestone_intervals_ms": {"candidate_generated": 80.0},
            "errors": [],
        },
        {
            "passed": False,
            "status": "completed",
            "strategy": "ollama_chain",
            "repair_rounds": 0,
            "degraded_mode": False,
            "duration_ms": 200.0,
            "backend_calls": 2,
            "model_call_durations_ms": [70.0, 80.0],
            "model_duration_ms": 150.0,
            "milestone_intervals_ms": {"candidate_generated": 170.0},
            "errors": ["semantic_mismatch"],
        },
    ]

    metrics = _aggregate_metrics(case_results)

    assert metrics["verified_completion_rate"] == 0.5
    assert metrics["semantic_pass_rate"] == 0.5
    assert metrics["invalid_success_rate"] == 0.5
    assert metrics["invalid_success_count"] == 1
    assert metrics["repair_rescue_rate"] == 1.0
    assert metrics["model_call_latency_ms"]["p50"] == 40.0
    assert metrics["model_duration_ms_total"] == 240.0
    assert metrics["latency_ms"]["cold_first"] == 100.0
    assert metrics["latency_ms"]["warm_p95"] == 200.0


def test_repeated_benchmark_reports_case_level_stability():
    report = run_repeated_dataset_benchmark(
        "evals/regression/model_backed_eval.jsonl",
        repeats=2,
        backend=DeterministicTestBackend(),
    )

    assert report["repeats"] == 2
    assert report["total_cases"] == 12
    assert report["stable_case_rate"] == 1.0
    assert report["consistent_pass_rate"] == 1.0
    assert report["invalid_success_count"] == 0
    assert report["ok"] is True
