import json

from app.evaluation.ablation import ABLATION_PROFILES, run_ablation_benchmark
from tests.support_backends import DeterministicTestBackend


def test_ablation_compares_layers_with_controlled_candidates(tmp_path):
    dataset = tmp_path / "ablation.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "normalize_launch_email",
                "family": "normalize_email_string",
                "prompt": (
                    "Нормализуй launch variable "
                    "wf.initVariables.userEmail: trim и lower-case."
                ),
                "context": {
                    "wf": {
                        "initVariables": {
                            "userEmail": "  USER@EXAMPLE.COM ",
                        }
                    }
                },
                "expected_output_style": "lua_block",
                "case_type": "public_benchmark",
                "expected_result": "user@example.com",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = run_ablation_benchmark(
        dataset,
        backend=DeterministicTestBackend(),
    )

    assert report["schema_version"] == 1
    assert tuple(report["method"]["profile_order"]) == ABLATION_PROFILES
    assert report["dataset_sha256"]
    assert report["case_count"] == 1
    assert report["backend_type"] == "fake_backend"
    assert report["ok"] is True
    assert set(report["profiles"]) == set(ABLATION_PROFILES)

    one_shot = report["profiles"]["one_shot_writer"]["metrics"]
    planner_writer = report["profiles"]["planner_writer"]["metrics"]
    validated = report["profiles"]["planner_writer_validation"]["metrics"]
    deterministic = report["profiles"]["deterministic_repair"]["metrics"]
    full = report["profiles"]["full_pipeline"]["metrics"]

    assert one_shot["model_calls_total"] == 1
    assert planner_writer["model_calls_total"] == 2
    assert validated["model_calls_total"] == 2
    assert deterministic["model_calls_total"] == 2
    assert full["model_calls_total"] >= 2
    assert full["invalid_success_count"] == 0
    assert "uplift_vs_one_shot_verified_cases" in full


def test_ablation_method_discloses_shared_candidate_control():
    assert ABLATION_PROFILES == (
        "one_shot_writer",
        "planner_writer",
        "planner_writer_validation",
        "deterministic_repair",
        "full_pipeline",
    )
