from collections import Counter

from app.core.public_eval import load_cases
from app.core.resources import materialized_resource


def _load_stress_cases():
    with materialized_resource("evals/regression/stress_eval.jsonl") as dataset_path:
        return load_cases(dataset_path)


def test_stress_eval_dataset_shape():
    cases = _load_stress_cases()
    assert len(cases) == 64

    counts = Counter(case["case_type"] for case in cases)
    assert counts == {
        "canonical": 20,
        "mutation": 16,
        "repair_focused": 12,
        "negative_static_trap": 8,
        "paraphrase_noisy": 8,
    }

    required_fields = {
        "id",
        "family",
        "case_type",
        "prompt",
        "context",
        "expected_output_style",
        "assertions",
        "forbidden_patterns",
        "allowed_assumptions",
        "difficulty",
        "repair_expected",
    }

    ids = set()
    for case in cases:
        assert required_fields.issubset(case.keys())
        assert case["id"] not in ids
        ids.add(case["id"])
        assert isinstance(case["assertions"], list)
        assert isinstance(case["forbidden_patterns"], list)
        assert isinstance(case["allowed_assumptions"], list)
        assert isinstance(case["repair_expected"], bool)


def test_stress_eval_repair_cases_are_flagged():
    cases = _load_stress_cases()

    repair_cases = [case for case in cases if case["case_type"] == "repair_focused"]
    assert len(repair_cases) == 12
    assert all(case["repair_expected"] for case in repair_cases)
