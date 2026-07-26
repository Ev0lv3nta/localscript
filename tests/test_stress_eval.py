from collections import Counter
from pathlib import Path

from app.core.public_eval import load_cases


def test_stress_eval_dataset_shape():
    dataset_path = Path(__file__).resolve().parents[1] / "datasets" / "stress_eval.jsonl"
    cases = load_cases(dataset_path)

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
    dataset_path = Path(__file__).resolve().parents[1] / "datasets" / "stress_eval.jsonl"
    cases = load_cases(dataset_path)

    repair_cases = [case for case in cases if case["case_type"] == "repair_focused"]
    assert len(repair_cases) == 12
    assert all(case["repair_expected"] for case in repair_cases)
