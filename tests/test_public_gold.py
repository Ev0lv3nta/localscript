from pathlib import Path

from app.core.public_eval import load_cases
from app.generation.extractor import TaskExtractor


def test_public_gold_dataset_declares_live_semantic_expectations():
    dataset_path = Path(__file__).resolve().parents[1] / "datasets" / "public_gold.jsonl"
    cases = load_cases(dataset_path)

    assert len(cases) == 8
    for case in cases:
        assert case["id"].startswith("synthetic_")
        assert case["source"] == "synthetic_v1"
        assert case["case_type"] == "live_semantic"
        assert case["expected_result"] is not None
        assert case["expected_strategy"] == "ollama_chain"
        assert case["forbidden_strategy"] == "safe_fallback"
        assert "expected_code" not in case


def test_public_gold_synthetic_cases_match_declared_families():
    dataset_path = Path(__file__).resolve().parents[1] / "datasets" / "public_gold.jsonl"
    extractor = TaskExtractor()

    for case in load_cases(dataset_path):
        task_spec = extractor.extract(case["prompt"], case.get("context"))

        assert task_spec.family == case["family"], case["id"]
