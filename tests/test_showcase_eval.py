from pathlib import Path

from app.core.public_eval import load_cases


def test_showcase_eval_dataset_declares_live_semantic_expectations():
    dataset_path = Path(__file__).resolve().parents[1] / "datasets" / "showcase_eval.jsonl"
    cases = load_cases(dataset_path)

    assert len(cases) == 16
    for case in cases:
        assert case["case_type"] == "live_semantic"
        assert case["expected_result"] is not None
        assert case["expected_strategy"] == "ollama_chain"
        assert case["forbidden_strategy"] == "safe_fallback"
