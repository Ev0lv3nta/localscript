from app.core.public_eval import load_cases
from app.core.resources import materialized_resource


def test_showcase_eval_dataset_declares_live_semantic_expectations():
    with materialized_resource("datasets/showcase_eval.jsonl") as dataset_path:
        cases = load_cases(dataset_path)

    assert len(cases) == 16
    for case in cases:
        assert case["case_type"] == "live_semantic"
        assert case["expected_result"] is not None
        assert case["expected_strategy"] == "ollama_chain"
        assert case["forbidden_strategy"] == "safe_fallback"
