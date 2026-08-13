from app.core.public_eval import evaluate_case
from app.evaluation.integrity import (
    _find_cross_corpus_overlaps,
    run_integrity_check,
)
from app.evaluation.manifest import dataset_specs, load_evaluation_manifest


def test_eval_integrity_separates_public_and_regression_corpora():
    report = run_integrity_check()

    assert report["ok"] is True
    assert report["errors"] == []
    assert report["overlaps"] == []
    assert report["private_holdout"] is None
    assert sum(
        item["case_count"]
        for item in report["datasets"]
        if item["corpus"] == "regression"
    ) == 140
    public = [
        item for item in report["datasets"] if item["corpus"] == "public_benchmark"
    ]
    assert len(public) == 1
    assert public[0]["name"] == "public_v1"
    assert public[0]["case_count"] == 12


def test_private_holdout_manifest_exposes_identity_but_not_content_path():
    holdouts = load_evaluation_manifest()["private_holdouts"]

    assert holdouts == [
        {
            "name": "holdout_v1",
            "external": True,
            "case_count": 8,
            "sha256": "690ce204ea542ca1e91e9049f4d3a9ba6d404de497d39121f6d42a0f92c69419",
            "gate": "release_only",
            "claim_scope": "private_holdout",
        }
    ]
    assert "path" not in holdouts[0]


def test_overlap_checker_detects_normalized_and_fuzzy_leakage():
    protected = [
        {"source": "public", "id": "p1", "prompt": "Верни последний элемент массива"},
        {"source": "public", "id": "p2", "prompt": "Нормализуй адрес электронной почты"},
    ]
    comparison = [
        {"source": "regression", "id": "r1", "prompt": "  ВЕРНИ последний элемент массива! "},
        {"source": "kb", "id": "k1", "prompt": "Нормализуй адрес электронной почты сейчас"},
    ]

    findings = _find_cross_corpus_overlaps(protected, comparison)

    assert {finding["protected_id"] for finding in findings} == {"p1", "p2"}
    assert {finding["kind"] for finding in findings} == {"normalized_exact", "fuzzy"}


def test_manifest_has_one_registered_path_per_dataset():
    specs = dataset_specs()

    assert len(specs) == 12
    assert len({spec.name for spec in specs}) == len(specs)
    assert len({spec.path for spec in specs}) == len(specs)
    assert all(spec.path.startswith("evals/") for spec in specs)


def test_private_holdout_identity_mismatch_fails_closed(tmp_path):
    altered = tmp_path / "holdout-v1.jsonl"
    altered.write_text(
        '{"id":"changed","family":"generic_lua","prompt":"changed"}\n',
        encoding="utf-8",
    )

    report = run_integrity_check(private_holdout_path=altered)

    assert report["ok"] is False
    assert "private_holdout_identity_mismatch" in report["errors"]


def test_public_benchmark_always_executes_explicit_semantic_oracle(monkeypatch):
    class Execution:
        ok = True
        degraded = False
        value = 999
        error_code = ""

    monkeypatch.setattr(
        "app.core.public_eval.execute_output",
        lambda *_args, **_kwargs: Execution(),
    )
    case = {
        "id": "public_wrong_code",
        "prompt": "Верни увеличенный счётчик.",
        "context": {"wf": {"vars": {"counter": 4}}},
        "expected_output_style": "lua_block",
        "case_type": "public_benchmark",
        "expected_result": 5,
        "forbidden_patterns": [],
    }

    failures = evaluate_case("return 999", case)

    assert "semantic_mismatch" in failures


def test_eval_without_explicit_expected_result_does_not_infer_prompt_intent():
    failures = evaluate_case(
        "return 5",
        {
            "id": "no_oracle",
            "prompt": "Return five.",
            "context": {"wf": {"vars": {}}},
        },
    )

    assert failures == ["dataset_missing_expected_result"]
