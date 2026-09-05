from app.core.benchmarks import quality_gate_failures
from app.core.public_eval import evaluate_case, load_cases
from app.core.resources import materialized_resource
from app.evaluation.integrity import (
    _find_cross_corpus_overlaps,
    run_integrity_check,
)
from app.evaluation.manifest import (
    dataset_specs,
    load_evaluation_manifest,
    stability_plan,
)


def test_live_corpus_covers_every_declared_scenario():
    report = run_integrity_check()

    assert report["ok"] is True
    assert report["errors"] == []
    assert report["overlaps"] == []
    assert report["private_holdout"] is None
    assert len(report["datasets"]) == 1
    dataset = report["datasets"][0]
    assert dataset["name"] == "live_v1"
    assert dataset["corpus"] == "live"
    assert dataset["case_count"] == 6

    with materialized_resource(dataset["path"]) as path:
        scenarios = {case["scenario"] for case in load_cases(path)}
    assert scenarios == {
        "scalar_transform",
        "filter_projection",
        "aggregation",
        "nested_object",
        "json_envelope",
        "clarification",
    }


def test_private_holdout_manifest_exposes_identity_but_not_content_path():
    holdouts = load_evaluation_manifest()["private_holdouts"]

    assert holdouts == [
        {
            "name": "holdout_v2",
            "external": True,
            "case_count": 8,
            "safety_case_count": 2,
            "sha256": "5aed110d22971d236bf99f750766925799bb45e07dee7b6cf86dafd4a37770b3",
            "gate": "release_only",
            "claim_scope": "synthetic_blind",
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


def test_manifest_declares_one_live_corpus_and_a_narrow_stability_plan():
    specs = dataset_specs()

    assert [spec.path for spec in specs] == ["evals/live/v1.jsonl"]

    dataset, case_ids, repeats = stability_plan()

    assert dataset == "live_v1"
    assert repeats == 2
    with materialized_resource(specs[0].path) as path:
        known = {case["id"] for case in load_cases(path)}
    assert set(case_ids) <= known
    assert len(case_ids) == 3


def test_private_holdout_identity_mismatch_fails_closed(tmp_path):
    altered = tmp_path / "holdout-v2.jsonl"
    altered.write_text(
        '{"id":"changed","family":"generic_lua","prompt":"changed"}\n',
        encoding="utf-8",
    )

    report = run_integrity_check(private_holdout_path=altered)

    assert report["ok"] is False
    assert "private_holdout_identity_mismatch" in report["errors"]


def test_live_case_always_executes_its_explicit_semantic_oracle(monkeypatch):
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
        "case_type": "live",
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


def test_live_threshold_is_declared_in_the_manifest():
    """Планка живого корпуса объявлена данными, а не выведена из результата прогона.

    Порог 5 из 6 — честная оценка локальной модели: шестой кейс каждый прогон разный, то есть
    за ним нет систематического дефекта. Менять планку можно только через ревью манифеста.
    """
    spec = dataset_specs()[0]

    assert spec.min_verified == 5
    assert spec.evidence_dict()["min_verified"] == 5


def test_gate_rejects_a_run_below_the_declared_threshold():
    manifest = [spec.evidence_dict() for spec in dataset_specs()]
    ok = {
        "eval_manifest": manifest,
        "live_v1": {"passed": 5, "metrics": {"invalid_success_count": 0}},
    }
    low = {
        "eval_manifest": manifest,
        "live_v1": {"passed": 4, "metrics": {"invalid_success_count": 0}},
    }
    invalid = {
        "eval_manifest": manifest,
        "live_v1": {"passed": 6, "metrics": {"invalid_success_count": 1}},
    }

    assert quality_gate_failures(ok) == []
    assert "live_v1_below_min_verified" in quality_gate_failures(low)
    assert "live_v1_invalid_success_detected" in quality_gate_failures(invalid)
