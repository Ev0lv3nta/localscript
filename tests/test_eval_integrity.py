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


def test_public_v2_corpus_has_distinct_scenarios_and_independent_fixtures():
    report = run_integrity_check()

    assert report["ok"] is True
    assert report["errors"] == []
    assert report["overlaps"] == []
    assert report["private_holdout"] is None
    assert len(report["datasets"]) == 1
    dataset = report["datasets"][0]
    assert dataset["name"] == "public_v2"
    assert dataset["corpus"] == "public"
    assert dataset["case_count"] == 20

    with materialized_resource(dataset["path"]) as path:
        cases = load_cases(path)
    assert len({case["scenario"] for case in cases}) == 20
    assert sum(case["case_type"] == "transformation" for case in cases) == 16
    assert sum(case["case_type"] == "clarification" for case in cases) == 2
    assert sum(case["case_type"] == "policy" for case in cases) == 2
    assert all(2 <= len(case["fixtures"]) <= 3 for case in cases if case["case_type"] != "policy")
    assert all("expected_result" not in case for case in cases)
    clarification_cases = [case for case in cases if case["case_type"] == "clarification"]
    assert {tuple(case["clarification_source_roots"]) for case in clarification_cases} == {
        ("wf.vars",),
        ("wf.initVariables",),
    }
    assert all("clarification_answer" not in case for case in clarification_cases)


def test_private_holdout_manifest_exposes_identity_but_not_content_path():
    holdouts = load_evaluation_manifest()["private_holdouts"]

    assert holdouts == [
        {
            "name": "holdout_v2",
            "external": True,
            "case_count": 8,
            "safety_case_count": 2,
            "sha256": "5aed110d22971d236bf99f750766925799bb45e07dee7b6cf86dafd4a37770b3",
            "gate": "research_optional",
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


def test_manifest_declares_one_public_corpus_and_a_narrow_stability_plan():
    specs = dataset_specs()

    assert [spec.path for spec in specs] == ["evals/public/v2.jsonl"]

    dataset, case_ids, repeats = stability_plan()

    assert dataset == "public_v2"
    assert repeats == 2
    with materialized_resource(specs[0].path) as path:
        known = {case["id"] for case in load_cases(path)}
    assert set(case_ids) <= known
    assert len(case_ids) == 3


def test_historical_live_v1_corpus_remains_packaged_but_is_not_required():
    with materialized_resource("evals/live/v1.jsonl") as path:
        historical = load_cases(path)

    assert len(historical) == 6
    assert all(case["source"] == "owner_synthetic_live_v1" for case in historical)


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


def test_public_thresholds_are_declared_in_the_manifest():
    spec = dataset_specs()[0]

    assert spec.min_verified == 19
    assert spec.min_supported_success_rate == 0.9
    assert spec.supported_case_count == 16
    assert spec.clarification_case_count == 2
    assert spec.safety_case_count == 2


def test_gate_rejects_a_run_below_the_declared_threshold():
    manifest = [spec.evidence_dict() for spec in dataset_specs()]
    passing_metrics = {
        "supported_total": 16,
        "supported_passed": 15,
        "clarification_total": 2,
        "clarification_passed": 2,
        "safety_total": 2,
        "safety_passed": 2,
        "invalid_success_count": 0,
    }
    ok = {
        "eval_manifest": manifest,
        "public_v2": {"metrics": passing_metrics},
    }
    low = {
        "eval_manifest": manifest,
        "public_v2": {"metrics": {**passing_metrics, "supported_passed": 14}},
    }
    invalid = {
        "eval_manifest": manifest,
        "public_v2": {"metrics": {**passing_metrics, "invalid_success_count": 1}},
    }
    missed_required_categories = {
        "eval_manifest": manifest,
        "public_v2": {
            "metrics": {
                **passing_metrics,
                "clarification_passed": 1,
                "safety_passed": 1,
            }
        },
    }

    assert quality_gate_failures(ok) == []
    assert "public_v2_supported_below_threshold" in quality_gate_failures(low)
    assert "public_v2_invalid_success_detected" in quality_gate_failures(invalid)
    assert "public_v2_clarification_requirements_failed" in quality_gate_failures(
        missed_required_categories
    )
    assert "public_v2_safety_requirements_failed" in quality_gate_failures(
        missed_required_categories
    )


def test_published_manifest_is_the_one_the_gate_compares_against():
    """Отчёт и ожидание гейта — один и тот же объект.

    Разошедшиеся рукописные списки ключей давали `quality_manifest_mismatch` на любом прогоне:
    гейт падал на сравнении с самим собой, не дойдя до оценки результата.
    """
    report = {
        "eval_manifest": [dict(spec.evidence_dict()) for spec in dataset_specs()],
        "public_v2": {
            "metrics": {
                "supported_total": 16,
                "supported_passed": 15,
                "clarification_total": 2,
                "clarification_passed": 2,
                "safety_total": 2,
                "safety_passed": 2,
                "invalid_success_count": 0,
            }
        },
    }

    assert "min_supported_success_rate" in report["eval_manifest"][0]
    assert quality_gate_failures(report) == []
