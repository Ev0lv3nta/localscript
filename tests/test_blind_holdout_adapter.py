import pytest

from app.evaluation.holdout import adapt_blind_holdout_cases


def holdout_case(**overrides):
    case = {
        "schema_version": 2,
        "id": "case-1",
        "prompt": "Верни значение.",
        "context": {"wf": {"vars": {"value": 1}}},
        "category": "transformation_scalar",
        "safety": False,
        "expected": {"status": "completed", "output_format": "lua_block", "result": 1},
    }
    case.update(overrides)
    return case


def test_completed_case_becomes_an_executable_oracle():
    adapted = adapt_blind_holdout_cases([holdout_case()])[0]

    assert adapted["expected_result"] == 1
    assert adapted["expected_output_style"] == "lua_block"
    assert "expected_status" not in adapted
    assert adapted["safety"] is False


def test_rejected_case_asserts_status_and_never_an_oracle():
    adapted = adapt_blind_holdout_cases(
        [
            holdout_case(
                id="case-2",
                safety=True,
                category="safety_dynamic_loading",
                expected={
                    "status": "validation_failed",
                    "output_format": "lua_block",
                    "result": None,
                },
            )
        ]
    )[0]

    assert adapted["expected_status"] == "validation_failed"
    assert "expected_result" not in adapted
    assert adapted["safety"] is True


def test_cases_in_the_repository_format_pass_through_untouched():
    live = [{"id": "live-1", "prompt": "x", "expected_result": 1}]

    assert adapt_blind_holdout_cases(live) is live


def test_unknown_output_format_fails_closed():
    with pytest.raises(ValueError, match="holdout_case_output_format_invalid"):
        adapt_blind_holdout_cases(
            [holdout_case(expected={"status": "completed", "output_format": "yaml", "result": 1})]
        )
