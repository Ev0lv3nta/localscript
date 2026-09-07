from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias

from pydantic import JsonValue, TypeAdapter, ValidationError

from app.validation.runtime_executor import execute_output
from app.workflow.contracts import (
    OutputContract,
    OutputFormat,
    OutputShape,
    WorkflowRoot,
    WorkflowStatus,
)

EvalCase: TypeAlias = dict[str, JsonValue]
CASES_ADAPTER: TypeAdapter[list[EvalCase]] = TypeAdapter(list[EvalCase])
PUBLIC_V2_SOURCE = "owner_synthetic_public_v2"
_CASE_TYPES = frozenset({"transformation", "clarification", "policy"})
_WORKFLOW_STATUSES = frozenset(status.value for status in WorkflowStatus)


class EvaluationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"


@dataclass(frozen=True)
class IndependentEvaluation:
    status: EvaluationStatus
    errors: tuple[str, ...]
    fixtures_total: int
    fixtures_attempted: int
    fixtures_passed: int


def load_cases(path: str | Path) -> list[EvalCase]:
    return load_cases_bytes(Path(path).read_bytes())


def _reject_nonfinite_json(_value: str) -> None:
    raise ValueError("non_finite_json_number")


def load_cases_bytes(payload: bytes) -> list[EvalCase]:
    decoded: list[object] = []
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("dataset_encoding_invalid") from error
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decoded.append(json.loads(line, parse_constant=_reject_nonfinite_json))
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"dataset_json_invalid::line_{line_number}") from error
    try:
        return CASES_ADAPTER.validate_python(decoded, strict=True)
    except ValidationError as error:
        raise ValueError("dataset_case_json_invalid") from error


def validate_dataset_cases(cases: list[EvalCase]) -> list[EvalCase]:
    """Validate the complete dataset before any model call is allowed."""
    if not cases:
        raise ValueError("dataset_cases_empty")
    errors: list[str] = []
    seen_ids: set[str] = set()
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            errors.append("dataset_case_id_missing")
            display_id = "unknown"
        else:
            display_id = case_id
            if case_id in seen_ids:
                errors.append(f"dataset_case_id_duplicate::{case_id}")
            seen_ids.add(case_id)
        if (
            case.get("source") == PUBLIC_V2_SOURCE
            or "fixtures" in case
            or "output" in case
            or case.get("case_type") in _CASE_TYPES
        ):
            errors.extend(_validate_public_v2_case(case, display_id))
        else:
            errors.extend(_validate_legacy_case(case, display_id))
    if errors:
        raise ValueError("dataset_invalid::" + ",".join(dict.fromkeys(errors)))
    return cases


def output_contract_for_case(case: EvalCase) -> OutputContract | None:
    raw = case.get("output")
    if raw is not None:
        try:
            return OutputContract.model_validate(raw)
        except ValidationError:
            return None
    style = case.get("expected_output_style")
    if style not in {OutputFormat.LUA_BLOCK.value, OutputFormat.JSON_ENVELOPE.value}:
        return None
    if "expected_result" not in case:
        return None
    expected = case["expected_result"]
    shape = OutputShape.OBJECT if style == OutputFormat.JSON_ENVELOPE.value else _shape_of(expected)
    return OutputContract(
        format=OutputFormat(str(style)),
        shape=shape,
        nullable=expected is None,
    )


def evaluate_case(code: str, case: EvalCase) -> list[str]:
    """Run the independent oracle; expected values never enter generation prompts."""
    return list(evaluate_case_detailed(code, case).errors)


def evaluate_case_detailed(code: str, case: EvalCase) -> IndependentEvaluation:
    fixtures = _oracle_fixtures(case)
    if fixtures is None:
        return IndependentEvaluation(
            status=EvaluationStatus.NOT_RUN,
            errors=("dataset_missing_expected_result",),
            fixtures_total=0,
            fixtures_attempted=0,
            fixtures_passed=0,
        )
    contract = output_contract_for_case(case)
    if contract is None:
        return IndependentEvaluation(
            status=EvaluationStatus.NOT_RUN,
            errors=("dataset_output_contract_invalid",),
            fixtures_total=len(fixtures),
            fixtures_attempted=0,
            fixtures_passed=0,
        )

    errors: list[str] = []
    attempted = 0
    passed = 0
    for index, fixture in enumerate(fixtures, start=1):
        fixture_name = str(fixture.get("name") or index)
        execution = execute_output(
            code,
            fixture.get("context"),
            contract.format.value,
            contract.shape.value,
        )
        if execution.degraded:
            return IndependentEvaluation(
                status=EvaluationStatus.NOT_RUN,
                errors=(execution.error_code or "semantic_degraded",),
                fixtures_total=len(fixtures),
                fixtures_attempted=attempted,
                fixtures_passed=passed,
            )
        attempted += 1
        if not execution.ok:
            code_value = execution.error_code or "semantic_runtime_error"
            errors.append(_fixture_error(fixture_name, code_value, len(fixtures)))
            continue
        if not _json_equal(execution.value, fixture["expected_result"]):
            errors.append(_fixture_error(fixture_name, "semantic_mismatch", len(fixtures)))
            continue
        passed += 1
    return IndependentEvaluation(
        status=EvaluationStatus.PASSED if not errors else EvaluationStatus.FAILED,
        errors=tuple(errors),
        fixtures_total=len(fixtures),
        fixtures_attempted=attempted,
        fixtures_passed=passed,
    )


def _fixture_error(name: str, code: str, fixture_count: int) -> str:
    return code if fixture_count == 1 else f"fixture::{name}::{code}"


def _oracle_fixtures(case: EvalCase) -> list[EvalCase] | None:
    raw_fixtures = case.get("fixtures")
    if isinstance(raw_fixtures, list):
        return [fixture for fixture in raw_fixtures if isinstance(fixture, dict)]
    if "expected_result" not in case:
        return None
    return [
        {
            "name": "primary",
            "context": case.get("context"),
            "expected_result": case["expected_result"],
        }
    ]


def _validate_public_v2_case(case: EvalCase, case_id: str) -> list[str]:
    errors = _validate_common_case(case, case_id)
    case_type = case.get("case_type")
    if case.get("source") != PUBLIC_V2_SOURCE:
        errors.append(f"dataset_case_source_invalid::{case_id}")
    if case_type not in _CASE_TYPES:
        errors.append(f"dataset_case_type_invalid::{case_id}")
        return errors
    allowed_fields = {
        "id",
        "prompt",
        "context",
        "output",
        "expected_status",
        "case_type",
        "category",
        "scenario",
        "safety",
        "source",
    }
    if case_type in {"transformation", "clarification"}:
        allowed_fields.add("fixtures")
    if case_type == "clarification":
        allowed_fields.update({"expected_final_status", "clarification_source_roots"})
    if set(case) - allowed_fields:
        errors.append(f"dataset_case_fields_invalid::{case_id}")
    for field in ("category", "scenario"):
        if not isinstance(case.get(field), str) or not str(case.get(field)).strip():
            errors.append(f"dataset_case_{field}_invalid::{case_id}")
    if type(case.get("safety")) is not bool or (
        case_type != "policy" and case.get("safety") is not False
    ):
        errors.append(f"dataset_case_safety_invalid::{case_id}")
    if "expected_result" in case or "examples" in case:
        errors.append(f"dataset_case_expected_leak_field::{case_id}")
    contract = output_contract_for_case(case)
    if contract is None:
        errors.append(f"dataset_output_contract_invalid::{case_id}")

    expected_status = case.get("expected_status")
    expected_final_status = case.get("expected_final_status")
    clarification_answer = case.get("clarification_answer")
    source_roots = case.get("clarification_source_roots")
    fixtures = case.get("fixtures")
    if case_type == "transformation":
        if expected_status != WorkflowStatus.COMPLETED.value:
            errors.append(f"dataset_expected_status_invalid::{case_id}")
        if (
            expected_final_status is not None
            or clarification_answer is not None
            or source_roots is not None
        ):
            errors.append(f"dataset_transformation_transition_invalid::{case_id}")
        errors.extend(_validate_fixtures(fixtures, contract, case.get("context"), case_id))
    elif case_type == "clarification":
        if (
            expected_status != WorkflowStatus.CLARIFICATION_REQUIRED.value
            or expected_final_status != WorkflowStatus.COMPLETED.value
            or not isinstance(source_roots, list)
            or not 1 <= len(source_roots) <= 2
            or len({str(item) for item in source_roots}) != len(source_roots)
            or any(
                not isinstance(item, str) or item not in {root.value for root in WorkflowRoot}
                for item in source_roots
            )
        ):
            errors.append(f"dataset_clarification_transition_invalid::{case_id}")
        errors.extend(_validate_fixtures(fixtures, contract, case.get("context"), case_id))
    else:
        if expected_status != WorkflowStatus.POLICY_REJECTED.value:
            errors.append(f"dataset_policy_status_invalid::{case_id}")
        if case.get("safety") is not True:
            errors.append(f"dataset_policy_safety_flag_missing::{case_id}")
        if (
            fixtures is not None
            or clarification_answer is not None
            or source_roots is not None
            or expected_final_status is not None
        ):
            errors.append(f"dataset_policy_oracle_invalid::{case_id}")
    return errors


def _validate_legacy_case(case: EvalCase, case_id: str) -> list[str]:
    errors = _validate_common_case(case, case_id)
    expected_status = case.get("expected_status", WorkflowStatus.COMPLETED.value)
    final_status = case.get("expected_final_status", expected_status)
    if expected_status not in _WORKFLOW_STATUSES or final_status not in _WORKFLOW_STATUSES:
        errors.append(f"dataset_expected_status_invalid::{case_id}")
    if final_status == WorkflowStatus.COMPLETED.value:
        if "expected_result" not in case:
            errors.append(f"dataset_expected_result_missing::{case_id}")
        elif output_contract_for_case(case) is None:
            errors.append(f"dataset_output_contract_invalid::{case_id}")
    if case.get("clarification_answer") is not None and (
        expected_status != WorkflowStatus.CLARIFICATION_REQUIRED.value
        or final_status != WorkflowStatus.COMPLETED.value
    ):
        errors.append(f"dataset_clarification_transition_invalid::{case_id}")
    return errors


def _validate_common_case(case: EvalCase, case_id: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(case.get("prompt"), str) or not str(case.get("prompt")).strip():
        errors.append(f"dataset_prompt_missing::{case_id}")
    if not _valid_workflow_context(case.get("context")):
        errors.append(f"dataset_context_invalid::{case_id}")
    if "expected_code" in case or "reference_code" in case:
        errors.append(f"dataset_reference_code_forbidden::{case_id}")
    return errors


def _validate_fixtures(
    raw_fixtures: JsonValue,
    contract: OutputContract | None,
    primary_context: JsonValue,
    case_id: str,
) -> list[str]:
    if not isinstance(raw_fixtures, list) or not 2 <= len(raw_fixtures) <= 3:
        return [f"dataset_fixture_count_invalid::{case_id}"]
    errors: list[str] = []
    names: set[str] = set()
    contexts: set[str] = set()
    for index, raw_fixture in enumerate(raw_fixtures, start=1):
        if not isinstance(raw_fixture, dict):
            errors.append(f"dataset_fixture_invalid::{case_id}::{index}")
            continue
        if set(raw_fixture) != {"name", "context", "expected_result"}:
            errors.append(f"dataset_fixture_fields_invalid::{case_id}::{index}")
        name = raw_fixture.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            errors.append(f"dataset_fixture_name_invalid::{case_id}::{index}")
        else:
            names.add(name)
        context = raw_fixture.get("context")
        if not _valid_workflow_context(context):
            errors.append(f"dataset_fixture_context_invalid::{case_id}::{index}")
        else:
            fingerprint = json.dumps(context, ensure_ascii=False, sort_keys=True)
            if fingerprint in contexts:
                errors.append(f"dataset_fixture_context_duplicate::{case_id}::{index}")
            contexts.add(fingerprint)
        if "expected_result" not in raw_fixture:
            errors.append(f"dataset_fixture_expected_missing::{case_id}::{index}")
        elif contract is not None and not _expected_matches_contract(
            raw_fixture["expected_result"], contract
        ):
            errors.append(f"dataset_fixture_expected_shape_invalid::{case_id}::{index}")
    first = raw_fixtures[0]
    if isinstance(first, dict) and primary_context != first.get("context"):
        errors.append(f"dataset_primary_context_mismatch::{case_id}")
    return errors


def _valid_workflow_context(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    workflow = value.get("wf")
    return (
        isinstance(workflow, dict)
        and any(root in workflow for root in ("vars", "initVariables"))
        and _runtime_json_supported(value)
    )


def _shape_of(value: object) -> OutputShape:
    if isinstance(value, list):
        return OutputShape.ARRAY
    if isinstance(value, dict):
        return OutputShape.OBJECT
    return OutputShape.SCALAR


def _expected_matches_contract(value: object, contract: OutputContract) -> bool:
    if value is None:
        return contract.shape is OutputShape.SCALAR and contract.nullable
    return _shape_of(value) is contract.shape and _runtime_json_supported(value)


def _runtime_json_supported(value: object, depth: int = 0) -> bool:
    if depth > 16 or value is None:
        return False
    if isinstance(value, bool | str):
        return True
    if isinstance(value, int | float):
        return math.isfinite(float(value)) and abs(value) <= 2**53 - 1
    if isinstance(value, list):
        return all(_runtime_json_supported(item, depth + 1) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _runtime_json_supported(item, depth + 1)
            for key, item in value.items()
        )
    return False


def _json_equal(actual: object, expected: object) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return actual == expected
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _json_equal(left, right) for left, right in zip(actual, expected, strict=True)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _json_equal(actual[key], expected[key]) for key in actual
        )
    return actual == expected


__all__ = [
    "EvalCase",
    "EvaluationStatus",
    "IndependentEvaluation",
    "evaluate_case",
    "evaluate_case_detailed",
    "load_cases",
    "load_cases_bytes",
    "output_contract_for_case",
    "validate_dataset_cases",
]
