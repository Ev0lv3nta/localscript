from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias

from pydantic import JsonValue, TypeAdapter

from app.validation.runtime_executor import execute_output

EvalCase: TypeAlias = dict[str, JsonValue]
CASES_ADAPTER: TypeAdapter[list[EvalCase]] = TypeAdapter(list[EvalCase])


def load_cases(path: str | Path) -> list[EvalCase]:
    return load_cases_bytes(Path(path).read_bytes())


def load_cases_bytes(payload: bytes) -> list[EvalCase]:
    decoded = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    return CASES_ADAPTER.validate_python(decoded, strict=True)


def evaluate_case(code: str, case: EvalCase) -> list[str]:
    """Evaluate only an explicit executable oracle; never infer intent from the prompt."""
    if "expected_result" not in case:
        return ["dataset_missing_expected_result"]
    output_format = case.get("expected_output_style", "lua_block")
    if output_format not in {"lua_block", "json_envelope"}:
        return ["dataset_output_format_invalid"]

    expected = case["expected_result"]
    execution = execute_output(
        code,
        case.get("context"),
        str(output_format),
        "array" if isinstance(expected, list) else None,
    )
    if execution.degraded:
        return [execution.error_code or "semantic_degraded"]
    if not execution.ok:
        return [execution.error_code or "semantic_runtime_error"]
    if not _json_equal(execution.value, case["expected_result"]):
        return ["semantic_mismatch"]
    return []


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
