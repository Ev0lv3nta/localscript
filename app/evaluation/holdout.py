"""Adapter for the frozen synthetic-blind holdout.

The holdout is authored against the public product specification, not against this repository, so
it uses its own small schema. Converting it here keeps that independence: the benchmark runner sees
ordinary cases and never learns that a dataset is the holdout.
"""

from __future__ import annotations

from typing import Any

HOLDOUT_SCHEMA_VERSION = 2
_OUTPUT_FORMATS = frozenset({"lua_block", "json_envelope"})


def is_blind_holdout_case(case: object) -> bool:
    return (
        isinstance(case, dict)
        and case.get("schema_version") == HOLDOUT_SCHEMA_VERSION
        and isinstance(case.get("expected"), dict)
    )


def adapt_blind_holdout_case(case: dict[str, Any]) -> dict[str, Any]:
    expected = case["expected"]
    status = expected.get("status")
    output_format = expected.get("output_format")
    if not isinstance(status, str) or not status:
        raise ValueError("holdout_case_status_invalid::{}".format(case.get("id")))
    if output_format is not None and output_format not in _OUTPUT_FORMATS:
        raise ValueError("holdout_case_output_format_invalid::{}".format(case.get("id")))

    adapted: dict[str, Any] = {
        "id": case.get("id"),
        "prompt": case.get("prompt"),
        "context": case.get("context"),
        "case_type": "private_holdout",
        "category": case.get("category"),
        "safety": bool(case.get("safety", False)),
    }
    if output_format is not None:
        adapted["expected_output_style"] = output_format
    if status == "completed":
        adapted["expected_result"] = expected.get("result")
    else:
        adapted["expected_status"] = status
    return adapted


def adapt_blind_holdout_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not cases or not all(is_blind_holdout_case(case) for case in cases):
        return cases
    return [adapt_blind_holdout_case(case) for case in cases]
