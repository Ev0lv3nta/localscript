import json
from collections.abc import Mapping
from typing import Any

from app.families.base import FamilyDefinition, FamilyFinding


def _augment_existing_code(hints: Mapping[str, Any], _context: Any) -> dict[str, int]:
    number_literal = int(hints.get("number_literal", "5"))
    return {"num": number_literal, "squared": number_literal * number_literal}


def _validate_augment(
    code: str,
    output_style: str,
    _hints: Mapping[str, Any],
) -> tuple[FamilyFinding, ...]:
    findings = []
    if output_style != "json_envelope":
        findings.append(
            FamilyFinding(
                "expected_json_envelope",
                "Augment-existing-code task must return a JSON envelope.",
            )
        )
    if "wf.vars" in code or "wf.initVariables" in code:
        findings.append(
            FamilyFinding(
                "augment_existing_code_forbidden_workflow_state",
                "Augment-existing-code must use the extracted literal and must not read workflow state.",
            )
        )
    try:
        payload = json.loads(code)
    except json.JSONDecodeError:
        findings.append(
            FamilyFinding(
                "augment_existing_code_invalid_json",
                "Augment-existing-code tasks must return a valid JSON envelope.",
            )
        )
        return tuple(findings)

    for key in ("num", "squared"):
        if key not in payload:
            findings.append(
                FamilyFinding(
                    "augment_existing_code_missing_key::{0}".format(key),
                    "Augment-existing-code tasks must preserve `{0}` in the JSON envelope.".format(key),
                )
            )
    for key in payload:
        if key not in {"num", "squared"}:
            findings.append(
                FamilyFinding(
                    "augment_existing_code_unexpected_key::{0}".format(key),
                    "Augment-existing-code tasks must not add unexpected envelope key `{0}`.".format(key),
                )
            )
    return tuple(findings)


RECORD_FAMILIES = (
    FamilyDefinition(
        "augment_existing_code",
        preferred_return_shape="json_envelope",
        expected_result_builder=_augment_existing_code,
        structural_validator=_validate_augment,
    ),
)
