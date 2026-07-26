import json

import pytest

from app.domain.outcomes import (
    Diagnostic,
    DiagnosticSeverity,
    GenerationOutcome,
    GenerationStatus,
    ValidationOutcome,
    ValidationStatus,
)


def _error(code="invalid_code"):
    return Diagnostic(
        code=code,
        message="Generated code did not pass validation.",
        severity=DiagnosticSeverity.ERROR,
        stage="validation",
    )


def test_completed_requires_non_empty_code():
    with pytest.raises(ValueError, match="requires code"):
        GenerationOutcome(
            status=GenerationStatus.COMPLETED,
            validation=ValidationOutcome(status=ValidationStatus.PASSED),
        )


def test_completed_requires_passed_validation():
    with pytest.raises(ValueError, match="requires passed validation"):
        GenerationOutcome(
            status=GenerationStatus.COMPLETED,
            validation=ValidationOutcome(
                status=ValidationStatus.FAILED,
                findings=(_error(),),
            ),
            code="return 1",
        )


def test_completed_rejects_error_diagnostics():
    with pytest.raises(ValueError, match="error diagnostics"):
        GenerationOutcome(
            status=GenerationStatus.COMPLETED,
            validation=ValidationOutcome(status=ValidationStatus.PASSED),
            code="return 1",
            diagnostics=(_error("generation_error"),),
        )


def test_clarification_requires_question_and_no_code():
    validation = ValidationOutcome(status=ValidationStatus.NOT_RUN)

    with pytest.raises(ValueError, match="requires a question"):
        GenerationOutcome(
            status=GenerationStatus.CLARIFICATION_REQUIRED,
            validation=validation,
        )
    with pytest.raises(ValueError, match="must not publish code"):
        GenerationOutcome(
            status=GenerationStatus.CLARIFICATION_REQUIRED,
            validation=validation,
            code="return 1",
            question="Which workflow root should be used?",
        )


@pytest.mark.parametrize(
    "validation_status",
    [ValidationStatus.FAILED, ValidationStatus.INCOMPLETE],
)
def test_validation_failed_accepts_failed_or_incomplete_validation(validation_status):
    findings = (_error(),) if validation_status is ValidationStatus.FAILED else ()

    outcome = GenerationOutcome(
        status=GenerationStatus.VALIDATION_FAILED,
        validation=ValidationOutcome(
            status=validation_status,
            findings=findings,
        ),
    )

    assert outcome.code is None


@pytest.mark.parametrize(
    "generation_status",
    [GenerationStatus.POLICY_REJECTED, GenerationStatus.BACKEND_UNAVAILABLE],
)
def test_non_success_outcomes_cannot_publish_code(generation_status):
    with pytest.raises(ValueError, match="must not publish code"):
        GenerationOutcome(
            status=generation_status,
            validation=ValidationOutcome(status=ValidationStatus.NOT_RUN),
            code="return 1",
        )


def test_failed_validation_requires_error_finding():
    with pytest.raises(ValueError, match="requires an error finding"):
        ValidationOutcome(status=ValidationStatus.FAILED)


def test_raw_or_cross_enum_statuses_are_rejected():
    with pytest.raises(TypeError, match="ValidationStatus"):
        ValidationOutcome(status="passed")
    with pytest.raises(TypeError, match="ValidationStatus"):
        ValidationOutcome(status=GenerationStatus.COMPLETED)
    with pytest.raises(TypeError, match="GenerationStatus"):
        GenerationOutcome(
            status="completed",
            validation=ValidationOutcome(status=ValidationStatus.PASSED),
            code="return 1",
        )


def test_generation_requires_typed_validation_outcome():
    with pytest.raises(TypeError, match="ValidationOutcome"):
        GenerationOutcome(
            status=GenerationStatus.COMPLETED,
            validation=object(),
            code="return 1",
        )


def test_diagnostic_requires_typed_severity():
    with pytest.raises(TypeError, match="DiagnosticSeverity"):
        Diagnostic(
            code="invalid_code",
            message="Generated code did not pass validation.",
            severity="error",
            stage="validation",
        )


def test_caller_owned_finding_lists_cannot_mutate_validation_outcome():
    findings = []
    outcome = ValidationOutcome(
        status=ValidationStatus.PASSED,
        findings=findings,
    )

    findings.append(_error())

    assert outcome.findings == ()


def test_caller_owned_diagnostic_lists_cannot_mutate_generation_outcome():
    diagnostics = []
    outcome = GenerationOutcome(
        status=GenerationStatus.COMPLETED,
        validation=ValidationOutcome(status=ValidationStatus.PASSED),
        code="return 1",
        diagnostics=diagnostics,
    )

    diagnostics.append(_error())

    assert outcome.diagnostics == ()


def test_string_enums_remain_json_serializable():
    payload = {
        "generation": GenerationStatus.VALIDATION_FAILED,
        "validation": ValidationStatus.INCOMPLETE,
        "severity": DiagnosticSeverity.WARNING,
    }

    assert json.loads(json.dumps(payload)) == {
        "generation": "validation_failed",
        "validation": "incomplete",
        "severity": "warning",
    }
