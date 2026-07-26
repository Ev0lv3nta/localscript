from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class GenerationStatus(str, Enum):
    COMPLETED = "completed"
    CLARIFICATION_REQUIRED = "clarification_required"
    VALIDATION_FAILED = "validation_failed"
    POLICY_REJECTED = "policy_rejected"
    BACKEND_UNAVAILABLE = "backend_unavailable"


class ValidationStatus(str, Enum):
    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


class DiagnosticSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    severity: DiagnosticSeverity
    stage: str

    def __post_init__(self):
        if not isinstance(self.code, str):
            raise TypeError("diagnostic code must be a string")
        if not isinstance(self.message, str):
            raise TypeError("diagnostic message must be a string")
        if not isinstance(self.severity, DiagnosticSeverity):
            raise TypeError("diagnostic severity must be DiagnosticSeverity")
        if not isinstance(self.stage, str):
            raise TypeError("diagnostic stage must be a string")
        if not self.code.strip():
            raise ValueError("diagnostic code must not be empty")
        if not self.message.strip():
            raise ValueError("diagnostic message must not be empty")
        if not self.stage.strip():
            raise ValueError("diagnostic stage must not be empty")


@dataclass(frozen=True)
class ValidationOutcome:
    status: ValidationStatus
    findings: Tuple[Diagnostic, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if not isinstance(self.status, ValidationStatus):
            raise TypeError("validation status must be ValidationStatus")
        try:
            findings = tuple(self.findings)
        except TypeError as exc:
            raise TypeError("validation findings must be iterable") from exc
        if not all(isinstance(finding, Diagnostic) for finding in findings):
            raise TypeError("validation findings must contain Diagnostic values")
        object.__setattr__(self, "findings", findings)

        has_errors = any(
            finding.severity is DiagnosticSeverity.ERROR
            for finding in findings
        )
        if self.status is ValidationStatus.PASSED and has_errors:
            raise ValueError("passed validation must not contain error findings")
        if self.status is ValidationStatus.FAILED and not has_errors:
            raise ValueError("failed validation requires an error finding")
        if self.status is ValidationStatus.NOT_RUN and self.findings:
            raise ValueError("validation that was not run must not contain findings")

    @property
    def ok(self):
        return self.status is ValidationStatus.PASSED


@dataclass(frozen=True)
class GenerationOutcome:
    status: GenerationStatus
    validation: ValidationOutcome
    code: Optional[str] = None
    question: Optional[str] = None
    diagnostics: Tuple[Diagnostic, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if not isinstance(self.status, GenerationStatus):
            raise TypeError("generation status must be GenerationStatus")
        if not isinstance(self.validation, ValidationOutcome):
            raise TypeError("generation validation must be ValidationOutcome")
        if self.code is not None and not isinstance(self.code, str):
            raise TypeError("generation code must be a string or None")
        if self.question is not None and not isinstance(self.question, str):
            raise TypeError("generation question must be a string or None")
        try:
            diagnostics = tuple(self.diagnostics)
        except TypeError as exc:
            raise TypeError("generation diagnostics must be iterable") from exc
        if not all(isinstance(diagnostic, Diagnostic) for diagnostic in diagnostics):
            raise TypeError("generation diagnostics must contain Diagnostic values")
        object.__setattr__(self, "diagnostics", diagnostics)

        has_errors = any(
            diagnostic.severity is DiagnosticSeverity.ERROR
            for diagnostic in diagnostics
        )

        if self.status is GenerationStatus.COMPLETED:
            if not self.code or not self.code.strip():
                raise ValueError("completed generation requires code")
            if not self.validation.ok:
                raise ValueError("completed generation requires passed validation")
            if has_errors:
                raise ValueError("completed generation must not contain error diagnostics")
        elif self.code is not None:
            raise ValueError("non-completed generation must not publish code")

        if self.status is GenerationStatus.CLARIFICATION_REQUIRED:
            if not self.question or not self.question.strip():
                raise ValueError("clarification outcome requires a question")
            if self.validation.status is not ValidationStatus.NOT_RUN:
                raise ValueError("clarification outcome requires validation not_run")
        elif self.question is not None:
            raise ValueError("only clarification outcome may contain a question")

        if (
            self.status is GenerationStatus.VALIDATION_FAILED
            and self.validation.status
            not in {ValidationStatus.FAILED, ValidationStatus.INCOMPLETE}
        ):
            raise ValueError(
                "validation_failed generation requires failed or incomplete validation"
            )
