from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

JsonScalar: TypeAlias = bool | int | float | str | None


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WorkflowRoot(StrEnum):
    VARS = "wf.vars"
    INIT_VARIABLES = "wf.initVariables"


class WorkflowPath(StrictModel):
    root: WorkflowRoot
    segments: tuple[str, ...] = ()

    @field_validator("segments")
    @classmethod
    def validate_segments(cls, segments: tuple[str, ...]) -> tuple[str, ...]:
        for segment in segments:
            if not segment or len(segment) > 128 or any(ord(char) < 32 for char in segment):
                raise ValueError("workflow path segments must be non-empty printable strings")
        return segments

    @property
    def dotted(self) -> str:
        return ".".join((self.root.value, *self.segments))


class OutputFormat(StrEnum):
    LUA_BLOCK = "lua_block"
    JSON_ENVELOPE = "json_envelope"


class OutputShape(StrEnum):
    SCALAR = "scalar"
    ARRAY = "array"
    OBJECT = "object"


class OutputContract(StrictModel):
    format: OutputFormat
    shape: OutputShape
    nullable: bool = False


class ContextValueType(StrEnum):
    NULL = "null"
    BOOLEAN = "boolean"
    NUMBER = "number"
    STRING = "string"
    ARRAY = "array"
    OBJECT = "object"


class ContextEntry(StrictModel):
    path: WorkflowPath
    value_type: ContextValueType


class ContextInventory(StrictModel):
    entries: tuple[ContextEntry, ...]
    truncated: bool = False


class PlanStep(StrictModel):
    description: str = Field(min_length=1, max_length=500)
    reads: tuple[WorkflowPath, ...] = ()


class AcceptanceCase(StrictModel):
    name: str = Field(min_length=1, max_length=80)
    context: dict[str, JsonValue]
    expected: JsonValue

    @field_validator("context")
    @classmethod
    def validate_workflow_context(
        cls,
        context: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        workflow = context.get("wf")
        if not isinstance(workflow, Mapping):
            raise ValueError("acceptance context must contain a wf object")
        if not any(root in workflow for root in ("vars", "initVariables")):
            raise ValueError(
                "acceptance context must contain wf.vars or wf.initVariables"
            )
        return context


class TaskPlan(StrictModel):
    kind: Literal["plan"] = "plan"
    objective: str = Field(min_length=1, max_length=1000)
    inputs: tuple[WorkflowPath, ...]
    output: OutputContract
    steps: tuple[PlanStep, ...] = Field(min_length=1, max_length=12)
    constraints: tuple[str, ...] = Field(default=(), max_length=12)
    acceptance_cases: tuple[AcceptanceCase, ...] = Field(min_length=1, max_length=3)

    @field_validator("constraints")
    @classmethod
    def validate_constraints(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() or len(value) > 500 for value in values):
            raise ValueError("constraints must be non-empty strings up to 500 characters")
        return values


class ClarificationRequest(StrictModel):
    kind: Literal["clarification"] = "clarification"
    question: str = Field(min_length=1, max_length=500)
    reason: str = Field(min_length=1, max_length=500)


PlanningDecision: TypeAlias = Annotated[
    TaskPlan | ClarificationRequest,
    Field(discriminator="kind"),
]


class CodeCandidate(StrictModel):
    code: str = Field(min_length=1, max_length=131_072)

    @field_validator("code")
    @classmethod
    def reject_markdown(cls, code: str) -> str:
        rendered = code.strip()
        if rendered.startswith("```") or rendered.endswith("```"):
            raise ValueError("candidate must not contain markdown fences")
        return rendered


class ReviewFinding(StrictModel):
    code: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    message: str = Field(min_length=1, max_length=500)


class ReviewApproved(StrictModel):
    kind: Literal["approved"] = "approved"


class ReviewRejected(StrictModel):
    kind: Literal["rejected"] = "rejected"
    findings: tuple[ReviewFinding, ...] = Field(min_length=1, max_length=8)


ReviewDecision: TypeAlias = Annotated[
    ReviewApproved | ReviewRejected,
    Field(discriminator="kind"),
]


class CheckStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class ValidationCheck(StrictModel):
    name: str
    status: CheckStatus
    code: str | None = None
    message: str | None = None

    @model_validator(mode="after")
    def validate_failure_details(self) -> ValidationCheck:
        if self.status is CheckStatus.FAILED and (not self.code or not self.message):
            raise ValueError("failed validation check requires code and message")
        if self.status is CheckStatus.PASSED and (self.code is not None or self.message is not None):
            raise ValueError("passed validation check must not contain failure details")
        return self


class ValidationResult(StrictModel):
    checks: tuple[ValidationCheck, ...] = Field(min_length=1)
    observations: tuple[JsonValue, ...] = ()

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(check.status is CheckStatus.PASSED for check in self.checks)


class WorkflowStage(StrEnum):
    RECEIVED = "received"
    PLANNED = "planned"
    GENERATED = "generated"
    VALIDATED = "validated"
    REVIEWED = "reviewed"
    REVISED = "revised"
    COMPLETED = "completed"
    FAILED = "failed"
    CLARIFICATION_REQUIRED = "clarification_required"


class WorkflowState(StrictModel):
    stage: WorkflowStage = WorkflowStage.RECEIVED
    plan: TaskPlan | None = None
    candidate: CodeCandidate | None = None
    validation: ValidationResult | None = None
    review: ReviewDecision | None = None
    revision_count: int = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_stage_payload(self) -> WorkflowState:
        empty = self.plan is None and self.candidate is None and self.validation is None
        if self.stage in {
            WorkflowStage.RECEIVED,
            WorkflowStage.CLARIFICATION_REQUIRED,
        }:
            if not empty or self.review is not None or self.revision_count != 0:
                raise ValueError("stage must not carry generated workflow artifacts")
            return self

        if self.plan is None:
            raise ValueError("workflow stage requires a task plan")
        if self.stage is WorkflowStage.PLANNED:
            if self.candidate is not None or self.validation is not None or self.review is not None:
                raise ValueError("planned stage may only carry a task plan")
        elif self.stage in {WorkflowStage.GENERATED, WorkflowStage.REVISED}:
            if self.candidate is None or self.validation is not None or self.review is not None:
                raise ValueError("candidate stage requires one unvalidated candidate")
            expected_revisions = 1 if self.stage is WorkflowStage.REVISED else 0
            if self.revision_count != expected_revisions:
                raise ValueError("candidate stage has an inconsistent revision count")
        elif self.stage is WorkflowStage.VALIDATED:
            if self.candidate is None or self.validation is None or self.review is not None:
                raise ValueError("validated stage requires candidate and validation")
        elif self.stage in {WorkflowStage.REVIEWED, WorkflowStage.COMPLETED}:
            if self.candidate is None or self.validation is None or self.review is None:
                raise ValueError("reviewed stage requires candidate, validation, and review")
        elif self.stage is WorkflowStage.FAILED and self.candidate is None:
            raise ValueError("failed generated workflow requires its rejected candidate internally")
        return self


class WorkflowStatus(StrEnum):
    COMPLETED = "completed"
    CLARIFICATION_REQUIRED = "clarification_required"
    VALIDATION_FAILED = "validation_failed"
    POLICY_REJECTED = "policy_rejected"
    BACKEND_UNAVAILABLE = "backend_unavailable"


class WorkflowDiagnostic(StrictModel):
    code: str
    message: str
    stage: WorkflowStage


class WorkflowResult(StrictModel):
    status: WorkflowStatus
    code: str | None = None
    question: str | None = None
    diagnostics: tuple[WorkflowDiagnostic, ...] = ()
    validation: ValidationResult | None = None
    revision_count: int = 0

    @model_validator(mode="after")
    def validate_public_shape(self) -> WorkflowResult:
        if self.status is WorkflowStatus.COMPLETED:
            if not self.code or self.validation is None or not self.validation.ok:
                raise ValueError("completed workflow requires code and successful validation")
        elif self.code is not None:
            raise ValueError("failed workflow must not publish candidate code")
        if self.status is WorkflowStatus.CLARIFICATION_REQUIRED:
            if not self.question:
                raise ValueError("clarification requires a question")
        elif self.question is not None:
            raise ValueError("only clarification may contain a question")
        return self


__all__ = [
    "AcceptanceCase",
    "CheckStatus",
    "ClarificationRequest",
    "CodeCandidate",
    "ContextEntry",
    "ContextInventory",
    "ContextValueType",
    "JsonScalar",
    "JsonValue",
    "OutputContract",
    "OutputFormat",
    "OutputShape",
    "PlanStep",
    "PlanningDecision",
    "ReviewApproved",
    "ReviewDecision",
    "ReviewFinding",
    "ReviewRejected",
    "StrictModel",
    "TaskPlan",
    "ValidationCheck",
    "ValidationResult",
    "WorkflowDiagnostic",
    "WorkflowPath",
    "WorkflowResult",
    "WorkflowRoot",
    "WorkflowStage",
    "WorkflowState",
    "WorkflowStatus",
]
