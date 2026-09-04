from typing import Annotated

from pydantic import BaseModel, Field, JsonValue, model_validator

from app.generation.results import SessionSummary
from app.workflow.contracts import (
    OutputContract,
    StrictModel,
    ValidationResult,
    WorkflowDiagnostic,
    WorkflowStatus,
)

MAX_SCHEMA_PROMPT_CHARS = 131072
MAX_SCHEMA_CODE_CHARS = 131072
MAX_SCHEMA_FEEDBACK_CHARS = 32768
MAX_SCHEMA_CLARIFICATION_ANSWER_CHARS = 32768
MAX_SCHEMA_SESSION_ID_CHARS = 64

PromptText = Annotated[str, Field(max_length=MAX_SCHEMA_PROMPT_CHARS)]
CodeText = Annotated[str, Field(max_length=MAX_SCHEMA_CODE_CHARS)]
FeedbackText = Annotated[str, Field(max_length=MAX_SCHEMA_FEEDBACK_CHARS)]
ClarificationAnswerText = Annotated[
    str,
    Field(max_length=MAX_SCHEMA_CLARIFICATION_ANSWER_CHARS),
]
SessionIdText = Annotated[str, Field(max_length=MAX_SCHEMA_SESSION_ID_CHARS)]


class GenerateRequest(BaseModel):
    prompt: PromptText | None = None
    context: JsonValue = None
    session_id: SessionIdText | None = None
    feedback: FeedbackText | None = None
    clarification_answer: ClarificationAnswerText | None = None


class GenerateResponse(StrictModel):
    status: WorkflowStatus
    session_id: str
    trace_id: str
    code: str | None = None
    question: str | None = None
    diagnostics: tuple[WorkflowDiagnostic, ...] = ()
    validation: ValidationResult | None = None
    revision_count: int = 0

    @model_validator(mode="after")
    def validate_public_shape(self) -> "GenerateResponse":
        if self.status is WorkflowStatus.COMPLETED:
            if not self.code or self.validation is None or not self.validation.ok:
                raise ValueError("completed response requires code and successful validation")
        elif self.code is not None:
            raise ValueError("only a completed response may carry code")
        if self.status is WorkflowStatus.CLARIFICATION_REQUIRED:
            if not self.question:
                raise ValueError("clarification response requires a question")
        elif self.question is not None:
            raise ValueError("only a clarification response may carry a question")
        return self


class HealthResponse(BaseModel):
    status: str
    profile: str


class ReadyResponse(BaseModel):
    status: str
    profile: str
    checks: dict[str, bool] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class TraceResponse(BaseModel):
    trace_id: str
    session_id: str
    status: str | None = None
    model: str | None = None
    revision_count: int = 0
    diagnostic_codes: list[str] = Field(default_factory=list)
    stage_events: list[dict[str, JsonValue]] = Field(default_factory=list)


class ProfileResponse(BaseModel):
    profile: str
    model: str
    fallback_model: str
    num_ctx: int
    num_predict: int
    ui_enabled: bool


class ExampleEntry(BaseModel):
    id: str
    title: str
    prompt: str
    context: JsonValue = None
    description: str | None = None


class ExamplesResponse(BaseModel):
    examples: list[ExampleEntry] = Field(default_factory=list)


class ValidateRequest(BaseModel):
    code: CodeText
    context: dict[str, JsonValue]
    output: OutputContract


class ValidateResponse(BaseModel):
    ok: bool
    validation: ValidationResult


__all__ = [
    "ExampleEntry",
    "ExamplesResponse",
    "GenerateRequest",
    "GenerateResponse",
    "HealthResponse",
    "ProfileResponse",
    "ReadyResponse",
    "SessionSummary",
    "TraceResponse",
    "ValidateRequest",
    "ValidateResponse",
]
