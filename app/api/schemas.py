from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.domain.outcomes import GenerationStatus, ValidationStatus

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
    prompt: PromptText
    context: Optional[Any] = None
    session_id: Optional[SessionIdText] = None
    feedback: Optional[FeedbackText] = None


class GenerateResponse(BaseModel):
    code: CodeText


class ValidationSummary(BaseModel):
    status: ValidationStatus
    ok: bool
    errors: List[str]
    degraded_mode: bool
    repair_rounds: int
    messages: List[Dict[str, Any]] = Field(default_factory=list)


class SessionStateSummary(BaseModel):
    session_id: str
    status: str
    original_task: Optional[str] = None
    latest_trace_id: Optional[str] = None
    last_strategy: Optional[str] = None
    open_clarification_question: Optional[str] = None
    clarification_history: List[Any] = Field(default_factory=list)
    feedback_history: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)


class GenerateRichRequest(BaseModel):
    prompt: Optional[PromptText] = None
    context: Optional[Any] = None
    session_id: Optional[SessionIdText] = None
    feedback: Optional[FeedbackText] = None
    clarification_answer: Optional[ClarificationAnswerText] = None


class GenerateRichResponse(BaseModel):
    status: GenerationStatus
    session_id: str
    trace_id: str
    strategy: str
    question: Optional[str] = None
    assumptions: List[str]
    code: Optional[str] = None
    validation: ValidationSummary
    session: SessionStateSummary


class AnalyzeRequest(BaseModel):
    prompt: PromptText
    context: Optional[Any] = None


class AnalyzeResponse(BaseModel):
    normalized_prompt: str
    suggested_strategy: str
    clarification_question: Optional[str] = None
    task_spec: Any
    reduced_context: Any
    available_paths: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    ambiguity_notes: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    profile: str


class ReadyResponse(BaseModel):
    status: str
    profile: str
    checks: Dict[str, bool] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)


class TraceResponse(BaseModel):
    trace_id: str
    session_id: str
    status: Optional[str] = None
    strategy: Optional[str] = None
    model: Optional[str] = None
    fallback_model: Optional[str] = None
    degraded_mode: bool = False
    repair_rounds: int = 0
    assumptions: List[str] = Field(default_factory=list)
    verification_errors: List[str] = Field(default_factory=list)
    validation_report: Dict[str, Any] = Field(default_factory=dict)
    planner: Dict[str, Any] = Field(default_factory=dict)
    critic: Dict[str, Any] = Field(default_factory=dict)
    repair_trace: List[Any] = Field(default_factory=list)
    rules_applied: List[str] = Field(default_factory=list)
    examples_used: List[str] = Field(default_factory=list)
    critic_rules_used: List[str] = Field(default_factory=list)
    semantic_checks: List[Any] = Field(default_factory=list)
    backend_error: Optional[str] = None
    code: str = ""


class ProfileResponse(BaseModel):
    profile: str
    model: str
    fallback_model: str
    num_ctx: int
    num_predict: int
    max_repair_rounds: int
    ui_enabled: bool


class ExampleEntry(BaseModel):
    id: str
    title: str
    mode: str
    prompt: str
    context: Any = None
    expected_strategy: Optional[str] = None
    description: Optional[str] = None


class ExamplesResponse(BaseModel):
    examples: List[ExampleEntry] = Field(default_factory=list)


class ValidateRequest(BaseModel):
    code: CodeText
    context: Optional[Any] = None
    output_style: Optional[str] = None


class SemanticResultSummary(BaseModel):
    ok: bool
    value: Any = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    degraded: bool = False


class ValidateResponse(BaseModel):
    ok: bool
    verification_errors: List[str] = Field(default_factory=list)
    validation_report: Dict[str, Any] = Field(default_factory=dict)
    degraded_mode: bool = False
    semantic_result: Optional[SemanticResultSummary] = None
