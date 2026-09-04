from __future__ import annotations

from enum import StrEnum

from app.workflow.contracts import JsonValue, StrictModel, WorkflowResult, WorkflowStage


class SessionStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    CLARIFICATION_REQUIRED = "clarification_required"
    VALIDATION_FAILED = "validation_failed"
    POLICY_REJECTED = "policy_rejected"
    BACKEND_UNAVAILABLE = "backend_unavailable"


class ClarificationExchange(StrictModel):
    question: str
    answer: str


class SessionSummary(StrictModel):
    session_id: str
    status: SessionStatus
    original_task: str
    latest_trace_id: str | None = None
    open_clarification_question: str | None = None
    clarification_history: tuple[ClarificationExchange, ...] = ()
    feedback_history: tuple[str, ...] = ()


class StageEvent(StrictModel):
    stage: WorkflowStage
    duration_ms: float


class GenerationResult(StrictModel):
    """Everything an adapter may show about one generation attempt.

    The workflow result carries its own invariants, so an adapter cannot assemble a response that
    publishes code without successful validation, or a question outside a clarification.
    """

    workflow: WorkflowResult
    session_id: str
    trace_id: str
    session: SessionSummary


__all__ = [
    "ClarificationExchange",
    "GenerationResult",
    "JsonValue",
    "SessionStatus",
    "SessionSummary",
    "StageEvent",
]
