from __future__ import annotations

import uuid
from pathlib import Path
from time import monotonic
from typing import Protocol

from app.core.config import RuntimeProfile
from app.core.sessions import SessionStore
from app.generation.results import (
    ClarificationExchange,
    GenerationResult,
    SessionStatus,
    SessionSummary,
    StageEvent,
)
from app.workflow.contracts import JsonValue, WorkflowResult, WorkflowStage, WorkflowStatus
from app.workflow.coordinator import CandidateValidator, WorkflowCoordinator
from app.workflow.roles import GeneratorRole, PlannerRole, ReviewerRole, StructuredModelClient
from app.workflow.validation import DeterministicCandidateValidator


class CompletionBackend(Protocol):
    def complete(self, prompt: str, *, response_format: object | None = None) -> str: ...


class TraceWriter(Protocol):
    root: Path

    def write(self, trace: dict[str, JsonValue]) -> str: ...


class _StageTimer:
    def __init__(self) -> None:
        self._last_stage: WorkflowStage | None = None
        self._last_time = monotonic()
        self._events: list[StageEvent] = []

    def observe(self, stage: WorkflowStage) -> None:
        now = monotonic()
        if self._last_stage is not None:
            self._events.append(
                StageEvent(
                    stage=self._last_stage,
                    duration_ms=round((now - self._last_time) * 1000, 3),
                )
            )
        self._last_stage = stage
        self._last_time = now

    def finish(self) -> tuple[StageEvent, ...]:
        if self._last_stage is not None:
            self._events.append(
                StageEvent(
                    stage=self._last_stage,
                    duration_ms=round((monotonic() - self._last_time) * 1000, 3),
                )
            )
            self._last_stage = None
        return tuple(self._events)


class GenerationEngine:
    """Application service around the typed agentic workflow.

    It owns session state, tracing and identifiers; every product decision belongs to the
    workflow coordinator. There is no prompt router, task label, code repair or task template.
    """

    def __init__(
        self,
        profile: RuntimeProfile,
        trace_store: TraceWriter,
        backend: CompletionBackend,
        session_store: SessionStore | None = None,
        validator: CandidateValidator | None = None,
    ) -> None:
        self.profile = profile
        self.trace_store = trace_store
        self.backend = backend
        if session_store is None:
            session_store = SessionStore(root=trace_store.root.parent / "sessions")
        self.session_store = session_store

        model = StructuredModelClient(self.backend.complete)
        self.workflow = WorkflowCoordinator(
            planner=PlannerRole(model),
            generator=GeneratorRole(model),
            reviewer=ReviewerRole(model),
            validator=validator or DeterministicCandidateValidator(),
        )

    def generate(
        self,
        prompt: str | None = None,
        context: object = None,
        session_id: str | None = None,
        feedback: str | None = None,
        clarification_answer: str | None = None,
    ) -> GenerationResult:
        resolved_session_id = session_id or uuid.uuid4().hex
        with self.session_store.transaction(resolved_session_id) as session_state:
            return self._generate_locked(
                prompt=prompt,
                context=context,
                session_id=resolved_session_id,
                feedback=feedback,
                clarification_answer=clarification_answer,
                session_state=session_state,
            )

    def _generate_locked(
        self,
        *,
        prompt: str | None,
        context: object,
        session_id: str,
        feedback: str | None,
        clarification_answer: str | None,
        session_state: dict[str, object],
    ) -> GenerationResult:
        self._prepare_session_state(
            session_id=session_id,
            prompt=prompt,
            context=context,
            feedback=feedback,
            clarification_answer=clarification_answer,
            session_state=session_state,
        )

        open_question = str(session_state.get("open_clarification_question") or "")
        if open_question and not clarification_answer:
            return GenerationResult(
                workflow=WorkflowResult(
                    status=WorkflowStatus.CLARIFICATION_REQUIRED,
                    question=open_question,
                ),
                session_id=session_id,
                trace_id=str(session_state.get("latest_trace_id") or ""),
                session=self.build_session_summary(session_state),
            )

        timer = _StageTimer()
        workflow = self.workflow.run(
            prompt=str(session_state["original_task"]),
            context=session_state.get("context"),
            clarification_answer=clarification_answer,
            feedback=feedback,
            observe=timer.observe,
        )
        stage_events = timer.finish()

        trace_id = self.trace_store.write(
            {
                "session_id": session_id,
                "status": workflow.status.value,
                "model": self.profile.model,
                "diagnostic_codes": [
                    diagnostic.code for diagnostic in workflow.diagnostics
                ],
                "revision_count": workflow.revision_count,
                "stage_events": [event.model_dump(mode="json") for event in stage_events],
            }
        )
        session_state["status"] = workflow.status.value
        session_state["latest_trace_id"] = trace_id
        trace_ids = session_state.setdefault("trace_ids", [])
        if isinstance(trace_ids, list):
            trace_ids.append(trace_id)
        session_state["open_clarification_question"] = workflow.question or ""

        return GenerationResult(
            workflow=workflow,
            session_id=session_id,
            trace_id=trace_id,
            session=self.build_session_summary(session_state),
        )

    @staticmethod
    def _prepare_session_state(
        *,
        session_id: str,
        prompt: str | None,
        context: object,
        feedback: str | None,
        clarification_answer: str | None,
        session_state: dict[str, object],
    ) -> None:
        if not session_state:
            if not prompt:
                raise ValueError("original task is required to initialize a session")
            session_state.update(
                {
                    "session_id": session_id,
                    "status": SessionStatus.PENDING.value,
                    "original_task": prompt,
                    "latest_prompt": prompt,
                    "context": context,
                    "open_clarification_question": "",
                    "clarification_history": [],
                    "feedback_history": [],
                    "latest_trace_id": None,
                    "trace_ids": [],
                }
            )
        elif prompt:
            session_state["latest_prompt"] = prompt

        if context is not None:
            session_state["context"] = context
        if feedback:
            history = session_state.setdefault("feedback_history", [])
            if isinstance(history, list):
                history.append(feedback)
        if clarification_answer:
            question = str(session_state.get("open_clarification_question") or "")
            history = session_state.setdefault("clarification_history", [])
            if isinstance(history, list):
                history.append({"question": question, "answer": clarification_answer})
            session_state["open_clarification_question"] = ""

        if not session_state.get("original_task"):
            raise ValueError("original task is required to initialize a session")

    @staticmethod
    def build_session_summary(session_state: dict[str, object]) -> SessionSummary:
        raw_status = str(session_state.get("status") or SessionStatus.PENDING.value)
        try:
            status = SessionStatus(raw_status)
        except ValueError:
            status = SessionStatus.PENDING
        history = session_state.get("clarification_history")
        exchanges = tuple(
            ClarificationExchange(
                question=str(item.get("question") or ""),
                answer=str(item.get("answer") or ""),
            )
            for item in (history if isinstance(history, list) else ())
            if isinstance(item, dict)
        )
        feedback = session_state.get("feedback_history")
        return SessionSummary(
            session_id=str(session_state["session_id"]),
            status=status,
            original_task=str(session_state.get("original_task") or ""),
            latest_trace_id=(
                str(session_state["latest_trace_id"])
                if session_state.get("latest_trace_id")
                else None
            ),
            open_clarification_question=(
                str(session_state["open_clarification_question"])
                if session_state.get("open_clarification_question")
                else None
            ),
            clarification_history=exchanges,
            feedback_history=tuple(
                str(item) for item in (feedback if isinstance(feedback, list) else ())
            ),
        )
