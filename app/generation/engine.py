from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Protocol

from pydantic import TypeAdapter

from app.core.config import RuntimeProfile
from app.core.sessions import SessionStore
from app.domain.outcomes import (
    Diagnostic,
    DiagnosticSeverity,
    GenerationOutcome,
    GenerationStatus,
    ValidationOutcome,
    ValidationStatus,
)
from app.generation.backend_errors import BackendUnavailable
from app.workflow.context import ContextInspector
from app.workflow.contracts import (
    CheckStatus,
    JsonValue,
    WorkflowResult,
    WorkflowStage,
)
from app.workflow.coordinator import CandidateValidator, WorkflowCoordinator
from app.workflow.roles import GeneratorRole, PlannerRole, ReviewerRole, StructuredModelClient
from app.workflow.validation import DeterministicCandidateValidator

JSON_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


class CompletionBackend(Protocol):
    def complete(self, prompt: str, *, response_format: object | None = None) -> str: ...


class TraceWriter(Protocol):
    root: Path

    def write(self, trace: dict[str, JsonValue]) -> str: ...


# Kept for callers during the API transition in PR3.
BackendUnavailableError = BackendUnavailable


@dataclass(frozen=True)
class GenerationResult:
    code: str = ""
    trace_id: str = ""
    session_id: str = ""
    strategy: str = ""
    verification_errors: list[str] = field(default_factory=list)
    validation_report: dict[str, JsonValue] = field(default_factory=dict)
    repair_rounds: int = 0
    degraded_mode: bool = False
    status: str = GenerationStatus.VALIDATION_FAILED.value
    question: str = ""
    assumptions: list[str] = field(default_factory=list)
    session_summary: dict[str, JsonValue] = field(default_factory=dict)
    clarification_suggested: bool = False
    assumption_risk: str = "low"
    outcome: GenerationOutcome | None = None


class _StageTimer:
    def __init__(self) -> None:
        self._last_stage: WorkflowStage | None = None
        self._last_time = monotonic()
        self._events: list[dict[str, JsonValue]] = []

    def observe(self, stage: WorkflowStage) -> None:
        now = monotonic()
        if self._last_stage is not None:
            self._events.append(
                {
                    "stage": self._last_stage.value,
                    "duration_ms": round((now - self._last_time) * 1000, 3),
                }
            )
        self._last_stage = stage
        self._last_time = now

    def finish(self) -> tuple[dict[str, JsonValue], ...]:
        if self._last_stage is not None:
            now = monotonic()
            self._events.append(
                {
                    "stage": self._last_stage.value,
                    "duration_ms": round((now - self._last_time) * 1000, 3),
                }
            )
            self._last_stage = None
        return tuple(self._events)


class GenerationEngine:
    """Compatibility facade over the typed agentic workflow.

    PR3 removes the legacy result fields and endpoints. This facade intentionally contains no
    prompt router, task labels, deterministic code repair, or task-specific templates.
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
            trace_root = trace_store.root
            session_store = SessionStore(root=trace_root.parent / "sessions")
        self.session_store = session_store

        model = StructuredModelClient(self.backend.complete)
        self.workflow = WorkflowCoordinator(
            planner=PlannerRole(model),
            generator=GeneratorRole(model),
            reviewer=ReviewerRole(model),
            validator=validator or DeterministicCandidateValidator(),
        )
        self.context_inspector = ContextInspector()

    def generate(
        self,
        prompt: str,
        context: object = None,
        session_id: str | None = None,
        feedback: str | None = None,
    ) -> GenerationResult:
        return self._run_generation(
            prompt=prompt,
            context=context,
            session_id=session_id,
            feedback=feedback,
            clarification_answer=None,
            rich_mode=False,
        )

    def generate_rich(
        self,
        prompt: str | None = None,
        context: object = None,
        session_id: str | None = None,
        feedback: str | None = None,
        clarification_answer: str | None = None,
    ) -> GenerationResult:
        return self._run_generation(
            prompt=prompt,
            context=context,
            session_id=session_id,
            feedback=feedback,
            clarification_answer=clarification_answer,
            rich_mode=True,
        )

    def _run_generation(
        self,
        *,
        prompt: str | None,
        context: object,
        session_id: str | None,
        feedback: str | None,
        clarification_answer: str | None,
        rich_mode: bool,
    ) -> GenerationResult:
        resolved_session_id = session_id or uuid.uuid4().hex
        with self.session_store.transaction(resolved_session_id) as session_state:
            return self._run_generation_locked(
                prompt=prompt,
                context=context,
                session_id=resolved_session_id,
                feedback=feedback,
                clarification_answer=clarification_answer,
                rich_mode=rich_mode,
                session_state=session_state,
            )

    def _run_generation_locked(
        self,
        *,
        prompt: str | None,
        context: object,
        session_id: str,
        feedback: str | None,
        clarification_answer: str | None,
        rich_mode: bool,
        session_state: dict[str, object],
    ) -> GenerationResult:
        del rich_mode
        self._prepare_session_state(
            session_id=session_id,
            prompt=prompt,
            context=context,
            feedback=feedback,
            clarification_answer=clarification_answer,
            session_state=session_state,
        )

        if session_state.get("open_clarification_question") and not clarification_answer:
            return self._cached_clarification_result(session_state)

        effective_prompt = str(session_state["original_task"])
        effective_context = session_state.get("context")
        timer = _StageTimer()
        workflow_result = self.workflow.run(
            prompt=effective_prompt,
            context=effective_context,
            clarification_answer=clarification_answer,
            feedback=feedback,
            observe=timer.observe,
        )
        stage_events = timer.finish()

        trace_id = self.trace_store.write(
            {
                "session_id": session_id,
                "status": workflow_result.status.value,
                "model": self.profile.model,
                "diagnostic_codes": [
                    diagnostic.code for diagnostic in workflow_result.diagnostics
                ],
                "revision_count": workflow_result.revision_count,
                "stage_events": list(stage_events),
            }
        )
        session_state["status"] = workflow_result.status.value
        session_state["latest_trace_id"] = trace_id
        trace_ids = session_state.setdefault("trace_ids", [])
        if isinstance(trace_ids, list):
            trace_ids.append(trace_id)
        session_state["open_clarification_question"] = workflow_result.question or ""

        result = self._build_result(
            workflow_result,
            trace_id=trace_id,
            session_id=session_id,
            session_state=session_state,
        )
        return result

    def analyze(self, prompt: str, context: object = None) -> dict[str, JsonValue]:
        """Temporary deterministic shape for the endpoint removed in PR3."""
        json_context = JSON_ADAPTER.validate_python(context, strict=True)
        inventory = self.context_inspector.inventory(json_context)
        sample = self.context_inspector.sample(json_context)
        return {
            "normalized_prompt": prompt.strip(),
            "suggested_strategy": "",
            "clarification_question": None,
            "task_spec": {"context_inventory": inventory.model_dump(mode="json")},
            "reduced_context": sample,
            "available_paths": [entry.path.dotted for entry in inventory.entries],
            "assumptions": [],
            "ambiguity_notes": [],
        }

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
                    "status": "pending",
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

    def _cached_clarification_result(
        self,
        session_state: dict[str, object],
    ) -> GenerationResult:
        question = str(session_state.get("open_clarification_question") or "")
        outcome = GenerationOutcome(
            status=GenerationStatus.CLARIFICATION_REQUIRED,
            validation=ValidationOutcome(status=ValidationStatus.NOT_RUN),
            question=question,
        )
        return GenerationResult(
            trace_id=str(session_state.get("latest_trace_id") or ""),
            session_id=str(session_state["session_id"]),
            status=outcome.status.value,
            question=question,
            session_summary=self.build_session_summary(session_state),
            clarification_suggested=True,
            assumption_risk="high",
            outcome=outcome,
        )

    def _build_result(
        self,
        workflow: WorkflowResult,
        *,
        trace_id: str,
        session_id: str,
        session_state: dict[str, object],
    ) -> GenerationResult:
        outcome = self._build_outcome(workflow)
        validation_report = self._validation_report(workflow)
        error_codes = [diagnostic.code for diagnostic in workflow.diagnostics]
        for check in workflow.validation.checks if workflow.validation else ():
            if check.status is CheckStatus.FAILED and check.code not in error_codes:
                error_codes.append(check.code or "validation_failed")
        is_clarification = outcome.status is GenerationStatus.CLARIFICATION_REQUIRED
        return GenerationResult(
            code=outcome.code or "",
            trace_id=trace_id,
            session_id=session_id,
            verification_errors=error_codes,
            validation_report=validation_report,
            repair_rounds=workflow.revision_count,
            status=outcome.status.value,
            question=outcome.question or "",
            session_summary=self.build_session_summary(session_state),
            clarification_suggested=is_clarification,
            assumption_risk="high" if is_clarification else "low",
            outcome=outcome,
        )

    @staticmethod
    def _build_outcome(workflow: WorkflowResult) -> GenerationOutcome:
        status = GenerationStatus(workflow.status.value)
        diagnostics = GenerationEngine._diagnostics(workflow)
        if status is GenerationStatus.COMPLETED:
            validation = ValidationOutcome(status=ValidationStatus.PASSED)
            return GenerationOutcome(
                status=status,
                validation=validation,
                code=workflow.code,
            )
        if status in {
            GenerationStatus.CLARIFICATION_REQUIRED,
            GenerationStatus.BACKEND_UNAVAILABLE,
        }:
            return GenerationOutcome(
                status=status,
                validation=ValidationOutcome(status=ValidationStatus.NOT_RUN),
                question=workflow.question,
                diagnostics=diagnostics,
            )

        if not diagnostics:
            diagnostics = (
                Diagnostic(
                    code="generation_failed",
                    message="Generation did not produce validated code.",
                    severity=DiagnosticSeverity.ERROR,
                    stage=WorkflowStage.FAILED.value,
                ),
            )
        return GenerationOutcome(
            status=status,
            validation=ValidationOutcome(
                status=ValidationStatus.FAILED,
                findings=diagnostics,
            ),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _diagnostics(workflow: WorkflowResult) -> tuple[Diagnostic, ...]:
        collected: list[Diagnostic] = []
        seen: set[tuple[str, str]] = set()

        def append(code: str, message: str, stage: str) -> None:
            key = (stage, code)
            if key in seen:
                return
            seen.add(key)
            collected.append(
                Diagnostic(
                    code=code,
                    message=message,
                    severity=DiagnosticSeverity.ERROR,
                    stage=stage,
                )
            )

        for diagnostic in workflow.diagnostics:
            append(diagnostic.code, diagnostic.message, diagnostic.stage.value)
        if workflow.validation is not None:
            for check in workflow.validation.checks:
                if check.status is CheckStatus.FAILED:
                    append(
                        check.code or "validation_failed",
                        check.message or "Validation failed.",
                        check.name,
                    )
        return tuple(collected)

    @staticmethod
    def _validation_report(workflow: WorkflowResult) -> dict[str, JsonValue]:
        diagnostics = GenerationEngine._diagnostics(workflow)
        return {
            "has_errors": bool(diagnostics),
            "has_warnings": False,
            "messages": [
                {
                    "validator": diagnostic.stage,
                    "level": diagnostic.severity.value,
                    "code": diagnostic.code,
                    "message": diagnostic.message,
                }
                for diagnostic in diagnostics
            ],
        }

    @staticmethod
    def build_session_summary(session_state: dict[str, object]) -> dict[str, JsonValue]:
        clarification_history = session_state.get("clarification_history")
        feedback_history = session_state.get("feedback_history")
        return {
            "session_id": str(session_state["session_id"]),
            "status": str(session_state.get("status", "pending")),
            "original_task": str(session_state.get("original_task") or ""),
            "latest_trace_id": (
                str(session_state["latest_trace_id"])
                if session_state.get("latest_trace_id")
                else None
            ),
            "last_strategy": None,
            "open_clarification_question": (
                str(session_state["open_clarification_question"])
                if session_state.get("open_clarification_question")
                else None
            ),
            "clarification_history": (
                JSON_ADAPTER.validate_python(clarification_history, strict=True)
                if clarification_history is not None
                else []
            ),
            "feedback_history": (
                JSON_ADAPTER.validate_python(feedback_history, strict=True)
                if feedback_history is not None
                else []
            ),
            "assumptions": [],
        }
