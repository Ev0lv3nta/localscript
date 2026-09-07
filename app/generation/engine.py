from __future__ import annotations

import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import monotonic
from typing import Protocol

from pydantic import TypeAdapter, ValidationError

from app.api.limits import APIConstraintError, validate_context, validate_prompt
from app.core.budgets import workflow_budget
from app.core.config import RuntimeProfile
from app.core.sessions import SessionStore
from app.generation.backend_errors import BackendBusy
from app.generation.results import (
    ClarificationExchange,
    GenerationResult,
    SessionStatus,
    SessionSummary,
    StageEvent,
)
from app.workflow.contracts import (
    AcceptanceCase,
    JsonValue,
    OutputContract,
    WorkflowResult,
    WorkflowRoot,
    WorkflowStage,
    WorkflowStatus,
)
from app.workflow.coordinator import CandidateValidator, WorkflowCoordinator
from app.workflow.roles import GeneratorRole, PlannerRole, ReviewerRole, StructuredModelClient
from app.workflow.validation import DeterministicCandidateValidator

MAX_SESSION_USER_TURNS = 10
_CONTEXT_UNSET = object()


class SessionStateError(ValueError):
    """A caller-correctable session error for HTTP/CLI adapters."""

    def __init__(self, *, code: str, status_code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.message = message


class SessionNotFoundError(SessionStateError):
    def __init__(self, session_id: str) -> None:
        super().__init__(
            code="session_not_found",
            status_code=404,
            message=f"Session {session_id!r} was not found.",
        )


class SessionConflictError(SessionStateError):
    def __init__(self, message: str) -> None:
        super().__init__(
            code="session_conflict",
            status_code=409,
            message=message,
        )


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
        self._admission = threading.Lock()
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
        context: object = _CONTEXT_UNSET,
        session_id: str | None = None,
        feedback: str | None = None,
        clarification_answer: str | None = None,
        output: OutputContract | None = None,
        examples: tuple[AcceptanceCase, ...] = (),
        source_roots: tuple[WorkflowRoot, ...] | None = None,
    ) -> GenerationResult:
        source_roots = self._normalize_source_roots(source_roots)
        resolved_session_id = session_id or uuid.uuid4().hex
        with (
            self.request_slot(),
            workflow_budget(),
            self.session_store.transaction(resolved_session_id) as session_state,
        ):
            return self._generate_locked(
                prompt=prompt,
                context=context,
                session_id=resolved_session_id,
                feedback=feedback,
                clarification_answer=clarification_answer,
                output=output,
                examples=examples,
                source_roots=source_roots,
                session_state=session_state,
                requested_existing_session=session_id is not None,
            )

    @contextmanager
    def request_slot(self) -> Iterator[None]:
        if not self._admission.acquire(blocking=False):
            raise BackendBusy()
        try:
            yield
        finally:
            self._admission.release()

    def _generate_locked(
        self,
        *,
        prompt: str | None,
        context: object,
        session_id: str,
        feedback: str | None,
        clarification_answer: str | None,
        output: OutputContract | None,
        examples: tuple[AcceptanceCase, ...],
        source_roots: tuple[WorkflowRoot, ...] | None,
        session_state: dict[str, object],
        requested_existing_session: bool,
    ) -> GenerationResult:
        if (
            requested_existing_session
            and prompt is None
            and context is _CONTEXT_UNSET
            and feedback is None
            and clarification_answer is None
            and output is None
            and not examples
            and source_roots is None
            and session_state.get("status") == SessionStatus.COMPLETED.value
        ):
            raise SessionConflictError(
                "A completed session needs feedback or an explicit context update to continue."
            )
        self._prepare_session_state(
            session_id=session_id,
            prompt=prompt,
            context=context,
            feedback=feedback,
            clarification_answer=clarification_answer,
            output=output,
            examples=examples,
            source_roots=source_roots,
            session_state=session_state,
            requested_existing_session=requested_existing_session,
        )
        self._validate_effective_request(session_state, feedback, clarification_answer)

        open_question = str(session_state.get("open_clarification_question") or "")
        if open_question and not clarification_answer:
            return GenerationResult(
                workflow=WorkflowResult(
                    status=WorkflowStatus.CLARIFICATION_REQUIRED,
                    question=open_question,
                    source_choices=(
                        (WorkflowRoot.VARS, WorkflowRoot.INIT_VARIABLES)
                        if session_state.get("open_clarification_kind") == "source_root"
                        else ()
                    ),
                ),
                session_id=session_id,
                trace_id=str(session_state.get("latest_trace_id") or ""),
                session=self.build_session_summary(session_state),
            )

        timer = _StageTimer()
        workflow = self.workflow.run(
            prompt=str(session_state["original_task"]),
            context=session_state.get("context"),
            clarification_answer=self._effective_clarification_history(session_state),
            feedback=self._effective_feedback_history(
                session_state,
                include_latest_code=feedback is not None,
            ),
            output=self._stored_output(session_state),
            examples=self._stored_examples(session_state),
            source_roots=self._stored_source_roots(session_state),
            observe=timer.observe,
        )
        stage_events = timer.finish()

        trace_id = self.trace_store.write(
            {
                "session_id": session_id,
                "status": workflow.status.value,
                "model": self.profile.model,
                "diagnostic_codes": [diagnostic.code for diagnostic in workflow.diagnostics],
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
        clarification_kind = ""
        if workflow.status is WorkflowStatus.CLARIFICATION_REQUIRED:
            if workflow.source_choices:
                clarification_kind = "source_root"
            else:
                clarification_kind = (
                    "missing_context" if session_state.get("context") is None else "planner"
                )
        session_state["open_clarification_kind"] = clarification_kind
        if workflow.status is WorkflowStatus.COMPLETED and workflow.code:
            session_state["latest_completed_code"] = workflow.code

        return GenerationResult(
            workflow=workflow,
            session_id=session_id,
            trace_id=trace_id,
            session=self.build_session_summary(session_state),
        )

    def _validate_effective_request(
        self, state: dict[str, object], feedback: str | None, answer: str | None
    ) -> None:
        try:
            for text in (str(state.get("original_task") or ""), feedback, answer):
                if text is not None and not text.strip():
                    raise APIConstraintError(
                        422, "empty_user_message", "User messages must not be empty."
                    )
                validate_prompt(text, self.profile.max_prompt_chars)
            adapter: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
            context = adapter.validate_python(state.get("context"), strict=True)
            validate_context(
                context,
                self.profile.max_context_bytes,
                self.profile.max_context_depth,
                self.profile.max_context_nodes,
            )
            examples = self._stored_examples(state)
            if len(examples) > 3:
                raise APIConstraintError(
                    422, "too_many_examples", "At most three caller examples are supported."
                )
            for example in examples:
                validate_context(
                    example.context,
                    self.profile.max_context_bytes,
                    self.profile.max_context_depth,
                    self.profile.max_context_nodes,
                )
            effective_text = (
                str(state.get("original_task") or "")
                + (self._effective_clarification_history(state) or "")
                + (
                    self._effective_feedback_history(state, include_latest_code=bool(feedback))
                    or ""
                )
            )
            if len(effective_text) > 12000:
                raise APIConstraintError(
                    422,
                    "effective_task_too_large",
                    "The accumulated task is too large; start a new session.",
                )
        except APIConstraintError as error:
            raise SessionStateError(
                code=error.code, status_code=error.status_code, message=error.message
            ) from None
        except ValidationError:
            raise SessionStateError(
                code="invalid_json_context", status_code=422, message="Context must be JSON."
            ) from None

    @staticmethod
    def _prepare_session_state(
        *,
        session_id: str,
        prompt: str | None,
        context: object,
        feedback: str | None,
        clarification_answer: str | None,
        output: OutputContract | None,
        examples: tuple[AcceptanceCase, ...],
        source_roots: tuple[WorkflowRoot, ...] | None,
        session_state: dict[str, object],
        requested_existing_session: bool,
    ) -> None:
        if not session_state:
            if requested_existing_session and not prompt:
                raise SessionNotFoundError(session_id)
            if not prompt:
                raise ValueError("original task is required to initialize a session")
            session_state.update(
                {
                    "session_id": session_id,
                    "status": SessionStatus.PENDING.value,
                    "original_task": prompt,
                    "latest_prompt": prompt,
                    "context": None if context is _CONTEXT_UNSET else context,
                    "output": output.model_dump(mode="json") if output is not None else None,
                    "examples": [item.model_dump(mode="json") for item in examples],
                    "source_roots": (
                        [root.value for root in source_roots] if source_roots is not None else None
                    ),
                    "open_clarification_question": "",
                    "open_clarification_kind": "",
                    "clarification_history": [],
                    "feedback_history": [],
                    "user_turn_history": [],
                    "latest_trace_id": None,
                    "trace_ids": [],
                }
            )
        elif prompt is not None:
            raise SessionConflictError(
                "A prompt cannot replace the task in an existing session; "
                "start a new session for a new task or use feedback to revise this one."
            )

        if session_state and session_state.get("original_task"):
            stored_output = GenerationEngine._stored_output(session_state)
            if output is not None and output != stored_output:
                raise SessionConflictError(
                    "The output contract cannot change within an existing session; "
                    "start a new session for a different contract."
                )
            stored_examples = GenerationEngine._stored_examples(session_state)
            if examples and examples != stored_examples:
                raise SessionConflictError(
                    "Acceptance examples cannot change within an existing session; "
                    "start a new session for different examples."
                )
            stored_source_roots = GenerationEngine._stored_source_roots(session_state)
            if source_roots is not None and stored_source_roots not in (None, source_roots):
                raise SessionConflictError(
                    "Workflow source selection cannot change within an existing session."
                )
            if (
                source_roots is not None
                and stored_source_roots is None
                and session_state.get("open_clarification_kind") != "source_root"
            ):
                raise SessionConflictError(
                    "Workflow sources can only be selected when the session requests that choice."
                )
            if source_roots is not None:
                session_state["source_roots"] = [root.value for root in source_roots]

        context_resolves_missing_request = (
            context is not _CONTEXT_UNSET
            and context is not None
            and session_state.get("open_clarification_kind") == "missing_context"
        )
        if context is not _CONTEXT_UNSET:
            session_state["context"] = context
        if context_resolves_missing_request and not clarification_answer:
            session_state["open_clarification_question"] = ""
            session_state["open_clarification_kind"] = ""

        source_roots_resolve_question = (
            source_roots is not None
            and session_state.get("open_clarification_kind") == "source_root"
        )

        GenerationEngine._ensure_user_turn_history(session_state)
        if source_roots_resolve_question and source_roots is not None and not clarification_answer:
            question = str(session_state.get("open_clarification_question") or "")
            GenerationEngine._append_user_turn(
                session_state,
                {
                    "kind": "clarification",
                    "question": question,
                    "text": ", ".join(root.value for root in source_roots),
                },
            )
            session_state["open_clarification_question"] = ""
            session_state["open_clarification_kind"] = ""
        if clarification_answer:
            question = str(session_state.get("open_clarification_question") or "")
            if not question:
                raise SessionConflictError(
                    "This session has no open clarification question to answer."
                )
            GenerationEngine._append_user_turn(
                session_state,
                {
                    "kind": "clarification",
                    "question": question,
                    "text": clarification_answer,
                },
            )
            session_state["open_clarification_question"] = ""
            session_state["open_clarification_kind"] = ""
        if feedback:
            GenerationEngine._append_user_turn(
                session_state,
                {"kind": "feedback", "text": feedback},
            )

        if not session_state.get("original_task"):
            raise ValueError("original task is required to initialize a session")

    @staticmethod
    def _stored_output(session_state: dict[str, object]) -> OutputContract | None:
        raw_output = session_state.get("output")
        if raw_output is None:
            return None
        try:
            return OutputContract.model_validate(raw_output)
        except (TypeError, ValidationError):
            raise SessionConflictError(
                "The persisted output contract is invalid; start a new session."
            ) from None

    @staticmethod
    def _stored_examples(session_state: dict[str, object]) -> tuple[AcceptanceCase, ...]:
        raw_examples = session_state.get("examples")
        if raw_examples is None:
            return ()
        if not isinstance(raw_examples, list):
            raise SessionConflictError(
                "The persisted acceptance examples are invalid; start a new session."
            )
        try:
            return tuple(AcceptanceCase.model_validate(item) for item in raw_examples)
        except (TypeError, ValidationError):
            raise SessionConflictError(
                "The persisted acceptance examples are invalid; start a new session."
            ) from None

    @staticmethod
    def _normalize_source_roots(
        source_roots: tuple[WorkflowRoot, ...] | None,
    ) -> tuple[WorkflowRoot, ...] | None:
        if source_roots is None:
            return None
        try:
            normalized = tuple(WorkflowRoot(root) for root in source_roots)
        except (TypeError, ValueError):
            raise SessionStateError(
                code="invalid_source_roots",
                status_code=422,
                message="source_roots must contain workflow root enum values.",
            ) from None
        if not 1 <= len(normalized) <= 2 or len(set(normalized)) != len(normalized):
            raise SessionStateError(
                code="invalid_source_roots",
                status_code=422,
                message="source_roots must contain one or two distinct workflow roots.",
            )
        selected = set(normalized)
        return tuple(
            root for root in (WorkflowRoot.VARS, WorkflowRoot.INIT_VARIABLES) if root in selected
        )

    @staticmethod
    def _stored_source_roots(
        session_state: dict[str, object],
    ) -> tuple[WorkflowRoot, ...] | None:
        raw_roots = session_state.get("source_roots")
        if raw_roots is None:
            return None
        if not isinstance(raw_roots, list):
            raise SessionConflictError(
                "The persisted workflow source selection is invalid; start a new session."
            )
        try:
            roots = tuple(WorkflowRoot(root) for root in raw_roots)
        except (TypeError, ValueError):
            raise SessionConflictError(
                "The persisted workflow source selection is invalid; start a new session."
            ) from None
        if not 1 <= len(roots) <= 2 or len(set(roots)) != len(roots):
            raise SessionConflictError(
                "The persisted workflow source selection is invalid; start a new session."
            )
        selected = set(roots)
        return tuple(
            root for root in (WorkflowRoot.VARS, WorkflowRoot.INIT_VARIABLES) if root in selected
        )

    @staticmethod
    def _ensure_user_turn_history(session_state: dict[str, object]) -> list[dict[str, str]]:
        raw_history = session_state.get("user_turn_history")
        normalized: list[dict[str, str]] = []
        if isinstance(raw_history, list):
            for item in raw_history:
                if not isinstance(item, dict):
                    continue
                kind = str(item.get("kind") or "")
                text = str(item.get("text") or "")
                if kind == "clarification" and text:
                    normalized.append(
                        {
                            "kind": kind,
                            "question": str(item.get("question") or ""),
                            "text": text,
                        }
                    )
                elif kind == "feedback" and text:
                    normalized.append({"kind": kind, "text": text})
        else:
            clarifications = session_state.get("clarification_history")
            feedback = session_state.get("feedback_history")
            if (
                isinstance(clarifications, list)
                and clarifications
                and isinstance(feedback, list)
                and feedback
            ):
                raise SessionConflictError(
                    "This legacy session contains unordered clarification and feedback history; "
                    "start a new session to avoid changing the meaning of prior instructions."
                )
            # A legacy state containing only one kind of turn has an
            # unambiguous order and can be migrated to the canonical list.
            if isinstance(clarifications, list):
                for item in clarifications:
                    if not isinstance(item, dict) or not item.get("answer"):
                        continue
                    normalized.append(
                        {
                            "kind": "clarification",
                            "question": str(item.get("question") or ""),
                            "text": str(item["answer"]),
                        }
                    )
            if isinstance(feedback, list):
                normalized.extend(
                    {"kind": "feedback", "text": str(item)} for item in feedback if item
                )

        if len(normalized) > MAX_SESSION_USER_TURNS:
            raise SessionConflictError(
                "This session contains more than 10 persisted user turns and cannot be "
                "continued safely; start a new session."
            )
        session_state["user_turn_history"] = normalized
        GenerationEngine._sync_public_history(session_state, normalized)
        return normalized

    @staticmethod
    def _append_user_turn(
        session_state: dict[str, object],
        turn: dict[str, str],
    ) -> None:
        history = GenerationEngine._ensure_user_turn_history(session_state)
        if len(history) >= MAX_SESSION_USER_TURNS:
            raise SessionConflictError(
                "This session already contains 10 user turns; start a new session instead of "
                "discarding confirmed clarification or feedback history."
            )
        history.append(turn)
        session_state["user_turn_history"] = history
        GenerationEngine._sync_public_history(session_state, history)

    @staticmethod
    def _sync_public_history(
        session_state: dict[str, object],
        history: list[dict[str, str]],
    ) -> None:
        session_state["clarification_history"] = [
            {"question": item.get("question", ""), "answer": item["text"]}
            for item in history
            if item["kind"] == "clarification"
        ]
        session_state["feedback_history"] = [
            item["text"] for item in history if item["kind"] == "feedback"
        ]

    @staticmethod
    def _effective_clarification_history(session_state: dict[str, object]) -> str | None:
        history = GenerationEngine._ensure_user_turn_history(session_state)
        entries = [item for item in history if item["kind"] == "clarification"]
        if not entries:
            return None
        lines = ["Confirmed clarification history (oldest to newest):"]
        for index, item in enumerate(entries, start=1):
            lines.append(
                f"{index}. Question: {item.get('question', '')}\n   Answer: {item['text']}"
            )
        return "\n".join(lines)

    @staticmethod
    def _effective_feedback_history(
        session_state: dict[str, object],
        *,
        include_latest_code: bool,
    ) -> str | None:
        history = GenerationEngine._ensure_user_turn_history(session_state)
        entries = [item["text"] for item in history if item["kind"] == "feedback"]
        if not entries:
            return None
        lines = ["Applied feedback history (oldest to newest):"]
        lines.extend(f"{index}. {text}" for index, text in enumerate(entries, start=1))
        latest_code = session_state.get("latest_completed_code")
        if include_latest_code and latest_code:
            lines.extend(
                (
                    "Previous completed Lua candidate to revise:",
                    str(latest_code),
                )
            )
        return "\n".join(lines)

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
