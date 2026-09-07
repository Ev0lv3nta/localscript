import threading

import pytest

from app.core.config import get_runtime_profile
from app.core.sessions import SessionStore
from app.core.traces import TraceStore
from app.generation.engine import (
    MAX_SESSION_USER_TURNS,
    GenerationEngine,
    SessionConflictError,
    SessionNotFoundError,
)
from app.workflow.contracts import (
    AcceptanceCase,
    CheckStatus,
    OutputContract,
    OutputFormat,
    OutputShape,
    ValidationCheck,
    ValidationResult,
    WorkflowResult,
    WorkflowStatus,
)


def _completed_result() -> WorkflowResult:
    return WorkflowResult(
        status=WorkflowStatus.COMPLETED,
        code="return wf.vars.value",
        validation=ValidationResult(
            checks=(ValidationCheck(name="all", status=CheckStatus.PASSED),),
        ),
    )


def _clarification_result(question: str = "Which root?") -> WorkflowResult:
    return WorkflowResult(
        status=WorkflowStatus.CLARIFICATION_REQUIRED,
        question=question,
    )


class UnusedBackend:
    def complete(self, _prompt, *, response_format=None):
        raise AssertionError("capturing workflow should bypass the model backend")


class CapturingWorkflow:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _engine(tmp_path, workflow):
    sessions = SessionStore(root=tmp_path / "sessions")
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=UnusedBackend(),
        session_store=sessions,
    )
    engine.workflow = workflow
    return engine


def test_existing_session_rejects_a_new_prompt_without_mutating_task(tmp_path):
    workflow = CapturingWorkflow([_completed_result()])
    engine = _engine(tmp_path, workflow)
    first = engine.generate(prompt="Task A", context={"wf": {"vars": {"value": 1}}})

    with pytest.raises(SessionConflictError) as raised:
        engine.generate(prompt="Task B", session_id=first.session_id)

    assert raised.value.code == "session_conflict"
    assert raised.value.status_code == 409
    assert "new session" in raised.value.message
    persisted = engine.session_store.read(first.session_id)
    assert persisted["original_task"] == "Task A"
    assert persisted["latest_prompt"] == "Task A"
    assert len(workflow.calls) == 1


def test_missing_continuation_session_raises_typed_not_found(tmp_path):
    engine = _engine(tmp_path, CapturingWorkflow([]))

    with pytest.raises(SessionNotFoundError) as raised:
        engine.generate(session_id="missing-session", feedback="Please revise it.")

    assert raised.value.code == "session_not_found"
    assert raised.value.status_code == 404
    assert engine.session_store.read("missing-session") is None


def test_answer_then_feedback_passes_accumulated_history_and_preserves_context(tmp_path):
    workflow = CapturingWorkflow(
        [_clarification_result(), _completed_result(), _completed_result()]
    )
    engine = _engine(tmp_path, workflow)
    context = {"wf": {"vars": {"value": 7, "token": "functional-token"}}}
    output = OutputContract(format=OutputFormat.LUA_BLOCK, shape=OutputShape.SCALAR)
    examples = (
        AcceptanceCase(
            name="provided",
            context={"wf": {"vars": {"value": 7}}},
            expected=7,
        ),
    )

    first = engine.generate(prompt="Task A", context=context, output=output, examples=examples)
    engine.generate(session_id=first.session_id, clarification_answer="Use wf.vars.")
    final = engine.generate(session_id=first.session_id, feedback="Return nil when absent.")

    second_call = workflow.calls[1]
    third_call = workflow.calls[2]
    assert second_call["context"] == context
    assert second_call["output"] == output
    assert second_call["examples"] == examples
    assert "Question: Which root?" in second_call["clarification_answer"]
    assert "Answer: Use wf.vars." in second_call["clarification_answer"]
    assert third_call["context"] == context
    assert third_call["output"] == output
    assert third_call["examples"] == examples
    assert "Answer: Use wf.vars." in third_call["clarification_answer"]
    assert "1. Return nil when absent." in third_call["feedback"]

    persisted = engine.session_store.read(first.session_id)
    assert persisted["context"]["wf"]["vars"]["token"] == "functional-token"
    assert persisted["clarification_history"] == [
        {"question": "Which root?", "answer": "Use wf.vars."}
    ]
    assert persisted["feedback_history"] == ["Return nil when absent."]
    assert final.session.clarification_history[0].answer == "Use wf.vars."
    assert final.session.feedback_history == ("Return nil when absent.",)


def test_existing_session_rejects_conflicting_output_or_examples(tmp_path):
    workflow = CapturingWorkflow([_completed_result()])
    engine = _engine(tmp_path, workflow)
    output = OutputContract(format=OutputFormat.LUA_BLOCK, shape=OutputShape.SCALAR)
    examples = (
        AcceptanceCase(
            name="original",
            context={"wf": {"vars": {"value": 1}}},
            expected=1,
        ),
    )
    first = engine.generate(prompt="Task A", output=output, examples=examples)

    with pytest.raises(SessionConflictError):
        engine.generate(
            session_id=first.session_id,
            output=OutputContract(format=OutputFormat.LUA_BLOCK, shape=OutputShape.ARRAY),
        )
    with pytest.raises(SessionConflictError):
        engine.generate(
            session_id=first.session_id,
            examples=(
                AcceptanceCase(
                    name="changed",
                    context={"wf": {"vars": {"value": 2}}},
                    expected=2,
                ),
            ),
        )

    persisted = engine.session_store.read(first.session_id)
    assert OutputContract.model_validate(persisted["output"]) == output
    assert tuple(AcceptanceCase.model_validate(item) for item in persisted["examples"]) == examples


def test_user_turn_history_keeps_only_ten_most_recent_meaningful_turns(tmp_path):
    workflow = CapturingWorkflow(
        [_clarification_result(), *[_clarification_result() for _ in range(12)]]
    )
    engine = _engine(tmp_path, workflow)
    first = engine.generate(prompt="Task A")

    for index in range(12):
        engine.generate(
            session_id=first.session_id,
            clarification_answer=f"answer-{index}",
        )

    persisted = engine.session_store.read(first.session_id)
    assert len(persisted["user_turn_history"]) == MAX_SESSION_USER_TURNS
    assert [item["text"] for item in persisted["user_turn_history"]] == [
        f"answer-{index}" for index in range(2, 12)
    ]
    assert len(persisted["clarification_history"]) == MAX_SESSION_USER_TURNS
    effective_history = workflow.calls[-1]["clarification_answer"]
    answer_lines = [line.strip() for line in effective_history.splitlines() if "Answer:" in line]
    assert "Answer: answer-1" not in answer_lines
    assert "Answer: answer-2" in answer_lines
    assert "Answer: answer-11" in answer_lines


def test_reading_unrelated_session_does_not_wait_for_generation(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    class BlockingWorkflow:
        def run(self, **_kwargs):
            entered.set()
            assert release.wait(timeout=5)
            return _completed_result()

    engine = _engine(tmp_path, BlockingWorkflow())
    engine.session_store.write("other-session", {"value": "available"})
    failures = []
    generation = threading.Thread(
        target=lambda: _run_and_collect_failure(
            failures,
            engine.generate,
            prompt="Slow task",
            session_id="blocked-session",
        )
    )
    generation.start()
    assert entered.wait(timeout=2)

    result = []
    reader = threading.Thread(
        target=lambda: result.append(engine.session_store.read("other-session"))
    )
    reader.start()
    reader.join(timeout=0.5)

    try:
        assert not reader.is_alive()
        assert result[0]["value"] == "available"
        assert result[0]["session_id"] == "other-session"
    finally:
        release.set()
        generation.join(timeout=5)
        reader.join(timeout=5)

    assert not failures


def _run_and_collect_failure(failures, function, **kwargs):
    try:
        function(**kwargs)
    except Exception as exc:  # pragma: no cover - assertion reports the captured exception
        failures.append(exc)
