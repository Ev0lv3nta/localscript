import json
import threading
import time

from app.core.config import get_runtime_profile
from app.core.traces import TraceStore
from app.generation.backend_errors import BackendUnavailable
from app.generation.engine import GenerationEngine
from app.workflow.contracts import (
    CheckStatus,
    ValidationCheck,
    ValidationResult,
    WorkflowStatus,
)


class SequenceBackend:
    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, _prompt, *, response_format=None):
        assert response_format is not None
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class PassingValidator:
    def validate(self, **_kwargs):
        return ValidationResult(
            checks=(ValidationCheck(name="all", status=CheckStatus.PASSED),),
            observations=({"case": "value", "actual": 4},),
        )


def planner_response():
    return json.dumps(
        {
            "kind": "plan",
            "objective": "Return the value.",
            "inputs": [{"root": "wf.vars", "segments": ["value"]}],
            "output": {"format": "lua_block", "shape": "scalar", "nullable": False},
            "steps": [
                {
                    "description": "Return the value.",
                    "reads": [{"root": "wf.vars", "segments": ["value"]}],
                }
            ],
            "constraints": [],
            "acceptance_cases": [
                {
                    "name": "value",
                    "context": {"wf": {"vars": {"value": 4}}},
                    "expected": 4,
                }
            ],
        }
    )


def test_engine_runs_typed_workflow_and_writes_sanitized_trace(tmp_path):
    store = TraceStore(root=tmp_path / "traces")
    backend = SequenceBackend(
        [
            planner_response(),
            json.dumps({"code": "return wf.vars.value"}),
            json.dumps({"kind": "approved"}),
        ]
    )
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=store,
        backend=backend,
        validator=PassingValidator(),
    )

    result = engine.generate(
        prompt="Return the private value.",
        context={"wf": {"vars": {"value": 4, "secret": "do-not-store"}}},
    )

    assert result.workflow.status is WorkflowStatus.COMPLETED
    assert result.workflow.code == "return wf.vars.value"
    trace = store.read(result.trace_id)
    encoded = json.dumps(trace, ensure_ascii=False)
    assert "do-not-store" not in encoded
    assert "Return the private value" not in encoded
    assert "return wf.vars.value" not in encoded
    assert trace["diagnostic_codes"] == []
    assert [event["stage"] for event in trace["stage_events"]] == [
        "received",
        "planned",
        "generated",
        "validated",
        "reviewed",
        "completed",
    ]


def test_engine_returns_typed_clarification_without_candidate(tmp_path):
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=SequenceBackend(
            [
                json.dumps(
                    {
                        "kind": "clarification",
                        "question": "Which workflow root should be used?",
                        "reason": "Both roots contain value.",
                    }
                )
            ]
        ),
        validator=PassingValidator(),
    )

    result = engine.generate(
        prompt="Return value.",
        context={
            "wf": {
                "vars": {"value": 1},
                "initVariables": {"value": 2},
            }
        },
    )

    assert result.workflow.status is WorkflowStatus.CLARIFICATION_REQUIRED
    assert result.workflow.code is None
    assert result.workflow.question == "Which workflow root should be used?"


def test_engine_converts_backend_outage_to_fail_closed_outcome(tmp_path):
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=SequenceBackend([BackendUnavailable(reason="transport_error")]),
        validator=PassingValidator(),
    )

    result = engine.generate(prompt="Return value.", context=None)

    assert result.workflow.status is WorkflowStatus.BACKEND_UNAVAILABLE
    assert result.workflow.code is None


def test_engine_serializes_updates_for_the_same_session(tmp_path, monkeypatch):
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=SequenceBackend([]),
        validator=PassingValidator(),
    )

    def increment_session(**kwargs):
        session_state = kwargs["session_state"]
        current = session_state.get("count", 0)
        time.sleep(0.01)
        session_state["count"] = current + 1

    monkeypatch.setattr(engine, "_generate_locked", increment_session)
    threads = [
        threading.Thread(
            target=engine.generate,
            kwargs={
                "prompt": "prompt",
                "context": None,
                "session_id": "shared-session",
            },
        )
        for _ in range(8)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert engine.session_store.read("shared-session")["count"] == 8
