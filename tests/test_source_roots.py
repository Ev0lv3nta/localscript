import json

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_runtime_profile
from app.core.traces import TraceStore
from app.main import create_app
from app.validation.runtime_executor import execute_output
from app.workflow.contracts import (
    AcceptanceCase,
    CodeCandidate,
    OutputContract,
    OutputFormat,
    OutputShape,
    PlanStep,
    ReviewApproved,
    TaskPlan,
    WorkflowRoot,
    WorkflowStatus,
)
from app.workflow.coordinator import WorkflowCoordinator
from app.workflow.validation import DeterministicCandidateValidator
from tests.support_backends import DeterministicTestBackend
from tests.test_workflow_coordinator import Generator, Planner, Reviewer


def _plan(*, expected, case_context, inputs=()):
    return TaskPlan(
        objective="Return the requested value.",
        inputs=inputs,
        output=OutputContract(format=OutputFormat.LUA_BLOCK, shape=OutputShape.SCALAR),
        steps=(PlanStep(description="Return the requested value."),),
        acceptance_cases=(AcceptanceCase(name="model", context=case_context, expected=expected),),
    )


@pytest.mark.parametrize(
    "code",
    [
        "local alias = wf.vars; return alias.value",
        ("local function read_value(source) return source.value end; return read_value(wf.vars)"),
    ],
)
def test_trusted_runtime_reports_root_reads_through_aliases_and_arguments(code):
    execution = execute_output(
        code,
        {"wf": {"vars": {"value": 4}, "initVariables": {"value": 9}}},
    )

    assert execution.ok
    assert execution.read_roots == (WorkflowRoot.VARS.value,)


def test_pairs_over_wf_reports_both_roots():
    execution = execute_output(
        "local count = 0; for _, _ in pairs(wf) do count = count + 1 end; return count",
        {"wf": {"vars": {"value": 4}, "initVariables": {"value": 9}}},
    )

    assert execution.ok
    assert execution.read_roots == (
        WorkflowRoot.VARS.value,
        WorkflowRoot.INIT_VARIABLES.value,
    )


def test_json_envelope_aggregates_roots_from_every_chunk():
    execution = execute_output(
        '{"left":"lua{return wf.vars.value}lua","right":"lua{return wf.initVariables.value}lua"}',
        {"wf": {"vars": {"value": 4}, "initVariables": {"value": 9}}},
        output_style="json_envelope",
    )

    assert execution.ok
    assert execution.read_roots == (
        WorkflowRoot.VARS.value,
        WorkflowRoot.INIT_VARIABLES.value,
    )


def test_model_cannot_hide_single_root_guess_by_omitting_plan_inputs():
    plan = _plan(
        expected=4,
        case_context={"wf": {"vars": {"value": 4}}},
        inputs=(),
    )
    workflow = WorkflowCoordinator(
        planner=Planner(plan),
        generator=Generator(initial="return wf.vars.value"),
        reviewer=Reviewer([]),
        validator=DeterministicCandidateValidator(),
    )

    result = workflow.run(
        prompt="Return value.",
        context={"wf": {"vars": {"value": 1}, "initVariables": {"value": 2}}},
    )

    assert result.status is WorkflowStatus.CLARIFICATION_REQUIRED
    assert result.code is None
    assert result.source_choices == (WorkflowRoot.VARS, WorkflowRoot.INIT_VARIABLES)


def test_code_using_both_roots_does_not_force_single_root_question():
    plan = _plan(
        expected=7,
        case_context={"wf": {"vars": {"value": 3}, "initVariables": {"value": 4}}},
    )
    workflow = WorkflowCoordinator(
        planner=Planner(plan),
        generator=Generator(initial="return wf.vars.value + wf.initVariables.value"),
        reviewer=Reviewer([ReviewApproved()]),
        validator=DeterministicCandidateValidator(),
    )

    result = workflow.run(
        prompt="Add both values.",
        context={"wf": {"vars": {"value": 1}, "initVariables": {"value": 2}}},
    )

    assert result.status is WorkflowStatus.COMPLETED
    assert result.code == "return wf.vars.value + wf.initVariables.value"


def test_code_with_no_workflow_reads_relies_on_examples_without_source_question():
    plan = _plan(
        expected=1,
        case_context={"wf": {"vars": {"value": 3}, "initVariables": {"value": 4}}},
    )
    workflow = WorkflowCoordinator(
        planner=Planner(plan),
        generator=Generator(initial="return 1"),
        reviewer=Reviewer([ReviewApproved()]),
        validator=DeterministicCandidateValidator(),
    )

    result = workflow.run(
        prompt="Return one.",
        context={"wf": {"vars": {"value": 1}, "initVariables": {"value": 2}}},
    )

    assert result.status is WorkflowStatus.COMPLETED


def test_selected_source_rejects_actual_access_to_other_root():
    plan = _plan(
        expected=4,
        case_context={"wf": {"initVariables": {"value": 4}}},
    )

    result = DeterministicCandidateValidator().validate(
        candidate=CodeCandidate(code="return wf.initVariables.value"),
        plan=plan,
        context={"wf": {"vars": {"value": 1}, "initVariables": {"value": 2}}},
        source_roots=(WorkflowRoot.VARS,),
    )

    assert not result.ok
    assert any(check.code == "source_root_not_selected" for check in result.checks)


def test_declared_plan_input_outside_selected_roots_is_rejected_before_generation():
    from app.workflow.contracts import WorkflowPath

    selected = (WorkflowRoot.VARS,)
    other_path = WorkflowPath(root=WorkflowRoot.INIT_VARIABLES, segments=("value",))
    plan = _plan(
        expected=4,
        case_context={"wf": {"initVariables": {"value": 4}}},
        inputs=(other_path,),
    )
    generator = Generator(initial="return wf.initVariables.value")
    workflow = WorkflowCoordinator(
        planner=Planner(plan),
        generator=generator,
        reviewer=Reviewer([]),
        validator=DeterministicCandidateValidator(),
    )

    result = workflow.run(
        prompt="Return value.",
        context={"wf": {"vars": {"value": 1}, "initVariables": {"value": 2}}},
        source_roots=selected,
    )

    assert result.status is WorkflowStatus.VALIDATION_FAILED
    assert any(item.code == "plan_source_root_not_selected" for item in result.diagnostics)
    assert generator.calls == 0


def test_api_exposes_and_accepts_structured_source_choice(tmp_path):
    app = create_app(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=DeterministicTestBackend(),
    )
    client = TestClient(app)
    first = client.post(
        "/api/generate",
        json={
            "prompt": "Return value.",
            "context": {
                "wf": {
                    "vars": {"value": 1},
                    "initVariables": {"value": 2},
                }
            },
        },
    )

    assert first.status_code == 200
    assert first.json()["status"] == "clarification_required"
    assert first.json()["source_choices"] == ["wf.vars", "wf.initVariables"]
    session_id = first.json()["session_id"]

    cached = client.post("/api/generate", json={"session_id": session_id})
    assert cached.json()["source_choices"] == ["wf.vars", "wf.initVariables"]

    continued = client.post(
        "/api/generate",
        json={"session_id": session_id, "source_roots": ["wf.vars"]},
    )

    assert continued.status_code == 200
    assert continued.json()["status"] == "completed"
    assert continued.json()["code"] == "return wf.vars.value"


def test_planner_source_question_can_be_resolved_without_free_text(tmp_path):
    class AskingBackend(DeterministicTestBackend):
        asked = False

        def complete(self, prompt, **kwargs):
            if "You are the planner" in prompt and not self.asked:
                self.asked = True
                return json.dumps(
                    {
                        "kind": "clarification",
                        "question": "Какой источник выбрать?",
                        "reason": "Совпадающие поля.",
                        "source_choices": ["wf.vars", "wf.initVariables"],
                    }
                )
            return super().complete(prompt, **kwargs)

    client = TestClient(
        create_app(
            profile=get_runtime_profile(),
            trace_store=TraceStore(root=tmp_path / "traces"),
            backend=AskingBackend(),
        )
    )
    first = client.post(
        "/api/generate",
        json={
            "prompt": "Return value.",
            "context": {"wf": {"vars": {"value": 1}, "initVariables": {"value": 2}}},
        },
    ).json()
    assert first["source_choices"] == ["wf.vars", "wf.initVariables"]
    continued = client.post(
        "/api/generate",
        json={
            "session_id": first["session_id"],
            "source_roots": ["wf.vars"],
        },
    )
    assert continued.status_code == 200
    assert continued.json()["status"] == "completed"
