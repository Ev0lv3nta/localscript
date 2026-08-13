import json

import pytest
from pydantic import ValidationError

from app.workflow.context import ContextInspector
from app.workflow.contracts import (
    AcceptanceCase,
    CheckStatus,
    ClarificationRequest,
    CodeCandidate,
    OutputContract,
    OutputFormat,
    OutputShape,
    PlanStep,
    ReviewApproved,
    ReviewFinding,
    ReviewRejected,
    TaskPlan,
    ValidationCheck,
    ValidationResult,
    WorkflowPath,
    WorkflowResult,
    WorkflowRoot,
    WorkflowStatus,
)


def build_plan() -> TaskPlan:
    source = WorkflowPath(root=WorkflowRoot.VARS, segments=("value",))
    return TaskPlan(
        objective="Return the input value in lower case.",
        inputs=(source,),
        output=OutputContract(
            format=OutputFormat.LUA_BLOCK,
            shape=OutputShape.SCALAR,
        ),
        steps=(PlanStep(description="Read and normalize the value.", reads=(source,)),),
        acceptance_cases=(
            AcceptanceCase(
                name="lowercase",
                context={"wf": {"vars": {"value": "ABC"}}},
                expected="abc",
            ),
        ),
    )


def test_workflow_path_uses_structured_root_and_segments():
    path = WorkflowPath(root=WorkflowRoot.INIT_VARIABLES, segments=("order.id", "value"))

    assert path.dotted == "wf.initVariables.order.id.value"


def test_workflow_path_rejects_empty_or_control_segments():
    with pytest.raises(ValidationError):
        WorkflowPath(root=WorkflowRoot.VARS, segments=("",))
    with pytest.raises(ValidationError):
        WorkflowPath(root=WorkflowRoot.VARS, segments=("bad\nsegment",))


def test_context_inspector_walks_json_without_prompt_classification():
    inspector = ContextInspector()

    inventory = inspector.inventory(
        {
            "wf": {
                "vars": {"orders": [{"id": 7, "active": True}]},
                "initVariables": {"region": "ru"},
            }
        }
    )

    entries = {(entry.path.dotted, entry.value_type.value) for entry in inventory.entries}
    assert ("wf.vars.orders", "array") in entries
    assert ("wf.vars.orders.[].id", "number") in entries
    assert ("wf.initVariables.region", "string") in entries


def test_context_sample_falls_back_to_typed_inventory_when_large():
    sample = ContextInspector(sample_chars=20).sample(
        {"wf": {"vars": {"value": "x" * 100}}}
    )

    assert isinstance(sample, dict)
    assert sample["truncated"] is True
    assert sample["paths"]


def test_code_candidate_rejects_markdown_fences():
    with pytest.raises(ValidationError):
        CodeCandidate(code="```lua\nreturn 1\n```")


def test_failed_validation_check_requires_structured_details():
    with pytest.raises(ValidationError):
        ValidationCheck(name="ast", status=CheckStatus.FAILED)


def test_public_result_never_exposes_rejected_candidate():
    with pytest.raises(ValidationError):
        WorkflowResult(status=WorkflowStatus.VALIDATION_FAILED, code="return 1")


def test_completed_result_requires_successful_validation():
    validation = ValidationResult(
        checks=(ValidationCheck(name="runtime", status=CheckStatus.PASSED),)
    )

    result = WorkflowResult(
        status=WorkflowStatus.COMPLETED,
        code="return 1",
        validation=validation,
    )

    assert result.code == "return 1"


def test_discriminated_decisions_have_consistent_shapes():
    assert ClarificationRequest(question="Which root?", reason="Two roots are present.").kind == "clarification"
    assert ReviewApproved().kind == "approved"
    assert ReviewRejected(
        findings=(ReviewFinding(code="wrong_result", message="The result is incorrect."),)
    ).kind == "rejected"


def test_plan_json_contains_no_family_or_router_fields():
    payload = json.loads(build_plan().model_dump_json())

    assert "family" not in payload
    assert "strategy" not in payload
    assert "confidence" not in payload
