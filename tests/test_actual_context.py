import pytest
from pydantic import ValidationError

from app.workflow.contracts import (
    AcceptanceCase,
    CodeCandidate,
    ReviewApproved,
    WorkflowStage,
    WorkflowState,
)
from app.workflow.coordinator import WorkflowCoordinator
from app.workflow.validation import DeterministicCandidateValidator
from tests.test_workflow_coordinator import Generator, Planner, Reviewer, plan, validation


def test_generated_code_must_execute_on_actual_context():
    workflow = WorkflowCoordinator(
        planner=Planner(plan()),
        generator=Generator(initial="return wf.vars.value * 1", revised="return wf.vars.value * 1"),
        reviewer=Reviewer([ReviewApproved()]),
        validator=DeterministicCandidateValidator(),
    )
    result = workflow.run(
        prompt="Return value as a number", context={"wf": {"vars": {"value": "bad"}}}
    )
    assert result.code is None
    assert result.status.value == "validation_failed"


def test_caller_expected_is_independent_of_model_examples():
    result = DeterministicCandidateValidator().validate(
        candidate=CodeCandidate(code="return 4"),
        plan=plan(),
        context={"wf": {"vars": {"value": 4}}},
        examples=(
            AcceptanceCase(name="other", context={"wf": {"vars": {"value": 9}}}, expected=9),
        ),
    )
    assert not result.ok
    assert any(
        check.name == "caller:other" and check.code == "acceptance_result_mismatch"
        for check in result.checks
    )


def test_completed_state_rejects_failed_validation():
    with pytest.raises(ValidationError):
        WorkflowState(
            stage=WorkflowStage.COMPLETED,
            plan=plan(),
            candidate=CodeCandidate(code="return 4"),
            validation=validation(False),
            review=ReviewApproved(),
        )
