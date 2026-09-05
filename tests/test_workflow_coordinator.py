from app.generation.backend_errors import BackendProtocol, BackendUnavailable
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
    WorkflowRoot,
    WorkflowStatus,
)
from app.workflow.coordinator import WorkflowCoordinator


def plan():
    path = WorkflowPath(root=WorkflowRoot.VARS, segments=("value",))
    return TaskPlan(
        objective="Return value.",
        inputs=(path,),
        output=OutputContract(format=OutputFormat.LUA_BLOCK, shape=OutputShape.SCALAR),
        steps=(PlanStep(description="Return the value.", reads=(path,)),),
        acceptance_cases=(
            AcceptanceCase(
                name="value",
                context={"wf": {"vars": {"value": 4}}},
                expected=4,
            ),
        ),
    )


def validation(ok=True):
    if ok:
        check = ValidationCheck(name="all", status=CheckStatus.PASSED)
    else:
        check = ValidationCheck(
            name="runtime",
            status=CheckStatus.FAILED,
            code="runtime_mismatch",
            message="Wrong result.",
        )
    return ValidationResult(checks=(check,))


class Planner:
    def __init__(self, result):
        self.result = result

    def run(self, **_kwargs):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class Generator:
    def __init__(self, initial="return 1", revised="return 2"):
        self.initial = CodeCandidate(code=initial)
        self.revised = CodeCandidate(code=revised)
        self.revisions = 0
        self.calls = 0

    def run(self, **_kwargs):
        self.calls += 1
        return self.initial

    def revise(self, **_kwargs):
        self.revisions += 1
        return self.revised


class Reviewer:
    def __init__(self, decisions):
        self.decisions = list(decisions)

    def run(self, **_kwargs):
        return self.decisions.pop(0)


class Validator:
    def __init__(self, results):
        self.results = list(results)

    def validate(self, **_kwargs):
        return self.results.pop(0)


def coordinator(planner_result, validator_results, review_results):
    return WorkflowCoordinator(
        planner=Planner(planner_result),
        generator=Generator(),
        reviewer=Reviewer(review_results),
        validator=Validator(validator_results),
    )


def test_coordinator_completes_only_after_validation_and_review():
    stages = []
    workflow = coordinator(plan(), [validation()], [ReviewApproved()])

    result = workflow.run(
        prompt="Return value",
        context={"wf": {"vars": {"value": 1}}},
        observe=stages.append,
    )

    assert result.status is WorkflowStatus.COMPLETED
    assert result.code == "return 1"
    assert stages[-1].value == "completed"


def test_coordinator_returns_clarification_without_generation():
    workflow = coordinator(
        ClarificationRequest(question="Which value?", reason="Two inputs are plausible."),
        [],
        [],
    )

    result = workflow.run(prompt="Return it", context={"wf": {"vars": {}}})

    assert result.status is WorkflowStatus.CLARIFICATION_REQUIRED
    assert result.question == "Which value?"
    assert result.code is None


def test_coordinator_revises_once_after_deterministic_failure():
    generator = Generator()
    workflow = WorkflowCoordinator(
        planner=Planner(plan()),
        generator=generator,
        reviewer=Reviewer([ReviewApproved()]),
        validator=Validator([validation(False), validation(True)]),
    )

    result = workflow.run(prompt="Return value", context={"wf": {"vars": {"value": 1}}})

    assert result.status is WorkflowStatus.COMPLETED
    assert result.code == "return 2"
    assert result.revision_count == 1
    assert generator.revisions == 1


def test_coordinator_revises_once_after_reviewer_rejection():
    rejected = ReviewRejected(
        findings=(ReviewFinding(code="wrong_semantics", message="Wrong value."),)
    )
    workflow = coordinator(
        plan(),
        [validation(), validation()],
        [rejected, ReviewApproved()],
    )

    result = workflow.run(prompt="Return value", context={"wf": {"vars": {"value": 1}}})

    assert result.status is WorkflowStatus.COMPLETED
    assert result.revision_count == 1


def test_second_validation_failure_does_not_publish_rejected_code():
    workflow = coordinator(
        plan(),
        [validation(False), validation(False)],
        [],
    )

    result = workflow.run(prompt="Return value", context={"wf": {"vars": {"value": 1}}})

    assert result.status is WorkflowStatus.VALIDATION_FAILED
    assert result.code is None
    assert result.revision_count == 1


def test_second_reviewer_rejection_does_not_publish_rejected_code():
    rejected = ReviewRejected(
        findings=(ReviewFinding(code="wrong_semantics", message="Wrong value."),)
    )
    workflow = coordinator(
        plan(),
        [validation(), validation()],
        [rejected, rejected],
    )

    result = workflow.run(prompt="Return value", context={"wf": {"vars": {"value": 1}}})

    assert result.status is WorkflowStatus.VALIDATION_FAILED
    assert result.code is None


def test_backend_outage_is_typed_and_fail_closed():
    workflow = coordinator(BackendUnavailable(reason="transport_error"), [], [])

    result = workflow.run(prompt="Return value", context=None)

    assert result.status is WorkflowStatus.BACKEND_UNAVAILABLE
    assert result.code is None


def test_invalid_structured_model_output_is_typed_and_fail_closed():
    workflow = coordinator(BackendProtocol(reason="structured_response_invalid"), [], [])

    result = workflow.run(prompt="Return value", context=None)

    assert result.status is WorkflowStatus.VALIDATION_FAILED
    assert result.code is None


def test_oversized_workflow_key_is_inspected_instead_of_rejected():
    workflow = coordinator(plan(), [validation()], [ReviewApproved()])

    result = workflow.run(
        prompt="Return value",
        context={"wf": {"vars": {"value": 1, "x" * 200: 2}}},
    )

    assert result.status is WorkflowStatus.COMPLETED


class BrokenValidator:
    def validate(self, **_kwargs):
        return "not a validation result"


def test_stage_contract_violation_is_not_reported_as_a_bad_request():
    workflow = WorkflowCoordinator(
        planner=Planner(plan()),
        generator=Generator(),
        reviewer=Reviewer([]),
        validator=BrokenValidator(),
    )

    result = workflow.run(prompt="Return value", context={"wf": {"vars": {"value": 1}}})

    assert result.status is WorkflowStatus.VALIDATION_FAILED
    assert [diagnostic.code for diagnostic in result.diagnostics] == ["workflow_contract_violation"]
    assert result.code is None


class RePlanner:
    """Возвращает сначала противоречивый план, потом исправленный."""

    def __init__(self):
        self.findings = None

    def run(self, **kwargs):
        self.findings = kwargs.get("rejected_plan_findings")
        if not self.findings:
            broken = plan()
            return broken.model_copy(
                update={
                    "acceptance_cases": (
                        AcceptanceCase(
                            name="wrapped",
                            context={"wf": {"vars": {"value": 4}}},
                            expected={"result": 4},
                        ),
                    )
                }
            )
        return plan()


def test_self_contradictory_plan_gets_one_correction_before_failing():
    planner = RePlanner()
    workflow = WorkflowCoordinator(
        planner=planner,
        generator=Generator(),
        reviewer=Reviewer([ReviewApproved()]),
        validator=Validator([validation()]),
    )

    result = workflow.run(prompt="Return value", context={"wf": {"vars": {"value": 1}}})

    assert result.status is WorkflowStatus.COMPLETED
    assert planner.findings
    assert any("acceptance_output_contract_mismatch" in item for item in planner.findings)


def test_plan_that_stays_contradictory_fails_without_generating():
    generator = Generator()

    class AlwaysBroken:
        def run(self, **_kwargs):
            broken = plan()
            return broken.model_copy(
                update={
                    "acceptance_cases": (
                        AcceptanceCase(
                            name="wrapped",
                            context={"wf": {"vars": {"value": 4}}},
                            expected={"result": 4},
                        ),
                    )
                }
            )

    workflow = WorkflowCoordinator(
        planner=AlwaysBroken(),
        generator=generator,
        reviewer=Reviewer([]),
        validator=Validator([]),
    )

    result = workflow.run(prompt="Return value", context={"wf": {"vars": {"value": 1}}})

    assert result.status is WorkflowStatus.VALIDATION_FAILED
    assert result.code is None
    assert generator.calls == 0
