from dataclasses import dataclass

from app.workflow.contracts import (
    AcceptanceCase,
    CodeCandidate,
    OutputContract,
    OutputFormat,
    OutputShape,
    PlanStep,
    TaskPlan,
    WorkflowPath,
    WorkflowRoot,
)
from app.workflow.validation import DeterministicCandidateValidator


@dataclass(frozen=True)
class Finding:
    code: str
    message: str


@dataclass(frozen=True)
class Policy:
    findings: tuple[Finding, ...] = ()


@dataclass(frozen=True)
class Execution:
    ok: bool
    value: object = None
    error_code: str = ""
    error_message: str = ""
    degraded: bool = False


def build_plan(
    *,
    output_format: OutputFormat = OutputFormat.LUA_BLOCK,
    shape: OutputShape = OutputShape.SCALAR,
    expected: object = 4,
) -> TaskPlan:
    source = WorkflowPath(root=WorkflowRoot.VARS, segments=("value",))
    return TaskPlan(
        objective="Return the value.",
        inputs=(source,),
        output=OutputContract(format=output_format, shape=shape),
        steps=(PlanStep(description="Return the value.", reads=(source,)),),
        acceptance_cases=(
            AcceptanceCase(
                name="value",
                context={"wf": {"vars": {"value": 4}}},
                expected=expected,
            ),
        ),
    )


def test_validator_runs_ast_luac_and_acceptance_case(monkeypatch):
    monkeypatch.setattr(
        "app.workflow.validation.subprocess.run",
        lambda *_args, **_kwargs: type("Completed", (), {"returncode": 0})(),
    )
    validator = DeterministicCandidateValidator(
        policy_analyzer=lambda _code, _style: Policy(),
        runtime_executor=lambda _code, _context, _style: Execution(ok=True, value=4),
        luac_locator=lambda: "/usr/bin/luac",
    )

    result = validator.validate(candidate=CodeCandidate(code="return 4"), plan=build_plan())

    assert result.ok is True
    assert [check.name for check in result.checks] == [
        "output_contract",
        "ast_policy",
        "luac",
        "acceptance:value",
    ]
    assert result.observations == ({"case": "value", "actual": 4},)


def test_validator_never_executes_candidate_rejected_by_ast():
    executed = False

    def runtime(_code, _context, _style):
        nonlocal executed
        executed = True
        return Execution(ok=True, value=4)

    validator = DeterministicCandidateValidator(
        policy_analyzer=lambda _code, _style: Policy(
            findings=(Finding("dangerous_stdlib_os_forbidden", "os is forbidden"),)
        ),
        runtime_executor=runtime,
        luac_locator=lambda: "/usr/bin/luac",
    )

    result = validator.validate(
        candidate=CodeCandidate(code='return os.execute("id")'),
        plan=build_plan(),
    )

    assert result.ok is False
    assert result.checks[-1].code == "dangerous_stdlib_os_forbidden"
    assert executed is False


def test_validator_fails_closed_when_luac_is_unavailable():
    validator = DeterministicCandidateValidator(
        policy_analyzer=lambda _code, _style: Policy(),
        runtime_executor=lambda _code, _context, _style: Execution(ok=True, value=4),
        luac_locator=lambda: None,
    )

    result = validator.validate(candidate=CodeCandidate(code="return 4"), plan=build_plan())

    assert result.ok is False
    assert result.checks[-1].code == "luac_runtime_missing"
    assert result.observations == ()


def test_validator_compares_json_results_structurally(monkeypatch):
    monkeypatch.setattr(
        "app.workflow.validation.subprocess.run",
        lambda *_args, **_kwargs: type("Completed", (), {"returncode": 0})(),
    )
    validator = DeterministicCandidateValidator(
        policy_analyzer=lambda _code, _style: Policy(),
        runtime_executor=lambda _code, _context, _style: Execution(
            ok=True,
            value={"result": [1, 2]},
        ),
        luac_locator=lambda: "/usr/bin/luac",
    )
    plan = build_plan(
        output_format=OutputFormat.JSON_ENVELOPE,
        shape=OutputShape.OBJECT,
        expected={"result": [1, 3]},
    )

    result = validator.validate(
        candidate=CodeCandidate(code='{"result":"lua{return {1, 2}}lua"}'),
        plan=plan,
    )

    assert result.ok is False
    assert result.checks[-1].code == "acceptance_result_mismatch"


def test_existing_code_validation_executes_without_semantic_oracle(monkeypatch):
    monkeypatch.setattr(
        "app.workflow.validation.subprocess.run",
        lambda *_args, **_kwargs: type("Completed", (), {"returncode": 0})(),
    )
    validator = DeterministicCandidateValidator(
        policy_analyzer=lambda _code, _style: Policy(),
        runtime_executor=lambda _code, _context, _style: Execution(ok=True, value=[1, 2]),
        luac_locator=lambda: "/usr/bin/luac",
    )

    result = validator.validate_existing(
        candidate=CodeCandidate(code="return {1, 2}"),
        output=OutputContract(
            format=OutputFormat.LUA_BLOCK,
            shape=OutputShape.ARRAY,
        ),
        context={"wf": {"vars": {}}},
    )

    assert result.ok is True
    assert result.observations == ({"actual": [1, 2]},)
