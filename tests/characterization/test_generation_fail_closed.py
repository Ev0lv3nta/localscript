from types import SimpleNamespace

import pytest

from app.core.config import get_runtime_profile
from app.core.traces import TraceStore
from app.domain.outcomes import (
    DiagnosticSeverity,
    GenerationStatus,
    ValidationStatus,
)
from app.generation.engine import GenerationEngine
from app.validation.base import ValidationReport


class UnusedBackend:
    pass


def _invalid_report():
    report = ValidationReport()
    report.add(
        validator="dangerous_stdlib",
        level="error",
        code="dangerous_stdlib_os_forbidden",
        message="Access to the os namespace is forbidden.",
    )
    return report


def _chain_result(report, code="return wf.vars.value"):
    return SimpleNamespace(
        status="completed",
        code=code,
        validation_report=report,
        history=[],
        rounds=0,
        rules_applied=[],
        examples_used=[],
        critic_rules_used=[],
        planner={"assumptions": []},
        critic={},
        semantic_checks=[],
    )


def _generate_with_report(monkeypatch, tmp_path, report, code="return wf.vars.value"):
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=UnusedBackend(),
    )
    chain_result = _chain_result(report, code=code)
    repair_result = SimpleNamespace(
        code=chain_result.code,
        validation_report=report,
        history=[],
        rounds=1,
    )
    monkeypatch.setattr(engine.model_chain, "run", lambda **kwargs: chain_result)
    monkeypatch.setattr(engine.repair_loop, "run", lambda **kwargs: repair_result)

    return engine.generate(
        prompt="Верни wf.vars.value.",
        context={"wf": {"vars": {"value": 1}}},
    )


def test_generation_with_validation_errors_is_not_completed(monkeypatch, tmp_path):
    code = 'return os.execute("echo blocked")'
    result = _generate_with_report(
        monkeypatch,
        tmp_path,
        report=_invalid_report(),
        code=code,
    )

    assert result.outcome.status is GenerationStatus.VALIDATION_FAILED
    assert result.outcome.validation.status is ValidationStatus.FAILED
    assert result.outcome.code is None
    assert len(result.outcome.validation.findings) == 1
    finding = result.outcome.validation.findings[0]
    assert finding.stage == "dangerous_stdlib"
    assert finding.severity is DiagnosticSeverity.ERROR
    assert finding.code == "dangerous_stdlib_os_forbidden"
    assert finding.message == "Access to the os namespace is forbidden."

    assert result.status == "completed"
    assert result.code == code


@pytest.mark.parametrize(
    "runtime_code",
    ["lua_runtime_missing", "semantic_runtime_missing"],
)
def test_generation_without_required_runtime_is_incomplete(
    monkeypatch,
    tmp_path,
    runtime_code,
):
    report = ValidationReport()
    report.add(
        validator="runtime",
        level="warning",
        code=runtime_code,
        message="Required runtime is unavailable.",
    )

    result = _generate_with_report(monkeypatch, tmp_path, report=report)

    assert result.outcome.status is GenerationStatus.VALIDATION_FAILED
    assert result.outcome.validation.status is ValidationStatus.INCOMPLETE
    assert result.outcome.code is None
    assert result.outcome.validation.findings[0].severity is DiagnosticSeverity.WARNING

    assert result.status == "degraded_completed"
    assert result.code == "return wf.vars.value"


def test_generation_with_passed_validation_publishes_typed_code(monkeypatch, tmp_path):
    result = _generate_with_report(
        monkeypatch,
        tmp_path,
        report=ValidationReport(),
    )

    assert result.outcome.status is GenerationStatus.COMPLETED
    assert result.outcome.validation.status is ValidationStatus.PASSED
    assert result.outcome.code == "return wf.vars.value"
