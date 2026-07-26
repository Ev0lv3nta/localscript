from types import SimpleNamespace

import pytest

from app.core.config import get_runtime_profile
from app.core.traces import TraceStore
from app.domain.outcomes import GenerationStatus
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


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="generation core still reports validation errors as success",
)
def test_generation_with_validation_errors_is_not_completed(monkeypatch, tmp_path):
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=UnusedBackend(),
    )
    report = _invalid_report()
    chain_result = SimpleNamespace(
        status="completed",
        code='return os.execute("echo blocked")',
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
    repair_result = SimpleNamespace(
        code=chain_result.code,
        validation_report=report,
        history=[],
        rounds=1,
    )
    monkeypatch.setattr(engine.model_chain, "run", lambda **kwargs: chain_result)
    monkeypatch.setattr(engine.repair_loop, "run", lambda **kwargs: repair_result)

    result = engine.generate(
        prompt="Верни wf.vars.value.",
        context={"wf": {"vars": {"value": 1}}},
    )

    assert result.status == GenerationStatus.VALIDATION_FAILED.value
    assert result.code == ""
