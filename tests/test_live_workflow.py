"""Integration tests against a live Ollama backend.

Unit tests cover the workflow with scripted backends; they cannot see the boundary where a
Pydantic JSON Schema becomes a sampling grammar. Every schema defect this project has hit lived
exactly there, so the release gate runs these separately with `LOCALSCRIPT_REQUIRE_LIVE=1`.
"""

import pytest

from app.core.config import get_runtime_profile
from app.core.traces import TraceStore
from app.generation.engine import GenerationEngine
from app.workflow.context import ContextInspector
from app.workflow.contracts import ClarificationRequest, TaskPlan, WorkflowStatus
from app.workflow.roles import PlannerRole, StructuredModelClient

pytestmark = pytest.mark.integration


@pytest.fixture
def engine(live_ollama_backend, tmp_path):
    return GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=live_ollama_backend,
    )


def test_planner_schema_survives_the_sampling_grammar(live_ollama_backend):
    """The model must be able to emit every branch the planner schema declares."""
    context = {"wf": {"vars": {"emails": ["a@example.com", "b@example.com"]}}}
    inspector = ContextInspector()
    inventory = inspector.inventory(context)
    planner = PlannerRole(StructuredModelClient(live_ollama_backend.complete))

    decision = planner.run(
        prompt="Верни последний адрес из массива wf.vars.emails.",
        context_sample=inspector.sample(context),
        inventory=inventory,
    )

    assert isinstance(decision, (TaskPlan, ClarificationRequest))
    if isinstance(decision, TaskPlan):
        assert decision.acceptance_cases
        assert decision.inputs


def test_live_generation_never_publishes_unverified_code(engine):
    """Fail-closed is a property of the result object, not of a happy path."""
    result = engine.generate(
        prompt="Верни последний адрес из массива wf.vars.emails.",
        context={"wf": {"vars": {"emails": ["a@example.com", "b@example.com"]}}},
    )

    workflow = result.workflow
    if workflow.status is WorkflowStatus.COMPLETED:
        assert workflow.code
        assert workflow.question is None
    else:
        assert workflow.code is None


def test_live_ambiguous_root_is_asked_about_not_guessed(engine):
    """The same path under both roots is the one ambiguity the planner must not resolve alone."""
    result = engine.generate(
        prompt="Нормализуй email и верни его в lower-case.",
        context={
            "wf": {
                "vars": {"email": "A@EXAMPLE.COM"},
                "initVariables": {"email": "B@EXAMPLE.COM"},
            }
        },
    )

    assert result.workflow.status is WorkflowStatus.CLARIFICATION_REQUIRED
    assert result.workflow.question
    assert result.workflow.code is None
