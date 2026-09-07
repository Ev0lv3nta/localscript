import pytest

from app.core import budgets
from app.generation.backend_errors import BackendTimeout
from app.workflow.roles import CODE_RESPONSE, StructuredModelClient


def test_model_response_after_overall_deadline_is_not_accepted(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(budgets, "monotonic", lambda: now[0])

    def late_completion(*_args, **_kwargs):
        now[0] = 181.0
        return '{"code":"return 1"}'

    with budgets.workflow_budget(), pytest.raises(BackendTimeout) as error:
        StructuredModelClient(late_completion).request("Write code", CODE_RESPONSE)
    assert error.value.reason == "workflow_deadline_exceeded"


def test_nested_validation_budget_cannot_extend_request_deadline(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(budgets, "monotonic", lambda: now[0])
    with budgets.workflow_budget(10):
        now[0] = 9
        with budgets.workflow_budget(20):
            assert budgets.remaining_seconds(5) == 1
