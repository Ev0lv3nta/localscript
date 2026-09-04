from fastapi.testclient import TestClient

from app.core.config import get_runtime_profile
from app.core.traces import TraceStore
from app.main import create_app
from app.workflow.contracts import CheckStatus, ValidationCheck, ValidationResult
from tests.support_backends import DeterministicTestBackend


def _make_client(tmp_path):
    app = create_app(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=DeterministicTestBackend(),
    )
    class PassingValidator:
        def validate(self, **_kwargs):
            return ValidationResult(
                checks=(ValidationCheck(name="all", status=CheckStatus.PASSED),)
            )

    app.state.engine.workflow.validator = PassingValidator()
    return TestClient(app)


def test_trace_endpoint_returns_sanitized_trace_payload(tmp_path):
    client = _make_client(tmp_path)
    response = client.post(
        "/api/generate",
        json={
            "prompt": "Return the value.",
            "context": {"wf": {"vars": {"value": 4}}},
        },
    )

    trace = client.get("/api/traces/{}".format(response.headers["X-Trace-Id"]))

    assert trace.status_code == 200
    payload = trace.json()
    assert payload["trace_id"] == response.headers["X-Trace-Id"]
    assert payload["session_id"] == response.headers["X-Session-Id"]
    assert payload["status"] == "completed"
    assert payload["diagnostic_codes"] == []
    assert next(event["stage"] for event in payload["stage_events"]) == "received"
    assert set(payload) == {
        "trace_id",
        "session_id",
        "status",
        "model",
        "revision_count",
        "diagnostic_codes",
        "stage_events",
    }


def test_trace_endpoint_rejects_invalid_trace_id(tmp_path):
    client = _make_client(tmp_path)

    response = client.get("/api/traces/not valid")

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_trace_id"
