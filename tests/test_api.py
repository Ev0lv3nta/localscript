from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_runtime_profile
from app.core.traces import TraceStore
from app.generation.backend_errors import BackendBusy, BackendModel, BackendTimeout
from app.main import create_app
from app.workflow.contracts import CheckStatus, ValidationCheck, ValidationResult
from tests.support_backends import DeterministicTestBackend, UnavailableBackend


class PassingValidator:
    def validate(self, **_kwargs):
        return ValidationResult(
            checks=(ValidationCheck(name="all", status=CheckStatus.PASSED),),
            observations=({"case": "value", "actual": 4},),
        )


class RejectingValidator:
    def validate(self, **_kwargs):
        return ValidationResult(
            checks=(
                ValidationCheck(
                    name="ast_policy",
                    status=CheckStatus.FAILED,
                    code="dangerous_stdlib_os_forbidden",
                    message="os is forbidden",
                ),
            )
        )


def make_client(tmp_path, backend=None, validator=None):
    app = create_app(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=backend or DeterministicTestBackend(),
    )
    app.state.engine.workflow.validator = validator or PassingValidator()
    return TestClient(app), app


def test_health_endpoint(tmp_path):
    client, _ = make_client(tmp_path)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "profile": "competition"}


def test_ready_endpoint_reports_runtime_and_backend(tmp_path, monkeypatch):
    monkeypatch.setattr("app.api.routes.find_lua_binary", lambda: "/tmp/lua")
    monkeypatch.setattr("app.api.routes.find_luac_binary", lambda: "/tmp/luac")

    client, _ = make_client(tmp_path)
    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["backend_reachable"] is True
    assert response.json()["checks"]["lua_runtime_present"] is True


def test_generate_endpoint_publishes_only_validated_code_and_sanitized_trace(tmp_path):
    client, app = make_client(tmp_path)

    response = client.post(
        "/api/generate",
        json={
            "prompt": "Return the value.",
            "context": {"wf": {"vars": {"value": 4, "secret": "not-in-trace"}}},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["code"] == "return wf.vars.value"
    assert "question" not in body
    trace = app.state.trace_store.read(response.headers["X-Trace-Id"])
    assert trace["diagnostic_codes"] == []
    assert "not-in-trace" not in str(trace)
    assert len(list(Path(app.state.trace_store.root).glob("**/*.json"))) == 1


def test_generate_endpoint_never_publishes_rejected_candidate(tmp_path):
    client, _ = make_client(tmp_path, validator=RejectingValidator())

    response = client.post(
        "/api/generate",
        json={"prompt": "Return the value.", "context": {"wf": {"vars": {"value": 4}}}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "policy_rejected"
    assert "code" not in body
    assert "return wf.vars.value" not in response.text
    assert "dangerous_stdlib_os_forbidden" in [item["code"] for item in body["diagnostics"]]


def test_generate_endpoint_maps_backend_outage_without_internal_reason(tmp_path):
    client, _ = make_client(tmp_path, backend=UnavailableBackend())

    response = client.post(
        "/api/generate",
        json={"prompt": "Return the value.", "context": {"wf": {"vars": {"value": 4}}}},
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "backend_unavailable"
    assert "test_backend_unavailable" not in response.text


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (BackendTimeout(), 504),
        (BackendModel(), 503),
        (BackendBusy(), 429),
    ],
)
def test_generate_endpoint_maps_typed_backend_errors(tmp_path, error, expected_status):
    client, app = make_client(tmp_path)

    class FailingEngine:
        def generate(self, **_kwargs):
            raise error

    app.state.engine = FailingEngine()
    response = client.post("/api/generate", json={"prompt": "Return one."})

    assert response.status_code == expected_status
    assert response.json()["detail"]["code"] == error.code
    if isinstance(error, BackendBusy):
        assert response.headers["Retry-After"] == str(error.retry_after_seconds)


def test_generate_endpoint_requires_a_prompt_or_a_session(tmp_path):
    client, _ = make_client(tmp_path)

    response = client.post("/api/generate", json={"context": {"wf": {"vars": {}}}})

    assert response.status_code == 422


def test_generate_endpoint_rejects_empty_continuation_text(tmp_path):
    client, _ = make_client(tmp_path)

    for field in ("feedback", "clarification_answer"):
        response = client.post(
            "/api/generate",
            json={"session_id": "12345678", field: "   \n"},
        )

        assert response.status_code == 422


def test_generate_endpoint_maps_unknown_valid_session_to_not_found(tmp_path):
    client, _ = make_client(tmp_path)

    response = client.post(
        "/api/generate",
        json={"session_id": str(uuid4()), "clarification_answer": "Use wf.vars."},
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "session_not_found"


def test_generate_endpoint_maps_prompt_replacement_to_conflict(tmp_path):
    client, _ = make_client(tmp_path)
    created = client.post(
        "/api/generate",
        json={"prompt": "Return one.", "context": {"wf": {"vars": {"value": 1}}}},
    )

    response = client.post(
        "/api/generate",
        json={"session_id": created.json()["session_id"], "prompt": "Return two."},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "session_conflict"
