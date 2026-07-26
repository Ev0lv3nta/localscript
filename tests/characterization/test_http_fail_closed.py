import json

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_runtime_profile
from app.core.traces import TraceStore
from app.domain.outcomes import (
    GenerationOutcome,
    GenerationStatus,
    ValidationOutcome,
    ValidationStatus,
)
from app.generation.engine import GenerationResult
from app.main import create_app


class DangerousBackend:
    def complete(self, prompt, response_format=None, model=None):
        if "You are the planner for a LocalScript/Lua generation pipeline." in prompt:
            return json.dumps(
                {
                    "family": "generic_lua",
                    "root": "wf.vars",
                    "source_paths": ["wf.vars.value"],
                    "return_shape": "scalar",
                    "constraints": [],
                    "assumptions": [],
                    "clarification_needed": False,
                    "clarification_question": "",
                    "semantic_checks": [],
                }
            )
        if "You are the critic for a LocalScript/Lua generation pipeline." in prompt:
            return json.dumps(
                {"repairable": False, "issues": [], "minimal_actions": []}
            )
        return 'return os.execute("echo blocked")'

    def generate(self, prompt, context=None):
        return self.complete(prompt)


def _make_client(tmp_path):
    app = create_app(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=DangerousBackend(),
    )
    return TestClient(app)


def _transition_result(outcome, legacy_code=""):
    return GenerationResult(
        code=legacy_code,
        trace_id="trace-transition",
        session_id="session-transition",
        strategy="ollama_chain",
        verification_errors=["legacy_error"],
        validation_report={
            "has_errors": True,
            "has_warnings": False,
            "messages": [
                {
                    "validator": "legacy",
                    "level": "error",
                    "code": "legacy_error",
                    "message": "Legacy validation error.",
                }
            ],
        },
        repair_rounds=0,
        degraded_mode=False,
        status="degraded_completed",
        assumptions=[],
        session_summary={
            "session_id": "session-transition",
            "status": "degraded_completed",
        },
        outcome=outcome,
    )


def test_rich_generate_does_not_publish_rejected_candidate(tmp_path):
    client = _make_client(tmp_path)
    response = client.post(
        "/api/generate",
        json={
            "prompt": "Верни wf.vars.value.",
            "context": {"wf": {"vars": {"value": 1}}},
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == GenerationStatus.VALIDATION_FAILED.value
    assert response.json()["code"] is None
    assert response.json()["validation"]["status"] == "failed"
    session = client.get(
        "/api/sessions/{0}".format(response.json()["session_id"])
    )
    assert session.status_code == 200
    assert session.json()["status"] == "validation_failed"


def test_compat_generate_rejects_invalid_candidate(tmp_path):
    response = _make_client(tmp_path).post(
        "/generate",
        json={
            "prompt": "Верни wf.vars.value.",
            "context": {"wf": {"vars": {"value": 1}}},
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "validation_failed"
    assert "os.execute" not in response.text


@pytest.mark.parametrize("code", ["[]", "null", "42", '"text"'])
def test_validate_skips_execution_after_structural_error(monkeypatch, tmp_path, code):
    def unexpected_execution(**kwargs):
        raise AssertionError("semantic execution must not run after structural failure")

    monkeypatch.setattr("app.api.routes.execute_output", unexpected_execution)
    response = _make_client(tmp_path).post(
        "/api/validate",
        json={"code": code, "output_style": "json_envelope"},
    )

    assert response.status_code == 200
    assert response.json()["ok"] is False


@pytest.mark.parametrize(
    ("path", "method_name"),
    [
        ("/generate", "generate"),
        ("/api/generate", "generate_rich"),
    ],
)
def test_typed_backend_unavailable_outcome_returns_503(
    monkeypatch,
    tmp_path,
    path,
    method_name,
):
    client = _make_client(tmp_path)
    result = _transition_result(
        GenerationOutcome(
            status=GenerationStatus.BACKEND_UNAVAILABLE,
            validation=ValidationOutcome(status=ValidationStatus.NOT_RUN),
        )
    )
    monkeypatch.setattr(
        client.app.state.engine,
        method_name,
        lambda **kwargs: result,
    )

    response = client.post(path, json={"prompt": "Верни wf.vars.value."})

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "backend_unavailable"
    assert response.headers["X-Trace-Id"] == "trace-transition"


def test_rich_response_uses_typed_outcome_instead_of_legacy_fields(
    monkeypatch,
    tmp_path,
):
    client = _make_client(tmp_path)
    result = _transition_result(
        GenerationOutcome(
            status=GenerationStatus.COMPLETED,
            validation=ValidationOutcome(status=ValidationStatus.PASSED),
            code="return 1",
        ),
        legacy_code='return os.execute("echo legacy")',
    )
    monkeypatch.setattr(
        client.app.state.engine,
        "generate_rich",
        lambda **kwargs: result,
    )

    response = client.post(
        "/api/generate",
        json={"prompt": "Верни единицу."},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["code"] == "return 1"
    assert response.json()["validation"] == {
        "status": "passed",
        "ok": True,
        "errors": [],
        "degraded_mode": False,
        "repair_rounds": 0,
        "messages": [],
    }
    assert "os.execute" not in response.text


def test_session_get_infers_failure_from_persisted_validation_report(tmp_path):
    client = _make_client(tmp_path)
    client.app.state.session_store.write(
        "persisted-failure",
        {
            "session_id": "persisted-failure",
            "status": "completed",
            "previous_validation_report": {
                "has_errors": True,
                "has_warnings": False,
                "messages": [],
            },
        },
    )

    response = client.get("/api/sessions/persisted-failure")

    assert response.status_code == 200
    assert response.json()["status"] == "validation_failed"
