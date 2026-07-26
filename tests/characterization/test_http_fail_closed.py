import json

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_runtime_profile
from app.core.traces import TraceStore
from app.domain.outcomes import GenerationStatus
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


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="rich generation still publishes a candidate rejected by validation",
)
def test_rich_generate_does_not_publish_rejected_candidate(tmp_path):
    response = _make_client(tmp_path).post(
        "/api/generate",
        json={
            "prompt": "Верни wf.vars.value.",
            "context": {"wf": {"vars": {"value": 1}}},
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == GenerationStatus.VALIDATION_FAILED.value
    assert response.json()["code"] is None


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="compatibility generation still returns rejected code with HTTP 200",
)
def test_compat_generate_rejects_invalid_candidate(tmp_path):
    response = _make_client(tmp_path).post(
        "/generate",
        json={
            "prompt": "Верни wf.vars.value.",
            "context": {"wf": {"vars": {"value": 1}}},
        },
    )

    assert response.status_code == 422


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="validation endpoint raises for non-object JSON envelopes",
)
@pytest.mark.parametrize("code", ["[]", "null", "42", '"text"'])
def test_validate_never_returns_500_for_json_shaped_code(tmp_path, code):
    response = _make_client(tmp_path).post(
        "/api/validate",
        json={"code": code, "output_style": "json_envelope"},
    )

    assert response.status_code == 200
    assert response.json()["ok"] is False
