import json

from fastapi.testclient import TestClient

from app.core.config import get_runtime_profile
from app.core.traces import TraceStore
from app.main import create_app
from app.workflow.contracts import CheckStatus, ValidationCheck, ValidationResult


class ClarificationBackend:
    def complete(self, prompt, response_format=None, model=None):
        if "You are the planner" in prompt:
            if '"clarification_answer": null' in prompt:
                return json.dumps(
                    {
                        "kind": "clarification",
                        "question": "Which workflow root should be used?",
                        "reason": "Both roots contain the requested value.",
                    }
                )
            return json.dumps(
                {
                    "kind": "plan",
                    "objective": "Return the selected value.",
                    "inputs": [{"root": "wf.vars", "segments": ["value"]}],
                    "output": {
                        "format": "lua_block",
                        "shape": "scalar",
                        "nullable": False,
                    },
                    "steps": [
                        {
                            "description": "Return the value.",
                            "reads": [{"root": "wf.vars", "segments": ["value"]}],
                        }
                    ],
                    "constraints": [],
                    "acceptance_cases": [
                        {
                            "name": "value",
                            "context": {"wf": {"vars": {"value": 4}}},
                            "expected": 4,
                        }
                    ],
                }
            )
        if "You are the generator" in prompt:
            return json.dumps({"code": "return wf.vars.value"})
        if "You are the reviewer" in prompt:
            return json.dumps({"kind": "approved"})
        raise AssertionError("unexpected role")

    def close(self):
        return None


class PassingValidator:
    def validate(self, **_kwargs):
        return ValidationResult(
            checks=(ValidationCheck(name="all", status=CheckStatus.PASSED),),
            observations=({"case": "value", "actual": 4},),
        )


def make_client(tmp_path):
    app = create_app(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=ClarificationBackend(),
    )
    app.state.engine.workflow.validator = PassingValidator()
    return TestClient(app)


def test_rich_api_persists_and_resumes_one_clarification(tmp_path):
    client = make_client(tmp_path)
    first = client.post(
        "/api/generate",
        json={
            "prompt": "Return value.",
            "context": {
                "wf": {
                    "vars": {"value": 4},
                    "initVariables": {"value": 5},
                }
            },
        },
    )

    assert first.status_code == 200
    assert first.json()["status"] == "clarification_required"
    assert first.json()["code"] is None
    session_id = first.json()["session_id"]

    continued = client.post(
        "/api/generate",
        json={
            "session_id": session_id,
            "clarification_answer": "Use wf.vars.",
        },
    )

    assert continued.status_code == 200
    assert continued.json()["status"] == "completed"
    assert continued.json()["code"] == "return wf.vars.value"
    session = client.get(f"/api/sessions/{session_id}").json()
    assert session["clarification_history"] == [
        {
            "question": "Which workflow root should be used?",
            "answer": "Use wf.vars.",
        }
    ]
