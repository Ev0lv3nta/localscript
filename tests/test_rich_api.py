import json

from fastapi.testclient import TestClient

from app.core.config import get_runtime_profile
from app.core.traces import TraceStore
from app.main import create_app
from tests.support_backends import UnavailableBackend


class ClarificationBackend:
    def complete(self, prompt, response_format=None, model=None):
        if "You are the planner for a LocalScript/Lua generation pipeline." in prompt:
            if "Use wf.vars for email root." in prompt:
                return json.dumps(
                    {
                        "family": "generic_lua",
                        "root": "wf.vars",
                        "source_paths": ["wf.vars.email"],
                        "return_shape": "scalar",
                        "constraints": ["Do not use JsonPath"],
                        "assumptions": ["User clarified to use wf.vars."],
                        "clarification_needed": False,
                        "clarification_question": "",
                        "semantic_checks": ["must return lower-case wf.vars.email"],
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "family": "generic_lua",
                    "root": "unknown_mixed",
                    "source_paths": [],
                    "return_shape": "scalar",
                    "constraints": ["Do not use JsonPath"],
                    "assumptions": [],
                    "clarification_needed": True,
                    "clarification_question": "Use wf.vars or wf.initVariables for email root?",
                    "semantic_checks": [],
                },
                ensure_ascii=False,
            )
        if "You are the critic for a LocalScript/Lua generation pipeline." in prompt:
            return json.dumps(
                {"repairable": False, "issues": [], "minimal_actions": []},
                ensure_ascii=False,
            )
        return 'local value = wf.vars.email or ""\nvalue = string.gsub(value, "^%s*(.-)%s*$", "%1")\nreturn string.lower(value)'

    def generate(self, prompt, context=None):
        return self.complete(prompt)


class ForceModelBackend:
    def complete(self, prompt, response_format=None, model=None):
        if "You are the planner for a LocalScript/Lua generation pipeline." in prompt:
            return json.dumps(
                {
                    "family": "generic_lua",
                    "root": "wf.vars",
                    "source_paths": ["wf.vars.emails"],
                    "return_shape": "scalar",
                    "constraints": ["Do not use JsonPath"],
                    "assumptions": [],
                    "clarification_needed": False,
                    "clarification_question": "",
                    "semantic_checks": [],
                },
                ensure_ascii=False,
            )
        if "You are the critic for a LocalScript/Lua generation pipeline." in prompt:
            return json.dumps({"repairable": False, "issues": [], "minimal_actions": []}, ensure_ascii=False)
        return "return wf.vars.emails[#wf.vars.emails]"

    def generate(self, prompt, context=None):
        return self.complete(prompt)


def _make_client(tmp_path, backend):
    app = create_app(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=backend,
    )
    return TestClient(app)


def test_rich_api_requires_prompt_or_session_id(tmp_path):
    client = _make_client(tmp_path, ClarificationBackend())

    response = client.post("/api/generate", json={})

    assert response.status_code == 400


def test_rich_api_returns_clarification_needed_and_persists_session(tmp_path):
    client = _make_client(tmp_path, ClarificationBackend())

    response = client.post(
        "/api/generate",
        json={
            "prompt": "Нормализуй email и верни его в lower-case.",
            "context": {
                "wf": {
                    "vars": {"email": "A@EXAMPLE.COM"},
                    "initVariables": {"email": "B@EXAMPLE.COM"},
                }
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "clarification_needed"
    assert payload["question"] == "Use wf.vars or wf.initVariables for email root?"
    assert payload["code"] is None
    assert payload["session"]["open_clarification_question"] == payload["question"]


def test_rich_api_replays_existing_clarification_state_until_answer(tmp_path):
    client = _make_client(tmp_path, ClarificationBackend())
    first = client.post(
        "/api/generate",
        json={
            "prompt": "Нормализуй email и верни его в lower-case.",
            "context": {
                "wf": {
                    "vars": {"email": "A@EXAMPLE.COM"},
                    "initVariables": {"email": "B@EXAMPLE.COM"},
                }
            },
        },
    )
    session_id = first.json()["session_id"]

    second = client.post("/api/generate", json={"session_id": session_id})

    assert second.status_code == 200
    assert second.json()["status"] == "clarification_needed"
    assert second.json()["question"] == first.json()["question"]


def test_rich_api_continues_after_clarification_answer(tmp_path):
    client = _make_client(tmp_path, ClarificationBackend())
    first = client.post(
        "/api/generate",
        json={
            "prompt": "Нормализуй email и верни его в lower-case.",
            "context": {
                "wf": {
                    "vars": {"email": "A@EXAMPLE.COM"},
                    "initVariables": {"email": "B@EXAMPLE.COM"},
                }
            },
        },
    )

    session_id = first.json()["session_id"]
    continued = client.post(
        "/api/generate",
        json={
            "session_id": session_id,
            "clarification_answer": "Use wf.vars for email root.",
        },
    )

    assert continued.status_code == 200
    payload = continued.json()
    assert payload["status"] == "completed"
    assert payload["code"] == 'local value = wf.vars.email or ""\nvalue = string.gsub(value, "^%s*(.-)%s*$", "%1")\nreturn string.lower(value)'
    assert payload["strategy"] == "ollama_chain"
    assert payload["validation"]["ok"] is True
    assert payload["session"]["open_clarification_question"] is None


def test_get_session_returns_clarification_history_after_answer(tmp_path):
    client = _make_client(tmp_path, ClarificationBackend())
    first = client.post(
        "/api/generate",
        json={
            "prompt": "Нормализуй email и верни его в lower-case.",
            "context": {
                "wf": {
                    "vars": {"email": "A@EXAMPLE.COM"},
                    "initVariables": {"email": "B@EXAMPLE.COM"},
                }
            },
        },
    )
    session_id = first.json()["session_id"]
    client.post(
        "/api/generate",
        json={
            "session_id": session_id,
            "clarification_answer": "Use wf.vars for email root.",
        },
    )

    session = client.get("/api/sessions/{0}".format(session_id))

    assert session.status_code == 200
    assert session.json()["clarification_history"] == [
        {
            "question": "Use wf.vars or wf.initVariables for email root?",
            "answer": "Use wf.vars for email root.",
        }
    ]


def test_get_unknown_session_returns_404(tmp_path):
    client = _make_client(tmp_path, ClarificationBackend())

    response = client.get("/api/sessions/missing-session")

    assert response.status_code == 404


def test_analyze_endpoint_returns_routing_and_context_summary(tmp_path):
    client = _make_client(tmp_path, ClarificationBackend())

    response = client.post(
        "/api/analyze",
        json={
            "prompt": "Нормализуй email и верни его в lower-case.",
            "context": {
                "wf": {
                    "vars": {"email": "A@EXAMPLE.COM"},
                    "initVariables": {"email": "B@EXAMPLE.COM"},
                }
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["suggested_strategy"] == "clarification"
    assert payload["clarification_question"] == "Use wf.vars or wf.initVariables for email root?"
    assert "wf.vars.email" in payload["available_paths"]
    assert "wf.initVariables.email" in payload["available_paths"]


def test_analyze_endpoint_defaults_to_agent_strategy_for_public_prompt(tmp_path):
    client = _make_client(tmp_path, ForceModelBackend())

    response = client.post(
        "/api/analyze",
        json={
            "prompt": "Верни последний адрес из массива wf.vars.emails; если массив пустой, верни nil.",
            "context": {"wf": {"vars": {"emails": ["a@example.com", "b@example.com"]}}},
        },
    )

    assert response.status_code == 200
    assert response.json()["suggested_strategy"] == "ollama_chain"


def test_minimal_generate_keeps_backward_compatible_contract_on_ambiguous_task(tmp_path):
    client = _make_client(tmp_path, ClarificationBackend())

    response = client.post(
        "/generate",
        json={
            "prompt": "Нормализуй email и верни его в lower-case.",
            "context": {
                "wf": {
                    "vars": {"email": "A@EXAMPLE.COM"},
                    "initVariables": {"email": "B@EXAMPLE.COM"},
                }
            },
        },
    )

    assert response.status_code == 200
    assert list(response.json().keys()) == ["code"]
    assert response.json()["code"] == 'local value = wf.vars.email or ""\nvalue = string.gsub(value, "^%s*(.-)%s*$", "%1")\nreturn string.lower(value)'
    assert response.headers["X-Clarification-Suggested"] == "true"
    assert response.headers["X-Assumption-Risk"] == "high"


def test_rich_api_returns_503_when_backend_unavailable(tmp_path):
    client = _make_client(tmp_path, UnavailableBackend())

    response = client.post(
        "/api/generate",
        json={"prompt": "Сделай что-нибудь нестандартное без готового шаблона"},
    )

    assert response.status_code == 503
    payload = response.json()
    assert payload["detail"]["code"] == "backend_unavailable"


def test_rich_api_completed_response_contains_validation_summary(tmp_path):
    client = _make_client(tmp_path, ClarificationBackend())
    first = client.post(
        "/api/generate",
        json={
            "prompt": "Нормализуй email и верни его в lower-case.",
            "context": {
                "wf": {
                    "vars": {"email": "A@EXAMPLE.COM"},
                    "initVariables": {"email": "B@EXAMPLE.COM"},
                }
            },
        },
    )

    completed = client.post(
        "/api/generate",
        json={
            "session_id": first.json()["session_id"],
            "clarification_answer": "Use wf.vars for email root.",
        },
    )

    assert completed.status_code == 200
    assert completed.json()["validation"] == {
        "ok": True,
        "errors": [],
        "degraded_mode": False,
        "repair_rounds": 0,
        "messages": [],
    }


def test_rich_api_records_feedback_history_in_session(tmp_path):
    client = _make_client(tmp_path, ClarificationBackend())
    first = client.post(
        "/api/generate",
        json={
            "prompt": "Нормализуй email и верни его в lower-case.",
            "context": {
                "wf": {
                    "vars": {"email": "A@EXAMPLE.COM"},
                    "initVariables": {"email": "B@EXAMPLE.COM"},
                }
            },
        },
    )
    session_id = first.json()["session_id"]
    client.post(
        "/api/generate",
        json={
            "session_id": session_id,
            "clarification_answer": "Use wf.vars for email root.",
        },
    )
    client.post(
        "/api/generate",
        json={
            "session_id": session_id,
            "feedback": "Keep using wf.vars and lower-case the result.",
        },
    )

    session = client.get("/api/sessions/{0}".format(session_id))

    assert session.status_code == 200
    assert session.json()["feedback_history"] == ["Keep using wf.vars and lower-case the result."]


def test_rich_api_uses_agent_chain_for_public_prompt_without_debug_flags(tmp_path):
    client = _make_client(tmp_path, ForceModelBackend())

    response = client.post(
        "/api/generate",
        json={
            "prompt": "Верни последний адрес из массива wf.vars.emails; если массив пустой, верни nil.",
            "context": {"wf": {"vars": {"emails": ["a@example.com", "b@example.com"]}}},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["strategy"] == "ollama_chain"
    assert payload["code"] == "return wf.vars.emails[#wf.vars.emails]"
