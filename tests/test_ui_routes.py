import json

from fastapi.testclient import TestClient

from app.core.config import get_runtime_profile
from app.core.traces import TraceStore
from app.main import create_app


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
                        "semantic_checks": [],
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
            return json.dumps({"repairable": False, "issues": [], "minimal_actions": []}, ensure_ascii=False)
        return 'local value = wf.vars.email or ""\nvalue = string.gsub(value, "^%s*(.-)%s*$", "%1")\nreturn string.lower(value)'

    def generate(self, prompt, context=None):
        return self.complete(prompt)


def _make_client(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALSCRIPT_UI_ENABLED", "1")
    app = create_app(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=ClarificationBackend(),
    )
    return TestClient(app)


def test_ui_routes_serve_operator_console_html(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)

    root = client.get("/")
    alias = client.get("/ui")

    assert root.status_code == 200
    assert alias.status_code == 200
    assert "Сгенерировать и проверить" in root.text
    assert "Проверенный результат" in root.text
    assert "/static/app.js" in root.text


def test_ui_support_endpoints_expose_profile_and_examples(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)

    profile = client.get("/api/profile")
    examples = client.get("/api/examples")

    assert profile.status_code == 200
    assert profile.json()["ui_enabled"] is True
    assert profile.json()["model"] in {"qwen3:8b-q4_K_M", "qwen3:4b-instruct-2507-q4_K_M"}
    assert examples.status_code == 200
    assert len(examples.json()["examples"]) >= 4
    assert all("template" not in (item["title"].lower() + (item.get("description") or "").lower()) for item in examples.json()["examples"])


def test_ui_flow_smoke_covers_analyze_clarify_trace_and_validate(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    context = {
        "wf": {
            "vars": {"email": "A@EXAMPLE.COM"},
            "initVariables": {"email": "B@EXAMPLE.COM"},
        }
    }

    analyze = client.post("/api/analyze", json={"prompt": "Нормализуй email и верни его в lower-case.", "context": context})
    assert analyze.status_code == 200
    assert analyze.json()["suggested_strategy"] == "clarification"

    first = client.post("/api/generate", json={"prompt": "Нормализуй email и верни его в lower-case.", "context": context})
    assert first.status_code == 200
    assert first.json()["status"] == "clarification_required"

    continued = client.post(
        "/api/generate",
        json={"session_id": first.json()["session_id"], "clarification_answer": "Use wf.vars for email root."},
    )
    assert continued.status_code == 200
    assert continued.json()["status"] == "completed"

    trace = client.get("/api/traces/{0}".format(continued.json()["trace_id"]))
    assert trace.status_code == 200
    assert trace.json()["trace_id"] == continued.json()["trace_id"]

    validate = client.post(
        "/api/validate",
        json={
            "code": continued.json()["code"],
            "context": context,
            "output_style": "lua_block",
        },
    )
    assert validate.status_code == 200
    assert validate.json()["ok"] is True
