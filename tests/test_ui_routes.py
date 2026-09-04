from fastapi.testclient import TestClient

from app.core.config import get_runtime_profile
from app.core.traces import TraceStore
from app.main import create_app
from tests.support_backends import DeterministicTestBackend


def _make_client(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALSCRIPT_UI_ENABLED", "1")
    app = create_app(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=DeterministicTestBackend(),
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
    assert len(examples.json()["examples"]) >= 3
    assert all("template" not in (item["title"].lower() + (item.get("description") or "").lower()) for item in examples.json()["examples"])
