from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import get_runtime_profile
from app.core.traces import TraceStore
from app.main import create_app
from tests.support_backends import DeterministicTestBackend


class ReadyBackend(DeterministicTestBackend):
    def ping(self):
        return True

    def list_tags(self):
        profile = get_runtime_profile()
        return [profile.model, profile.fallback_model]


def test_health_endpoint():
    app = create_app()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["profile"] == "competition"


def test_ready_endpoint_reports_ready(tmp_path, monkeypatch):
    monkeypatch.setattr("app.api.routes._find_lua_binary", lambda: "/tmp/lua")
    monkeypatch.setattr("app.api.routes._find_luac_binary", lambda: "/tmp/luac")
    app = create_app(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=ReadyBackend(),
    )
    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["checks"]["backend_reachable"] is True


def test_ready_endpoint_reports_not_ready(tmp_path, monkeypatch):
    class EmptyBackend:
        def ping(self):
            return False

        def list_tags(self):
            return []

    monkeypatch.setattr("app.api.routes._find_lua_binary", lambda: None)
    monkeypatch.setattr("app.api.routes._find_luac_binary", lambda: None)
    app = create_app(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=EmptyBackend(),
    )
    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "not_ready"
    assert "backend_reachable" in payload["errors"]


def test_generate_endpoint_writes_trace(tmp_path):
    trace_store = TraceStore(root=tmp_path / "traces")
    app = create_app(
        profile=get_runtime_profile(),
        trace_store=trace_store,
        backend=DeterministicTestBackend(),
    )
    client = TestClient(app)

    response = client.post(
        "/generate",
        json={
            "prompt": "Возьми wf.vars.contacts и подготовь список таблиц для активных контактов с email: поле id оставь как есть, а email переведи в lower case. Если вход пустой, нужен пустой список через _utils.array.new().",
            "context": {
                "wf": {
                    "vars": {
                        "contacts": [
                            {"id": "C1", "active": True, "email": "ADMIN@EXAMPLE.COM"},
                            {"id": "C2", "active": False, "email": "skip@example.com"},
                        ]
                    }
                }
            },
        },
    )

    assert response.status_code == 200
    assert "string.lower(item.email)" in response.json()["code"]
    assert response.headers["X-Strategy"] == "ollama_chain"
    assert "X-Trace-Id" in response.headers
    assert "X-Session-Id" in response.headers
    assert "X-Strategy" in response.headers
    assert "X-Repair-Rounds" in response.headers
    assert "X-Degraded-Mode" in response.headers

    trace_files = list(Path(trace_store.root).glob("**/*.json"))
    assert len(trace_files) == 1


class UnsupportedRootBackend:
    def generate(self, prompt, context=None):
        return "local items = ctx.body.items\nreturn items[1]"


def test_generate_endpoint_repairs_unsupported_root_backend_output(tmp_path):
    trace_store = TraceStore(root=tmp_path / "traces")
    app = create_app(
        profile=get_runtime_profile(),
        trace_store=trace_store,
        backend=UnsupportedRootBackend(),
    )
    client = TestClient(app)

    response = client.post(
        "/generate",
        json={
            "prompt": "Iterate over the incoming items collection and return the first value.",
            "context": {"wf": {"vars": {"items": [1, 2, 3]}}},
        },
    )

    assert response.status_code == 200
    assert "ctx.body" not in response.json()["code"]
    assert "wf.vars.items" in response.json()["code"]
    assert response.headers["X-Repair-Rounds"] != "0"


class FailingBackend:
    def generate(self, prompt, context=None):
        raise RuntimeError("backend_down")


def test_generate_endpoint_returns_503_for_backend_failure(tmp_path):
    trace_store = TraceStore(root=tmp_path / "traces")
    app = create_app(
        profile=get_runtime_profile(),
        trace_store=trace_store,
        backend=FailingBackend(),
    )
    client = TestClient(app)

    response = client.post(
        "/generate",
        json={"prompt": "Return first item", "context": {"wf": {"vars": {"items": [1, 2, 3]}}}},
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "backend_unavailable"
