from fastapi.testclient import TestClient

from app.core.config import get_runtime_profile
from app.core.traces import TraceStore
from app.main import create_app
from tests.support_backends import DeterministicTestBackend


def _make_client(tmp_path):
    app = create_app(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=DeterministicTestBackend(),
    )
    return TestClient(app)


def test_trace_endpoint_returns_sanitized_trace_payload(tmp_path):
    client = _make_client(tmp_path)
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

    trace = client.get("/api/traces/{0}".format(response.headers["X-Trace-Id"]))

    assert trace.status_code == 200
    payload = trace.json()
    assert payload["trace_id"] == response.headers["X-Trace-Id"]
    assert payload["session_id"] == response.headers["X-Session-Id"]
    assert payload["strategy"] == response.headers["X-Strategy"]
    assert "code" in payload
    assert "planner" in payload
    assert "validation_report" in payload


def test_trace_endpoint_rejects_invalid_trace_id(tmp_path):
    client = _make_client(tmp_path)

    response = client.get("/api/traces/not valid")

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_trace_id"
