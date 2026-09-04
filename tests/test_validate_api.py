from fastapi.testclient import TestClient

from app.core.config import get_runtime_profile
from app.core.traces import TraceStore
from app.main import create_app
from tests.support_backends import DeterministicTestBackend


def make_client(tmp_path):
    app = create_app(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=DeterministicTestBackend(),
    )
    return TestClient(app)


def test_validate_endpoint_uses_explicit_output_contract(tmp_path):
    client = make_client(tmp_path)

    response = client.post(
        "/api/validate",
        json={
            "code": "return wf.vars.items[1]",
            "context": {"wf": {"vars": {"items": [10, 20, 30]}}},
            "output": {"format": "lua_block", "shape": "scalar", "nullable": False},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["validation"]["observations"] == [{"actual": 10}]


def test_validate_endpoint_rejects_real_dangerous_call(tmp_path):
    client = make_client(tmp_path)

    response = client.post(
        "/api/validate",
        json={
            "code": 'return os.execute("echo blocked")',
            "context": {"wf": {"vars": {"value": 1}}},
            "output": {"format": "lua_block", "shape": "scalar", "nullable": True},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    failed = [
        check["code"] for check in body["validation"]["checks"] if check["status"] == "failed"
    ]
    assert "dangerous_stdlib_os_forbidden" in failed


def test_validate_endpoint_requires_contract_and_context(tmp_path):
    client = make_client(tmp_path)

    response = client.post("/api/validate", json={"code": "return 1"})

    assert response.status_code == 422
