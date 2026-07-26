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


def test_validate_endpoint_returns_semantic_result_for_valid_code(tmp_path):
    client = _make_client(tmp_path)

    response = client.post(
        "/api/validate",
        json={
            "code": "return wf.vars.items[1]",
            "context": {"wf": {"vars": {"items": [10, 20, 30]}}},
            "output_style": "lua_block",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["semantic_result"]["ok"] is True
    assert payload["semantic_result"]["value"] == 10


def test_validate_endpoint_flags_dangerous_stdlib_usage(tmp_path):
    client = _make_client(tmp_path)

    response = client.post(
        "/api/validate",
        json={
            "code": 'return os.execute("echo hacked")',
            "context": {"wf": {"vars": {"value": 1}}},
            "output_style": "lua_block",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert "dangerous_stdlib_os_forbidden" in payload["verification_errors"] or any(
        message["code"] == "dangerous_stdlib_os_forbidden"
        for message in payload["validation_report"]["messages"]
    )


def test_validate_endpoint_auto_detects_json_envelope(tmp_path):
    client = _make_client(tmp_path)

    response = client.post(
        "/api/validate",
        json={
            "code": '{"code":"lua{return wf.vars.items[1]}lua"}',
            "context": {"wf": {"vars": {"items": [10, 20, 30]}}},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["semantic_result"]["ok"] is True
    assert payload["semantic_result"]["value"] == {"code": 10}


def test_validate_endpoint_does_not_auto_treat_plain_json_object_as_envelope(tmp_path):
    client = _make_client(tmp_path)

    response = client.post(
        "/api/validate",
        json={
            "code": '{"code":"return wf.vars.items[1]"}',
            "context": {"wf": {"vars": {"items": [10, 20, 30]}}},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert "json_envelope_value_not_lua_wrapper" not in payload["verification_errors"]
    assert not any(
        message["code"] == "json_envelope_value_not_lua_wrapper"
        for message in payload["validation_report"]["messages"]
    )
