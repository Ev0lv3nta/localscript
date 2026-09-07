import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.middleware import RequestBodyLimitMiddleware
from app.core.config import get_runtime_profile
from app.main import create_app
from tests.support_backends import DeterministicTestBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clear_runtime_profile_cache():
    get_runtime_profile.cache_clear()
    yield
    get_runtime_profile.cache_clear()


def _app(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALSCRIPT_STATE_DIR", str(tmp_path / "state"))
    profile = get_runtime_profile()
    return create_app(
        profile=profile,
        backend=DeterministicTestBackend(),
    )


def test_remote_mode_requires_a_strong_token(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALSCRIPT_REMOTE_MODE", "1")
    monkeypatch.setenv("LOCALSCRIPT_REMOTE_TOKEN", "too-short")

    with pytest.raises(RuntimeError, match="remote_token_missing_or_too_short"):
        _app(monkeypatch, tmp_path)


def test_remote_mode_protects_non_health_routes(monkeypatch, tmp_path):
    token = "a-secure-random-token-with-32-characters"
    monkeypatch.setenv("LOCALSCRIPT_REMOTE_MODE", "1")
    monkeypatch.setenv("LOCALSCRIPT_REMOTE_TOKEN", token)
    client = TestClient(_app(monkeypatch, tmp_path))

    assert client.get("/health").status_code == 200
    assert client.get("/api/profile").status_code == 401
    assert (
        client.get(
            "/api/profile",
            headers={"Authorization": "Bearer wrong-token"},
        ).status_code
        == 401
    )
    assert (
        client.get(
            "/api/profile",
            headers={"Authorization": f"Bearer {token}"},
        ).status_code
        == 200
    )


def test_request_body_limit_rejects_content_length_before_route(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALSCRIPT_MAX_REQUEST_BODY_BYTES", "32")
    profile = get_runtime_profile().model_copy(update={"max_request_body_bytes": 32})
    app = create_app(profile=profile, backend=DeterministicTestBackend())
    client = TestClient(app)

    response = client.post(
        "/api/v1/generate",
        content=b"x" * 33,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "request_too_large"


def test_request_body_limit_stops_chunked_stream_at_limit_plus_one():
    consumed = []
    sent = []
    messages = [
        {"type": "http.request", "body": b"a" * 20, "more_body": True},
        {"type": "http.request", "body": b"b" * 13, "more_body": True},
        {"type": "http.request", "body": b"must-not-be-read", "more_body": False},
    ]

    async def downstream(_scope, receive, _send):
        while True:
            message = await receive()
            consumed.append(message["body"])
            if not message.get("more_body"):
                break

    async def receive():
        return messages.pop(0)

    async def send(message):
        sent.append(message)

    middleware = RequestBodyLimitMiddleware(downstream, max_bytes=32)
    asyncio.run(
        middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/v1/generate",
                "headers": [],
            },
            receive,
            send,
        )
    )

    assert consumed == [b"a" * 20]
    assert messages[-1]["body"] == b"must-not-be-read"
    assert sent[0]["status"] == 413


def test_runtime_scripts_default_to_loopback_and_compose_publishes_loopback():
    start = (PROJECT_ROOT / "scripts" / "start.sh").read_text(encoding="utf-8")
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert '"UVI_HOST": profile.bind_host' in start
    assert "non-loopback bind requires LOCALSCRIPT_REMOTE_MODE=1" in start
    assert '"127.0.0.1:${LOCALSCRIPT_PORT:-8080}' in compose
    assert "LOCALSCRIPT_OLLAMA_CONTAINER_ALIAS=ollama" in dockerfile


def test_validate_api_blocks_os_in_an_unexecuted_branch(monkeypatch, tmp_path):
    client = TestClient(_app(monkeypatch, tmp_path))
    response = client.post(
        "/api/validate",
        json={
            "code": 'if false then os.execute("unused") end return 1',
            "context": {"wf": {"vars": {}}},
            "output": {"format": "lua_block", "shape": "scalar", "nullable": False},
        },
    )
    assert response.status_code == 200
    assert not response.json()["ok"]
    assert any(
        check["code"] == "dangerous_stdlib_os_forbidden"
        for check in response.json()["validation"]["checks"]
    )
