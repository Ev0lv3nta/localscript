from fastapi.testclient import TestClient

from app.core.config import get_runtime_profile
from app.core.traces import TraceStore
from app.main import create_app
from tests.support_backends import DeterministicTestBackend


def _make_client(tmp_path, **limits):
    profile = get_runtime_profile().model_copy(update=limits)
    app = create_app(
        profile=profile,
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=DeterministicTestBackend(),
    )
    return TestClient(app)


def test_request_body_limit_rejects_oversized_payload(tmp_path):
    client = _make_client(tmp_path, max_request_body_bytes=64)

    response = client.post(
        "/api/generate",
        content='{"prompt":"' + ("x" * 256) + '"}',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "request_too_large"


def test_prompt_length_limit_rejects_long_prompt(tmp_path):
    client = _make_client(tmp_path, max_prompt_chars=12)

    response = client.post("/api/generate", json={"prompt": "x" * 100})

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "prompt_too_long"


def test_context_depth_limit_rejects_deep_context(tmp_path):
    client = _make_client(tmp_path, max_context_depth=4)
    context = {"wf": {"vars": {"a": {"b": {"c": {"d": {"e": 1}}}}}}}

    response = client.post("/api/generate", json={"prompt": "Return wf.vars.a", "context": context})

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "context_too_deep"


def test_context_width_limit_rejects_wide_context(tmp_path):
    client = _make_client(tmp_path, max_context_nodes=8)
    context = {"wf": {"vars": {"items": list(range(20))}}}

    response = client.post(
        "/api/generate", json={"prompt": "Return wf.vars.items[1]", "context": context}
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "context_too_wide"
