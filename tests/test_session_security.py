import pytest
from fastapi.testclient import TestClient

from app.core.config import get_runtime_profile
from app.core.sessions import SessionStore
from app.core.storage import InvalidIdentifierError
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


def test_session_store_rejects_traversal_like_identifier(tmp_path):
    store = SessionStore(root=tmp_path / "sessions")

    with pytest.raises(InvalidIdentifierError):
        store.path_for("../../etc/passwd")


def test_generate_rejects_invalid_session_id(tmp_path):
    client = _make_client(tmp_path)

    response = client.post(
        "/api/generate",
        json={
            "prompt": "Return the first item from wf.vars.items.",
            "context": {"wf": {"vars": {"items": [1, 2, 3]}}},
            "session_id": "../../etc/passwd",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_session_id"


def test_get_session_rejects_invalid_identifier(tmp_path):
    client = _make_client(tmp_path)

    response = client.get("/api/sessions/not valid")

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_session_id"
