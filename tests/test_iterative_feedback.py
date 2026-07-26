from fastapi.testclient import TestClient

from app.core.config import get_runtime_profile
from app.core.traces import TraceStore
from app.main import create_app


class RevisionBackend:
    def generate(self, prompt, context=None):
        if "feedback_history" in prompt and "Use wf.vars.items[1] explicitly." in prompt:
            return "return wf.vars.items[1]"
        return "return nil"


def test_generate_supports_feedback_revision_in_same_session(tmp_path):
    trace_store = TraceStore(root=tmp_path / "traces")
    app = create_app(
        profile=get_runtime_profile(),
        trace_store=trace_store,
        backend=RevisionBackend(),
    )
    client = TestClient(app)

    first = client.post(
        "/generate",
        json={
            "prompt": "Return the first item from wf.vars.items.",
            "context": {"wf": {"vars": {"items": [1, 2, 3]}}},
        },
    )
    session_id = first.headers["X-Session-Id"]
    assert first.status_code == 200
    assert session_id

    revised = client.post(
        "/generate",
        json={
            "prompt": "Return the first item from wf.vars.items.",
            "context": {"wf": {"vars": {"items": [1, 2, 3]}}},
            "session_id": session_id,
            "feedback": "Use wf.vars.items[1] explicitly.",
        },
    )

    assert revised.status_code == 200
    assert revised.json()["code"] == "return wf.vars.items[1]"
    assert revised.headers["X-Session-Id"] == session_id
    assert revised.headers["X-Strategy"] == "feedback_revision"
