import json
import multiprocessing
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.core.sessions import SessionStore
from app.core.state import UnsafeStatePathError
from app.core.storage import (
    REDACTED,
    CorruptStateError,
    InvalidIdentifierError,
    validate_identifier,
)
from app.core.traces import TraceStore

SESSION_ID = "parallel-session"


def _process_increment(root, iterations):
    store = SessionStore(root=root)
    for _ in range(iterations):
        store.update(
            SESSION_ID,
            lambda payload: payload.update(count=payload.get("count", 0) + 1),
            default={"count": 0},
        )


class MutableClock:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, **kwargs):
        self.value += timedelta(**kwargs)


def test_session_update_has_no_lost_thread_updates(tmp_path):
    store = SessionStore(root=tmp_path / "sessions")

    def increment():
        for _ in range(40):
            store.update(
                SESSION_ID,
                lambda payload: payload.update(count=payload.get("count", 0) + 1),
                default={"count": 0},
            )

    threads = [threading.Thread(target=increment) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert store.read(SESSION_ID)["count"] == 320


@pytest.mark.skipif(os.name == "nt", reason="spawn setup is covered by thread test on Windows")
def test_session_update_has_no_lost_process_updates(tmp_path):
    root = tmp_path / "sessions"
    processes = [
        multiprocessing.Process(target=_process_increment, args=(root, 20))
        for _ in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0

    assert SessionStore(root=root).read(SESSION_ID)["count"] == 80


def test_session_transaction_rolls_back_on_error(tmp_path):
    store = SessionStore(root=tmp_path / "sessions")
    store.write(SESSION_ID, {"value": "before"})

    with pytest.raises(RuntimeError):
        with store.transaction(SESSION_ID) as payload:
            payload["value"] = "after"
            raise RuntimeError("stop")

    assert store.read(SESSION_ID)["value"] == "before"


def test_nested_secrets_are_redacted_without_breaking_session_context(tmp_path):
    sentinel = "STATE-SECRET-SENTINEL"
    sessions = SessionStore(root=tmp_path / "sessions")
    sessions.write(
        SESSION_ID,
        {
            "original_task": "continue this prompt",
            "context": {
                "useful": {"items": [1, 2]},
                "credentials": {"authorization": sentinel, "api-key": sentinel},
            },
        },
    )

    persisted = sessions.read(SESSION_ID)
    assert persisted["original_task"] == "continue this prompt"
    assert persisted["context"]["useful"] == {"items": [1, 2]}
    assert persisted["context"]["credentials"] == {
        "authorization": REDACTED,
        "api-key": REDACTED,
    }
    assert sentinel not in sessions.path_for(SESSION_ID).read_text(encoding="utf-8")


def test_trace_redaction_public_projection_and_uuid4_ids(tmp_path):
    sentinel = "TRACE-SECRET-SENTINEL"
    traces = TraceStore(root=tmp_path / "traces")
    trace_id = traces.write(
        {
            "session_id": uuid.uuid4().hex,
            "prompt": sentinel,
            "context": {"token": sentinel},
            "feedback": sentinel,
            "planner": {"authorization": sentinel, "family": "generic_lua"},
            "code": "return " + repr(sentinel),
            "status": "completed",
        }
    )

    assert uuid.UUID(trace_id).version == 4
    persisted = traces.read(trace_id)
    assert persisted["prompt"] == REDACTED
    assert persisted["context"] == REDACTED
    assert persisted["planner"] == REDACTED
    assert persisted["code"] == REDACTED
    public = traces.sanitize_trace(persisted)
    assert public["code"] == ""
    assert public["planner"] == {}
    assert sentinel not in json.dumps(public)
    assert persisted["created_at"].endswith("Z")
    assert datetime.fromisoformat(persisted["created_at"].replace("Z", "+00:00")).tzinfo


def test_uuid_shaped_ids_must_be_version_four_but_legacy_slugs_remain_compatible():
    validate_identifier("persisted-session", "invalid_session_id")
    with pytest.raises(InvalidIdentifierError):
        validate_identifier(str(uuid.uuid1()), "invalid_session_id")


def test_trace_lookup_uses_index_without_recursive_glob(monkeypatch, tmp_path):
    traces = TraceStore(root=tmp_path / "traces")
    session_id = uuid.uuid4().hex
    trace_id = traces.write({"session_id": session_id, "status": "completed"})

    monkeypatch.setattr(
        Path,
        "glob",
        lambda *args, **kwargs: pytest.fail("trace lookup used glob"),
    )
    assert traces.read(trace_id)["trace_id"] == trace_id
    assert traces.latest_for_session(session_id)["trace_id"] == trace_id


def test_retention_count_and_ttl_use_injected_aware_clock(tmp_path):
    clock = MutableClock(datetime(2026, 7, 27, tzinfo=timezone.utc))
    traces = TraceStore(
        root=tmp_path / "traces",
        retention_count=2,
        retention_ttl_seconds=60,
        clock=clock,
    )
    ids = []
    for _ in range(3):
        ids.append(traces.write({"session_id": uuid.uuid4().hex}))
        clock.advance(seconds=10)

    assert traces.read(ids[0]) is None
    clock.advance(seconds=61)
    assert set(traces.cleanup()) == set(ids[1:])
    assert traces.cleanup() == []


def test_corrupt_session_is_quarantined_with_typed_error(tmp_path):
    store = SessionStore(root=tmp_path / "sessions")
    path = store.path_for(SESSION_ID)
    path.write_text("{broken", encoding="utf-8")

    with pytest.raises(CorruptStateError) as raised:
        store.read(SESSION_ID)

    assert raised.value.code == "corrupt_state_json"
    assert raised.value.quarantine_path.exists()
    assert not path.exists()


def test_structurally_corrupt_session_is_quarantined(tmp_path):
    store = SessionStore(root=tmp_path / "sessions")
    path = store.path_for(SESSION_ID)
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(CorruptStateError):
        store.read(SESSION_ID)

    assert not path.exists()


def test_symlink_roots_and_state_files_fail_closed(tmp_path):
    real_root = tmp_path / "real"
    real_root.mkdir()
    root_link = tmp_path / "linked"
    root_link.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(UnsafeStatePathError):
        SessionStore(root=root_link)

    store = SessionStore(root=tmp_path / "sessions")
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    store.path_for(SESSION_ID).symlink_to(outside)
    with pytest.raises(Exception) as raised:
        store.read(SESSION_ID)
    assert getattr(raised.value, "code", None) == "unsafe_storage_path"
