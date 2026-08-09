from pathlib import Path

import pytest

from app.core.runtime_lock import get_runtime_lock_path
from app.core.sessions import SessionStore
from app.core.state import get_state_root
from app.core.traces import TraceStore


STATE_ENV_NAMES = (
    "LOCALSCRIPT_STATE_DIR",
    "XDG_STATE_HOME",
    "LOCALSCRIPT_TRACE_DIR",
    "LOCALSCRIPT_SESSION_DIR",
    "LOCALSCRIPT_RUNTIME_LOCK_PATH",
)


def _clear_state_environment(monkeypatch):
    for name in STATE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_state_root_uses_home_fallback(monkeypatch, tmp_path):
    _clear_state_environment(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    assert get_state_root() == (tmp_path / "home" / ".local" / "state" / "localscript").resolve()


def test_xdg_state_home_precedes_home_fallback(monkeypatch, tmp_path):
    _clear_state_environment(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))

    assert get_state_root() == (tmp_path / "xdg" / "localscript").resolve()


def test_localscript_state_dir_precedes_xdg_state_home(monkeypatch, tmp_path):
    _clear_state_environment(monkeypatch)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("LOCALSCRIPT_STATE_DIR", str(tmp_path / "state"))

    assert get_state_root() == (tmp_path / "state").resolve()


def test_default_writable_paths_share_state_root(monkeypatch, tmp_path):
    _clear_state_environment(monkeypatch)
    state_root = tmp_path / "state"
    monkeypatch.setenv("LOCALSCRIPT_STATE_DIR", str(state_root))

    assert TraceStore().root == (state_root / "traces").resolve()
    assert SessionStore().root == (state_root / "sessions").resolve()
    assert get_runtime_lock_path() == (state_root / "runtime_profile.lock.json").resolve()


def test_constructor_roots_remain_explicit_and_testable(monkeypatch, tmp_path):
    _clear_state_environment(monkeypatch)
    monkeypatch.setenv("LOCALSCRIPT_STATE_DIR", str(tmp_path / "state"))

    assert TraceStore(root=tmp_path / "custom-traces").root == (tmp_path / "custom-traces").resolve()
    assert SessionStore(root=tmp_path / "custom-sessions").root == (tmp_path / "custom-sessions").resolve()


def test_specific_overrides_precede_constructor_and_state_roots(monkeypatch, tmp_path):
    _clear_state_environment(monkeypatch)
    trace_dir = tmp_path / "trace-override"
    session_dir = tmp_path / "session-override"
    lock_path = tmp_path / "lock-override.json"
    monkeypatch.setenv("LOCALSCRIPT_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LOCALSCRIPT_TRACE_DIR", str(trace_dir))
    monkeypatch.setenv("LOCALSCRIPT_SESSION_DIR", str(session_dir))
    monkeypatch.setenv("LOCALSCRIPT_RUNTIME_LOCK_PATH", str(lock_path))

    assert TraceStore(root=tmp_path / "custom-traces").root == trace_dir.resolve()
    assert SessionStore(root=tmp_path / "custom-sessions").root == session_dir.resolve()
    assert get_runtime_lock_path() == lock_path.resolve()


def test_top_level_system_symlink_is_canonicalized(monkeypatch):
    system_link = next(
        (path for path in (Path("/var"), Path("/tmp")) if path.is_symlink()),
        None,
    )
    if system_link is None:
        pytest.skip("platform has no top-level compatibility symlink")

    _clear_state_environment(monkeypatch)
    configured = system_link / "localscript-state"
    monkeypatch.setenv("LOCALSCRIPT_STATE_DIR", str(configured))

    assert get_state_root() == configured.resolve()
