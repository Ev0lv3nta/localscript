from pathlib import Path

import pytest

from app.validation import runtime

RUNTIME_CASES = (
    (
        runtime.find_lua_binary,
        "LOCALSCRIPT_LUA_BIN",
        Path("lua54") / "bin" / "lua",
        "lua5.4",
    ),
    (
        runtime.find_luac_binary,
        "LOCALSCRIPT_LUAC_BIN",
        Path("lua54") / "bin" / "luac",
        "luac5.4",
    ),
)


def _make_executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path.resolve()


@pytest.mark.parametrize(("finder", "env_name", "local_path", "path_name"), RUNTIME_CASES)
def test_explicit_runtime_override_precedes_checkout_and_path(
    monkeypatch,
    tmp_path,
    finder,
    env_name,
    local_path,
    path_name,
):
    override = _make_executable(tmp_path / "override")
    _make_executable(tmp_path / ".tools" / local_path)
    monkeypatch.setattr(runtime, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runtime.shutil, "which", lambda command: "/from-path/{0}".format(command))
    monkeypatch.setenv(env_name, str(override))

    assert finder() == str(override)


@pytest.mark.parametrize(("finder", "env_name", "local_path", "path_name"), RUNTIME_CASES)
def test_invalid_explicit_runtime_override_does_not_fall_back(
    monkeypatch,
    tmp_path,
    finder,
    env_name,
    local_path,
    path_name,
):
    _make_executable(tmp_path / ".tools" / local_path)
    missing_override = str(tmp_path / "missing")
    monkeypatch.setattr(runtime, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        runtime.shutil,
        "which",
        lambda command: None if command == missing_override else "/from-path/{0}".format(command),
    )
    monkeypatch.setenv(env_name, missing_override)

    assert finder() is None


@pytest.mark.parametrize(("finder", "env_name", "local_path", "path_name"), RUNTIME_CASES)
def test_checkout_runtime_precedes_path(
    monkeypatch,
    tmp_path,
    finder,
    env_name,
    local_path,
    path_name,
):
    local_binary = _make_executable(tmp_path / ".tools" / local_path)
    monkeypatch.setattr(runtime, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runtime.shutil, "which", lambda command: "/from-path/{0}".format(command))
    monkeypatch.delenv(env_name, raising=False)

    assert finder() == str(local_binary)


@pytest.mark.parametrize(("finder", "env_name", "local_path", "path_name"), RUNTIME_CASES)
def test_path_runtime_is_used_without_override_or_checkout(
    monkeypatch,
    tmp_path,
    finder,
    env_name,
    local_path,
    path_name,
):
    monkeypatch.setattr(runtime, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        runtime.shutil,
        "which",
        lambda command: "/from-path/{0}".format(command) if command == path_name else None,
    )
    monkeypatch.delenv(env_name, raising=False)

    assert finder() == "/from-path/{0}".format(path_name)
