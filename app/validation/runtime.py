import os
import shutil
from pathlib import Path

from app.core.config import PROJECT_ROOT


def _is_executable(path):
    return path.is_file() and os.access(str(path), os.X_OK)


def _resolve_override(value):
    candidate = Path(value).expanduser()
    if _is_executable(candidate):
        return str(candidate.resolve())

    resolved = shutil.which(value)
    if resolved:
        return resolved
    return None


def _find_runtime_binary(env_name, local_paths, path_names):
    override = os.getenv(env_name)
    if override:
        return _resolve_override(override)

    tools_root = PROJECT_ROOT / ".tools"
    if tools_root.is_dir():
        for relative_path in local_paths:
            candidate = tools_root / relative_path
            if _is_executable(candidate):
                return str(candidate.resolve())

    for command in path_names:
        resolved = shutil.which(command)
        if resolved:
            return resolved
    return None


def find_lua_binary():
    return _find_runtime_binary(
        "LOCALSCRIPT_LUA_BIN",
        (
            Path("lua54") / "bin" / "lua",
            Path("lua-5.4.6") / "src" / "lua",
        ),
        ("lua5.4", "lua"),
    )


def find_luac_binary():
    return _find_runtime_binary(
        "LOCALSCRIPT_LUAC_BIN",
        (
            Path("lua54") / "bin" / "luac",
            Path("lua-5.4.6") / "src" / "luac",
        ),
        ("luac5.4", "luac"),
    )
