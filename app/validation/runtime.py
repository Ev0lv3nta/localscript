from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from pathlib import Path

from app.core.config import PROJECT_ROOT


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(str(path), os.X_OK)


def _resolve_override(value: str) -> str | None:
    candidate = Path(value).expanduser()
    if _is_executable(candidate):
        return str(candidate.resolve())

    resolved = shutil.which(value)
    if resolved:
        return resolved
    return None


def _find_runtime_binary(
    env_name: str,
    local_paths: Sequence[Path],
    path_names: Sequence[str],
) -> str | None:
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


def find_lua_binary() -> str | None:
    return _find_runtime_binary(
        "LOCALSCRIPT_LUA_BIN",
        (
            Path("lua54") / "bin" / "lua",
            Path("lua-5.4.6") / "src" / "lua",
        ),
        ("lua5.4", "lua"),
    )


def find_luac_binary() -> str | None:
    return _find_runtime_binary(
        "LOCALSCRIPT_LUAC_BIN",
        (
            Path("lua54") / "bin" / "luac",
            Path("lua-5.4.6") / "src" / "luac",
        ),
        ("luac5.4", "luac"),
    )
