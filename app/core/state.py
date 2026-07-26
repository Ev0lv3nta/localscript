import os
from pathlib import Path


def get_state_root():
    configured_root = os.getenv("LOCALSCRIPT_STATE_DIR")
    if configured_root:
        return Path(configured_root).expanduser().resolve()

    xdg_state_home = os.getenv("XDG_STATE_HOME")
    if xdg_state_home:
        return (Path(xdg_state_home).expanduser() / "localscript").resolve()

    return (Path.home() / ".local" / "state" / "localscript").resolve()


def resolve_state_path(override_name, default_name, root=None):
    configured_path = os.getenv(override_name)
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    if root is not None:
        return Path(root).expanduser().resolve()
    return (get_state_root() / default_name).resolve()
