import os
from pathlib import Path


class UnsafeStatePathError(RuntimeError):
    def __init__(self, path):
        self.code = "unsafe_state_path"
        self.path = Path(path)
        super().__init__("state path contains a symbolic link: {0}".format(self.path))


def _resolve_state_location(path):
    candidate = Path(path).expanduser()
    absolute = candidate if candidate.is_absolute() else Path.cwd() / candidate
    current = Path(absolute.anchor)
    for index, part in enumerate(absolute.parts[1:]):
        current = current / part
        # macOS exposes system locations such as /var and /tmp through a
        # top-level compatibility symlink. Canonicalize that trusted prefix,
        # but still reject links inside the caller-selected state path.
        if index > 0 and current.is_symlink():
            raise UnsafeStatePathError(current)
    return absolute.resolve()


def get_state_root():
    configured_root = os.getenv("LOCALSCRIPT_STATE_DIR")
    if configured_root:
        return _resolve_state_location(configured_root)

    xdg_state_home = os.getenv("XDG_STATE_HOME")
    if xdg_state_home:
        return _resolve_state_location(Path(xdg_state_home).expanduser() / "localscript")

    return _resolve_state_location(Path.home() / ".local" / "state" / "localscript")


def resolve_state_path(override_name, default_name, root=None):
    configured_path = os.getenv(override_name)
    if configured_path:
        return _resolve_state_location(configured_path)
    if root is not None:
        return _resolve_state_location(root)
    return _resolve_state_location(get_state_root() / default_name)
