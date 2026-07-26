import json
import os
import re
import tempfile
from pathlib import Path


SAFE_IDENTIFIER_RE = re.compile(r"^(?:[a-f0-9]{32}|[0-9a-fA-F-]{36}|[A-Za-z0-9_-]{8,64})$")


class InvalidIdentifierError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def validate_identifier(value, code):
    if not isinstance(value, str) or not SAFE_IDENTIFIER_RE.fullmatch(value):
        raise InvalidIdentifierError(code=code, message=code)
    return value


def resolve_within_root(root, relative_name, code):
    resolved_root = Path(root).resolve()
    resolved_path = (resolved_root / relative_name).resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise InvalidIdentifierError(code=code, message=code)
    return resolved_path


def atomic_write_json(path, payload):
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=".localscript-",
        suffix=".tmp",
        dir=str(target_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
