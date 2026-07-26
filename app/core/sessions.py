import json
import os
from pathlib import Path

from app.core.config import PROJECT_ROOT
from app.core.storage import atomic_write_json, resolve_within_root, validate_identifier


class SessionStore:
    def __init__(self, root=None):
        default_root = PROJECT_ROOT / "sessions"
        self.root = Path(os.getenv("LOCALSCRIPT_SESSION_DIR", root or default_root)).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, session_id):
        validate_identifier(session_id, "invalid_session_id")
        return resolve_within_root(self.root, "{0}.json".format(session_id), "invalid_session_id")

    def read(self, session_id):
        path = self.path_for(session_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def write(self, session_id, payload):
        path = self.path_for(session_id)
        atomic_write_json(path, payload)
        return path
