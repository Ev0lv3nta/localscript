import json
import uuid
from datetime import datetime

from app.core.state import resolve_state_path
from app.core.storage import atomic_write_json, validate_identifier


class TraceStore:
    def __init__(self, root=None):
        self.root = resolve_state_path("LOCALSCRIPT_TRACE_DIR", "traces", root=root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, trace):
        now = datetime.utcnow()
        trace_id = trace.get("trace_id") or now.strftime("%Y%m%dT%H%M%S%fZ")
        session_id = trace.get("session_id") or uuid.uuid4().hex
        date_dir = self.root / now.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        trace_path = date_dir / "{0}.json".format(trace_id)
        payload = dict(trace)
        payload["trace_id"] = trace_id
        payload["session_id"] = session_id
        atomic_write_json(trace_path, payload)
        return trace_id

    def read(self, trace_id):
        validate_identifier(trace_id, "invalid_trace_id")
        for trace_path in sorted(self.root.glob("**/{0}.json".format(trace_id))):
            return json.loads(trace_path.read_text(encoding="utf-8"))
        return None

    def latest_for_session(self, session_id):
        validate_identifier(session_id, "invalid_session_id")
        latest_path = None
        latest_payload = None
        for trace_path in sorted(self.root.glob("**/*.json")):
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            if payload.get("session_id") != session_id:
                continue
            if latest_path is None or trace_path.name > latest_path.name:
                latest_path = trace_path
                latest_payload = payload
        return latest_payload

    @staticmethod
    def sanitize_trace(payload):
        if payload is None:
            return None
        return {
            "trace_id": payload.get("trace_id"),
            "session_id": payload.get("session_id"),
            "status": payload.get("status"),
            "strategy": payload.get("strategy"),
            "model": payload.get("model"),
            "fallback_model": payload.get("fallback_model"),
            "degraded_mode": bool(payload.get("degraded_mode", False)),
            "repair_rounds": int(payload.get("repair_rounds", 0) or 0),
            "assumptions": list(payload.get("assumptions", [])),
            "verification_errors": list(payload.get("verification_errors", [])),
            "validation_report": payload.get("validation_report", {}),
            "planner": payload.get("planner", {}),
            "critic": payload.get("critic", {}),
            "repair_trace": payload.get("repair_trace", []),
            "rules_applied": list(payload.get("rules_applied", [])),
            "examples_used": list(payload.get("examples_used", [])),
            "critic_rules_used": list(payload.get("critic_rules_used", [])),
            "semantic_checks": list(payload.get("semantic_checks", [])),
            "backend_error": payload.get("backend_error"),
            "code": payload.get("code", ""),
        }
