import json
from datetime import datetime, timezone

from app.core.state import resolve_state_path
from app.core.storage import atomic_write_json


def get_runtime_lock_path():
    return resolve_state_path(
        "LOCALSCRIPT_RUNTIME_LOCK_PATH",
        "runtime_profile.lock.json",
    )


def load_runtime_lock():
    lock_path = get_runtime_lock_path()
    if not lock_path.exists():
        return None
    return json.loads(lock_path.read_text(encoding="utf-8"))


def write_runtime_lock(payload):
    lock_path = get_runtime_lock_path()
    payload = dict(payload)
    payload.setdefault(
        "locked_at",
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(lock_path, payload)
    return lock_path


def build_runtime_lock(
    profile,
    selected_model,
    selection_reason,
    quality_report,
    vram_report,
    primary_vram_report=None,
    fallback_vram_report=None,
    available_tags=None,
    hard_gate_failures=None,
):
    failures = list(hard_gate_failures or [])
    return {
        "profile": profile.name,
        "locked": not failures,
        "artifact_role": "generated_validation_snapshot",
        "generated_by": "localscript doctor --judge",
        "startup_required": False,
        "selected_model": selected_model,
        "fallback_model": profile.fallback_model,
        "selection_reason": selection_reason,
        "quality_report": quality_report,
        "vram_report": vram_report,
        "primary_vram_report": primary_vram_report or vram_report,
        "fallback_vram_report": fallback_vram_report,
        "available_tags": available_tags or [],
        "hard_gate_failures": failures,
    }
