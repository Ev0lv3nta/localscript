from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.state import resolve_state_path
from app.core.storage import atomic_write_json

if TYPE_CHECKING:
    from app.core.config import RuntimeProfile


def get_runtime_lock_path() -> Path:
    return resolve_state_path(
        "LOCALSCRIPT_RUNTIME_LOCK_PATH",
        "runtime_profile.lock.json",
    )


def load_runtime_lock() -> dict[str, Any] | None:
    lock_path = get_runtime_lock_path()
    if not lock_path.exists():
        return None
    payload: dict[str, Any] = json.loads(lock_path.read_text(encoding="utf-8"))
    return payload


def write_runtime_lock(payload: dict[str, Any]) -> Path:
    lock_path = get_runtime_lock_path()
    payload = dict(payload)
    payload.setdefault(
        "locked_at",
        datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(lock_path, payload)
    return lock_path


def build_runtime_lock(
    profile: RuntimeProfile,
    selected_model: str,
    selection_reason: str,
    quality_report: dict[str, Any],
    vram_report: dict[str, Any],
    primary_vram_report: dict[str, Any] | None = None,
    fallback_vram_report: dict[str, Any] | None = None,
    available_tags: Sequence[str] | None = None,
    hard_gate_failures: Sequence[str] | None = None,
) -> dict[str, Any]:
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
        "available_tags": list(available_tags or []),
        "hard_gate_failures": failures,
    }
