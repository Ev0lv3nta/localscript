from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterator
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.core.state import resolve_state_path
from app.core.storage import (
    SECRET_KEYS,
    atomic_write_json,
    delete_file,
    ensure_directory,
    file_lock,
    isoformat_utc,
    parse_utc,
    read_json,
    redact_nested,
    resolve_within_root,
    utc_now,
    validate_identifier,
)


def _retention_value(explicit: int | None, environment_name: str) -> int | None:
    value: int | None
    if explicit is not None:
        value = int(explicit)
    else:
        configured = os.getenv(environment_name)
        value = int(configured) if configured not in (None, "") else None
    if value is not None and value < 0:
        raise ValueError(f"{environment_name} must be non-negative")
    return value


class SessionStore:
    def __init__(
        self,
        root: Path | str | None = None,
        retention_count: int | None = None,
        retention_ttl_seconds: int | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.root = resolve_state_path("LOCALSCRIPT_SESSION_DIR", "sessions", root=root)
        ensure_directory(self.root)
        self._lock_path = self.root / ".locks" / "store.lock"
        self.retention_count = _retention_value(
            retention_count, "LOCALSCRIPT_SESSION_RETENTION_COUNT"
        )
        self.retention_ttl_seconds = _retention_value(
            retention_ttl_seconds, "LOCALSCRIPT_SESSION_RETENTION_TTL_SECONDS"
        )
        self._clock = clock

    def path_for(self, session_id: str) -> Path:
        validate_identifier(session_id, "invalid_session_id")
        return resolve_within_root(self.root, f"{session_id}.json", "invalid_session_id")

    def _read_unlocked(self, session_id: str) -> dict[str, Any] | None:
        path = self.path_for(session_id)
        if not path.exists():
            return None
        payload: dict[str, Any] = read_json(path, expected_type=dict)
        return payload

    def read(self, session_id: str) -> dict[str, Any] | None:
        with file_lock(self._lock_path):
            payload = self._read_unlocked(session_id)
            return deepcopy(payload)

    def _write_unlocked(
        self,
        session_id: str,
        payload: dict[str, Any],
        now: datetime | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        path = self.path_for(session_id)
        current = self._read_unlocked(session_id) if path.exists() else None
        timestamp = isoformat_utc(now or utc_now(self._clock))
        persisted = redact_nested(deepcopy(payload), SECRET_KEYS)
        if not isinstance(persisted, dict):
            raise TypeError("session payload must be a mapping")
        persisted["session_id"] = session_id
        persisted["_state_created_at"] = (
            current.get("_state_created_at", timestamp) if isinstance(current, dict) else timestamp
        )
        persisted["_state_updated_at"] = timestamp
        atomic_write_json(path, persisted)
        return path, persisted

    def write(self, session_id: str, payload: dict[str, Any]) -> Path:
        with file_lock(self._lock_path):
            path, _ = self._write_unlocked(session_id, payload)
            self._cleanup_unlocked()
            return path

    @contextlib.contextmanager
    def transaction(
        self,
        session_id: str,
        default: dict[str, Any] | Callable[[], dict[str, Any]] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield a locked mutable session and atomically commit on normal exit."""
        with file_lock(self._lock_path):
            current = self._read_unlocked(session_id)
            if current is None:
                current = default() if callable(default) else deepcopy(default)
                if current is None:
                    current = {}
            working = deepcopy(current)
            yield working
            self._write_unlocked(session_id, working)
            self._cleanup_unlocked()

    def update(
        self,
        session_id: str,
        updater: Callable[[dict[str, Any]], dict[str, Any] | None],
        default: dict[str, Any] | Callable[[], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run updater under the store lock and return the committed payload."""
        committed: dict[str, Any] = {}
        with self.transaction(session_id, default=default) as payload:
            replacement = updater(payload)
            if replacement is not None:
                if not isinstance(replacement, dict):
                    raise TypeError("session updater must return a mapping or None")
                payload.clear()
                payload.update(replacement)
            committed = payload
        return deepcopy(committed)

    def _session_entries_unlocked(self) -> list[tuple[datetime, Path]]:
        entries: list[tuple[datetime, Path]] = []
        for path in self.root.iterdir():
            if path.is_symlink():
                continue
            if not path.is_file() or path.suffix != ".json" or path.name.startswith("."):
                continue
            payload = read_json(path, expected_type=dict)
            timestamp = None
            if isinstance(payload, dict):
                with contextlib.suppress(TypeError, ValueError):
                    timestamp = parse_utc(payload.get("_state_updated_at"))
            if timestamp is None:
                timestamp = datetime.fromtimestamp(path.stat().st_mtime, UTC)
            entries.append((timestamp, path))
        return entries

    def _cleanup_unlocked(self) -> list[str]:
        if self.retention_count is None and self.retention_ttl_seconds is None:
            return []
        now = utc_now(self._clock)
        entries = sorted(self._session_entries_unlocked(), reverse=True)
        expired: set[Path] = set()
        if self.retention_ttl_seconds is not None:
            cutoff = now - timedelta(seconds=self.retention_ttl_seconds)
            expired.update(path for timestamp, path in entries if timestamp < cutoff)
        if self.retention_count is not None:
            survivors = [(timestamp, path) for timestamp, path in entries if path not in expired]
            expired.update(path for _, path in survivors[self.retention_count :])
        removed: list[str] = []
        for path in sorted(expired):
            if delete_file(path):
                removed.append(path.stem)
        return removed

    def cleanup(self) -> list[str]:
        """Apply configured count/TTL retention; repeated calls are idempotent."""
        with file_lock(self._lock_path):
            return self._cleanup_unlocked()
