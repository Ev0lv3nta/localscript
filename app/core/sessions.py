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
    CorruptStateError,
    atomic_write_json,
    delete_file,
    ensure_directory,
    file_lock,
    isoformat_utc,
    parse_utc,
    read_json,
    resolve_within_root,
    try_file_lock,
    utc_now,
    validate_identifier,
)

DEFAULT_RETENTION_COUNT = 100
DEFAULT_RETENTION_TTL_SECONDS = 7 * 24 * 60 * 60


def _retention_value(explicit: int | None, environment_name: str, default: int) -> int:
    value: int | None
    if explicit is not None:
        value = int(explicit)
    else:
        configured = os.getenv(environment_name)
        value = int(configured) if configured not in (None, "") else default
    if value < 0:
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
        with contextlib.suppress(OSError):
            os.chmod(self.root, 0o700)
        self._locks_root = ensure_directory(self.root / ".locks")
        with contextlib.suppress(OSError):
            os.chmod(self._locks_root, 0o700)
        self._store_lock_path = self._locks_root / "store.lock"
        self.retention_count = _retention_value(
            retention_count,
            "LOCALSCRIPT_SESSION_RETENTION_COUNT",
            DEFAULT_RETENTION_COUNT,
        )
        self.retention_ttl_seconds = _retention_value(
            retention_ttl_seconds,
            "LOCALSCRIPT_SESSION_RETENTION_TTL_SECONDS",
            DEFAULT_RETENTION_TTL_SECONDS,
        )
        self._clock = clock

    def path_for(self, session_id: str) -> Path:
        validate_identifier(session_id, "invalid_session_id")
        return resolve_within_root(self.root, f"{session_id}.json", "invalid_session_id")

    def _lock_path_for(self, session_id: str) -> Path:
        validate_identifier(session_id, "invalid_session_id")
        return resolve_within_root(
            self.root,
            f"{self._locks_root.name}/{session_id}.lock",
            "invalid_session_id",
        )

    def _read_unlocked(self, session_id: str) -> dict[str, Any] | None:
        path = self.path_for(session_id)
        if not path.exists():
            return None
        payload: dict[str, Any] = read_json(path, expected_type=dict)
        return payload

    def read(self, session_id: str) -> dict[str, Any] | None:
        # Atomic replacement makes the last committed snapshot safe to read
        # without waiting for an in-flight generation on this session.
        try:
            payload = self._read_unlocked(session_id)
        except FileNotFoundError:
            # Retention may remove the snapshot between exists() and open().
            return None
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
        # Session files are private functional state. Redacting keys such as
        # ``token`` here changes the input seen by a later workflow turn and can
        # silently change the generated program. Public traces are redacted at
        # their own storage boundary; sessions preserve the caller's JSON value.
        persisted = deepcopy(payload)
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
        with file_lock(self._lock_path_for(session_id)):
            path, _ = self._write_unlocked(session_id, payload)
        self.cleanup()
        return path

    @contextlib.contextmanager
    def transaction(
        self,
        session_id: str,
        default: dict[str, Any] | Callable[[], dict[str, Any]] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield one locked mutable session and atomically commit on normal exit.

        The lock is deliberately scoped to ``session_id``. A generation may
        keep this transaction open while it calls the model, but it must not
        block reads or updates for unrelated sessions.
        """
        committed = False
        with file_lock(self._lock_path_for(session_id)):
            current = self._read_unlocked(session_id)
            if current is None:
                current = default() if callable(default) else deepcopy(default)
                if current is None:
                    current = {}
            working = deepcopy(current)
            yield working
            self._write_unlocked(session_id, working)
            committed = True
        if committed:
            self.cleanup()

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
            try:
                payload = read_json(path, expected_type=dict)
            except (CorruptStateError, FileNotFoundError):
                # Retention is maintenance for all sessions. A malformed or
                # concurrently removed unrelated entry must not turn a healthy
                # session's already committed write into an error response.
                continue
            timestamp = None
            if isinstance(payload, dict):
                with contextlib.suppress(TypeError, ValueError):
                    timestamp = parse_utc(payload.get("_state_updated_at"))
            if timestamp is None:
                timestamp = datetime.fromtimestamp(path.stat().st_mtime, UTC)
            entries.append((timestamp, path))
        return entries

    def _expired_paths_unlocked(self) -> set[Path]:
        now = utc_now(self._clock)
        entries = sorted(self._session_entries_unlocked(), reverse=True)
        expired: set[Path] = set()
        cutoff = now - timedelta(seconds=self.retention_ttl_seconds)
        expired.update(path for timestamp, path in entries if timestamp < cutoff)
        survivors = [(timestamp, path) for timestamp, path in entries if path not in expired]
        expired.update(path for _, path in survivors[self.retention_count :])
        return expired

    def _cleanup_unlocked(self) -> list[str]:
        removed: list[str] = []
        for path in sorted(self._expired_paths_unlocked()):
            with try_file_lock(self._lock_path_for(path.stem)) as acquired:
                if not acquired:
                    continue
                # The transaction might have committed between the first scan
                # and our lock attempt. Recheck its fresh timestamp/ranking.
                if path in self._expired_paths_unlocked() and delete_file(path):
                    removed.append(path.stem)
        return removed

    def cleanup(self) -> list[str]:
        """Apply configured count/TTL retention; repeated calls are idempotent."""
        with file_lock(self._store_lock_path):
            return self._cleanup_unlocked()
