from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterator
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.core.state import resolve_state_path
from app.core.storage import (
    REDACTED,
    TRACE_PRIVATE_KEYS,
    atomic_write_json,
    delete_file,
    ensure_directory,
    file_lock,
    generate_identifier,
    isoformat_utc,
    parse_utc,
    read_json,
    redact_nested,
    resolve_within_root,
    utc_now,
    validate_identifier,
)

DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


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


class TraceStore:
    def __init__(
        self,
        root: Path | str | None = None,
        retention_count: int | None = None,
        retention_ttl_seconds: int | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.root = resolve_state_path("LOCALSCRIPT_TRACE_DIR", "traces", root=root)
        ensure_directory(self.root)
        self._index_root = ensure_directory(self.root / ".index")
        self._trace_index = ensure_directory(self._index_root / "traces")
        self._session_index = ensure_directory(self._index_root / "sessions")
        self._lock_path = self.root / ".locks" / "store.lock"
        self.retention_count = _retention_value(
            retention_count, "LOCALSCRIPT_TRACE_RETENTION_COUNT"
        )
        self.retention_ttl_seconds = _retention_value(
            retention_ttl_seconds, "LOCALSCRIPT_TRACE_RETENTION_TTL_SECONDS"
        )
        self._clock = clock

    def _trace_index_path(self, trace_id: str) -> Path:
        validate_identifier(trace_id, "invalid_trace_id")
        return resolve_within_root(
            self._trace_index, f"{trace_id}.idx", "invalid_trace_id"
        )

    def _session_index_dir(self, session_id: str) -> Path:
        validate_identifier(session_id, "invalid_session_id")
        return resolve_within_root(self._session_index, session_id, "invalid_session_id")

    def write(self, trace: dict[str, Any]) -> str:
        if not isinstance(trace, dict):
            raise TypeError("trace payload must be a mapping")
        now = utc_now(self._clock)
        trace_id = trace.get("trace_id") or generate_identifier()
        session_id = trace.get("session_id") or generate_identifier()
        validate_identifier(trace_id, "invalid_trace_id")
        validate_identifier(session_id, "invalid_session_id")
        created_at = isoformat_utc(now)
        date_name = now.strftime("%Y-%m-%d")
        relative_path = f"{date_name}/{trace_id}.json"
        with file_lock(self._lock_path):
            date_dir = ensure_directory(
                resolve_within_root(self.root, date_name, "invalid_trace_path")
            )
            trace_path = resolve_within_root(
                date_dir, f"{trace_id}.json", "invalid_trace_id"
            )
            payload = redact_nested(deepcopy(trace), TRACE_PRIVATE_KEYS)
            if "code" in payload:
                payload["code"] = REDACTED
            payload["trace_id"] = trace_id
            payload["session_id"] = session_id
            payload["created_at"] = created_at
            atomic_write_json(trace_path, payload)
            pointer = {
                "trace_id": trace_id,
                "session_id": session_id,
                "created_at": created_at,
                "relative_path": relative_path,
            }
            atomic_write_json(self._trace_index_path(trace_id), pointer)
            session_dir = ensure_directory(self._session_index_dir(session_id))
            atomic_write_json(session_dir / f"{trace_id}.idx", pointer)
            self._cleanup_unlocked()
        return trace_id

    def _path_from_pointer(self, pointer: object) -> Path:
        relative_path = pointer.get("relative_path") if isinstance(pointer, dict) else None
        if not isinstance(relative_path, str):
            raise ValueError("invalid trace index pointer")
        return resolve_within_root(self.root, relative_path, "invalid_trace_path")

    def _find_legacy_path_unlocked(self, trace_id: str) -> Path | None:
        # Compatibility scan is bounded to immediate YYYY-MM-DD directories.
        filename = f"{trace_id}.json"
        for directory in sorted(self.root.iterdir(), reverse=True):
            if (
                directory.is_symlink()
                or not directory.is_dir()
                or not DATE_DIR_RE.fullmatch(directory.name)
            ):
                continue
            candidate = resolve_within_root(directory, filename, "invalid_trace_id")
            if candidate.exists():
                return candidate
        return None

    def read(self, trace_id: str) -> dict[str, Any] | None:
        validate_identifier(trace_id, "invalid_trace_id")
        with file_lock(self._lock_path):
            index_path = self._trace_index_path(trace_id)
            trace_path: Path | None
            if index_path.exists():
                trace_path = self._path_from_pointer(
                    read_json(index_path, expected_type=dict)
                )
            else:
                trace_path = self._find_legacy_path_unlocked(trace_id)
            if trace_path is None or not trace_path.exists():
                return None
            payload: dict[str, Any] = read_json(trace_path, expected_type=dict)
            return payload

    def latest_for_session(self, session_id: str) -> dict[str, Any] | None:
        validate_identifier(session_id, "invalid_session_id")
        with file_lock(self._lock_path):
            session_dir = self._session_index_dir(session_id)
            if session_dir.exists():
                pointers: list[tuple[datetime, dict[str, Any]]] = []
                for pointer_path in session_dir.iterdir():
                    if (
                        pointer_path.is_symlink()
                        or not pointer_path.is_file()
                        or pointer_path.suffix != ".idx"
                    ):
                        continue
                    pointer = read_json(pointer_path, expected_type=dict)
                    try:
                        pointers.append((parse_utc(pointer["created_at"]), pointer))
                    except (KeyError, TypeError, ValueError):
                        continue
                for _, pointer in sorted(pointers, key=lambda item: item[0], reverse=True):
                    indexed_path = self._path_from_pointer(pointer)
                    if indexed_path.exists():
                        indexed: dict[str, Any] = read_json(indexed_path, expected_type=dict)
                        return indexed

            latest: tuple[datetime, dict[str, Any]] | None = None
            for trace_path in self._iter_trace_paths_unlocked():
                payload = read_json(trace_path, expected_type=dict)
                if payload.get("session_id") != session_id:
                    continue
                try:
                    created_at = parse_utc(payload.get("created_at"))
                except (TypeError, ValueError):
                    created_at = datetime.fromtimestamp(
                        trace_path.stat().st_mtime, UTC
                    )
                if latest is None or created_at > latest[0]:
                    latest = (created_at, payload)
            return latest[1] if latest else None

    def _iter_trace_paths_unlocked(self) -> Iterator[Path]:
        for directory in self.root.iterdir():
            if (
                directory.is_symlink()
                or not directory.is_dir()
                or not DATE_DIR_RE.fullmatch(directory.name)
            ):
                continue
            for path in directory.iterdir():
                if (
                    not path.is_symlink()
                    and path.is_file()
                    and path.suffix == ".json"
                    and not path.name.startswith(".")
                ):
                    yield path

    def _delete_trace_unlocked(self, trace_path: Path, payload: dict[str, Any]) -> str:
        trace_id = str(payload.get("trace_id", trace_path.stem))
        session_id = payload.get("session_id")
        delete_file(trace_path)
        delete_file(self._trace_index_path(trace_id))
        if session_id:
            session_dir = self._session_index_dir(session_id)
            delete_file(session_dir / f"{trace_id}.idx")
        return trace_id

    def _cleanup_unlocked(self) -> list[str]:
        if self.retention_count is None and self.retention_ttl_seconds is None:
            return []
        now = utc_now(self._clock)
        entries: list[tuple[datetime, Path, dict[str, Any]]] = []
        for path in self._iter_trace_paths_unlocked():
            payload = read_json(path, expected_type=dict)
            try:
                timestamp = parse_utc(payload.get("created_at"))
            except (TypeError, ValueError):
                timestamp = datetime.fromtimestamp(path.stat().st_mtime, UTC)
            entries.append((timestamp, path, payload))
        entries.sort(key=lambda item: item[0], reverse=True)
        expired: set[Path] = set()
        if self.retention_ttl_seconds is not None:
            cutoff = now - timedelta(seconds=self.retention_ttl_seconds)
            expired.update(path for timestamp, path, _ in entries if timestamp < cutoff)
        if self.retention_count is not None:
            survivors = [item for item in entries if item[1] not in expired]
            expired.update(path for _, path, _ in survivors[self.retention_count :])
        removed: list[str] = []
        for _, path, payload in entries:
            if path in expired:
                removed.append(self._delete_trace_unlocked(path, payload))
        return removed

    def cleanup(self) -> list[str]:
        with file_lock(self._lock_path):
            return self._cleanup_unlocked()

    @staticmethod
    def sanitize_trace(payload: dict[str, Any] | None) -> dict[str, Any] | None:
        """Return a stable public projection without request or model output."""
        if payload is None:
            return None
        stage_events = payload.get("stage_events")
        diagnostic_codes = payload.get("diagnostic_codes")
        return {
            "trace_id": payload.get("trace_id"),
            "session_id": payload.get("session_id"),
            "status": payload.get("status"),
            "model": payload.get("model"),
            "revision_count": int(payload.get("revision_count", 0) or 0),
            "diagnostic_codes": (
                [str(code) for code in diagnostic_codes]
                if isinstance(diagnostic_codes, list)
                else []
            ),
            "stage_events": (
                [event for event in stage_events if isinstance(event, dict)]
                if isinstance(stage_events, list)
                else []
            ),
        }
