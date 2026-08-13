import contextlib
import json
import os
import re
import tempfile
import threading
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

UUID_RE = re.compile(
    r"^(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})$",
    re.IGNORECASE,
)
UUID_SHAPE_RE = re.compile(
    r"^(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$",
    re.IGNORECASE,
)
LEGACY_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
REDACTED = "[REDACTED]"
SECRET_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "cookie",
        "password",
        "secret",
        "token",
    }
)
TRACE_PRIVATE_KEYS = SECRET_KEYS | frozenset(
    {
        "candidate",
        "clarification_answer",
        "context",
        "critic",
        "effective_prompt",
        "feedback",
        "history",
        "planner",
        "prompt",
        "raw",
        "repair_trace",
        "response",
        "writer",
    }
)

_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class InvalidIdentifierError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


class UnsafeStoragePathError(RuntimeError):
    def __init__(self, path):
        self.code = "unsafe_storage_path"
        self.path = Path(path)
        super().__init__("storage path is outside its root or contains a symbolic link: {0}".format(path))


class CorruptStateError(RuntimeError):
    def __init__(self, path, quarantine_path, cause):
        self.code = "corrupt_state_json"
        self.path = Path(path)
        self.quarantine_path = Path(quarantine_path) if quarantine_path else None
        self.cause = cause
        message = "corrupt state JSON at {0}".format(self.path)
        if self.quarantine_path is not None:
            message += "; quarantined at {0}".format(self.quarantine_path)
        super().__init__(message)


def validate_identifier(value, code, allow_legacy=True):
    """Validate IDs without normalizing them.

    UUIDv4 (compact or canonical form) is the current format. The restricted
    legacy slug format remains readable for persisted sessions and API
    compatibility; it cannot contain separators or dot segments.
    """
    valid_uuid = False
    if isinstance(value, str) and UUID_RE.fullmatch(value):
        try:
            parsed = uuid.UUID(value)
            valid_uuid = parsed.version == 4
        except ValueError:
            valid_uuid = False
    uuid_shaped = bool(isinstance(value, str) and UUID_SHAPE_RE.fullmatch(value))
    valid_legacy = bool(
        allow_legacy
        and not uuid_shaped
        and isinstance(value, str)
        and LEGACY_IDENTIFIER_RE.fullmatch(value)
    )
    if not (valid_uuid or valid_legacy):
        raise InvalidIdentifierError(code=code, message=code)
    return value


def generate_identifier():
    return uuid.uuid4().hex


def _assert_not_symlink(path):
    if Path(path).is_symlink():
        raise UnsafeStoragePathError(path)


def ensure_directory(path):
    directory = Path(path)
    _assert_not_symlink(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _assert_not_symlink(directory)
    if not directory.is_dir():
        raise UnsafeStoragePathError(directory)
    return directory


def resolve_within_root(root, relative_name, code):
    resolved_root = Path(root).resolve()
    _assert_not_symlink(root)
    candidate = resolved_root / relative_name
    current = resolved_root
    try:
        relative_parts = candidate.relative_to(resolved_root).parts
    except ValueError:
        raise InvalidIdentifierError(code=code, message=code)
    for part in relative_parts:
        current = current / part
        _assert_not_symlink(current)
    resolved_path = candidate.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise InvalidIdentifierError(code=code, message=code)
    return resolved_path


def _fsync_directory(directory):
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        fd = os.open(str(directory), flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_json(path, payload):
    target_path = Path(path)
    ensure_directory(target_path.parent)
    _assert_not_symlink(target_path)
    fd, temp_path = tempfile.mkstemp(
        prefix=".localscript-",
        suffix=".tmp",
        dir=str(target_path.parent),
    )
    try:
        try:
            os.fchmod(fd, 0o600)
        except (AttributeError, OSError):
            pass
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target_path)
        _fsync_directory(target_path.parent)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _thread_lock_for(path):
    key = str(Path(path).resolve())
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


@contextlib.contextmanager
def file_lock(path):
    lock_path = Path(path)
    ensure_directory(lock_path.parent)
    _assert_not_symlink(lock_path)
    thread_lock = _thread_lock_for(lock_path)
    with thread_lock:
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(str(lock_path), flags, 0o600)
        try:
            if os.name == "nt":
                import msvcrt

                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"\0")
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            if os.name == "nt":
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def _quarantine_corrupt_file(path):
    source = Path(path)
    if not source.exists():
        return None
    quarantine = source.with_name(
        ".{0}.corrupt-{1}".format(source.name, generate_identifier())
    )
    os.replace(source, quarantine)
    _fsync_directory(source.parent)
    return quarantine


def read_json(path, quarantine=True, expected_type=None):
    source = Path(path)
    _assert_not_symlink(source)
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(str(source), flags)
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if expected_type is not None and not isinstance(payload, expected_type):
            raise TypeError(
                "state JSON must contain {0}".format(expected_type.__name__)
            )
        return payload
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        quarantine_path = _quarantine_corrupt_file(source) if quarantine else None
        raise CorruptStateError(source, quarantine_path, exc) from exc


def delete_file(path):
    target = Path(path)
    _assert_not_symlink(target)
    try:
        target.unlink()
    except FileNotFoundError:
        return False
    _fsync_directory(target.parent)
    return True


def utc_now(clock=None):
    value = clock() if clock is not None else datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("state clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def isoformat_utc(value):
    aware = value.astimezone(timezone.utc)
    return aware.isoformat(timespec="microseconds").replace("+00:00", "Z")


def parse_utc(value):
    if not isinstance(value, str):
        raise ValueError("timestamp must be a string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _normalized_key(key):
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def redact_nested(value, sensitive_keys=SECRET_KEYS):
    if isinstance(value, Mapping):
        redacted = {}
        for key, nested_value in value.items():
            normalized = _normalized_key(key)
            if (
                normalized in sensitive_keys
                or normalized.endswith("_token")
                or normalized.endswith("_password")
                or normalized.endswith("_secret")
            ):
                redacted[key] = REDACTED
            else:
                redacted[key] = redact_nested(nested_value, sensitive_keys)
        return redacted
    if isinstance(value, list):
        return [redact_nested(item, sensitive_keys) for item in value]
    if isinstance(value, tuple):
        return [redact_nested(item, sensitive_keys) for item in value]
    return value
