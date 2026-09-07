"""Request-local deadlines shared by model and Lua operations."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic

from app.generation.backend_errors import BackendTimeout

_deadline: ContextVar[float | None] = ContextVar("workflow_deadline", default=None)


def remaining_seconds(cap: float) -> float:
    deadline = _deadline.get()
    remaining = cap if deadline is None else min(cap, deadline - monotonic())
    if remaining <= 0:
        raise BackendTimeout(reason="workflow_deadline_exceeded")
    return remaining


@contextmanager
def workflow_budget(seconds: float = 180.0) -> Iterator[None]:
    previous = _deadline.get()
    deadline = monotonic() + seconds
    token = _deadline.set(min(previous, deadline) if previous is not None else deadline)
    try:
        yield
    finally:
        _deadline.reset(token)
