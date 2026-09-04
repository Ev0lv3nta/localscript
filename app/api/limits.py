from __future__ import annotations

import json

from pydantic import JsonValue


class APIConstraintError(ValueError):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def validate_prompt(prompt: str | None, max_chars: int) -> None:
    if prompt is None:
        return
    if len(prompt) > int(max_chars):
        raise APIConstraintError(
            status_code=422,
            code="prompt_too_long",
            message="prompt exceeds maximum allowed length",
        )


def validate_context(
    context: JsonValue,
    max_bytes: int,
    max_depth: int,
    max_nodes: int,
) -> dict[str, int]:
    if context is None:
        return {
            "serialized_bytes": 0,
            "depth": 0,
            "nodes": 0,
        }

    serialized = json.dumps(context, ensure_ascii=False, sort_keys=True)
    if len(serialized.encode("utf-8")) > int(max_bytes):
        raise APIConstraintError(
            status_code=422,
            code="context_too_large",
            message="context exceeds maximum serialized size",
        )

    depth, nodes = _measure_context(context)
    if depth > int(max_depth):
        raise APIConstraintError(
            status_code=422,
            code="context_too_deep",
            message="context exceeds maximum nesting depth",
        )
    if nodes > int(max_nodes):
        raise APIConstraintError(
            status_code=422,
            code="context_too_wide",
            message="context exceeds maximum node budget",
        )
    return {
        "serialized_bytes": len(serialized.encode("utf-8")),
        "depth": depth,
        "nodes": nodes,
    }


def _measure_context(value: JsonValue, depth: int = 1) -> tuple[int, int]:
    if isinstance(value, dict):
        total_nodes = len(value)
        max_depth = depth
        for nested in value.values():
            nested_depth, nested_nodes = _measure_context(nested, depth + 1)
            max_depth = max(max_depth, nested_depth)
            total_nodes += nested_nodes
        return max_depth, total_nodes
    if isinstance(value, list):
        total_nodes = len(value)
        max_depth = depth
        for nested in value:
            nested_depth, nested_nodes = _measure_context(nested, depth + 1)
            max_depth = max(max_depth, nested_depth)
            total_nodes += nested_nodes
        return max_depth, total_nodes
    return depth, 1
