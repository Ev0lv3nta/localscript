"""One parser for raw Lua and the supported JSON envelope."""

from __future__ import annotations

import json
from dataclasses import dataclass


class OutputParseError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ParsedOutput:
    chunks: tuple[str, ...]
    keys: tuple[str, ...] = ()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise OutputParseError("json_envelope_duplicate_key", "Envelope keys must be unique.")
        result[key] = value
    return result


def parse_output(code: object, output_style: str) -> ParsedOutput:
    if not isinstance(code, str) or not code.strip():
        raise OutputParseError("lua_chunk_empty", "Code must contain a non-empty Lua chunk.")
    if len(code.encode("utf-8")) > 131072:
        raise OutputParseError("code_too_large", "Code exceeds 128 KiB.")
    code = code.strip()
    if output_style == "lua_block":
        if code.startswith("lua{") and code.endswith("}lua"):
            raise OutputParseError("lua_block_wrapper_forbidden", "Raw Lua must not be wrapped.")
        return ParsedOutput((code,))
    if output_style != "json_envelope":
        raise OutputParseError("output_format_invalid", "Unsupported output format.")
    try:
        payload = json.loads(code, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, RecursionError) as error:
        raise OutputParseError("json_envelope_invalid", "Invalid JSON envelope.") from error
    if not isinstance(payload, dict) or not payload:
        raise OutputParseError("json_envelope_not_object", "Envelope must be a non-empty object.")
    if len(payload) > 16:
        raise OutputParseError(
            "envelope_too_many_chunks", "At most 16 envelope chunks are supported."
        )
    chunks: list[str] = []
    for key, value in payload.items():
        if not key or not isinstance(value, str):
            raise OutputParseError(
                "json_envelope_value_invalid", "Each key must contain wrapped Lua."
            )
        if not value.startswith("lua{") or not value.endswith("}lua"):
            raise OutputParseError(
                "json_envelope_wrapper_invalid", "Use lua{...}lua for each value."
            )
        chunk = value[4:-4]
        if not chunk.strip():
            raise OutputParseError("lua_chunk_empty", "Envelope chunks must not be empty.")
        chunks.append(chunk)
    return ParsedOutput(tuple(chunks), tuple(payload))
