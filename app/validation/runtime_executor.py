from __future__ import annotations

import contextlib
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import BinaryIO

from app.core.budgets import remaining_seconds
from app.validation.lua_ast import analyze_lua_chunk
from app.validation.output import OutputParseError, parse_output
from app.validation.runtime import find_lua_binary, runtime_version


@dataclass
class RuntimeExecutionResult:
    ok: bool
    value: object = None
    error_code: str = ""
    error_message: str = ""
    degraded: bool = False
    read_roots: tuple[str, ...] = ()


def _find_lua_binary() -> str | None:
    binary = find_lua_binary()
    return binary if binary and runtime_version(binary) else None


def _lua_string_literal(value: str) -> str:
    chunks = ['"']
    for byte in value.encode("utf-8"):
        if byte == 34:
            chunks.append('\\"')
        elif byte == 92:
            chunks.append("\\\\")
        elif 32 <= byte <= 126:
            chunks.append(chr(byte))
        else:
            chunks.append(f"\\{byte:03d}")
    chunks.append('"')
    return "".join(chunks)


def _lua_number_literal(value: float) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        if abs(value) > 2**53 - 1:
            raise ValueError("integer_out_of_range")
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("non_finite_number")
        if abs(value) > 2**53 - 1:
            raise ValueError("number_out_of_range")
        return repr(value)
    return "nil"


def _serialize_to_lua(value: object, path: str = "$", depth: int = 0) -> str:
    if depth > 16:
        raise ValueError(f"context_too_deep: {path}")
    if value is None:
        raise ValueError(f"nested_null_unsupported: {path}")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _lua_number_literal(value)
    if isinstance(value, str):
        return _lua_string_literal(value)
    if isinstance(value, list):
        inner = ", ".join(
            _serialize_to_lua(item, f"{path}[{i}]", depth + 1) for i, item in enumerate(value)
        )
        # Процентный формат: шаблон сам состоит из фигурных скобок Lua.
        return "setmetatable({%s}, { __localscript_array = true })" % inner  # noqa: UP031
    if isinstance(value, dict):
        items = []
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError(f"object_key_not_string: {path}")
            items.append(
                f"[{_lua_string_literal(key)}] = {_serialize_to_lua(nested, f'{path}[{key!r}]', depth + 1)}"
            )
        return "setmetatable({%s}, { __localscript_object = true })" % ", ".join(items)  # noqa: UP031
    raise ValueError(f"unsupported_json_value: {path}")


def _build_runner(chunk: str, context: object, output_shape: str | None = None) -> str:
    serialized_context = _serialize_to_lua(context or {})
    declared_array = "true" if output_shape == "array" else "false"
    serialized_chunk = _lua_string_literal(chunk)
    return f"""
local wf_container = {serialized_context}
local raw_wf = wf_container.wf or {{}}
if raw_wf.vars == nil then
  raw_wf.vars = {{}}
end
if raw_wf.initVariables == nil then
  raw_wf.initVariables = {{}}
end

local _ls_read_roots = {{
  ["wf.vars"] = false,
  ["wf.initVariables"] = false,
}}

local function _ls_child_root(parent_root, key)
  if parent_root == "wf" and key == "vars" then
    _ls_read_roots["wf.vars"] = true
    return "wf.vars"
  end
  if parent_root == "wf" and key == "initVariables" then
    _ls_read_roots["wf.initVariables"] = true
    return "wf.initVariables"
  end
  return parent_root
end

local function _ls_readonly(value, cache, source_root)
  if type(value) ~= "table" then
    return value
  end
  cache = cache or {{}}
  if cache[value] ~= nil then
    return cache[value]
  end

  local proxy = {{}}
  cache[value] = proxy
  local source_mt = getmetatable(value)
  local mt = {{
    __index = function(_, key)
      return _ls_readonly(value[key], cache, _ls_child_root(source_root, key))
    end,
    __newindex = function()
      error("workflow input is read-only", 2)
    end,
    __len = function()
      return #value
    end,
    __pairs = function()
      local function iterate(_, previous)
        local key, nested = next(value, previous)
        if key == nil then
          return nil
        end
        return key, _ls_readonly(nested, cache, _ls_child_root(source_root, key))
      end
      return iterate, proxy, nil
    end,
    __localscript_array = source_mt and source_mt.__localscript_array or nil,
    __localscript_object = source_mt and source_mt.__localscript_object or nil,
    __localscript_readonly = true,
  }}
  return setmetatable(proxy, mt)
end

local wf = _ls_readonly(raw_wf, nil, "wf")

local function _ls_is_array(tbl)
  if type(tbl) ~= "table" then
    return false
  end
  local mt = getmetatable(tbl)
  if mt and mt.__localscript_object then
    return false
  end
  local max_index = 0
  local count = 0
  for key, _ in pairs(tbl) do
    if type(key) ~= "number" or key < 1 or math.floor(key) ~= key then
      return false
    end
    if key > max_index then
      max_index = key
    end
    count = count + 1
  end
  if count == 0 then
    return mt and mt.__localscript_array or false
  end
  return max_index == count
end

local function _ls_array(value)
  value = value or {{}}
  local mt = getmetatable(value)
  if mt and mt.__localscript_readonly then
    local copy = {{}}
    for index, nested in ipairs(value) do
      copy[index] = nested
    end
    value = copy
  end
  return setmetatable(value, {{ __localscript_array = true }})
end

local _utils = {{
  array = {{
    new = function(arg1, arg2)
      if arg1 == nil then
        return _ls_array({{}})
      end
      if type(arg1) == "function" then
        local result = _ls_array({{}})
        local source = arg2 or {{}}
        for _, item in ipairs(source) do
          local produced = arg1(item)
          if produced ~= nil then
            table.insert(result, produced)
          end
        end
        return result
      end
      if type(arg1) == "table" then
        if _ls_is_array(arg1) then
          return _ls_array(arg1)
        end
        return arg1
      end
      return _ls_array({{arg1}})
    end,
    markAsArray = function(arr)
      return _ls_array(arr or {{}})
    end
  }}
}}

local safe_table = {{
  insert = table.insert,
  concat = table.concat,
  sort = table.sort,
  remove = table.remove,
  unpack = table.unpack or unpack,
}}

local safe_env = {{
  wf = wf,
  _utils = _ls_readonly(_utils),
  math = _ls_readonly(math),
  string = _ls_readonly(string),
  table = _ls_readonly(safe_table),
  tonumber = tonumber,
  tostring = tostring,
  type = type,
  pairs = pairs,
  ipairs = ipairs,
  select = select,
  assert = assert,
  error = error,
  pcall = pcall,
  xpcall = xpcall,
  utf8 = _ls_readonly(utf8),
}}

local function _ls_read_roots_json()
  local roots = {{}}
  if _ls_read_roots["wf.vars"] then
    roots[#roots + 1] = '"wf.vars"'
  end
  if _ls_read_roots["wf.initVariables"] then
    roots[#roots + 1] = '"wf.initVariables"'
  end
  return "[" .. table.concat(roots, ",") .. "]"
end

local function _ls_escape_string(value)
  value = value:gsub("\\\\", "\\\\\\\\")
  value = value:gsub('"', '\\\\"')
  value = value:gsub("\\b", "\\\\b")
  value = value:gsub("\\f", "\\\\f")
  value = value:gsub("\\n", "\\\\n")
  value = value:gsub("\\r", "\\\\r")
  value = value:gsub("\\t", "\\\\t")
  value = value:gsub("[%z\\1-\\31]", function(ch)
    return string.format("\\\\u%04x", string.byte(ch))
  end)
  return value
end

local active_tables = {{}}
local function _ls_to_json(value, depth)
  depth = depth or 0
  if depth > 16 then error("result_too_deep") end
  local value_type = type(value)
  if value == nil then
    return "null"
  end
  if value_type == "boolean" then
    return value and "true" or "false"
  end
  if value_type == "number" then
    if value ~= value or value == math.huge or value == -math.huge or math.abs(value) > 9007199254740991 then
      error("unsupported_json_number")
    end
    if math.type(value) == "float" then return string.format("%.17g", value) end
    return tostring(value)
  end
  if value_type == "string" then
    return '"' .. _ls_escape_string(value) .. '"'
  end
  if value_type ~= "table" then
    error("unsupported_json_result_type")
  end
  if active_tables[value] then error("cyclic_result") end
  active_tables[value] = true
  if _ls_is_array(value) then
    local parts = {{}}
    for index = 1, #value do
      parts[#parts + 1] = _ls_to_json(value[index], depth + 1)
    end
    active_tables[value] = nil
    return "[" .. table.concat(parts, ",") .. "]"
  end

  local entries = {{}}
  local labels = {{}}
  for key, _ in pairs(value) do
    local key_type = type(key)
    if key_type ~= "string" then
      error("JSON object keys must be strings; sparse/mixed arrays are unsupported")
    end
    local label = tostring(key)
    if labels[label] then
      error("JSON object keys collide after string conversion")
    end
    labels[label] = true
    entries[#entries + 1] = {{ key = key, label = label }}
  end
  table.sort(entries, function(left, right)
    return left.label < right.label
  end)

  local parts = {{}}
  for _, entry in ipairs(entries) do
    parts[#parts + 1] = '"' .. _ls_escape_string(entry.label) .. '":' .. _ls_to_json(value[entry.key], depth + 1)
  end
  active_tables[value] = nil
  return "{{" .. table.concat(parts, ",") .. "}}"
end

local chunk, load_error = load({serialized_chunk}, "localscript_generated", "t", safe_env)
if not chunk then
  io.write('{{"ok":false,"error_code":"lua_load_error","error_message":' .. _ls_to_json(load_error) .. ',"read_roots":' .. _ls_read_roots_json() .. '}}')
  return
end

local ok, result = pcall(chunk)
if not ok then
  io.write('{{"ok":false,"error_code":"lua_runtime_error","error_message":' .. _ls_to_json(result) .. ',"read_roots":' .. _ls_read_roots_json() .. '}}')
  return
end

-- Lua не различает пустой массив и пустой объект: и то и другое — таблица без ключей.
-- Когда контракт объявляет массив, пустой результат обязан приехать как [], иначе требование
-- «на пустом входе верни пустой массив» невыполнимо в принципе.
local result_mt = type(result) == "table" and getmetatable(result)
local result_empty = type(result) == "table" and pairs(result)(result) == nil
if {declared_array} and result_empty and not (result_mt and result_mt.__localscript_object) then
  result = _ls_array(result)
end

local serialization_ok, serialized_result = pcall(_ls_to_json, result)
if not serialization_ok then
  io.write('{{"ok":false,"error_code":"lua_result_serialization_error","error_message":"' .. _ls_escape_string(tostring(serialized_result)) .. '","read_roots":' .. _ls_read_roots_json() .. '}}')
  return
end

if #serialized_result > 65536 then
  io.write('{{"ok":false,"error_code":"lua_result_too_large","error_message":"Result exceeds 64 KiB.","read_roots":' .. _ls_read_roots_json() .. '}}')
  return
end

io.write('{{"ok":true,"value":' .. serialized_result .. ',"read_roots":' .. _ls_read_roots_json() .. '}}')
"""


def _execute_process(
    command: list[str], stdout: BinaryIO, stderr: BinaryIO, timeout: float
) -> subprocess.CompletedProcess[bytes]:
    if sys.platform != "darwin":
        return subprocess.run(
            command,
            stdout=stdout,
            stderr=stderr,
            timeout=timeout,
            close_fds=True,
            env={"LC_ALL": "C.UTF-8"},
        )
    # Development on macOS: RSS monitoring is not Linux's hard address-space limit.
    # Failure to measure an active child also stops execution rather than removing the cap.
    with subprocess.Popen(
        command, stdout=stdout, stderr=stderr, close_fds=True, env={"LC_ALL": "C.UTF-8"}
    ) as child:
        deadline = monotonic() + timeout
        try:
            while True:
                try:
                    child.wait(timeout=min(0.025, max(0.001, deadline - monotonic())))
                    break
                except subprocess.TimeoutExpired:
                    if monotonic() >= deadline:
                        raise subprocess.TimeoutExpired(command, timeout) from None
                    sample = subprocess.run(
                        ["/bin/ps", "-o", "rss=", "-p", str(child.pid)],
                        capture_output=True,
                        timeout=0.5,
                    )
                    if child.poll() is not None:
                        break
                    if sample.returncode or not sample.stdout.strip().isdigit():
                        raise OSError("lua_memory_monitor_unavailable") from None
                    if int(sample.stdout) > 256 * 1024:
                        raise OSError("lua_memory_limit_exceeded") from None
        except BaseException:
            child.kill()
            child.wait()
            raise
        return subprocess.CompletedProcess(command, child.returncode)


def _run_chunk(
    chunk: str,
    context: object,
    output_shape: str | None = None,
    deadline: float | None = None,
) -> RuntimeExecutionResult:
    policy_result = analyze_lua_chunk(chunk)
    if not policy_result.ok:
        finding = policy_result.findings[0]
        return RuntimeExecutionResult(
            ok=False,
            error_code=finding.code,
            error_message=f"Line {finding.line}, column {finding.column}: {finding.message}",
        )

    lua_binary = _find_lua_binary()
    if not lua_binary:
        return RuntimeExecutionResult(
            ok=False,
            error_code="lua_runtime_missing",
            error_message="Lua runtime is unavailable; semantic execution ran in degraded mode.",
            degraded=True,
        )

    try:
        runner = _build_runner(chunk, context, output_shape)
    except (ValueError, RecursionError) as error:
        return RuntimeExecutionResult(
            ok=False, error_code="unsupported_json_context", error_message=str(error)
        )
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".lua", delete=False) as handle:
        handle.write(runner)
        temp_path = handle.name

    try:
        try:
            remaining = (
                min(remaining_seconds(5.0), deadline - monotonic())
                if deadline is not None
                else remaining_seconds(5.0)
            )
            if remaining <= 0:
                raise subprocess.TimeoutExpired(lua_binary, 0)
            with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
                completed = _execute_process(
                    [
                        sys.executable,
                        "-I",
                        str(Path(__file__).with_name("worker.py")),
                        lua_binary,
                        temp_path,
                    ],
                    stdout_file,
                    stderr_file,
                    remaining,
                )
                stdout_file.seek(0)
                stderr_file.seek(0)
                completed.stdout = stdout_file.read(65536 + 4097)
                completed.stderr = stderr_file.read(4096)
        except subprocess.TimeoutExpired:
            return RuntimeExecutionResult(
                ok=False,
                error_code="lua_runtime_timeout",
                error_message="Lua execution exceeded the 5 second timeout.",
            )
        except OSError as error:
            return RuntimeExecutionResult(
                ok=False, error_code="lua_process_failed", error_message=str(error)
            )
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp_path)

    try:
        stdout = (completed.stdout or b"").decode("utf-8")
    except UnicodeDecodeError:
        return RuntimeExecutionResult(
            ok=False,
            error_code="lua_runtime_invalid_utf8",
            error_message="Lua subprocess stdout is not valid UTF-8.",
        )
    try:
        stderr = (completed.stderr or b"").decode("utf-8")
    except UnicodeDecodeError:
        return RuntimeExecutionResult(
            ok=False,
            error_code="lua_runtime_invalid_utf8",
            error_message="Lua subprocess stderr is not valid UTF-8.",
        )

    if completed.returncode != 0:
        return RuntimeExecutionResult(
            ok=False,
            error_code="lua_runtime_error",
            error_message=stderr.strip() or stdout.strip(),
        )

    stdout = stdout.strip()
    try:
        payload = json.loads(stdout or "{}")
    except json.JSONDecodeError:
        return RuntimeExecutionResult(
            ok=False,
            error_code="lua_runtime_invalid_output",
            error_message=stdout,
        )

    if not isinstance(payload, dict):
        return RuntimeExecutionResult(
            ok=False,
            error_code="lua_runtime_invalid_output",
            error_message="Lua subprocess output must be a JSON object.",
        )
    raw_read_roots = payload.get("read_roots", [])
    if not isinstance(raw_read_roots, list) or any(
        not isinstance(root, str) or root not in {"wf.vars", "wf.initVariables"}
        for root in raw_read_roots
    ):
        return RuntimeExecutionResult(
            ok=False,
            error_code="lua_runtime_invalid_output",
            error_message="Lua subprocess returned invalid workflow root metadata.",
        )
    read_roots = tuple(root for root in ("wf.vars", "wf.initVariables") if root in raw_read_roots)

    return RuntimeExecutionResult(
        ok=payload.get("ok") is True,
        value=payload.get("value"),
        error_code=payload.get("error_code", ""),
        error_message=payload.get("error_message", ""),
        read_roots=read_roots,
    )


def execute_output(
    code: object,
    context: object = None,
    output_style: str = "lua_block",
    output_shape: str | None = None,
) -> RuntimeExecutionResult:
    deadline = monotonic() + 20.0
    try:
        parsed = parse_output(code, output_style)
    except OutputParseError as error:
        return RuntimeExecutionResult(ok=False, error_code=error.code, error_message=str(error))
    if not parsed.keys:
        return _run_chunk(parsed.chunks[0], context, output_shape, deadline)
    result: dict[str, object] = {}
    read_roots: set[str] = set()
    for key, chunk in zip(parsed.keys, parsed.chunks, strict=True):
        execution = _run_chunk(chunk, context, deadline=deadline)
        if not execution.ok:
            return execution
        read_roots.update(execution.read_roots)
        result[key] = execution.value
    if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > 65536:
        return RuntimeExecutionResult(
            ok=False, error_code="lua_result_too_large", error_message="Result exceeds 64 KiB."
        )
    return RuntimeExecutionResult(
        ok=True,
        value=result,
        read_roots=tuple(root for root in ("wf.vars", "wf.initVariables") if root in read_roots),
    )
