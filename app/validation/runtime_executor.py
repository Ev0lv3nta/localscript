import json
import math
import os
import resource
import subprocess
import tempfile
from dataclasses import dataclass

from app.validation.lua_ast import analyze_lua_chunk
from app.validation.runtime import find_lua_binary


@dataclass
class RuntimeExecutionResult:
    ok: bool
    value: object = None
    error_code: str = ""
    error_message: str = ""
    degraded: bool = False


def _strip_lua_wrapper(code):
    if not isinstance(code, str):
        return ""
    stripped = code.strip()
    if stripped.startswith("lua{") and stripped.endswith("}lua"):
        return stripped[4:-4]
    return stripped


def _extract_lua_chunks(code, output_style):
    if output_style != "json_envelope":
        if not isinstance(code, str) or not code.strip():
            return []
        return [_strip_lua_wrapper(code)]

    if not isinstance(code, str):
        return []

    try:
        payload = json.loads(code)
    except (ValueError, TypeError, RecursionError):
        return []

    if not isinstance(payload, dict) or not payload:
        return []

    chunks = []
    for value in payload.values():
        if not (isinstance(value, str) and value.startswith("lua{") and value.endswith("}lua")):
            return []
        chunks.append(_strip_lua_wrapper(value))
    return chunks


def _find_lua_binary():
    return find_lua_binary()


def _lua_string_literal(value):
    chunks = ['"']
    for byte in value.encode("utf-8"):
        if byte == 34:
            chunks.append('\\"')
        elif byte == 92:
            chunks.append("\\\\")
        elif 32 <= byte <= 126:
            chunks.append(chr(byte))
        else:
            chunks.append("\\{0:03d}".format(byte))
    chunks.append('"')
    return "".join(chunks)


def _lua_number_literal(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "nil"
        return repr(value)
    return "nil"


def _serialize_to_lua(value):
    if value is None:
        return "nil"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _lua_number_literal(value)
    if isinstance(value, str):
        return _lua_string_literal(value)
    if isinstance(value, list):
        inner = ", ".join(_serialize_to_lua(item) for item in value)
        return "setmetatable({%s}, { __localscript_array = true })" % inner
    if isinstance(value, dict):
        items = []
        for key, nested in value.items():
            items.append(
                "[{0}] = {1}".format(_lua_string_literal(str(key)), _serialize_to_lua(nested))
            )
        return "{%s}" % ", ".join(items)
    return "nil"


def _build_runner(chunk, context):
    serialized_context = _serialize_to_lua(context or {})
    serialized_chunk = _lua_string_literal(chunk)
    return """
local wf_container = {context_literal}
local raw_wf = wf_container.wf or {{}}
if raw_wf.vars == nil then
  raw_wf.vars = {{}}
end
if raw_wf.initVariables == nil then
  raw_wf.initVariables = {{}}
end

local function _ls_readonly(value, cache)
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
      return _ls_readonly(value[key], cache)
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
        return key, _ls_readonly(nested, cache)
      end
      return iterate, proxy, nil
    end,
    __localscript_array = source_mt and source_mt.__localscript_array or nil,
    __localscript_readonly = true,
  }}
  return setmetatable(proxy, mt)
end

local wf = _ls_readonly(raw_wf)

local function _ls_is_array(tbl)
  if type(tbl) ~= "table" then
    return false
  end
  local mt = getmetatable(tbl)
  if mt and mt.__localscript_array then
    return true
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
    return false
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
          if produced ~= nil and produced ~= false then
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
  _utils = _utils,
  math = math,
  string = string,
  table = safe_table,
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
  utf8 = utf8,
}}

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

local function _ls_to_json(value)
  local value_type = type(value)
  if value == nil then
    return "null"
  end
  if value_type == "boolean" then
    return value and "true" or "false"
  end
  if value_type == "number" then
    return tostring(value)
  end
  if value_type == "string" then
    return '"' .. _ls_escape_string(value) .. '"'
  end
  if value_type ~= "table" then
    return '"' .. _ls_escape_string(tostring(value)) .. '"'
  end
  if _ls_is_array(value) then
    local parts = {{}}
    for index = 1, #value do
      parts[#parts + 1] = _ls_to_json(value[index])
    end
    return "[" .. table.concat(parts, ",") .. "]"
  end

  local entries = {{}}
  local labels = {{}}
  for key, _ in pairs(value) do
    local key_type = type(key)
    if key_type ~= "string" and key_type ~= "number" then
      error("JSON object keys must be strings or numbers")
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
    parts[#parts + 1] = '"' .. _ls_escape_string(entry.label) .. '":' .. _ls_to_json(value[entry.key])
  end
  return "{{" .. table.concat(parts, ",") .. "}}"
end

local chunk, load_error = load({chunk_literal}, "localscript_generated", "t", safe_env)
if not chunk then
  io.write('{{"ok":false,"error_code":"lua_load_error","error_message":' .. _ls_to_json(load_error) .. '}}')
  return
end

local ok, result = pcall(chunk)
if not ok then
  io.write('{{"ok":false,"error_code":"lua_runtime_error","error_message":' .. _ls_to_json(result) .. '}}')
  return
end

local serialization_ok, serialized_result = pcall(_ls_to_json, result)
if not serialization_ok then
  io.write('{{"ok":false,"error_code":"lua_result_serialization_error","error_message":"' .. _ls_escape_string(tostring(serialized_result)) .. '"}}')
  return
end

io.write('{{"ok":true,"value":' .. serialized_result .. '}}')
""".format(context_literal=serialized_context, chunk_literal=serialized_chunk)


def _subprocess_limits():
    def _apply_limits():
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
        except Exception:
            pass

    return _apply_limits


def _run_chunk(chunk, context):
    policy_result = analyze_lua_chunk(chunk)
    if not policy_result.ok:
        finding = policy_result.findings[0]
        return RuntimeExecutionResult(
            ok=False,
            error_code=finding.code,
            error_message="Line {0}, column {1}: {2}".format(
                finding.line,
                finding.column,
                finding.message,
            ),
        )

    lua_binary = _find_lua_binary()
    if not lua_binary:
        return RuntimeExecutionResult(
            ok=False,
            error_code="lua_runtime_missing",
            error_message="Lua runtime is unavailable; semantic execution ran in degraded mode.",
            degraded=True,
        )

    runner = _build_runner(chunk, context)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".lua", delete=False) as handle:
        handle.write(runner)
        temp_path = handle.name

    try:
        try:
            completed = subprocess.run(
                [lua_binary, temp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                close_fds=True,
                env={"LC_ALL": "C.UTF-8"},
                preexec_fn=_subprocess_limits(),
            )
        except subprocess.TimeoutExpired:
            return RuntimeExecutionResult(
                ok=False,
                error_code="lua_runtime_timeout",
                error_message="Lua execution exceeded the 5 second timeout.",
            )
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

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

    return RuntimeExecutionResult(
        ok=payload.get("ok") is True,
        value=payload.get("value"),
        error_code=payload.get("error_code", ""),
        error_message=payload.get("error_message", ""),
    )


def execute_output(code, context=None, output_style="lua_block"):
    if not isinstance(code, str):
        return RuntimeExecutionResult(
            ok=False,
            error_code="contract_not_string",
            error_message="Generated output must be a string.",
        )

    if output_style == "json_envelope":
        try:
            payload = json.loads(code)
        except (ValueError, TypeError, RecursionError) as exc:
            return RuntimeExecutionResult(
                ok=False,
                error_code="json_envelope_invalid",
                error_message=str(exc),
            )

        if not isinstance(payload, dict):
            return RuntimeExecutionResult(
                ok=False,
                error_code="json_envelope_not_object",
                error_message="Envelope must be a JSON object.",
            )
        if not payload:
            return RuntimeExecutionResult(
                ok=False,
                error_code="json_envelope_empty",
                error_message="Envelope must contain at least one Lua value.",
            )

        for key, chunk in payload.items():
            if not isinstance(chunk, str):
                return RuntimeExecutionResult(
                    ok=False,
                    error_code="json_envelope_value_not_string",
                    error_message="Envelope value for `{0}` must be a string.".format(key),
                )
            if not chunk.startswith("lua{") or not chunk.endswith("}lua"):
                return RuntimeExecutionResult(
                    ok=False,
                    error_code="json_envelope_value_not_lua_wrapper",
                    error_message="Envelope value for `{0}` must use `lua{{...}}lua`.".format(key),
                )

        result = {}
        chunks = _extract_lua_chunks(code, "json_envelope")
        if not chunks:
            return RuntimeExecutionResult(
                ok=False,
                error_code="lua_chunk_missing",
                error_message="No executable Lua chunk could be extracted.",
            )
        if any(not chunk.strip() for chunk in chunks):
            return RuntimeExecutionResult(
                ok=False,
                error_code="lua_chunk_missing",
                error_message="No executable Lua chunk could be extracted.",
            )
        for (key, _), chunk in zip(payload.items(), chunks):
            chunk_result = _run_chunk(chunk, context)
            if not chunk_result.ok:
                return chunk_result
            result[key] = chunk_result.value
        return RuntimeExecutionResult(ok=True, value=result)

    chunks = _extract_lua_chunks(code, output_style)
    if not chunks:
        return RuntimeExecutionResult(
            ok=False,
            error_code="lua_chunk_missing",
            error_message="No executable Lua chunk could be extracted.",
        )
    return _run_chunk(chunks[0], context)
