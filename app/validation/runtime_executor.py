import json
import math
import os
import resource
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

from app.core.config import PROJECT_ROOT


@dataclass
class RuntimeExecutionResult:
    ok: bool
    value: object = None
    error_code: str = ""
    error_message: str = ""
    degraded: bool = False


DANGEROUS_LUA_PATTERNS = [
    ("os.", "dangerous_stdlib_os_forbidden", "Access to the `os` namespace is forbidden."),
    ("io.", "dangerous_stdlib_io_forbidden", "Access to the `io` namespace is forbidden."),
    ("package.", "dangerous_stdlib_package_forbidden", "Access to the `package` namespace is forbidden."),
    ("require(", "dangerous_stdlib_require_forbidden", "`require` is forbidden."),
    ("debug.", "dangerous_stdlib_debug_forbidden", "Access to the `debug` namespace is forbidden."),
    ("dofile(", "dangerous_stdlib_dofile_forbidden", "`dofile` is forbidden."),
    ("loadfile(", "dangerous_stdlib_loadfile_forbidden", "`loadfile` is forbidden."),
    ("collectgarbage(", "dangerous_stdlib_collectgarbage_forbidden", "`collectgarbage` is forbidden."),
]


def detect_unsafe_lua_usage(code):
    for token, error_code, message in DANGEROUS_LUA_PATTERNS:
        if token in (code or ""):
            return token, error_code, message
    return None


def _strip_lua_wrapper(code):
    stripped = (code or "").strip()
    if stripped.startswith("lua{") and stripped.endswith("}lua"):
        return stripped[4:-4]
    return stripped


def _extract_lua_chunks(code, output_style):
    if output_style != "json_envelope":
        return [_strip_lua_wrapper(code)]

    try:
        payload = json.loads(code)
    except json.JSONDecodeError:
        return []

    chunks = []
    for value in payload.values():
        if isinstance(value, str):
            chunks.append(_strip_lua_wrapper(value))
    return chunks


def _find_lua_binary():
    local_candidates = [
        PROJECT_ROOT / ".tools" / "lua54" / "bin" / "lua",
        PROJECT_ROOT / ".tools" / "lua-5.4.6" / "src" / "lua",
    ]
    for candidate in local_candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)

    preferred = os.getenv("LOCALSCRIPT_LUA_BIN")
    if preferred:
        if os.path.isfile(preferred) and os.access(preferred, os.X_OK):
            return preferred

    for candidate in ["lua5.4", "lua"]:
        path = shutil.which(candidate)
        if path:
            return path
    return None


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
        return 'setmetatable({%s}, { __localscript_array = true })' % inner
    if isinstance(value, dict):
        items = []
        for key, nested in value.items():
            items.append("[{0}] = {1}".format(_lua_string_literal(str(key)), _serialize_to_lua(nested)))
        return "{%s}" % ", ".join(items)
    return "nil"


def _build_runner(chunk, context):
    serialized_context = _serialize_to_lua(context or {})
    serialized_chunk = _lua_string_literal(chunk)
    return """
local wf_container = {context_literal}
wf = wf_container.wf or {{}}
if wf.vars == nil then
  wf.vars = {{}}
end
if wf.initVariables == nil then
  wf.initVariables = {{}}
end

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
  return setmetatable(value or {{}}, {{ __localscript_array = true }})
end

_utils = {{
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
  next = next,
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

  local keys = {{}}
  for key, _ in pairs(value) do
    keys[#keys + 1] = tostring(key)
  end
  table.sort(keys)

  local parts = {{}}
  for _, key in ipairs(keys) do
    parts[#parts + 1] = '"' .. _ls_escape_string(key) .. '":' .. _ls_to_json(value[key])
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

io.write('{{"ok":true,"value":' .. _ls_to_json(result) .. '}}')
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
    unsafe_usage = detect_unsafe_lua_usage(chunk)
    if unsafe_usage:
        _, error_code, message = unsafe_usage
        return RuntimeExecutionResult(
            ok=False,
            error_code=error_code,
            error_message=message,
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
        completed = subprocess.run(
            [lua_binary, temp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            close_fds=True,
            env={"LC_ALL": "C.UTF-8"},
            preexec_fn=_subprocess_limits(),
        )
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    if completed.returncode != 0:
        return RuntimeExecutionResult(
            ok=False,
            error_code="lua_runtime_error",
            error_message=completed.stderr.strip() or completed.stdout.strip(),
        )

    stdout = completed.stdout.strip()
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
    if output_style == "json_envelope":
        try:
            payload = json.loads(code)
        except json.JSONDecodeError as exc:
            return RuntimeExecutionResult(
                ok=False,
                error_code="json_envelope_invalid",
                error_message=str(exc),
            )

        result = {}
        for key, chunk in payload.items():
            chunk_result = _run_chunk(_extract_lua_chunks(json.dumps({key: chunk}), "json_envelope")[0], context)
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
