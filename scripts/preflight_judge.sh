#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PORT="${LOCALSCRIPT_PORT:-8080}"
OLLAMA_HOST="${LOCALSCRIPT_OLLAMA_HOST:-http://127.0.0.1:11434}"
PRIMARY_MODEL="${LOCALSCRIPT_PRIMARY_MODEL:-hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M}"
FALLBACK_MODEL="${LOCALSCRIPT_FALLBACK_MODEL:-qwen3:8b-q4_K_M}"
PYTHON_BIN="${LOCALSCRIPT_PYTHON_BIN:-}"

if [ -z "${PYTHON_BIN}" ]; then
  if [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  elif [ -x "/opt/venv/bin/python" ]; then
    PYTHON_BIN="/opt/venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

PASS_COUNT=0
FAIL_COUNT=0

pass() {
  PASS_COUNT=$((PASS_COUNT + 1))
  printf 'PASS %s\n' "$1"
}

fail_check() {
  FAIL_COUNT=$((FAIL_COUNT + 1))
  printf 'FAIL %s\n' "$1"
}

check_ollama_host() {
  if curl -fsS "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    pass "ollama_host_reachable ${OLLAMA_HOST}"
  else
    fail_check "ollama_host_unreachable ${OLLAMA_HOST}"
  fi
}

check_model_tags() {
  local tags
  tags="$("${PYTHON_BIN}" - <<PY
import json
from urllib.request import urlopen
host = "${OLLAMA_HOST}".rstrip("/")
try:
    with urlopen(host + "/api/tags", timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
except Exception:
    print("")
    raise SystemExit(0)
for item in payload.get("models", []):
    name = item.get("name")
    if isinstance(name, str) and name:
        print(name)
PY
)"
  if printf '%s\n' "${tags}" | grep -Fx "${PRIMARY_MODEL}" >/dev/null 2>&1; then
    pass "primary_model_tag_present ${PRIMARY_MODEL}"
  else
    fail_check "primary_model_tag_missing ${PRIMARY_MODEL}"
  fi
  if printf '%s\n' "${tags}" | grep -Fx "${FALLBACK_MODEL}" >/dev/null 2>&1; then
    pass "fallback_model_tag_present ${FALLBACK_MODEL}"
  else
    fail_check "fallback_model_tag_missing ${FALLBACK_MODEL}"
  fi
}

check_lua_runtime() {
  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
from app.validation.runtime import find_lua_binary, find_luac_binary
assert find_lua_binary()
assert find_luac_binary()
PY
  then
    pass "lua_runtime_available"
  else
    fail_check "lua_runtime_missing"
  fi
}

check_port_free() {
  if "${PYTHON_BIN}" - <<PY >/dev/null 2>&1
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.5)
try:
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", int("${PORT}"))) != 0 else 1)
finally:
    sock.close()
PY
  then
    pass "service_port_free ${PORT}"
  else
    fail_check "service_port_in_use ${PORT}"
  fi
}

check_runtime_profile() {
  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
from app.core.config import get_runtime_profile
profile = get_runtime_profile()
assert profile.model
assert profile.fallback_model
assert profile.num_ctx > 0
assert profile.num_predict > 0
PY
  then
    pass "runtime_profile_valid"
  else
    fail_check "runtime_profile_invalid"
  fi
}

check_doctor_judge() {
  local tmp_json tmp_err
  tmp_json="$(mktemp)"
  tmp_err="$(mktemp)"
  if LOCALSCRIPT_IGNORE_LOCK=1 "${PYTHON_BIN}" -m app.cli.main doctor --judge >"${tmp_json}" 2>"${tmp_err}"; then
    if "${PYTHON_BIN}" - <<PY >/dev/null 2>&1
import json
from pathlib import Path
payload = json.loads(Path("${tmp_json}").read_text(encoding="utf-8"))
assert payload.get("ok") is True
PY
    then
      pass "doctor_judge_green"
    else
      fail_check "doctor_judge_report_not_green"
    fi
  else
    fail_check "doctor_judge_failed $(tr '\n' ' ' < "${tmp_err}")"
  fi
  rm -f "${tmp_json}" "${tmp_err}"
}

check_ollama_host
check_model_tags
check_lua_runtime
check_port_free
check_runtime_profile
check_doctor_judge

printf 'SUMMARY pass=%s fail=%s\n' "${PASS_COUNT}" "${FAIL_COUNT}"
[ "${FAIL_COUNT}" -eq 0 ]
