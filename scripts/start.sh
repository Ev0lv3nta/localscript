#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

fail() {
  printf 'localscript start: %s\n' "$1" >&2
  exit 1
}

resolve_python_bin() {
  if [ -n "${LOCALSCRIPT_PYTHON_BIN:-}" ]; then
    printf '%s\n' "${LOCALSCRIPT_PYTHON_BIN}"
    return 0
  fi
  if [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
    printf '%s\n' "${ROOT_DIR}/.venv/bin/python"
    return 0
  fi
  if [ -x "/opt/venv/bin/python" ]; then
    printf '%s\n' "/opt/venv/bin/python"
    return 0
  fi
  command -v python3 >/dev/null 2>&1 || fail "python3 was not found"
  command -v python3
}

load_effective_config() {
  "${PYTHON_BIN}" - <<'PY'
import shlex

from app.core.config import get_runtime_profile

profile = get_runtime_profile()
values = {
    "PORT": profile.port,
    "OLLAMA_HOST": profile.ollama_host,
    "OLLAMA_MODE": profile.ollama_mode,
    "PRIMARY_MODEL": profile.model,
    "STARTUP_TIMEOUT_SECONDS": profile.startup_timeout_seconds,
    "OLLAMA_POLL_INTERVAL_SECONDS": profile.ollama_poll_interval_seconds,
    "UVI_HOST": profile.bind_host,
    "REMOTE_MODE": "1" if profile.remote_mode else "0",
    "REMOTE_TOKEN": profile.remote_token,
    "UI_ENABLED": "1" if profile.ui_enabled else "0",
    "PROFILE_NAME": profile.name,
}
for name, value in values.items():
    print(f"{name}={shlex.quote(str(value))}")
PY
}

python_version_triplet() {
  "${PYTHON_BIN}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
PY
}

validate_python_runtime() {
  "${PYTHON_BIN}" - <<PY
import sys
minimum = (3, int("${SUPPORTED_PYTHON_MIN_MINOR}"))
maximum = (3, int("${SUPPORTED_PYTHON_MAX_MINOR}"))
current = sys.version_info[:2]
if current < minimum or current > maximum:
    raise SystemExit(
        "unsupported_python::{0}.{1}::expected >=3.{2},<=3.{3}".format(
            current[0], current[1], minimum[1], maximum[1]
        )
    )
PY
}

fetch_model_tags() {
  "${PYTHON_BIN}" - <<PY
import json
from urllib.request import ProxyHandler, build_opener

host = "${OLLAMA_HOST}".rstrip("/")
opener = build_opener(ProxyHandler({}))
with opener.open(host + "/api/tags", timeout=5) as response:
    payload = json.loads(response.read().decode("utf-8"))

for item in payload.get("models", []):
    name = item.get("name")
    if isinstance(name, str) and name:
        print(name)
PY
}

wait_for_ollama() {
  local remaining request_timeout sleep_for
  SECONDS=0
  while [ "${SECONDS}" -lt "${STARTUP_TIMEOUT_SECONDS}" ]; do
    remaining=$((STARTUP_TIMEOUT_SECONDS - SECONDS))
    request_timeout="${remaining}"
    if [ "${request_timeout}" -gt 5 ]; then
      request_timeout=5
    fi
    if curl --noproxy '*' --connect-timeout 3 --max-time "${request_timeout}" \
      -fsS "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
      return 0
    fi
    remaining=$((STARTUP_TIMEOUT_SECONDS - SECONDS))
    [ "${remaining}" -gt 0 ] || break
    sleep_for="${OLLAMA_POLL_INTERVAL_SECONDS}"
    if [ "${sleep_for}" -gt "${remaining}" ]; then
      sleep_for="${remaining}"
    fi
    sleep "${sleep_for}"
  done
  fail "Ollama did not become reachable at the configured endpoint within ${STARTUP_TIMEOUT_SECONDS}s"
}

validate_bind_policy() {
  case "${UVI_HOST}" in
    127.0.0.1|localhost|::1)
      return 0
      ;;
  esac
  if [ "${REMOTE_MODE}" != "1" ]; then
    fail "non-loopback bind requires LOCALSCRIPT_REMOTE_MODE=1"
  fi
  if [ "${#REMOTE_TOKEN}" -lt 32 ]; then
    fail "remote mode requires LOCALSCRIPT_REMOTE_TOKEN with at least 32 characters"
  fi
}

ensure_service_port_free() {
  "${PYTHON_BIN}" - <<PY
import socket

port = int("${PORT}")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.5)
try:
    if sock.connect_ex(("127.0.0.1", port)) == 0:
        raise SystemExit(f"port_in_use::{port}")
finally:
    sock.close()
PY
}

PYTHON_BIN="$(resolve_python_bin)"
if [ ! -x "${PYTHON_BIN}" ]; then
  fail "python interpreter \`${PYTHON_BIN}\` is not executable"
fi

SUPPORTED_PYTHON_MIN_MINOR="${LOCALSCRIPT_PYTHON_MIN_MINOR:-11}"
SUPPORTED_PYTHON_MAX_MINOR="${LOCALSCRIPT_PYTHON_MAX_MINOR:-12}"

EFFECTIVE_CONFIG="$(load_effective_config 2>&1)" || fail "${EFFECTIVE_CONFIG}"
eval "${EFFECTIVE_CONFIG}"

command -v curl >/dev/null 2>&1 || fail "required command \`curl\` was not found"
validate_bind_policy

PYTHON_VERSION_CHECK="$(validate_python_runtime 2>&1 || true)"
if [ -n "${PYTHON_VERSION_CHECK}" ]; then
  fail "${PYTHON_VERSION_CHECK}"
fi

wait_for_ollama
AVAILABLE_TAGS="$(fetch_model_tags || true)"
if ! printf '%s\n' "${AVAILABLE_TAGS}" | grep -Fx "${PRIMARY_MODEL}" >/dev/null 2>&1; then
  fail "required model tag \`${PRIMARY_MODEL}\` is missing; run \`make model-setup\` first"
fi
ensure_service_port_free

export PYTHONUNBUFFERED=1
export LOCALSCRIPT_PROFILE="${PROFILE_NAME}"
export LOCALSCRIPT_OLLAMA_HOST="${OLLAMA_HOST}"
export LOCALSCRIPT_OLLAMA_MODE="${OLLAMA_MODE}"
export LOCALSCRIPT_UI_ENABLED="${UI_ENABLED}"
export LOCALSCRIPT_REMOTE_MODE="${REMOTE_MODE}"

printf 'localscript start: python %s\n' "$(python_version_triplet)"
printf 'localscript start: profile %s\n' "${PROFILE_NAME}"
printf 'localscript start: model %s\n' "${PRIMARY_MODEL}"
printf 'localscript start: Ollama mode %s\n' "${OLLAMA_MODE}"
printf 'localscript start: service URL http://127.0.0.1:%s\n' "${PORT}"
printf 'localscript start: Swagger URL http://127.0.0.1:%s/docs\n' "${PORT}"

exec "${PYTHON_BIN}" -m uvicorn app.main:app --host "${UVI_HOST}" --port "${PORT}"
