#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PORT="${LOCALSCRIPT_PORT:-8080}"
OLLAMA_HOST="${LOCALSCRIPT_OLLAMA_HOST:-http://127.0.0.1:11434}"
OLLAMA_MODE="${LOCALSCRIPT_OLLAMA_MODE:-auto}"
PRIMARY_MODEL="${LOCALSCRIPT_PRIMARY_MODEL:-qwen3:8b-q4_K_M}"
FALLBACK_MODEL="${LOCALSCRIPT_FALLBACK_MODEL:-qwen3:4b-instruct-2507-q4_K_M}"
STARTUP_TIMEOUT_SECONDS="${LOCALSCRIPT_STARTUP_TIMEOUT_SECONDS:-120}"
OLLAMA_POLL_INTERVAL_SECONDS="${LOCALSCRIPT_OLLAMA_POLL_INTERVAL_SECONDS:-2}"
SUPPORTED_PYTHON_MIN_MINOR="${LOCALSCRIPT_PYTHON_MIN_MINOR:-11}"
SUPPORTED_PYTHON_MAX_MINOR="${LOCALSCRIPT_PYTHON_MAX_MINOR:-12}"
UVI_HOST="${LOCALSCRIPT_BIND_HOST:-0.0.0.0}"

fail() {
  printf 'localscript judge_up: %s\n' "$1" >&2
  exit 1
}

warn() {
  printf 'localscript judge_up: %s\n' "$1" >&2
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "required command \`$1\` was not found"
  fi
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

ollama_host_kind() {
  "${PYTHON_BIN}" - <<PY
from urllib.parse import urlparse
parsed = urlparse("${OLLAMA_HOST}")
host = (parsed.hostname or "").lower()
print("local" if host in {"127.0.0.1", "localhost", "::1"} else "remote")
PY
}

detect_mode() {
  if [ "${OLLAMA_MODE}" != "auto" ]; then
    printf '%s\n' "${OLLAMA_MODE}"
    return 0
  fi

  local host_kind
  host_kind="$(ollama_host_kind)"
  if [ "${host_kind}" = "local" ] && command -v ollama >/dev/null 2>&1; then
    printf 'local_cli\n'
    return 0
  fi
  printf 'remote_api\n'
}

fetch_model_tags() {
  "${PYTHON_BIN}" - <<PY
import json
from urllib.request import urlopen

host = "${OLLAMA_HOST}".rstrip("/")
with urlopen(host + "/api/tags", timeout=10) as response:
    payload = json.loads(response.read().decode("utf-8"))

for item in payload.get("models", []):
    name = item.get("name")
    if isinstance(name, str) and name:
        print(name)
PY
}

wait_for_ollama() {
  local deadline elapsed
  deadline="${STARTUP_TIMEOUT_SECONDS}"
  elapsed=0
  while [ "${elapsed}" -lt "${deadline}" ]; do
    if curl -fsS "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
      return 0
    fi
    sleep "${OLLAMA_POLL_INTERVAL_SECONDS}"
    elapsed=$((elapsed + OLLAMA_POLL_INTERVAL_SECONDS))
  done
  fail "Ollama did not become reachable at ${OLLAMA_HOST} within ${STARTUP_TIMEOUT_SECONDS}s"
}

require_model_tag() {
  local model="$1"
  if printf '%s\n' "${AVAILABLE_TAGS}" | grep -Fx "${model}" >/dev/null 2>&1; then
    return 0
  fi

  if [ "${SELECTED_OLLAMA_MODE}" = "local_cli" ]; then
    printf 'localscript judge_up: missing model tag `%s`, pulling via local ollama CLI...\n' "${model}"
    ollama pull "${model}" >/dev/null
    AVAILABLE_TAGS="$(fetch_model_tags || true)"
    if printf '%s\n' "${AVAILABLE_TAGS}" | grep -Fx "${model}" >/dev/null 2>&1; then
      return 0
    fi
  fi

  fail "required model tag \`${model}\` is missing on ${OLLAMA_HOST}"
}

ensure_service_port_free() {
  "${PYTHON_BIN}" - <<PY
import socket
import sys

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

PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PATH

require_command curl
require_command uvicorn

PYTHON_VERSION_CHECK="$(validate_python_runtime 2>&1 || true)"
if [ -n "${PYTHON_VERSION_CHECK}" ]; then
  fail "${PYTHON_VERSION_CHECK}"
fi

SELECTED_OLLAMA_MODE="$(detect_mode)"
case "${SELECTED_OLLAMA_MODE}" in
  remote_api)
    ;;
  local_cli)
    require_command ollama
    ;;
  *)
    fail "unsupported LOCALSCRIPT_OLLAMA_MODE=${SELECTED_OLLAMA_MODE}; supported values are remote_api, local_cli, or auto"
    ;;
esac

wait_for_ollama
AVAILABLE_TAGS="$(fetch_model_tags || true)"
require_model_tag "${PRIMARY_MODEL}"
if [ "${FALLBACK_MODEL}" != "${PRIMARY_MODEL}" ]; then
  require_model_tag "${FALLBACK_MODEL}"
fi
ensure_service_port_free

export PYTHONUNBUFFERED=1
export LOCALSCRIPT_PROFILE="${LOCALSCRIPT_PROFILE:-competition}"
export LOCALSCRIPT_OLLAMA_HOST="${OLLAMA_HOST}"
export LOCALSCRIPT_OLLAMA_MODE="${SELECTED_OLLAMA_MODE}"
export LOCALSCRIPT_UI_ENABLED="${LOCALSCRIPT_UI_ENABLED:-1}"

printf 'localscript judge_up: python %s\n' "$(python_version_triplet)"
printf 'localscript judge_up: Ollama mode %s\n' "${SELECTED_OLLAMA_MODE}"
printf 'localscript judge_up: Ollama host %s\n' "${OLLAMA_HOST}"
printf 'localscript judge_up: service URL http://127.0.0.1:%s\n' "${PORT}"
printf 'localscript judge_up: Swagger URL http://127.0.0.1:%s/docs\n' "${PORT}"
printf 'localscript judge_up: run ./scripts/preflight_judge.sh for full judged preflight\n'

exec uvicorn app.main:app --host "${UVI_HOST}" --port "${PORT}"
