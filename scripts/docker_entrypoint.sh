#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PORT="${LOCALSCRIPT_PORT:-8080}"
OLLAMA_HOST="${LOCALSCRIPT_OLLAMA_HOST:-http://ollama:11434}"
PRIMARY_MODEL="${LOCALSCRIPT_PRIMARY_MODEL:-qwen3:8b-q4_K_M}"
FALLBACK_MODEL="${LOCALSCRIPT_FALLBACK_MODEL:-qwen3:4b-instruct-2507-q4_K_M}"
STARTUP_TIMEOUT_SECONDS="${LOCALSCRIPT_STARTUP_TIMEOUT_SECONDS:-180}"
POLL_INTERVAL_SECONDS="${LOCALSCRIPT_OLLAMA_POLL_INTERVAL_SECONDS:-2}"

fail() {
  printf 'localscript docker_entrypoint: %s\n' "$1" >&2
  exit 1
}

require_python() {
  if [ -x "/opt/venv/bin/python" ]; then
    printf '/opt/venv/bin/python\n'
    return 0
  fi
  command -v python3 >/dev/null 2>&1 || fail "python3 was not found"
  command -v python3
}

fetch_tags() {
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
  local elapsed=0
  while [ "${elapsed}" -lt "${STARTUP_TIMEOUT_SECONDS}" ]; do
    if curl -fsS "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
      return 0
    fi
    sleep "${POLL_INTERVAL_SECONDS}"
    elapsed=$((elapsed + POLL_INTERVAL_SECONDS))
  done
  fail "Ollama did not become reachable at ${OLLAMA_HOST} within ${STARTUP_TIMEOUT_SECONDS}s"
}

require_model() {
  local model="$1"
  if printf '%s\n' "${AVAILABLE_TAGS}" | grep -Fx "${model}" >/dev/null 2>&1; then
    return 0
  fi
  fail "required model tag \`${model}\` is missing at ${OLLAMA_HOST}"
}

PYTHON_BIN="$(require_python)"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export LOCALSCRIPT_PROFILE="${LOCALSCRIPT_PROFILE:-competition}"
export LOCALSCRIPT_OLLAMA_MODE="remote_api"
export LOCALSCRIPT_OLLAMA_HOST="${OLLAMA_HOST}"

wait_for_ollama
AVAILABLE_TAGS="$(fetch_tags || true)"
require_model "${PRIMARY_MODEL}"
if [ "${FALLBACK_MODEL}" != "${PRIMARY_MODEL}" ]; then
  require_model "${FALLBACK_MODEL}"
fi

printf 'localscript docker_entrypoint: python=%s\n' "$("${PYTHON_BIN}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
PY
)"
printf 'localscript docker_entrypoint: ollama=%s\n' "${OLLAMA_HOST}"
printf 'localscript docker_entrypoint: service=http://127.0.0.1:%s\n' "${PORT}"

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT}"
