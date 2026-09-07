#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf 'localscript docker_entrypoint: %s\n' "$1" >&2
  exit 1
}

PYTHON_BIN="${LOCALSCRIPT_PYTHON_BIN:-/opt/venv/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
  fail "python interpreter \`${PYTHON_BIN}\` is not executable"
fi

EFFECTIVE_CONFIG="$("${PYTHON_BIN}" - <<'PY'
import shlex
from app.core.config import get_runtime_profile

profile = get_runtime_profile()
values = {
    "PORT": profile.port,
    "OLLAMA_HOST": profile.ollama_host,
    "PRIMARY_MODEL": profile.model,
    "STARTUP_TIMEOUT_SECONDS": profile.startup_timeout_seconds,
    "POLL_INTERVAL_SECONDS": profile.ollama_poll_interval_seconds,
    "PROFILE_NAME": profile.name,
}
for name, value in values.items():
    print(f"{name}={shlex.quote(str(value))}")
PY
)" || fail "invalid effective configuration"
eval "${EFFECTIVE_CONFIG}"

fetch_tags() {
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
    sleep_for="${POLL_INTERVAL_SECONDS}"
    if [ "${sleep_for}" -gt "${remaining}" ]; then
      sleep_for="${remaining}"
    fi
    sleep "${sleep_for}"
  done
  fail "Ollama did not become reachable at the configured endpoint within ${STARTUP_TIMEOUT_SECONDS}s"
}

wait_for_ollama
AVAILABLE_TAGS="$(fetch_tags || true)"
if ! printf '%s\n' "${AVAILABLE_TAGS}" | grep -Fx "${PRIMARY_MODEL}" >/dev/null 2>&1; then
  fail "required model tag \`${PRIMARY_MODEL}\` is missing; pull it before starting LocalScript"
fi

printf 'localscript docker_entrypoint: profile=%s\n' "${PROFILE_NAME}"
printf 'localscript docker_entrypoint: model=%s\n' "${PRIMARY_MODEL}"
printf 'localscript docker_entrypoint: service=http://127.0.0.1:%s\n' "${PORT}"

exec "${PYTHON_BIN}" -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT}"
