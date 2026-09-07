#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-localscript:ci}"
SUFFIX="${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}-$$"
NETWORK="localscript-ci-${SUFFIX}"
MOCK_CONTAINER="localscript-ollama-${SUFFIX}"
APP_CONTAINER="localscript-app-${SUFFIX}"
STATE_VOLUME="localscript-state-${SUFFIX}"
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(dirname -- "${SCRIPT_DIR}")"

cleanup() {
  docker rm --force "${APP_CONTAINER}" "${MOCK_CONTAINER}" >/dev/null 2>&1 || true
  docker network rm "${NETWORK}" >/dev/null 2>&1 || true
  docker volume rm --force "${STATE_VOLUME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker compose --file "${PROJECT_ROOT}/docker-compose.yml" config --quiet
docker network create "${NETWORK}" >/dev/null
docker volume create "${STATE_VOLUME}" >/dev/null

docker run --detach \
  --name "${MOCK_CONTAINER}" \
  --network "${NETWORK}" \
  --network-alias ollama \
  --entrypoint python \
  "${IMAGE}" \
  -c 'import json
from http.server import BaseHTTPRequestHandler, HTTPServer

MODELS = [
    {"name": "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"},
]
PLAN = {
    "kind": "plan",
    "objective": "Return the workflow value.",
    "inputs": [{"root": "wf.vars", "segments": ["value"]}],
    "output": {"format": "lua_block", "shape": "scalar", "nullable": False},
    "steps": [{
        "description": "Read and return the value.",
        "reads": [{"root": "wf.vars", "segments": ["value"]}],
    }],
    "constraints": [],
    "acceptance_cases": [{
        "name": "value",
        "context": {"wf": {"vars": {"value": 7}}},
        "expected": 7,
    }],
}

class Handler(BaseHTTPRequestHandler):
    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        payload = {"models": MODELS} if self.path == "/api/tags" else {"version": "ci-mock"}
        self.send_json(payload)

    def do_POST(self):
        if self.path != "/api/generate":
            self.send_json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length).decode("utf-8"))
        prompt = request.get("prompt", "")
        if "You are the planner" in prompt:
            response = PLAN
        elif "You are the generator" in prompt or "You are revising" in prompt:
            response = {"code": "return wf.vars.value"}
        elif "You are the reviewer" in prompt:
            response = {"kind": "approved"}
        else:
            self.send_json({"error": "unexpected role"}, status=400)
            return
        self.send_json({
            "model": MODELS[0]["name"],
            "response": json.dumps(response),
            "done": True,
            "done_reason": "stop",
        })

    def log_message(self, *_args):
        pass

HTTPServer(("0.0.0.0", 11434), Handler).serve_forever()'

start_app() {
  docker run --detach \
    --name "${APP_CONTAINER}" \
    --network "${NETWORK}" \
    --mount "type=volume,source=${STATE_VOLUME},target=/var/lib/localscript" \
    --env LOCALSCRIPT_OLLAMA_HOST=http://ollama:11434 \
    --env LOCALSCRIPT_STARTUP_TIMEOUT_SECONDS=30 \
    --env LOCALSCRIPT_OLLAMA_POLL_INTERVAL_SECONDS=1 \
    --env LOCALSCRIPT_UI_ENABLED=0 \
    "${IMAGE}" >/dev/null
}

wait_for_app() {
  for _attempt in $(seq 1 30); do
    if docker exec "${APP_CONTAINER}" curl --fail --silent http://127.0.0.1:8080/ready >/dev/null; then
      return 0
    fi
    if ! docker inspect --format '{{.State.Running}}' "${APP_CONTAINER}" | grep -Fx true >/dev/null; then
      docker logs "${APP_CONTAINER}" >&2
      return 1
    fi
    sleep 1
  done
  docker logs "${APP_CONTAINER}" >&2
  return 1
}

start_app
wait_for_app

docker exec "${APP_CONTAINER}" curl --fail --silent http://127.0.0.1:8080/health >/dev/null
docker exec "${APP_CONTAINER}" curl --fail --silent http://127.0.0.1:8080/ready >/dev/null
docker exec "${APP_CONTAINER}" localscript --help >/dev/null
docker exec "${APP_CONTAINER}" lua -e 'assert(_VERSION == "Lua 5.4")'
docker exec "${APP_CONTAINER}" sh -c 'test -w "$LOCALSCRIPT_STATE_DIR"'
docker exec "${APP_CONTAINER}" curl --fail --silent \
  --header 'Content-Type: application/json' \
  --data '{"code":"return wf.vars.value","context":{"wf":{"vars":{"value":7}}},"output":{"format":"lua_block","shape":"scalar","nullable":false}}' \
  http://127.0.0.1:8080/api/validate \
  | grep -F '"ok":true' >/dev/null

SESSION_RESPONSE="$(docker exec "${APP_CONTAINER}" curl --fail --silent \
  --header 'Content-Type: application/json' \
  --data '{"prompt":"Return wf.vars.value.","context":{"wf":{"vars":{"value":7}}},"output":{"format":"lua_block","shape":"scalar","nullable":false}}' \
  http://127.0.0.1:8080/api/generate)"
SESSION_ID="$(printf '%s' "${SESSION_RESPONSE}" | docker exec --interactive "${APP_CONTAINER}" \
  python -c 'import json, sys
payload = json.load(sys.stdin)
assert payload["status"] == "completed"
assert payload["code"] == "return wf.vars.value"
print(payload["session_id"])')"

docker rm --force "${APP_CONTAINER}" >/dev/null
start_app
wait_for_app

docker exec "${APP_CONTAINER}" curl --fail --silent \
  "http://127.0.0.1:8080/api/sessions/${SESSION_ID}" \
  | docker exec --interactive "${APP_CONTAINER}" python -c 'import json, sys
payload = json.load(sys.stdin)
assert payload["status"] == "completed"
assert payload["original_task"] == "Return wf.vars.value."'

test "$(docker inspect --format '{{.Config.User}}' "${APP_CONTAINER}")" = "appuser"
test "$(docker exec "${APP_CONTAINER}" id -u)" != "0"
