#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-localscript:ci}"
SUFFIX="${GITHUB_RUN_ID:-$$}"
NETWORK="localscript-ci-${SUFFIX}"
MOCK_CONTAINER="localscript-ollama-${SUFFIX}"
APP_CONTAINER="localscript-app-${SUFFIX}"

cleanup() {
  docker rm --force "${APP_CONTAINER}" "${MOCK_CONTAINER}" >/dev/null 2>&1 || true
  docker network rm "${NETWORK}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker network create "${NETWORK}" >/dev/null

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
    {"name": "qwen3:8b-q4_K_M"},
]

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = {"models": MODELS} if self.path == "/api/tags" else {"version": "ci-mock"}
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass

HTTPServer(("0.0.0.0", 11434), Handler).serve_forever()'

docker run --detach \
  --name "${APP_CONTAINER}" \
  --network "${NETWORK}" \
  --env LOCALSCRIPT_OLLAMA_HOST=http://ollama:11434 \
  --env LOCALSCRIPT_STARTUP_TIMEOUT_SECONDS=30 \
  --env LOCALSCRIPT_OLLAMA_POLL_INTERVAL_SECONDS=1 \
  --env LOCALSCRIPT_UI_ENABLED=0 \
  "${IMAGE}" >/dev/null

for _attempt in $(seq 1 30); do
  if docker exec "${APP_CONTAINER}" curl --fail --silent http://127.0.0.1:8080/ready >/dev/null; then
    break
  fi
  if ! docker inspect --format '{{.State.Running}}' "${APP_CONTAINER}" | grep -Fx true >/dev/null; then
    docker logs "${APP_CONTAINER}" >&2
    exit 1
  fi
  sleep 1
done

docker exec "${APP_CONTAINER}" curl --fail --silent http://127.0.0.1:8080/health >/dev/null
docker exec "${APP_CONTAINER}" curl --fail --silent http://127.0.0.1:8080/ready >/dev/null
docker exec "${APP_CONTAINER}" localscript --help >/dev/null
docker exec "${APP_CONTAINER}" ./.tools/lua54/bin/lua -e 'assert(_VERSION == "Lua 5.4")'

test "$(docker inspect --format '{{.Config.User}}' "${APP_CONTAINER}")" = "appuser"
test "$(docker exec "${APP_CONTAINER}" id -u)" != "0"
