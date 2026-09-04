#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PORT="${LOCALSCRIPT_SMOKE_PORT:-$((18080 + RANDOM % 1000))}"
BASE_URL="${LOCALSCRIPT_BASE_URL:-http://127.0.0.1:${PORT}}"
USE_EXISTING="${LOCALSCRIPT_SMOKE_USE_EXISTING:-0}"
STARTUP_WAIT_SECONDS="${LOCALSCRIPT_SMOKE_STARTUP_WAIT_SECONDS:-120}"
resolve_python_bin() {
  if [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
    printf '%s\n' "${ROOT_DIR}/.venv/bin/python"
    return 0
  fi
  command -v python3 >/dev/null 2>&1 || {
    echo "judge_smoke_failed: python3 not found" >&2
    exit 1
  }
  command -v python3
}

PYTHON_BIN="$(resolve_python_bin)"
mkdir -p "${ROOT_DIR}/traces"
TRACE_DIR="$(mktemp -d "${ROOT_DIR}/traces/judge_smoke.XXXXXX")"
LOG_FILE="${TRACE_DIR}/server.log"
export BASE_URL TRACE_DIR ROOT_DIR
SERVER_PID=""

if [ "${USE_EXISTING}" != "1" ]; then
  LOCALSCRIPT_PORT="${PORT}" \
  LOCALSCRIPT_TRACE_DIR="${TRACE_DIR}" \
  LOCALSCRIPT_SKIP_INSTALL="${LOCALSCRIPT_SKIP_INSTALL:-0}" \
  ./scripts/judge_up.sh >"$LOG_FILE" 2>&1 &
  SERVER_PID=$!
fi

cleanup() {
  if [ -n "${SERVER_PID}" ] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  rm -rf "$TRACE_DIR"
}
trap cleanup EXIT

for _ in $(seq 1 "${STARTUP_WAIT_SECONDS}"); do
  if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
  tail -n 50 "$LOG_FILE" >&2 || true
  echo "judge_smoke_failed: server did not become healthy" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import json
import os
import urllib.request
from pathlib import Path

base_url = os.environ["BASE_URL"]
trace_dir = Path(os.environ["TRACE_DIR"])
use_existing = os.environ.get("LOCALSCRIPT_SMOKE_USE_EXISTING") == "1"

if not use_existing:
    from fastapi.testclient import TestClient

    from app.core.config import get_runtime_profile
    from app.core.traces import TraceStore
    from app.main import create_app

EMAILS_CONTEXT = {"wf": {"vars": {"emails": ["a@example.com", "b@example.com"]}}}
CLARIFY_CONTEXT = {
    "wf": {
        "vars": {"email": "A@EXAMPLE.COM"},
        "initVariables": {"email": "B@EXAMPLE.COM"},
    }
}


def post_json(path, payload):
    request = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response), dict(response.headers)


def fail(reason, detail):
    raise SystemExit("{0}::{1}".format(reason, json.dumps(detail, ensure_ascii=False)))


class ScriptedBackend:
    """Return a fixed sequence of structured role responses.

    The smoke check must prove that a rejected candidate never reaches the caller, so the danger
    case needs a valid plan followed by code the AST policy must refuse.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self._index = 0

    def complete(self, prompt, response_format=None, model=None):
        del prompt, response_format, model
        index = min(self._index, len(self._responses) - 1)
        self._index += 1
        return self._responses[index]


DANGEROUS_PLAN = json.dumps(
    {
        "kind": "plan",
        "objective": "Return a value from the workflow context.",
        "inputs": [{"root": "wf.vars", "segments": ["value"]}],
        "output": {"format": "lua_block", "shape": "scalar", "nullable": False},
        "steps": [{"description": "Return the value.", "reads": []}],
        "constraints": [],
        "acceptance_cases": [
            {"name": "value", "context": {"wf": {"vars": {"value": 1}}}, "expected": 1}
        ],
    }
)
DANGEROUS_CODE = json.dumps({"code": 'return os.execute("echo hacked")'})

health = json.load(urllib.request.urlopen(base_url + "/health"))
if health.get("status") != "ok":
    fail("health_failed", health)

public_body, public_headers = post_json(
    "/api/generate",
    {
        "prompt": "Верни последний адрес из массива wf.vars.emails; если массив пустой, верни nil.",
        "context": EMAILS_CONTEXT,
    },
)
if public_body.get("status") != "completed":
    fail("public_generation_failed", public_body)
if not public_headers.get("X-Trace-Id") or not public_headers.get("X-Session-Id"):
    fail("public_headers_missing", dict(public_headers))

envelope_body, _ = post_json(
    "/api/validate",
    {
        "code": '{"code":"lua{return wf.vars.emails[#wf.vars.emails]}lua"}',
        "context": EMAILS_CONTEXT,
        "output": {"format": "json_envelope", "shape": "object", "nullable": False},
    },
)
if envelope_body.get("ok") is not True:
    fail("json_envelope_validation_failed", envelope_body)

clarify_first, _ = post_json(
    "/api/generate",
    {"prompt": "Нормализуй email и верни его в lower-case.", "context": CLARIFY_CONTEXT},
)
if clarify_first.get("status") != "clarification_required":
    fail("clarification_status_failed", clarify_first)
if not clarify_first.get("question"):
    fail("clarification_question_missing", clarify_first)
clarify_second, _ = post_json(
    "/api/generate",
    {
        "session_id": clarify_first["session_id"],
        "clarification_answer": "Use wf.vars for the email root.",
    },
)
if clarify_second.get("status") != "completed":
    fail("clarification_continue_failed", clarify_second)

if use_existing:
    danger_case = {"mode": "skipped_existing_service", "diagnostic_codes": []}
else:
    danger_app = create_app(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=trace_dir / "danger_api"),
        backend=ScriptedBackend([DANGEROUS_PLAN, DANGEROUS_CODE, DANGEROUS_CODE]),
    )
    danger_response = TestClient(danger_app).post(
        "/api/generate",
        json={"prompt": "Return dangerous code.", "context": {"wf": {"vars": {"value": 1}}}},
    )
    if danger_response.status_code != 200:
        fail("danger_api_unexpected_status", {"status": danger_response.status_code})
    danger_body = danger_response.json()
    if danger_body.get("status") == "completed" or danger_body.get("code") is not None:
        fail("danger_api_fail_open", danger_body)
    danger_codes = [item.get("code") for item in danger_body.get("diagnostics", [])]
    if "dangerous_stdlib_os_forbidden" not in danger_codes:
        fail("danger_policy_not_applied", danger_body)
    danger_case = {"mode": "api_end_to_end", "diagnostic_codes": danger_codes}

print(
    json.dumps(
        {
            "ok": True,
            "health": health,
            "public_case": {
                "status": public_body["status"],
                "revision_count": public_body["revision_count"],
            },
            "json_envelope_case": {"ok": envelope_body["ok"]},
            "clarification_case": {
                "question": clarify_first["question"],
                "final_status": clarify_second["status"],
            },
            "danger_case": danger_case,
        },
        ensure_ascii=False,
    )
)
PY
