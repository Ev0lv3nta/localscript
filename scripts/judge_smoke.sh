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
import sys
import time
import urllib.request
from pathlib import Path

base_url = os.environ.get("BASE_URL")
trace_dir = Path(os.environ.get("TRACE_DIR"))
root_dir = Path(os.environ.get("ROOT_DIR"))
use_existing = os.environ.get("LOCALSCRIPT_SMOKE_USE_EXISTING") == "1"

if not use_existing:
    from app.validation.runtime_executor import execute_output
    from fastapi.testclient import TestClient
    from app.core.config import get_runtime_profile
    from app.core.traces import TraceStore
    from app.main import create_app


def post_generate(prompt, context=None):
    payload = {"prompt": prompt}
    if context is not None:
        payload["context"] = context
    request = urllib.request.Request(
        base_url + "/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        body = json.load(response)
        if use_existing:
            return body, {
                "strategy": response.headers.get("X-Strategy"),
                "repair_rounds": int(response.headers.get("X-Repair-Rounds", "0")),
            }
        trace_id = response.headers.get("X-Trace-Id")
    if not trace_id:
        raise SystemExit("trace_id_missing")
    trace_path = None
    for _ in range(25):
        trace_path = next(trace_dir.rglob(trace_id + ".json"), None)
        if trace_path is not None:
            break
        time.sleep(0.2)
    if trace_path is None:
        raise SystemExit("trace_not_found::{0}".format(trace_id))
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    return body, trace_payload


def post_json(path, payload):
    request = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response), dict(response.headers)


def semantic_value(code, context, output_style=None):
    if use_existing:
        payload = {
            "code": code,
            "context": context,
        }
        if output_style is not None:
            payload["output_style"] = output_style
        body, _ = post_json("/api/validate", payload)
        semantic = body.get("semantic_result") or {}
        return {
            "ok": semantic.get("ok", False),
            "value": semantic.get("value"),
            "error_code": semantic.get("error_code"),
            "error_message": semantic.get("error_message"),
            "degraded": semantic.get("degraded", False),
        }
    result = execute_output(
        code,
        context,
        output_style=output_style,
    )
    return {
        "ok": result.ok,
        "value": result.value,
        "error_code": result.error_code,
        "error_message": result.error_message,
        "degraded": result.degraded,
    }


class UnsupportedRootBackend:
    def generate(self, prompt, context=None):
        return "local items = ctx.body.items\nreturn items[1]"


class DangerousStdlibBackend:
    def generate(self, prompt, context=None):
        return 'return os.execute("echo hacked")'


health = json.load(urllib.request.urlopen(base_url + "/health"))
if health.get("status") != "ok":
    raise SystemExit("health_failed")

public_body, public_trace = post_generate(
    "Верни последний адрес из массива wf.vars.emails; если массив пустой, верни nil.",
    {"wf": {"vars": {"emails": ["a@example.com", "b@example.com"]}}},
)
if public_trace.get("strategy") not in {"template", "ollama_chain", "feedback_revision"}:
    raise SystemExit("public_strategy_failed: {0}".format(public_trace.get("strategy")))
public_execution = semantic_value(
    public_body.get("code", ""),
    {"wf": {"vars": {"emails": ["a@example.com", "b@example.com"]}}},
)
if not public_execution["ok"] or public_execution["value"] != "b@example.com":
    raise SystemExit("public_semantic_failed: {0}".format(public_execution))

if use_existing:
    model_body, _ = post_json(
        "/api/generate",
        {
            "prompt": "Возьми wf.vars.contacts и подготовь список таблиц для активных контактов с email: поле id оставь как есть, а email переведи в lower case. Если вход пустой, нужен пустой список через _utils.array.new().",
            "context": {
                "wf": {
                    "vars": {
                        "contacts": [
                            {"id": "C1", "active": True, "email": "ADMIN@EXAMPLE.COM"},
                            {"id": "C2", "active": False, "email": "skip@example.com"},
                            {"id": "C3", "active": True, "email": "Owner@Example.com"},
                        ]
                    }
                }
            },
        },
    )
    model_trace = {
        "strategy": model_body.get("strategy"),
        "repair_rounds": model_body.get("validation", {}).get("repair_rounds", 0),
    }
else:
    model_body, model_trace = post_generate(
        "Возьми wf.vars.contacts и подготовь список таблиц для активных контактов с email: поле id оставь как есть, а email переведи в lower case. Если вход пустой, нужен пустой список через _utils.array.new().",
        {
            "wf": {
                "vars": {
                    "contacts": [
                        {"id": "C1", "active": True, "email": "ADMIN@EXAMPLE.COM"},
                        {"id": "C2", "active": False, "email": "skip@example.com"},
                        {"id": "C3", "active": True, "email": "Owner@Example.com"},
                    ]
                }
            }
        },
    )
if model_trace.get("strategy") not in {"ollama_chain", "feedback_revision"}:
    raise SystemExit("model_strategy_failed: {0}".format(model_trace.get("strategy")))
model_execution = semantic_value(
    model_body.get("code", ""),
    {
        "wf": {
            "vars": {
                "contacts": [
                    {"id": "C1", "active": True, "email": "ADMIN@EXAMPLE.COM"},
                    {"id": "C2", "active": False, "email": "skip@example.com"},
                    {"id": "C3", "active": True, "email": "Owner@Example.com"},
                ]
            }
        }
    },
)
if not model_execution["ok"] or model_execution["value"] != [
    {"id": "C1", "email": "admin@example.com"},
    {"id": "C3", "email": "owner@example.com"},
]:
    raise SystemExit("model_semantic_failed: {0}".format(model_execution))

validate_request = urllib.request.Request(
    base_url + "/api/validate",
    data=json.dumps(
        {
            "code": '{"code":"lua{return wf.vars.emails[#wf.vars.emails]}lua"}',
            "context": {"wf": {"vars": {"emails": ["a@example.com", "b@example.com"]}}},
            "output_style": "json_envelope",
        }
    ).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(validate_request) as validate_response:
    envelope_payload = json.load(validate_response)
if envelope_payload.get("ok") is not True:
    raise SystemExit("json_envelope_validation_failed::{0}".format(envelope_payload))
if envelope_payload.get("semantic_result", {}).get("value") != {"code": "b@example.com"}:
    raise SystemExit("json_envelope_semantic_failed::{0}".format(envelope_payload))

clarify_context = {
    "wf": {
        "vars": {"email": "A@EXAMPLE.COM"},
        "initVariables": {"email": "B@EXAMPLE.COM"},
    }
}
clarify_first, clarify_first_headers = post_json(
    "/api/generate",
    {
        "prompt": "Нормализуй email и верни его в lower-case.",
        "context": clarify_context,
    },
)
if clarify_first.get("status") != "clarification_needed":
    raise SystemExit("clarification_status_failed::{0}".format(clarify_first))
if "wf.vars or wf.initVariables" not in (clarify_first.get("question") or ""):
    raise SystemExit("clarification_question_failed::{0}".format(clarify_first))
clarify_second, _ = post_json(
    "/api/generate",
    {
        "session_id": clarify_first["session_id"],
        "clarification_answer": "Use wf.vars for email root.",
    },
)
if clarify_second.get("status") != "completed":
    raise SystemExit("clarification_continue_failed::{0}".format(clarify_second))
clarify_execution = semantic_value(
    clarify_second.get("code", ""),
    clarify_context,
)
if not clarify_execution["ok"] or clarify_execution["value"] != "a@example.com":
    raise SystemExit("clarification_semantic_failed::{0}".format(clarify_execution))

if use_existing:
    repair_trace = {"strategy": "skipped_existing_service"}
    repair_code = None
    repair_execution = None
    danger_codes = ["skipped_existing_service"]
else:
    repair_trace_store = TraceStore(root=trace_dir / "repair_api")
    repair_app = create_app(
        profile=get_runtime_profile(),
        trace_store=repair_trace_store,
        backend=UnsupportedRootBackend(),
    )
    repair_client = TestClient(repair_app)
    repair_response = repair_client.post(
        "/generate",
        json={
            "prompt": "Iterate over the incoming items collection and return the first value.",
            "context": {"wf": {"vars": {"items": [1, 2, 3]}}},
        },
    )
    if repair_response.status_code != 200:
        raise SystemExit("repair_api_failed::{0}".format(repair_response.text))
    repair_trace_id = repair_response.headers.get("X-Trace-Id")
    if not repair_trace_id:
        raise SystemExit("repair_trace_id_missing")
    repair_trace = json.loads((next((trace_dir / "repair_api").rglob(repair_trace_id + ".json"))).read_text(encoding="utf-8"))
    repair_code = repair_response.json().get("code", "")
    if "ctx.body" in repair_code:
        raise SystemExit("repair_ctx_body_not_removed::{0}".format(repair_code))
    if "wf.vars.items" not in repair_code:
        raise SystemExit("repair_expected_root_missing::{0}".format(repair_code))
    if repair_response.headers.get("X-Repair-Rounds") in {None, "0"}:
        raise SystemExit("repair_rounds_missing::{0}".format(repair_response.headers.get("X-Repair-Rounds")))
    repair_execution = semantic_value(repair_code, {"wf": {"vars": {"items": [1, 2, 3]}}})
    if not repair_execution["ok"] or repair_execution["value"] != 1:
        raise SystemExit("repair_semantic_failed::{0}".format(repair_execution))

    danger_trace_store = TraceStore(root=trace_dir / "danger_api")
    danger_app = create_app(
        profile=get_runtime_profile(),
        trace_store=danger_trace_store,
        backend=DangerousStdlibBackend(),
    )
    danger_client = TestClient(danger_app)
    danger_response = danger_client.post(
        "/generate",
        json={
            "prompt": "Return dangerous code.",
            "context": {"wf": {"vars": {"value": 1}}},
        },
    )
    if danger_response.status_code != 200:
        raise SystemExit("danger_api_failed::{0}".format(danger_response.text))
    danger_trace_id = danger_response.headers.get("X-Trace-Id")
    if not danger_trace_id:
        raise SystemExit("danger_trace_id_missing")
    danger_trace = json.loads((next((trace_dir / "danger_api").rglob(danger_trace_id + ".json"))).read_text(encoding="utf-8"))
    danger_codes = [
        message.get("code")
        for message in danger_trace.get("validation_report", {}).get("messages", [])
    ]
    if "os_execute_forbidden" not in danger_codes:
        if "dangerous_stdlib_os_forbidden" not in danger_codes:
            raise SystemExit("danger_validator_failed::{0}".format(danger_trace))

print(
    json.dumps(
        {
            "ok": True,
            "health": health,
            "public_case": {
                "code": public_body["code"],
                "strategy": public_trace["strategy"],
                "repair_rounds": public_trace["repair_rounds"],
                "semantic_value": public_execution["value"],
            },
            "model_case": {
                "code": model_body["code"],
                "strategy": model_trace["strategy"],
                "repair_rounds": model_trace["repair_rounds"],
                "semantic_value": model_execution["value"],
            },
            "json_envelope_case": {
                "ok": envelope_payload["ok"],
                "semantic_value": envelope_payload["semantic_result"]["value"],
            },
            "clarification_case": {
                "status": clarify_second["status"],
                "question": clarify_first["question"],
                "strategy": clarify_second["strategy"],
                "code": clarify_second["code"],
                "semantic_value": clarify_execution["value"],
            },
            "repair_case": {
                "mode": "skipped_existing_service" if use_existing else "api_end_to_end",
                "strategy": repair_trace["strategy"],
                "repair_rounds": None if use_existing else int(repair_response.headers["X-Repair-Rounds"]),
                "code": repair_code,
                "semantic_value": None if use_existing else repair_execution["value"],
            },
            "danger_case": {
                "validation_codes": danger_codes,
                "strategy": "skipped_existing_service" if use_existing else danger_trace["strategy"],
            },
        },
        ensure_ascii=False,
    )
)
PY
