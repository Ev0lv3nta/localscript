#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M}"
FALLBACK_MODEL="${2:-qwen3:8b-q4_K_M}"
MODE="${3:-both}"
OLLAMA_HOST="${LOCALSCRIPT_OLLAMA_HOST:-http://127.0.0.1:11434}"
PROMPT="${LOCALSCRIPT_VRAM_PROMPT:-Return Lua code only: return 1}"
THINK="${LOCALSCRIPT_OLLAMA_THINK:-false}"
NUM_CTX=8192
# Порог подбирается под карту, а не под модель: 20 ГБ оставляют запас на 24-гигабайтной.
CAP_GB="${LOCALSCRIPT_VRAM_CAP_GB:-20}"
SHORT_NUM_PREDICT=64
JUDGED_NUM_PREDICT=256

if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf '{"status":"skipped","reason":"nvidia_smi_unavailable","model":"%s","fallback_model":"%s"}\n' "$MODEL" "$FALLBACK_MODEL"
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  printf '{"status":"skipped","reason":"curl_unavailable","model":"%s","fallback_model":"%s"}\n' "$MODEL" "$FALLBACK_MODEL"
  exit 0
fi

if ! curl -fsS "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
  printf '{"status":"skipped","reason":"ollama_unreachable","model":"%s","fallback_model":"%s"}\n' "$MODEL" "$FALLBACK_MODEL"
  exit 0
fi

run_probe() {
  local probe_name="$1"
  local num_predict="$2"
  local sample_file
  sample_file="$(mktemp)"

  cleanup_probe() {
    rm -f "$sample_file"
  }
  trap cleanup_probe RETURN

  curl -fsS "${OLLAMA_HOST}/api/generate" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"think\":${THINK},\"stream\":false,\"prompt\":\"${PROMPT}\",\"options\":{\"num_ctx\":${NUM_CTX},\"num_predict\":${num_predict}}}" \
    >/dev/null 2>&1 &
  GEN_PID=$!

  while kill -0 "$GEN_PID" >/dev/null 2>&1; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{m=0} {if ($1>m) m=$1} END{print m+0}' >> "$sample_file"
    sleep 0.2
  done
  wait "$GEN_PID" || true

  if [ ! -s "$sample_file" ]; then
    python3 - <<PY
import json
print(json.dumps({
    "name": "${probe_name}",
    "status": "skipped",
    "reason": "no_samples_collected",
    "model": "${MODEL}",
    "fallback_model": "${FALLBACK_MODEL}",
    "num_ctx": ${NUM_CTX},
    "num_predict": ${num_predict},
    "think": ${THINK},
    "cap_gb": float(${CAP_GB}),
}))
PY
    return
  fi

  PEAK_MB="$(awk 'BEGIN{m=0} {if ($1>m) m=$1} END{print m+0}' "$sample_file")"
  STATUS="ok"
  if [ "$PEAK_MB" -gt "$(( CAP_GB * 1024 ))" ]; then
    STATUS="over_cap"
  fi

  python3 - <<PY
import json
peak_mb = int(${PEAK_MB})
print(json.dumps({
    "name": "${probe_name}",
    "status": "${STATUS}",
    "model": "${MODEL}",
    "fallback_model": "${FALLBACK_MODEL}",
    "peak_vram_mb": peak_mb,
    "peak_vram_gb": round(peak_mb / 1024.0, 3),
    "cap_gb": float(${CAP_GB}),
    "num_ctx": ${NUM_CTX},
    "num_predict": ${num_predict},
    "think": ${THINK@Q} == "true",
}))
PY
}

short_probe_json=""
judged_probe_json=""

if [ "${MODE}" = "short_probe" ] || [ "${MODE}" = "both" ]; then
  short_probe_json="$(run_probe "short_probe" "${SHORT_NUM_PREDICT}")"
fi

if [ "${MODE}" = "judged_probe" ] || [ "${MODE}" = "both" ]; then
  judged_probe_json="$(run_probe "judged_probe" "${JUDGED_NUM_PREDICT}")"
fi

SHORT_PROBE_JSON="${short_probe_json}" JUDGED_PROBE_JSON="${judged_probe_json}" python3 - <<'PY'
import json

short_probe = json.loads(__import__("os").environ["SHORT_PROBE_JSON"]) if __import__("os").environ["SHORT_PROBE_JSON"] else None
judged_probe = json.loads(__import__("os").environ["JUDGED_PROBE_JSON"]) if __import__("os").environ["JUDGED_PROBE_JSON"] else None
selected = judged_probe or short_probe or {}
print(json.dumps({
    "status": selected.get("status", "skipped"),
    "probe_mode": selected.get("name"),
    "model": selected.get("model", "${MODEL}"),
    "fallback_model": selected.get("fallback_model", "${FALLBACK_MODEL}"),
    "peak_vram_mb": selected.get("peak_vram_mb"),
    "peak_vram_gb": selected.get("peak_vram_gb"),
    "cap_gb": selected.get("cap_gb", 8.0),
    "num_ctx": selected.get("num_ctx"),
    "num_predict": selected.get("num_predict"),
    "think": selected.get("think"),
    "short_probe": short_probe,
    "judged_probe": judged_probe,
}))
PY
