#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${LOCALSCRIPT_PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="$(command -v python3)"
fi
command -v ollama >/dev/null 2>&1 || {
  printf 'localscript setup_model: required command `ollama` was not found\n' >&2
  exit 1
}

EFFECTIVE="$("${PYTHON_BIN}" - <<'PY'
import shlex
from app.core.config import get_runtime_profile

profile = get_runtime_profile()
print("MODEL=" + shlex.quote(profile.model))
print("OLLAMA_ENDPOINT=" + shlex.quote(profile.ollama_host))
PY
)"
eval "${EFFECTIVE}"

printf 'localscript setup_model: pulling %s via %s\n' "${MODEL}" "${OLLAMA_ENDPOINT}"
OLLAMA_HOST="${OLLAMA_ENDPOINT}" ollama pull "${MODEL}"
