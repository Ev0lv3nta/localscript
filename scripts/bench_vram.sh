#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

select_python() {
  if [ -n "${LOCALSCRIPT_PYTHON_BIN:-}" ]; then
    printf '%s\n' "${LOCALSCRIPT_PYTHON_BIN}"
    return
  fi
  if [ -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
    printf '%s\n' "${PROJECT_ROOT}/.venv/bin/python"
    return
  fi
  if [ -x "/opt/venv/bin/python" ]; then
    printf '%s\n' "/opt/venv/bin/python"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  printf 'Python 3 is required to locate the packaged VRAM probe.\n' >&2
  return 1
}

PYTHON_BIN="$(select_python)"
PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - "$@" <<'PY'
import subprocess
import sys

from app.core.resources import materialized_resource


with materialized_resource("scripts/bench_vram.sh") as script:
    raise SystemExit(subprocess.call(["bash", str(script), *sys.argv[1:]]))
PY
