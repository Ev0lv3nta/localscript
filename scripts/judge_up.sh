#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
printf 'localscript: scripts/judge_up.sh is a compatibility alias; use scripts/start.sh\n' >&2
exec "${ROOT_DIR}/scripts/start.sh" "$@"
