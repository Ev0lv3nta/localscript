#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="${ROOT_DIR}/.tools"
LUA_VERSION="${LOCALSCRIPT_LUA_VERSION:-5.4.6}"
LUA_ROOT="${TOOLS_DIR}/lua54"
LUA_BIN="${LUA_ROOT}/bin/lua"
LUAC_BIN="${LUA_ROOT}/bin/luac"
LUA_ARCHIVE="lua-${LUA_VERSION}.tar.gz"
LUA_SRC_DIR="${TOOLS_DIR}/lua-${LUA_VERSION}"

log() {
  printf 'localscript bootstrap_lua54: %s\n' "$1"
}

fail() {
  log "$1" >&2
  exit 1
}

resolve_archive() {
  local candidates=()
  if [ -n "${LOCALSCRIPT_LUA_ARCHIVE:-}" ]; then
    candidates+=("${LOCALSCRIPT_LUA_ARCHIVE}")
  fi
  candidates+=(
    "${ROOT_DIR}/third_party/lua/${LUA_ARCHIVE}"
    "${TOOLS_DIR}/${LUA_ARCHIVE}"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [ -f "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

if [ -x "${LUA_BIN}" ] && [ -x "${LUAC_BIN}" ]; then
  log "using existing local Lua runtime at ${LUA_ROOT}"
  exit 0
fi

if ! command -v make >/dev/null 2>&1; then
  fail "required command \`make\` was not found. Install build-essential/make and retry."
fi

mkdir -p "${TOOLS_DIR}"

ARCHIVE_PATH="$(resolve_archive || true)"
if [ -z "${ARCHIVE_PATH}" ]; then
  fail "offline archive ${LUA_ARCHIVE} was not found. Checked LOCALSCRIPT_LUA_ARCHIVE, third_party/lua/, and legacy .tools/."
fi

log "building Lua ${LUA_VERSION} from ${ARCHIVE_PATH}"
cd "${TOOLS_DIR}"
rm -rf "${LUA_SRC_DIR}"
tar -xzf "${ARCHIVE_PATH}"
cd "${LUA_SRC_DIR}"
case "$(uname -s)" in
  Darwin)
    MAKE_TARGET="macosx"
    ;;
  Linux)
    MAKE_TARGET="linux"
    ;;
  *)
    fail "unsupported build platform: $(uname -s)"
    ;;
esac
make "${MAKE_TARGET}" >/dev/null

mkdir -p "${LUA_ROOT}/bin"
ln -sf "../../lua-${LUA_VERSION}/src/lua" "${LUA_ROOT}/bin/lua"
ln -sf "../../lua-${LUA_VERSION}/src/luac" "${LUA_ROOT}/bin/luac"
log "ready: ${LUA_BIN}"
