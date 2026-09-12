#!/usr/bin/env bash
# Build, install, and clean neat_ai_backpropagation for GRQ (#152 / GRQ#4774).
#
# Installs both artefacts this crate ships:
#   ~/.cargo/bin/neat_ai_backpropagation
#   ~/.cargo/lib/libneat_ai_backpropagation.{dylib,so}
# each stamped with .neat_ai_backpropagation.version beside it.
#
# stdout is the CLI binary path only. Diagnostics go to stderr.
# A second run whose stamps match prints
#   [neat_ai_backpropagation] already installed v<x>
# and runs no cargo command, then removes nothing (target/ is already gone).
#
# Cross-platform: macOS bash 3.2, Ubuntu, AWS Linux.
set -euo pipefail

PKG="neat_ai_backpropagation"

_repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

_lib_filename() {
  case "$(uname -s)" in
    Darwin) echo "lib${PKG}.dylib" ;;
    *) echo "lib${PKG}.so" ;;
  esac
}

_require_tools() {
  export PATH="${HOME}/.cargo/bin:${PATH}"
  if ! command -v jq >/dev/null 2>&1; then
    echo "ERROR: jq is not available. Please install jq." >&2
    exit 1
  fi
  if ! command -v cargo >/dev/null 2>&1; then
    echo "ERROR: cargo is not available. Install Rust (rustup) first." >&2
    exit 1
  fi
}

# Echo "<version>" from the workspace member. Fails loud on zero/many matches.
_crate_version() {
  local root="$1"
  local meta ver count
  meta="$(cargo metadata --no-deps --format-version 1 --manifest-path "${root}/Cargo.toml")"
  count="$(printf '%s\n' "${meta}" | jq --arg n "${PKG}" '[.packages[] | select(.name==$n)] | length')"
  if [[ "${count}" != "1" ]]; then
    echo "ERROR: expected exactly one package named ${PKG}, got ${count}" >&2
    exit 1
  fi
  ver="$(printf '%s\n' "${meta}" | jq -r --arg n "${PKG}" '.packages[] | select(.name==$n) | .version')"
  if [[ -z "${ver}" || "${ver}" == "null" ]]; then
    echo "ERROR: cannot read version for ${PKG}" >&2
    exit 1
  fi
  printf '%s\n' "${ver}"
}

_find_built_lib() {
  local root="$1" lib_file="$2"
  if [[ -f "${root}/target/release/deps/${lib_file}" ]]; then
    echo "${root}/target/release/deps/${lib_file}"
  elif [[ -f "${root}/target/release/${lib_file}" ]]; then
    echo "${root}/target/release/${lib_file}"
  else
    echo ""
  fi
}

_already_installed() {
  local desired="$1" bin="$2" lib="$3" bin_mark="$4" lib_mark="$5"
  local bv lv
  [[ -x "${bin}" && -f "${lib}" && -f "${bin_mark}" && -f "${lib_mark}" ]] || return 1
  bv="$(cat "${bin_mark}" 2>/dev/null || echo "")"
  lv="$(cat "${lib_mark}" 2>/dev/null || echo "")"
  [[ "${bv}" == "${desired}" && "${lv}" == "${desired}" ]]
}

_install_macos_lib() {
  local lib_path="$1" lib_file="$2"
  install_name_tool -id "@rpath/${lib_file}" "${lib_path}" >&2 2>/dev/null || true
  codesign --force --sign - --timestamp=none --preserve-metadata=entitlements \
    "${lib_path}" >&2 2>/dev/null || true
}

ensure_installed() {
  _require_tools
  local root desired lib_file bin_dir lib_dir bin lib bin_mark lib_mark built_bin built_lib
  root="$(_repo_root)"
  desired="$(_crate_version "${root}")"
  lib_file="$(_lib_filename)"

  bin_dir="${HOME}/.cargo/bin"
  lib_dir="${HOME}/.cargo/lib"
  bin="${bin_dir}/${PKG}"
  lib="${lib_dir}/${lib_file}"
  bin_mark="${bin_dir}/.${PKG}.version"
  lib_mark="${lib_dir}/.${PKG}.version"

  if _already_installed "${desired}" "${bin}" "${lib}" "${bin_mark}" "${lib_mark}"; then
    echo "[${PKG}] already installed v${desired}" >&2
    echo "${bin}"
    return 0
  fi

  echo "Building ${PKG} v${desired}" >&2
  (
    cd "${root}"
    cargo build --release -p "${PKG}" --bin "${PKG}" --lib
  ) >&2

  built_bin="${root}/target/release/${PKG}"
  built_lib="$(_find_built_lib "${root}" "${lib_file}")"
  if [[ ! -x "${built_bin}" ]]; then
    echo "ERROR: build produced no executable at ${built_bin}" >&2
    exit 1
  fi
  if [[ -z "${built_lib}" || ! -f "${built_lib}" ]]; then
    echo "ERROR: build produced no cdylib ${lib_file} under target/release" >&2
    exit 1
  fi

  mkdir -p "${bin_dir}" "${lib_dir}" >&2
  cp "${built_bin}" "${bin}" >&2
  chmod +x "${bin}" >&2
  cp "${built_lib}" "${lib}" >&2
  if [[ "$(uname -s)" == "Darwin" ]]; then
    _install_macos_lib "${lib}" "${lib_file}"
  fi
  printf '%s\n' "${desired}" >"${bin_mark}"
  printf '%s\n' "${desired}" >"${lib_mark}"

  echo "Removing ${root}/target after install" >&2
  rm -rf "${root}/target"

  echo "${bin}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  ensure_installed
fi
