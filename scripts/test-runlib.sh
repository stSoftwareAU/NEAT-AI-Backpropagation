#!/usr/bin/env bash
# Hermetic tests for scripts/runlib.sh (Issue #152 / GRQ#4774).
#
# scripts/runlib.sh is the canonical NEAT-AI-core copy (core #680): it is
# byte-identical to `scripts/runlib.sh` on NEAT-AI-core `Develop` and is never
# edited here — .github/workflows/family-sync.yml re-copies it. These tests
# therefore assert the *contract* that copy owes this crate, not its internals,
# so a refreshed copy keeps passing:
#
#   * an up-to-date install compiles nothing — the cargo shim below fails loud
#     on any cargo command that is not `metadata`, and the invocation count is
#     asserted, so a build (or a second metadata call) is a test failure;
#   * stdout is the CLI path and nothing else, with the already-installed line
#     on stderr;
#   * the *whole* shape is checked — a missing cdylib, or a missing stamp, is
#     not "already installed".
#
# Why `metadata` is tolerated rather than refused: today's canonical copy
# declines its no-cargo fast path on a manifest carrying an explicit `[[bin]]`
# table — this crate's shape — and falls through to `cargo metadata`, which is
# always the authority. NEAT-AI-core #690 teaches the fast path to read a
# single `[[bin]]` table naming the crate, after which the skip runs no cargo
# command at all. Both readings satisfy the assertions below, so the family
# sync that brings #690 in cannot turn this suite red.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNLIB="${SCRIPT_DIR}/runlib.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
REAL_PATH="${PATH}"

# runlib.sh resolves the workspace from $PWD, so every case runs from the
# repository root regardless of where the suite was invoked.
cd "${REPO_ROOT}"

PASSED=0
FAILED=0

if [[ ! -x "${RUNLIB}" ]]; then
  echo "FAIL: runlib not found or not executable: ${RUNLIB}" >&2
  exit 2
fi

assert_eq() {
  local desc="$1" expected="$2" actual="$3"
  if [[ "${expected}" == "${actual}" ]]; then
    echo "  PASS: ${desc}"
    PASSED=$((PASSED + 1))
  else
    echo "  FAIL: ${desc}"
    echo "    expected: '${expected}'"
    echo "    actual:   '${actual}'"
    FAILED=$((FAILED + 1))
  fi
}

CRATE="neat_ai_backpropagation"
LIB_FILE="lib${CRATE}.so"
if [[ "$(uname -s)" == "Darwin" ]]; then
  LIB_FILE="lib${CRATE}.dylib"
fi

# The version the crate manifest declares — the same value both the fast path
# (read from the manifest) and the metadata path (answered by the shim below)
# must agree on, so neither reading is special-cased here.
VERSION="$(awk '
  /^\[package\]/ { in_package = 1; next }
  /^\[/ { in_package = 0 }
  in_package && /^[[:space:]]*version[[:space:]]*=/ {
    gsub(/^[^"]*"|".*$/, "")
    print
    exit
  }
' "${REPO_ROOT}/backpropagation/Cargo.toml")"

if [[ -z "${VERSION}" ]]; then
  echo "FAIL: could not read the crate version from backpropagation/Cargo.toml" >&2
  exit 2
fi

SHIM_LOG="${WORK_DIR}/cargo-invocations.log"
export RUNLIB_SHIM_LOG="${SHIM_LOG}"

# Answers `cargo metadata` with this workspace's real shape and fails loud on
# every other cargo command — a compile on the already-installed path is the
# regression this suite exists to catch. Every invocation is logged, so the
# count can be asserted rather than inferred.
install_cargo_shim() {
  local bin_dir="$1"
  mkdir -p "${bin_dir}"
  cat >"${bin_dir}/cargo" <<EOF
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "\$*" >>"\${RUNLIB_SHIM_LOG}"
if [[ "\${1:-}" == "metadata" ]]; then
  cat <<'JSON'
{
  "packages": [
    {
      "name": "${CRATE}",
      "version": "${VERSION}",
      "manifest_path": "${REPO_ROOT}/backpropagation/Cargo.toml",
      "targets": [
        { "kind": ["bin"], "name": "${CRATE}" },
        { "kind": ["cdylib", "rlib"], "name": "${CRATE}" }
      ]
    }
  ],
  "target_directory": "${WORK_DIR}/target"
}
JSON
  exit 0
fi
echo "UNEXPECTED cargo: \$*" >&2
exit 99
EOF
  chmod +x "${bin_dir}/cargo"
}

# Number of cargo invocations the shim recorded since the last reset.
cargo_invocations() {
  [[ -f "${SHIM_LOG}" ]] || { printf '0'; return 0; }
  awk 'END { print NR }' "${SHIM_LOG}"
}

# Every cargo invocation that was not `metadata` — a build, an install, …
cargo_non_metadata_invocations() {
  [[ -f "${SHIM_LOG}" ]] || { printf '0'; return 0; }
  awk '$1 != "metadata" { count++ } END { print count + 0 }' "${SHIM_LOG}"
}

# Both are set: runlib.sh installs under $CARGO_HOME, falling back to
# $HOME/.cargo, and this container exports a real CARGO_HOME that would
# otherwise make the suite write into the developer's own toolchain.
HOME="${WORK_DIR}/home"
export HOME
CARGO_HOME="${HOME}/.cargo"
export CARGO_HOME
BIN_DIR="${CARGO_HOME}/bin"
LIB_DIR="${CARGO_HOME}/lib"

# A complete, stamped install of both artefacts at the crate's own version.
stage_installed() {
  rm -rf "${HOME}"
  mkdir -p "${BIN_DIR}" "${LIB_DIR}"
  printf 'fake\n' >"${BIN_DIR}/${CRATE}"
  chmod +x "${BIN_DIR}/${CRATE}"
  printf 'fake\n' >"${LIB_DIR}/${LIB_FILE}"
  printf '%s\n' "${VERSION}" >"${BIN_DIR}/.${CRATE}.version"
  printf '%s\n' "${VERSION}" >"${LIB_DIR}/.${CRATE}.version"
  : >"${SHIM_LOG}"
}

install_cargo_shim "${WORK_DIR}/shim"
PATH="${WORK_DIR}/shim:${REAL_PATH}"
export PATH

echo "=== already installed: compiles nothing, CLI path on stdout ==="
stage_installed
RC=0
OUT="$(bash "${RUNLIB}" 2>"${WORK_DIR}/already.err")" || RC=$?
assert_eq "already-installed exits 0" "0" "${RC}"
assert_eq "already-installed stdout is the CLI path" \
  "${BIN_DIR}/${CRATE}" "${OUT}"
assert_eq "already-installed names the version on stderr" "0" \
  "$(grep -q "\[${CRATE}\] already installed v${VERSION}" "${WORK_DIR}/already.err"; echo $?)"
assert_eq "already-installed runs no cargo build" "0" \
  "$(cargo_non_metadata_invocations)"
# One `cargo metadata` is the most the canonical copy may cost on this path
# (zero once NEAT-AI-core #690 lands and the fast path reads the [[bin]]
# table); anything more means the skip is doing real work.
if [[ "$(cargo_invocations)" -le 1 ]]; then
  assert_eq "already-installed costs at most one cargo metadata call" "ok" "ok"
else
  assert_eq "already-installed costs at most one cargo metadata call" "ok" \
    "$(cargo_invocations) cargo invocations"
fi

echo ""
echo "=== missing stamp is not treated as already installed ==="
stage_installed
rm -f "${BIN_DIR}/.${CRATE}.version"
RC=0
OUT="$(bash "${RUNLIB}" 2>"${WORK_DIR}/rebuild.err")" || RC=$?
assert_eq "missing stamp attempts a build and the shim refuses it" "99" "${RC}"
assert_eq "refused build names unexpected cargo" "0" \
  "$(grep -q 'UNEXPECTED cargo: build' "${WORK_DIR}/rebuild.err"; echo $?)"

echo ""
echo "=== a missing cdylib is not treated as already installed ==="
stage_installed
rm -f "${LIB_DIR}/${LIB_FILE}"
RC=0
OUT="$(bash "${RUNLIB}" 2>"${WORK_DIR}/half.err")" || RC=$?
assert_eq "half-installed crate rebuilds rather than reporting complete" "99" "${RC}"
assert_eq "half-installed rebuild names unexpected cargo" "0" \
  "$(grep -q 'UNEXPECTED cargo: build' "${WORK_DIR}/half.err"; echo $?)"

echo ""
echo "=== a stale stamp is not treated as already installed ==="
stage_installed
printf '0.0.1\n' >"${BIN_DIR}/.${CRATE}.version"
printf '0.0.1\n' >"${LIB_DIR}/.${CRATE}.version"
RC=0
OUT="$(bash "${RUNLIB}" 2>"${WORK_DIR}/stale.err")" || RC=$?
assert_eq "stale stamp rebuilds rather than reporting complete" "99" "${RC}"

# The install path itself, hermetically: a synthetic checkout with this crate's
# shape (an explicit `[[bin]]` table beside a cdylib `[lib]`), and a cargo shim
# whose "build" drops the artefacts where the real one would. Nothing else in
# the repository asserts that both artefacts land, that both stamps are written
# or that `target/` is removed — the cases above all stop at the decision to
# build.
echo ""
echo "=== a cold run installs both artefacts, stamps them and removes target/ ==="
FIXTURE="${WORK_DIR}/fixture"
mkdir -p "${FIXTURE}/member/src" "${FIXTURE}/shim"
cat >"${FIXTURE}/Cargo.toml" <<'TOML'
[workspace]
members = ["member"]
resolver = "2"
TOML
cat >"${FIXTURE}/member/Cargo.toml" <<TOML
[package]
name = "${CRATE}"
version = "${VERSION}"
edition = "2024"

[lib]
name = "${CRATE}"
path = "src/lib.rs"
crate-type = ["cdylib", "rlib"]

[[bin]]
name = "${CRATE}"
path = "src/main.rs"
TOML
: >"${FIXTURE}/member/src/lib.rs"
: >"${FIXTURE}/member/src/main.rs"

# `build` writes the artefacts the real cargo would, so the staging, commit,
# stamp and target/ removal steps all run for real against them.
cat >"${FIXTURE}/shim/cargo" <<EOF
#!/usr/bin/env bash
set -euo pipefail
printf '%s
' "\$*" >>"\${RUNLIB_SHIM_LOG}"
case "\${1:-}" in
  metadata)
    cat <<'JSON'
{
  "packages": [
    {
      "name": "${CRATE}",
      "version": "${VERSION}",
      "manifest_path": "${FIXTURE}/member/Cargo.toml",
      "targets": [
        { "kind": ["bin"], "name": "${CRATE}" },
        { "kind": ["cdylib", "rlib"], "name": "${CRATE}" }
      ]
    }
  ],
  "target_directory": "${FIXTURE}/target"
}
JSON
    ;;
  build)
    mkdir -p "${FIXTURE}/target/release"
    printf 'built-bin
' >"${FIXTURE}/target/release/${CRATE}"
    chmod +x "${FIXTURE}/target/release/${CRATE}"
    printf 'built-lib
' >"${FIXTURE}/target/release/${LIB_FILE}"
    ;;
  *)
    echo "UNEXPECTED cargo: \$*" >&2
    exit 99
    ;;
esac
EOF
chmod +x "${FIXTURE}/shim/cargo"

rm -rf "${HOME}"
mkdir -p "${BIN_DIR}" "${LIB_DIR}"
: >"${SHIM_LOG}"
RC=0
OUT="$(cd "${FIXTURE}" && PATH="${FIXTURE}/shim:${REAL_PATH}" bash "${RUNLIB}" 2>"${WORK_DIR}/install.err")" || RC=$?
assert_eq "cold install exits 0" "0" "${RC}"
assert_eq "cold install stdout is the CLI path" "${BIN_DIR}/${CRATE}" "${OUT}"
assert_eq "the CLI binary is installed" "built-bin" "$(cat "${BIN_DIR}/${CRATE}" 2>/dev/null || true)"
assert_eq "the cdylib is installed" "built-lib" "$(cat "${LIB_DIR}/${LIB_FILE}" 2>/dev/null || true)"
BIN_EXECUTABLE="no"
[[ -x "${BIN_DIR}/${CRATE}" ]] && BIN_EXECUTABLE="yes"
assert_eq "the CLI binary is executable" "yes" "${BIN_EXECUTABLE}"
assert_eq "a stamp sits beside the binary" "${VERSION}" \
  "$(cat "${BIN_DIR}/.${CRATE}.version" 2>/dev/null || true)"
assert_eq "a stamp sits beside the cdylib" "${VERSION}" \
  "$(cat "${LIB_DIR}/.${CRATE}.version" 2>/dev/null || true)"
TARGET_PRESENT="yes"
[[ -d "${FIXTURE}/target" ]] || TARGET_PRESENT="no"
assert_eq "target/ is removed after a successful install" "no" "${TARGET_PRESENT}"
assert_eq "the removal names the freed bytes on stderr" "0" \
  "$(grep -q "\[${CRATE}\] removed ${FIXTURE}/target (freed [0-9]* bytes)" "${WORK_DIR}/install.err"; echo $?)"

echo ""
echo "=== the run after that install compiles nothing ==="
: >"${SHIM_LOG}"
RC=0
OUT="$(cd "${FIXTURE}" && PATH="${FIXTURE}/shim:${REAL_PATH}" bash "${RUNLIB}" 2>"${WORK_DIR}/install-warm.err")" || RC=$?
assert_eq "the warm run exits 0" "0" "${RC}"
assert_eq "the warm run names the installed version" "0" \
  "$(grep -q "\[${CRATE}\] already installed v${VERSION}" "${WORK_DIR}/install-warm.err"; echo $?)"
assert_eq "the warm run runs no cargo build" "0" "$(cargo_non_metadata_invocations)"

echo ""
echo "=== summary: ${PASSED} passed, ${FAILED} failed ==="
[[ "${FAILED}" -eq 0 ]]
