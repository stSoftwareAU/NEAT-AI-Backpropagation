#!/usr/bin/env bash
# Tests for scripts/generate-synthetic-corpus.sh (Issue #178).
#
# The generator is the single source of truth for the synthetic corpus every
# experiment script writes. These cases run it for real and decode the records
# it writes, so a drifted layout, record count or target formula fails here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATOR="${SCRIPT_DIR}/generate-synthetic-corpus.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

PASSED=0
FAILED=0

pass() {
  echo "  PASS: $1"
  PASSED=$((PASSED + 1))
}

fail() {
  echo "  FAIL: $1" >&2
  FAILED=$((FAILED + 1))
}

# expect_exit DESCRIPTION EXPECTED_CODE CMD...
expect_exit() {
  local description="$1" expected="$2"
  shift 2
  local actual=0
  "$@" >"${WORK_DIR}/out.log" 2>&1 || actual=$?
  if [[ "${actual}" -eq "${expected}" ]]; then
    pass "${description}"
  else
    fail "${description} (expected exit ${expected}, got ${actual})"
    sed 's/^/    /' "${WORK_DIR}/out.log" >&2
  fi
}

# Decode a corpus directory and check every record against a reference target.
# Prints nothing and exits 0 only when the layout and every value match.
verify_corpus() {
  local dir="$1" reference="$2"
  python3 - "${dir}" "${reference}" <<'PY'
import struct
import sys
from pathlib import Path

data = Path(sys.argv[1])
reference = sys.argv[2]
targets = {
    "contaminated": lambda i, x: -2.0 * x + 1.0 if i < 5 else 1.0 * x + 0.1,
    "linear": lambda i, x: 0.5 * x + 0.25,
}[reference]

files = sorted(p.name for p in data.iterdir())
if files != ["0.bin", "1.bin", "2.bin", "3.bin"]:
    sys.exit(f"unexpected files: {files}")
for name in files:
    raw = (data / name).read_bytes()
    if len(raw) != 250 * 8:
        sys.exit(f"{name}: {len(raw)} bytes, expected {250 * 8}")
    for i in range(250):
        x, y = struct.unpack_from("<ff", raw, i * 8)
        want_x = i / 250 * 2 - 1
        want_y = targets(i, want_x)
        if x != struct.unpack("<f", struct.pack("<f", want_x))[0]:
            sys.exit(f"{name}[{i}]: x={x}, expected {want_x}")
        if y != struct.unpack("<f", struct.pack("<f", want_y))[0]:
            sys.exit(f"{name}[{i}]: y={y}, expected {want_y}")
PY
}

CONTAMINATED='-2.0 * x + 1.0 if i < 5 else 1.0 * x + 0.1'
LINEAR='0.5 * x + 0.25'

echo "=== generate-synthetic-corpus.sh ==="

echo "--- happy path ---"
expect_exit "contaminated target writes a corpus" 0 \
  "${GENERATOR}" "${WORK_DIR}/contaminated" "${CONTAMINATED}"
expect_exit "contaminated corpus decodes to the expected records" 0 \
  verify_corpus "${WORK_DIR}/contaminated" contaminated
expect_exit "linear target writes a corpus" 0 \
  "${GENERATOR}" "${WORK_DIR}/linear" "${LINEAR}"
expect_exit "linear corpus decodes to the expected records" 0 \
  verify_corpus "${WORK_DIR}/linear" linear

echo "--- edge cases ---"
mkdir -p "${WORK_DIR}/existing"
expect_exit "an existing empty directory is filled" 0 \
  "${GENERATOR}" "${WORK_DIR}/existing" "${LINEAR}"
expect_exit "the filled directory decodes correctly" 0 \
  verify_corpus "${WORK_DIR}/existing" linear
expect_exit "a nested, not-yet-created directory is created" 0 \
  "${GENERATOR}" "${WORK_DIR}/a/b/c" "${LINEAR}"
expect_exit "the nested directory decodes correctly" 0 \
  verify_corpus "${WORK_DIR}/a/b/c" linear

echo "--- error paths ---"
expect_exit "no arguments is a usage error" 2 "${GENERATOR}"
expect_exit "a missing target is a usage error" 2 \
  "${GENERATOR}" "${WORK_DIR}/missing-target"
expect_exit "an extra argument is a usage error" 2 \
  "${GENERATOR}" "${WORK_DIR}/extra" "${LINEAR}" surplus
expect_exit "an empty directory argument is a usage error" 2 \
  "${GENERATOR}" "" "${LINEAR}"
expect_exit "an empty target is a usage error" 2 \
  "${GENERATOR}" "${WORK_DIR}/empty-target" ""
expect_exit "a target naming anything but x and i is refused" 2 \
  "${GENERATOR}" "${WORK_DIR}/bad-name" "y + 1"
expect_exit "a target calling a function is refused" 2 \
  "${GENERATOR}" "${WORK_DIR}/call" "__import__('os').getcwd()"
expect_exit "a target with an attribute access is refused" 2 \
  "${GENERATOR}" "${WORK_DIR}/attr" "x.real"
expect_exit "a target that is not an expression is refused" 2 \
  "${GENERATOR}" "${WORK_DIR}/syntax" "x +"
expect_exit "a target with a string constant is refused" 2 \
  "${GENERATOR}" "${WORK_DIR}/string" "'a'"
if [[ -e "${WORK_DIR}/call/0.bin" || -e "${WORK_DIR}/bad-name/0.bin" ]]; then
  fail "a refused target left records behind"
else
  pass "a refused target writes no records"
fi
expect_exit "a target that fails at evaluation fails loud" 1 \
  "${GENERATOR}" "${WORK_DIR}/divide" "x / (i - i)"
if [[ -e "${WORK_DIR}/divide/0.bin" ]]; then
  fail "a failed evaluation left a partial corpus behind"
else
  pass "a failed evaluation leaves no partial corpus"
fi
touch "${WORK_DIR}/a-file"
expect_exit "a directory path that is a file fails loud" 1 \
  "${GENERATOR}" "${WORK_DIR}/a-file" "${LINEAR}"

echo
echo "Passed: ${PASSED}  Failed: ${FAILED}"
if [[ "${FAILED}" -gt 0 ]]; then
  exit 1
fi
