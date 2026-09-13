#!/usr/bin/env bash
# Tests for scripts/check-runlib-canonical.sh (issue #152).
#
# Each case runs the real checker over real files and asserts the exit code and
# the message, so the drift gate cannot pass by saying nothing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKER="$SCRIPT_DIR/check-runlib-canonical.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

PASSED=0
FAILED=0

if [[ ! -x "$CHECKER" ]]; then
  echo "FAIL: checker not found or not executable: $CHECKER" >&2
  exit 2
fi

# expect_exit DESCRIPTION EXPECTED_CODE COPY CANONICAL [NEEDLE]
expect_exit() {
  local description="$1" expected="$2" copy="$3" canonical="$4" needle="${5:-}"
  local output status=0
  output="$(RUNLIB_COPY="$copy" "$CHECKER" "$canonical" 2>&1)" || status=$?
  if [[ "$status" -ne "$expected" ]]; then
    echo "FAIL $description: expected exit $expected, got $status" >&2
    printf '%s\n' "$output" >&2
    FAILED=$((FAILED + 1))
    return
  fi
  if [[ -n "$needle" && "$output" != *"$needle"* ]]; then
    echo "FAIL $description: output did not mention '$needle'" >&2
    printf '%s\n' "$output" >&2
    FAILED=$((FAILED + 1))
    return
  fi
  echo "OK   $description"
  PASSED=$((PASSED + 1))
}

printf '#!/usr/bin/env bash\necho canonical\n' >"$WORK_DIR/canonical.sh"
cp "$WORK_DIR/canonical.sh" "$WORK_DIR/identical.sh"
printf '#!/usr/bin/env bash\necho canonical\n# a downstream edit\n' >"$WORK_DIR/drifted.sh"
# One byte apart — drift is not always a visible line.
printf '#!/usr/bin/env bash\necho Canonical\n' >"$WORK_DIR/one-byte.sh"

expect_exit "accepts a byte-identical copy" 0 \
  "$WORK_DIR/identical.sh" "$WORK_DIR/canonical.sh" "byte-identical"

expect_exit "rejects a copy with an extra line" 1 \
  "$WORK_DIR/drifted.sh" "$WORK_DIR/canonical.sh" "has drifted"

expect_exit "rejects a copy differing by a single byte" 1 \
  "$WORK_DIR/one-byte.sh" "$WORK_DIR/canonical.sh" "has drifted"

expect_exit "names the copy contract in the failure" 1 \
  "$WORK_DIR/drifted.sh" "$WORK_DIR/canonical.sh" "has one home"

expect_exit "reports an unreadable canonical copy with exit 2" 2 \
  "$WORK_DIR/identical.sh" "$WORK_DIR/absent.sh" "not found"

expect_exit "reports a missing copy with exit 2" 2 \
  "$WORK_DIR/no-such-copy.sh" "$WORK_DIR/canonical.sh" "no copy to check"

# The committed copy against the real sibling, when this checkout has one —
# the same comparison CI makes. Skipped loudly rather than silently when the
# sibling is absent.
if [[ -d "$REPO_ROOT/../NEAT-AI-core" ]]; then
  expect_exit "the committed scripts/runlib.sh matches the NEAT-AI-core sibling" 0 \
    "$REPO_ROOT/scripts/runlib.sh" "" "byte-identical"
else
  echo "SKIP the committed copy vs the sibling — no ../NEAT-AI-core checkout (CI runs this for real)"
fi

echo "check-runlib-canonical tests: $PASSED passed, $FAILED failed"
if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
