#!/usr/bin/env bash
# Tests for scripts/check-canonical-copies.sh (issues #152 and #153).
#
# Each case runs the real checker over real files and asserts the exit code and
# the message, so the drift gate cannot pass by saying nothing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKER="$SCRIPT_DIR/check-canonical-copies.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

PASSED=0
FAILED=0

if [[ ! -x "$CHECKER" ]]; then
  echo "FAIL: checker not found or not executable: $CHECKER" >&2
  exit 2
fi

# expect_exit DESCRIPTION EXPECTED_CODE COPIES_DIR CANONICAL_DIR [NEEDLE]
expect_exit() {
  local description="$1" expected="$2" copies="$3" canonical="$4" needle="${5:-}"
  local output status=0
  output="$(CANONICAL_COPIES_DIR="$copies" "$CHECKER" "$canonical" 2>&1)" || status=$?
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

# new_pair NAME → a canonical directory and a matching copies directory.
new_pair() {
  local name="$1"
  mkdir -p "$WORK_DIR/$name/canonical" "$WORK_DIR/$name/copies"
  printf '#!/usr/bin/env bash\necho runlib\n' >"$WORK_DIR/$name/canonical/runlib.sh"
  printf '#!/usr/bin/env bash\necho family-pins\n' >"$WORK_DIR/$name/canonical/family-pins.sh"
  cp "$WORK_DIR/$name/canonical/runlib.sh" "$WORK_DIR/$name/copies/runlib.sh"
  cp "$WORK_DIR/$name/canonical/family-pins.sh" "$WORK_DIR/$name/copies/family-pins.sh"
  printf '%s' "$WORK_DIR/$name"
}

test_accepts_byte_identical_copies() {
  local dir
  dir="$(new_pair identical)"
  expect_exit "accepts byte-identical copies of both scripts" 0 \
    "$dir/copies" "$dir/canonical" "byte-identical"
}

test_rejects_a_drifted_runlib() {
  local dir
  dir="$(new_pair runlib-drift)"
  printf '# a downstream edit\n' >>"$dir/copies/runlib.sh"
  expect_exit "rejects a runlib.sh with an extra line" 1 \
    "$dir/copies" "$dir/canonical" "runlib.sh has drifted"
}

test_rejects_a_drifted_family_pins() {
  local dir
  dir="$(new_pair family-pins-drift)"
  printf '# a downstream edit\n' >>"$dir/copies/family-pins.sh"
  expect_exit "rejects a family-pins.sh with an extra line" 1 \
    "$dir/copies" "$dir/canonical" "family-pins.sh has drifted"
}

test_rejects_a_single_byte_difference() {
  local dir
  dir="$(new_pair one-byte)"
  printf '#!/usr/bin/env bash\necho Family-pins\n' >"$dir/copies/family-pins.sh"
  expect_exit "rejects a copy differing by a single byte" 1 \
    "$dir/copies" "$dir/canonical" "has drifted"
}

test_names_the_copy_contract_in_the_failure() {
  local dir
  dir="$(new_pair contract)"
  printf '# a downstream edit\n' >>"$dir/copies/family-pins.sh"
  expect_exit "names the copy contract in the failure" 1 \
    "$dir/copies" "$dir/canonical" "has one home"
}

test_reports_an_unreadable_canonical_copy_with_exit_2() {
  local dir
  dir="$(new_pair unreadable)"
  rm "$dir/canonical/family-pins.sh"
  expect_exit "reports an unreadable canonical copy with exit 2" 2 \
    "$dir/copies" "$dir/canonical" "not found"
}

test_reports_a_missing_copy_with_exit_2() {
  local dir
  dir="$(new_pair missing-copy)"
  rm "$dir/copies/family-pins.sh"
  expect_exit "reports a missing copy with exit 2" 2 \
    "$dir/copies" "$dir/canonical" "no copy to check"
}

# The committed copies against the real sibling, when this checkout has one —
# the same comparison CI makes. Skipped loudly rather than silently when the
# sibling is absent.
test_the_committed_copies_match_the_sibling() {
  if [[ -d "$REPO_ROOT/../NEAT-AI-core" ]]; then
    expect_exit "the committed copies match the NEAT-AI-core sibling" 0 \
      "$REPO_ROOT/scripts" "" "byte-identical"
  else
    echo "SKIP the committed copies vs the sibling — no ../NEAT-AI-core checkout (CI runs this for real)"
  fi
}

for test_case in \
  test_accepts_byte_identical_copies \
  test_rejects_a_drifted_runlib \
  test_rejects_a_drifted_family_pins \
  test_rejects_a_single_byte_difference \
  test_names_the_copy_contract_in_the_failure \
  test_reports_an_unreadable_canonical_copy_with_exit_2 \
  test_reports_a_missing_copy_with_exit_2 \
  test_the_committed_copies_match_the_sibling; do
  "$test_case"
done

echo "check-canonical-copies tests: $PASSED passed, $FAILED failed"
if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
