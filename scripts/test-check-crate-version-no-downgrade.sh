#!/usr/bin/env bash
# WHAT-tests for crate version no-downgrade (Issue #87).
#
# Covers:
#   * behind  → fail
#   * equal   → pass
#   * ahead   → pass
# Plus the bump script's exit contract: behind fails CI (exit 2), ahead skips
# without forcing another bump (exit 1).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKER="$SCRIPT_DIR/check-crate-version-no-downgrade.sh"
BUMPER="$SCRIPT_DIR/bump-backpropagation-version.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

PASSED=0
FAILED=0

if [[ ! -x "$CHECKER" ]]; then
  echo "FAIL: checker not found or not executable: $CHECKER" >&2
  exit 2
fi
if [[ ! -x "$BUMPER" ]]; then
  echo "FAIL: bumper not found or not executable: $BUMPER" >&2
  exit 2
fi

# expect_exit DESCRIPTION EXPECTED_CODE COMMAND...
expect_exit() {
  local description="$1" expected="$2"
  shift 2
  local output status=0
  output="$("$@" 2>&1)" || status=$?
  if [[ "$status" -ne "$expected" ]]; then
    echo "FAIL $description: expected exit $expected, got $status" >&2
    printf '%s\n' "$output" >&2
    FAILED=$((FAILED + 1))
    return
  fi
  echo "OK   $description"
  PASSED=$((PASSED + 1))
}

# One checker case per row: BASE|HEAD|EXPECTED EXIT|DESCRIPTION. `|` separates
# for the same reason as the bump-script table — a description may carry any
# punctuation. Expected exit 0 = the head version is acceptable, 1 = it is a
# downgrade the checker must refuse.
CHECKER_CASES=$(
  cat <<'EOF'
0.1.18|0.1.17|1|behind fails
0.1.18|0.1.18|0|equal passes
0.1.18|0.1.19|0|ahead passes
0.2.0|0.1.99|1|minor-behind fails (sort -V)
EOF
)

SAW_PASS_CASE=0
SAW_FAIL_CASE=0
while IFS= read -r row; do
  IFS='|' read -r base head expected description <<<"$row"
  if [[ -z "$base" || -z "$head" || -z "$expected" || -z "$description" ]]; then
    echo "FAIL: malformed case row: '$row'" >&2
    exit 2
  fi
  expect_exit "checker: $description" "$expected" \
    "$CHECKER" --base-version "$base" --head-version "$head"
  if [[ "$expected" -eq 0 ]]; then
    SAW_PASS_CASE=1
  else
    SAW_FAIL_CASE=1
  fi
done <<<"$CHECKER_CASES"

# An emptied or mis-parsed table would otherwise run no cases and still report
# a pass, so both sides of the checker's contract must have been exercised.
if [[ "$SAW_PASS_CASE" -eq 1 && "$SAW_FAIL_CASE" -eq 1 ]]; then
  echo "OK   checker: both accept and reject cases ran"
  PASSED=$((PASSED + 1))
else
  echo "FAIL: checker case table exercised only one side of the contract" >&2
  FAILED=$((FAILED + 1))
fi

# Fixture repo for the bump script: Develop at 0.1.10, branch variants.
FIXTURE="$WORK_DIR/repo"
mkdir -p "$FIXTURE/backpropagation/src"
cd "$FIXTURE"
git init -q -b Develop
git config user.name "test"
git config user.email "test@example.com"
cat >backpropagation/Cargo.toml <<'EOF'
[package]
name = "neat_ai_backpropagation"
version = "0.1.10"
edition = "2021"
EOF
echo '// stub' >backpropagation/src/lib.rs
git add backpropagation
git commit -q -m "base Develop 0.1.10"
git branch -M Develop

# Behind: merge-conflict style downgrade must fail, not skip.
git checkout -q -b behind
sed -i.bak 's/version = "0.1.10"/version = "0.1.9"/' backpropagation/Cargo.toml
rm -f backpropagation/Cargo.toml.bak
git add backpropagation/Cargo.toml
git commit -q -m "accidentally take older version"
expect_exit "bump: behind fails (exit 2, not skip)" 2 \
  "$BUMPER" --repo-root "$FIXTURE" --base-ref Develop --check

# Ahead: accept without forcing another bump.
git checkout -q Develop
git checkout -q -b ahead
sed -i.bak 's/version = "0.1.10"/version = "0.1.11"/' backpropagation/Cargo.toml
rm -f backpropagation/Cargo.toml.bak
git add backpropagation/Cargo.toml
git commit -q -m "manual bump ahead of Develop"
expect_exit "bump: ahead skips without further bump" 1 \
  "$BUMPER" --repo-root "$FIXTURE" --base-ref Develop --check

# Equal + src change: still eligible to auto-patch-bump.
git checkout -q Develop
git checkout -q -b equal-src
echo '// change' >>backpropagation/src/lib.rs
git add backpropagation/src/lib.rs
git commit -q -m "src change, version still equal"
expect_exit "bump: equal with src changes would bump" 0 \
  "$BUMPER" --repo-root "$FIXTURE" --base-ref Develop --check

echo
echo "Passed: $PASSED  Failed: $FAILED"
if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
exit 0
