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

expect_exit "checker: behind fails" 1 \
  "$CHECKER" --base-version 0.1.18 --head-version 0.1.17

expect_exit "checker: equal passes" 0 \
  "$CHECKER" --base-version 0.1.18 --head-version 0.1.18

expect_exit "checker: ahead passes" 0 \
  "$CHECKER" --base-version 0.1.18 --head-version 0.1.19

expect_exit "checker: minor-behind fails (sort -V)" 1 \
  "$CHECKER" --base-version 0.2.0 --head-version 0.1.99

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
