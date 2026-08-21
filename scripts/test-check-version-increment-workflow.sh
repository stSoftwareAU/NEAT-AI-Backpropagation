#!/usr/bin/env bash
# WHAT-tests for the version-increment workflow validator (issue #95).
#
# The committed workflow must pass, and each rule must actually fail a
# workflow that breaks it — including the `paths:` filter, which has to list
# every build-affecting path so the bump job runs at all.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKER="$SCRIPT_DIR/check-version-increment-workflow.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

PASSED=0
FAILED=0

if [[ ! -x "$CHECKER" ]]; then
  echo "FAIL: checker not found or not executable: $CHECKER" >&2
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

expect_exit "committed workflow passes" 0 \
  "$CHECKER" "$REPO_ROOT/.github/workflows/version-increment.yml"

expect_exit "missing workflow file fails loudly" 2 \
  "$CHECKER" "$WORK_DIR/absent.yml"

# A compliant fixture; each mutation below removes exactly one rule.
write_workflow() {
  local target="$1"
  cat >"$target" <<'EOF'
name: Version Increment

on:
  pull_request:
    types: [opened, synchronize, reopened]
    branches:
      - Develop
    paths:
      - "backpropagation/src/**"
      - "backpropagation/Cargo.toml"
      - "Cargo.toml"
      - "Cargo.lock"
      - ".cargo/config.toml"
      - "rust-toolchain.toml"
      - "include/**"

permissions:
  contents: write

jobs:
  version-increment:
    runs-on: ubuntu-latest
    if: github.event.pull_request.head.repo.full_name == github.repository
    steps:
      - name: Bump
        id: bump
        run: |
          set -euo pipefail
          ./scripts/bump-backpropagation-version.sh --base-ref origin/Develop
      - name: Commit
        if: steps.bump.outputs.changed == 'true'
        run: |
          set -euo pipefail
          git commit -m "chore: auto-increment versions for changed projects"
EOF
}

FIXTURE="$WORK_DIR/version-increment.yml"
write_workflow "$FIXTURE"
expect_exit "compliant fixture passes" 0 "$CHECKER" "$FIXTURE"

write_workflow "$FIXTURE"
# Drop the lockfile from the paths filter: a dependency-only PR would then
# never run the bump job, and remotes would keep a stale library.
grep -v '"Cargo.lock"' "$FIXTURE" >"$FIXTURE.tmp" && mv "$FIXTURE.tmp" "$FIXTURE"
expect_exit "paths filter missing Cargo.lock fails" 1 "$CHECKER" "$FIXTURE"

write_workflow "$FIXTURE"
grep -v '".cargo/config.toml"' "$FIXTURE" >"$FIXTURE.tmp" && mv "$FIXTURE.tmp" "$FIXTURE"
expect_exit "paths filter missing .cargo/config.toml fails" 1 "$CHECKER" "$FIXTURE"

write_workflow "$FIXTURE"
sed -i.bak 's/^  pull_request:/  push:/' "$FIXTURE" && rm -f "$FIXTURE.bak"
expect_exit "no pull_request trigger fails" 1 "$CHECKER" "$FIXTURE"

write_workflow "$FIXTURE"
sed -i.bak 's/^  contents: write$/  contents: read/' "$FIXTURE" && rm -f "$FIXTURE.bak"
expect_exit "no contents: write permission fails" 1 "$CHECKER" "$FIXTURE"

write_workflow "$FIXTURE"
grep -v 'bump-backpropagation-version.sh' "$FIXTURE" >"$FIXTURE.tmp" && mv "$FIXTURE.tmp" "$FIXTURE"
expect_exit "no bump script invocation fails" 1 "$CHECKER" "$FIXTURE"

write_workflow "$FIXTURE"
grep -v "steps.bump.outputs.changed" "$FIXTURE" >"$FIXTURE.tmp" && mv "$FIXTURE.tmp" "$FIXTURE"
expect_exit "unconditional commit/push fails" 1 "$CHECKER" "$FIXTURE"

write_workflow "$FIXTURE"
grep -v 'head.repo.full_name' "$FIXTURE" >"$FIXTURE.tmp" && mv "$FIXTURE.tmp" "$FIXTURE"
expect_exit "no fork guard fails" 1 "$CHECKER" "$FIXTURE"

echo
echo "Passed: $PASSED  Failed: $FAILED"
if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
exit 0
