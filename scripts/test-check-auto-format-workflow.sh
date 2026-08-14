#!/usr/bin/env bash
# Tests for scripts/check-auto-format-workflow.sh (Issue #27).
#
# Each case writes a fixture workflow to a temporary directory, runs the real
# checker against it, and asserts the exit code (and, where it matters, that
# the failure names the rule that broke). The cases here cover the
# milestone-branch coverage rule: milestone sub-issue PRs target a shared
# `milestone/<slug>` branch, so a filter that lists only the default branch
# leaves every intermediate PR ungated.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKER="$SCRIPT_DIR/check-auto-format-workflow.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

PASSED=0
FAILED=0

if [[ ! -x "$CHECKER" ]]; then
  echo "FAIL: checker not found or not executable: $CHECKER" >&2
  exit 2
fi

# write_workflow NAME <<'YAML' … YAML  → echoes the fixture path.
write_workflow() {
  local path="$WORK_DIR/$1.yml"
  cat >"$path"
  printf '%s' "$path"
}

# expect_exit DESCRIPTION EXPECTED_CODE WORKFLOW_PATH [EXPECTED_OUTPUT_SUBSTRING]
expect_exit() {
  local description="$1" expected="$2" workflow="$3" needle="${4:-}"
  local output status=0
  output="$("$CHECKER" "$workflow" 2>&1)" || status=$?
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

milestone_block=$(write_workflow milestone-block <<'YAML'
name: Auto Format
on:
  pull_request:
    types: [opened, synchronize, reopened]
    branches:
      - Develop
      - "milestone/**"
    paths-ignore:
      - "**.md"
permissions:
  contents: write
jobs:
  auto-format:
    if: github.event.pull_request.head.repo.full_name == github.repository
    steps:
      - name: Apply cargo fmt
        run: |
          set -euo pipefail
          cargo fmt --all
          cargo update -p neat-core
      - name: Commit
        if: steps.detect.outputs.changed == 'true'
        env:
          GH_PAT: ${{ secrets.ACTIONS_PUSH || secrets.GITHUB_TOKEN }}
        run: |
          set -euo pipefail
          git push origin HEAD
YAML
)
expect_exit "accepts a block-style filter covering milestone branches" 0 "$milestone_block"

milestone_flow=$(write_workflow milestone-flow <<'YAML'
name: Auto Format
on:
  pull_request:
    types: [opened, synchronize, reopened]
    branches: [Develop, "milestone/*"]
permissions:
  contents: write
jobs:
  auto-format:
    if: github.event.pull_request.head.repo.full_name == github.repository
    steps:
      - name: Apply cargo fmt
        run: |
          set -euo pipefail
          cargo fmt --all
          cargo update -p neat-core
      - name: Commit
        if: steps.detect.outputs.changed == 'true'
        env:
          GH_PAT: ${{ secrets.ACTIONS_PUSH || secrets.GITHUB_TOKEN }}
        run: |
          set -euo pipefail
          git push origin HEAD
YAML
)
expect_exit "accepts a flow-style filter covering milestone branches" 0 "$milestone_flow"

no_filter=$(write_workflow no-filter <<'YAML'
name: Auto Format
on:
  pull_request:
    types: [opened, synchronize, reopened]
permissions:
  contents: write
jobs:
  auto-format:
    if: github.event.pull_request.head.repo.full_name == github.repository
    steps:
      - name: Apply cargo fmt
        run: |
          set -euo pipefail
          cargo fmt --all
          cargo update -p neat-core
      - name: Commit
        if: steps.detect.outputs.changed == 'true'
        env:
          GH_PAT: ${{ secrets.ACTIONS_PUSH || secrets.GITHUB_TOKEN }}
        run: |
          set -euo pipefail
          git push origin HEAD
YAML
)
expect_exit "accepts no branch filter at all (every PR is gated)" 0 "$no_filter"

develop_only=$(write_workflow develop-only <<'YAML'
name: Auto Format
on:
  pull_request:
    types: [opened, synchronize, reopened]
    branches:
      - Develop
    paths-ignore:
      - "**.md"
permissions:
  contents: write
jobs:
  auto-format:
    if: github.event.pull_request.head.repo.full_name == github.repository
    steps:
      - name: Apply cargo fmt
        run: |
          set -euo pipefail
          cargo fmt --all
          cargo update -p neat-core
      - name: Commit
        if: steps.detect.outputs.changed == 'true'
        env:
          GH_PAT: ${{ secrets.ACTIONS_PUSH || secrets.GITHUB_TOKEN }}
        run: |
          set -euo pipefail
          git push origin HEAD
YAML
)
expect_exit "rejects a filter that skips milestone branches" 1 \
  "$develop_only" "milestone"

literal_milestone=$(write_workflow literal-milestone <<'YAML'
name: Auto Format
on:
  pull_request:
    types: [opened, synchronize, reopened]
    branches:
      - Develop
      - milestone/only-this-one
permissions:
  contents: write
jobs:
  auto-format:
    if: github.event.pull_request.head.repo.full_name == github.repository
    steps:
      - name: Apply cargo fmt
        run: |
          set -euo pipefail
          cargo fmt --all
          cargo update -p neat-core
      - name: Commit
        if: steps.detect.outputs.changed == 'true'
        env:
          GH_PAT: ${{ secrets.ACTIONS_PUSH || secrets.GITHUB_TOKEN }}
        run: |
          set -euo pipefail
          git push origin HEAD
YAML
)
expect_exit "rejects a single literal milestone branch as coverage" 1 \
  "$literal_milestone" "milestone"

expect_exit "reports a missing workflow with exit 2" 2 \
  "$WORK_DIR/does-not-exist.yml" "not found"

echo "check-auto-format-workflow tests: $PASSED passed, $FAILED failed"
if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
