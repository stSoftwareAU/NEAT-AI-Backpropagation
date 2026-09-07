#!/usr/bin/env bash
# Tests for scripts/check-codeql-workflow.sh (Issue #20).
#
# Each case writes a fixture workflow to a temporary directory, runs the real
# checker against it, and asserts the exit code (and, where it matters, that
# the failure names the rule that broke).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKER="$SCRIPT_DIR/check-codeql-workflow.sh"
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

valid=$(write_workflow valid <<'YAML'
name: CodeQL
on:
  pull_request:
    branches:
      - Develop
  schedule:
    - cron: "30 4 * * 1"
permissions:
  contents: read
jobs:
  analyse:
    permissions:
      contents: read
      security-events: write
    steps:
      - uses: actions/checkout@93cb6efe18208431cddfb8368fd83d5badbf9bfd  # v5
      - uses: ./.github/actions/setup-rust-workspace
      - name: Initialise CodeQL
        uses: github/codeql-action/init@c16c0f3f2812ec4bb3750a5ed64873fe2ce0fbef  # v4
        with:
          languages: rust
      - name: Analyse
        uses: github/codeql-action/analyze@c16c0f3f2812ec4bb3750a5ed64873fe2ce0fbef  # v4
YAML
)
expect_exit "accepts a weekly Rust analysis on PRs to Develop" 0 "$valid"

daily=$(write_workflow daily <<'YAML'
name: CodeQL
on:
  pull_request:
    branches:
      - Develop
  schedule:
    - cron: "30 4 * * *"
permissions:
  contents: read
jobs:
  analyse:
    permissions:
      security-events: write
    steps:
      - uses: github/codeql-action/init@c16c0f3f2812ec4bb3750a5ed64873fe2ce0fbef  # v4
        with:
          languages: rust
      - uses: github/codeql-action/analyze@c16c0f3f2812ec4bb3750a5ed64873fe2ce0fbef  # v4
YAML
)
expect_exit "accepts a cadence more frequent than weekly" 0 "$daily"

matrix=$(write_workflow matrix <<'YAML'
name: CodeQL
on:
  pull_request:
    branches:
      - Develop
  schedule:
    - cron: "30 4 * * 1"
permissions:
  contents: read
jobs:
  analyse:
    permissions:
      security-events: write
    strategy:
      matrix:
        include:
          - language: rust
            build-mode: none
    steps:
      - uses: github/codeql-action/init@c16c0f3f2812ec4bb3750a5ed64873fe2ce0fbef  # v4
        with:
          languages: ${{ matrix.language }}
      - uses: github/codeql-action/analyze@c16c0f3f2812ec4bb3750a5ed64873fe2ce0fbef  # v4
YAML
)
expect_exit "accepts Rust declared through a build matrix" 0 "$matrix"

expect_exit "reports a missing workflow with exit 2" 2 \
  "$WORK_DIR/does-not-exist.yml" "not found"

no_pr=$(write_workflow no-pr <<'YAML'
name: CodeQL
on:
  schedule:
    - cron: "30 4 * * 1"
permissions:
  contents: read
jobs:
  analyse:
    permissions:
      security-events: write
    steps:
      - uses: github/codeql-action/init@c16c0f3f2812ec4bb3750a5ed64873fe2ce0fbef  # v4
        with:
          languages: rust
      - uses: github/codeql-action/analyze@c16c0f3f2812ec4bb3750a5ed64873fe2ce0fbef  # v4
YAML
)
expect_exit "rejects a workflow with no pull_request trigger" 1 "$no_pr" "pull_request"

wrong_branch=$(write_workflow wrong-branch <<'YAML'
name: CodeQL
on:
  pull_request:
    branches:
      - main
  schedule:
    - cron: "30 4 * * 1"
permissions:
  contents: read
jobs:
  analyse:
    permissions:
      security-events: write
    steps:
      - uses: github/codeql-action/init@c16c0f3f2812ec4bb3750a5ed64873fe2ce0fbef  # v4
        with:
          languages: rust
      - uses: github/codeql-action/analyze@c16c0f3f2812ec4bb3750a5ed64873fe2ce0fbef  # v4
YAML
)
expect_exit "rejects a pull_request trigger that skips the default branch" 1 \
  "$wrong_branch" "Develop"

no_schedule=$(write_workflow no-schedule <<'YAML'
name: CodeQL
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  analyse:
    permissions:
      security-events: write
    steps:
      - uses: github/codeql-action/init@c16c0f3f2812ec4bb3750a5ed64873fe2ce0fbef  # v4
        with:
          languages: rust
      - uses: github/codeql-action/analyze@c16c0f3f2812ec4bb3750a5ed64873fe2ce0fbef  # v4
YAML
)
expect_exit "rejects a workflow that only runs when a PR is open" 1 \
  "$no_schedule" "schedule"

monthly=$(write_workflow monthly <<'YAML'
name: CodeQL
on:
  pull_request:
    branches:
      - Develop
  schedule:
    - cron: "30 4 1 * *"
permissions:
  contents: read
jobs:
  analyse:
    permissions:
      security-events: write
    steps:
      - uses: github/codeql-action/init@c16c0f3f2812ec4bb3750a5ed64873fe2ce0fbef  # v4
        with:
          languages: rust
      - uses: github/codeql-action/analyze@c16c0f3f2812ec4bb3750a5ed64873fe2ce0fbef  # v4
YAML
)
expect_exit "rejects a monthly schedule as slower than weekly" 1 "$monthly" "weekly"

bad_cron=$(write_workflow bad-cron <<'YAML'
name: CodeQL
on:
  pull_request:
    branches:
      - Develop
  schedule:
    - cron: "30 4 *"
permissions:
  contents: read
jobs:
  analyse:
    permissions:
      security-events: write
    steps:
      - uses: github/codeql-action/init@c16c0f3f2812ec4bb3750a5ed64873fe2ce0fbef  # v4
        with:
          languages: rust
      - uses: github/codeql-action/analyze@c16c0f3f2812ec4bb3750a5ed64873fe2ce0fbef  # v4
YAML
)
expect_exit "rejects a cron expression that is not five fields" 1 "$bad_cron" "cron"

no_permission=$(write_workflow no-permission <<'YAML'
name: CodeQL
on:
  pull_request:
    branches:
      - Develop
  schedule:
    - cron: "30 4 * * 1"
permissions:
  contents: read
jobs:
  analyse:
    steps:
      - uses: github/codeql-action/init@c16c0f3f2812ec4bb3750a5ed64873fe2ce0fbef  # v4
        with:
          languages: rust
      - uses: github/codeql-action/analyze@c16c0f3f2812ec4bb3750a5ed64873fe2ce0fbef  # v4
YAML
)
expect_exit "rejects a workflow that cannot upload its results" 1 \
  "$no_permission" "security-events"

no_rust=$(write_workflow no-rust <<'YAML'
name: CodeQL
on:
  pull_request:
    branches:
      - Develop
  schedule:
    - cron: "30 4 * * 1"
permissions:
  contents: read
jobs:
  analyse:
    permissions:
      security-events: write
    steps:
      - uses: github/codeql-action/init@c16c0f3f2812ec4bb3750a5ed64873fe2ce0fbef  # v4
        with:
          languages: javascript
      - uses: github/codeql-action/analyze@c16c0f3f2812ec4bb3750a5ed64873fe2ce0fbef  # v4
YAML
)
expect_exit "rejects a workflow that never analyses Rust" 1 "$no_rust" "rust"

no_analyze=$(write_workflow no-analyze <<'YAML'
name: CodeQL
on:
  pull_request:
    branches:
      - Develop
  schedule:
    - cron: "30 4 * * 1"
permissions:
  contents: read
jobs:
  analyse:
    permissions:
      security-events: write
    steps:
      - uses: github/codeql-action/init@c16c0f3f2812ec4bb3750a5ed64873fe2ce0fbef  # v4
        with:
          languages: rust
YAML
)
expect_exit "rejects an init step with no matching analyze step" 1 \
  "$no_analyze" "analyze"

unpinned=$(write_workflow unpinned <<'YAML'
name: CodeQL
on:
  pull_request:
    branches:
      - Develop
  schedule:
    - cron: "30 4 * * 1"
permissions:
  contents: read
jobs:
  analyse:
    permissions:
      security-events: write
    steps:
      - uses: github/codeql-action/init@v4
        with:
          languages: rust
      - uses: github/codeql-action/analyze@c16c0f3f2812ec4bb3750a5ed64873fe2ce0fbef  # v4
YAML
)
expect_exit "rejects an action pinned to a movable tag" 1 "$unpinned" "github/codeql-action/init@v4"

echo "check-codeql-workflow tests: $PASSED passed, $FAILED failed"
if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
