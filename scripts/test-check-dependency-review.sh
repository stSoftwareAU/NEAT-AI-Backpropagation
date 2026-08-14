#!/usr/bin/env bash
# Tests for scripts/check-dependency-review.sh (Issue #28).
#
# Each case writes a fixture workflow directory, runs the real checker against
# it, and asserts the exit code (and, where it matters, that the failure names
# the rule that broke). The final case runs the checker against this
# repository's own committed workflows.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKER="$SCRIPT_DIR/check-dependency-review.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

PASSED=0
FAILED=0

if [[ ! -x "$CHECKER" ]]; then
  echo "FAIL: checker not found or not executable: $CHECKER" >&2
  exit 2
fi

DEPENDENCY_REVIEW_SHA="a1d282b36b6f3519aa1f3fc636f609c47dddb294"

# fixture NAME → creates a fixture directory holding a compliant security.yml
# and ci.yml pair, and echoes its path. Individual cases then overwrite the one
# file they are breaking.
fixture() {
  local dir="$WORK_DIR/$1"
  mkdir -p "$dir"
  cat >"$dir/security.yml" <<YAML
name: Security Reusable Workflow
on:
  workflow_call:
    inputs:
      include-dependency-review:
        description: "Include dependency review"
        required: false
        type: boolean
        default: true
jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - name: Dependency review
        if: \${{ inputs.include-dependency-review && github.event_name == 'pull_request' }}
        uses: actions/dependency-review-action@${DEPENDENCY_REVIEW_SHA}  # v5.0.0
YAML
  cat >"$dir/ci.yml" <<'YAML'
name: CI
on:
  pull_request:
    branches:
      - Develop
jobs:
  security:
    uses: ./.github/workflows/security.yml
YAML
  printf '%s' "$dir"
}

# expect_exit DESCRIPTION EXPECTED_CODE WORKFLOW_DIR [EXPECTED_OUTPUT_SUBSTRING]
expect_exit() {
  local description="$1" expected="$2" workflow_dir="$3" needle="${4:-}"
  local output status=0
  output="$("$CHECKER" "$workflow_dir" 2>&1)" || status=$?
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

valid=$(fixture valid)
expect_exit "accepts a caller that inherits the enabled default" 0 "$valid"

explicit=$(fixture explicit)
cat >"$explicit/ci.yml" <<'YAML'
name: CI
on:
  pull_request:
    branches:
      - Develop
jobs:
  security:
    uses: ./.github/workflows/security.yml
    with:
      include-dependency-review: true
YAML
expect_exit "accepts a caller that opts in explicitly" 0 "$explicit"

expect_exit "reports a missing workflow directory with exit 2" 2 \
  "$WORK_DIR/does-not-exist" "not found"

no_security=$(fixture no-security)
rm "$no_security/security.yml"
expect_exit "reports a missing security workflow with exit 2" 2 \
  "$no_security" "security.yml"

disabled=$(fixture disabled)
cat >"$disabled/ci.yml" <<'YAML'
name: CI
on:
  pull_request:
    branches:
      - Develop
jobs:
  security:
    uses: ./.github/workflows/security.yml
    with:
      include-dependency-review: false
YAML
expect_exit "rejects a caller that switches dependency review off" 1 \
  "$disabled" "switched off"

quoted_disabled=$(fixture quoted-disabled)
cat >"$quoted_disabled/ci.yml" <<'YAML'
name: CI
on:
  pull_request:
    branches:
      - Develop
jobs:
  security:
    uses: ./.github/workflows/security.yml
    with:
      include-dependency-review: "false"
YAML
expect_exit "rejects a quoted false as readily as a bare one" 1 \
  "$quoted_disabled" "switched off"

default_false=$(fixture default-false)
cat >"$default_false/security.yml" <<YAML
name: Security Reusable Workflow
on:
  workflow_call:
    inputs:
      include-dependency-review:
        description: "Include dependency review"
        required: false
        type: boolean
        default: false
jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - name: Dependency review
        if: \${{ inputs.include-dependency-review }}
        uses: actions/dependency-review-action@${DEPENDENCY_REVIEW_SHA}  # v5.0.0
YAML
expect_exit "rejects an input that defaults to off" 1 "$default_false" "defaults to 'false'"

no_default=$(fixture no-default)
cat >"$no_default/security.yml" <<YAML
name: Security Reusable Workflow
on:
  workflow_call:
    inputs:
      include-dependency-review:
        description: "Include dependency review"
        required: false
        type: boolean
jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - name: Dependency review
        if: \${{ inputs.include-dependency-review }}
        uses: actions/dependency-review-action@${DEPENDENCY_REVIEW_SHA}  # v5.0.0
YAML
expect_exit "rejects an input with no default at all" 1 "$no_default" "no default"

no_step=$(fixture no-step)
cat >"$no_step/security.yml" <<'YAML'
name: Security Reusable Workflow
on:
  workflow_call:
    inputs:
      include-dependency-review:
        description: "Include dependency review"
        required: false
        type: boolean
        default: true
jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - name: Cargo Security Audit
        uses: rustsec/audit-check@858dc40f52ca2b8570b7a997c1c4e35c6fc9a432  # v2
YAML
expect_exit "rejects a security workflow with no dependency-review step" 1 \
  "$no_step" "no actions/dependency-review-action step"

unpinned=$(fixture unpinned)
cat >"$unpinned/security.yml" <<'YAML'
name: Security Reusable Workflow
on:
  workflow_call:
    inputs:
      include-dependency-review:
        description: "Include dependency review"
        required: false
        type: boolean
        default: true
jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - name: Dependency review
        uses: actions/dependency-review-action@v5
YAML
expect_exit "rejects a step pinned to a movable tag" 1 \
  "$unpinned" "actions/dependency-review-action@v5"

no_caller=$(fixture no-caller)
rm "$no_caller/ci.yml"
expect_exit "rejects a reusable workflow nothing calls" 1 \
  "$no_caller" "no workflow calls security.yml"

schedule_only=$(fixture schedule-only)
cat >"$schedule_only/ci.yml" <<'YAML'
name: CI
on:
  schedule:
    - cron: "30 4 * * 1"
jobs:
  security:
    uses: ./.github/workflows/security.yml
YAML
expect_exit "rejects a caller that never runs on pull requests" 1 \
  "$schedule_only" "pull_request"

expect_exit "accepts this repository's committed workflows" 0 \
  "$REPO_ROOT/.github/workflows"

echo "check-dependency-review tests: $PASSED passed, $FAILED failed"
if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
