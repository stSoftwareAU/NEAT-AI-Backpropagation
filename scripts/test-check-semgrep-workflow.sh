#!/usr/bin/env bash
# Tests for scripts/check-semgrep-workflow.sh (Issue #43).
#
# Each case writes a fixture workflow to a temporary directory, runs the real
# checker against it, and asserts the exit code (and, where it matters, that
# the failure names the rule that broke).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKER="$SCRIPT_DIR/check-semgrep-workflow.sh"
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
name: Semgrep
on:
  pull_request:
    branches:
      - Develop
      - "milestone/**"
permissions:
  contents: read
jobs:
  semgrep:
    runs-on: ubuntu-latest
    container:
      image: semgrep/semgrep@sha256:67319956da3dcb58baf5b322899c15458e3963e7018a86aeeb5cd224e69cb77a
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
        with:
          fetch-depth: 0
      - name: Semgrep scan
        env:
          SEMGREP_APP_TOKEN: ${{ secrets.SEMGREP_APP_TOKEN }}
        run: |
          set -euo pipefail
          semgrep ci --config p/default --no-suppress-errors
YAML
)
expect_exit "accepts a digest-pinned container running 'semgrep ci'" 0 "$valid"

# A different shape entirely — pip-installed CLI, `semgrep scan`, wildcard base
# branches — must also pass, so the checker enforces the policy rather than one
# hard-coded file.
pip_variant=$(write_workflow pip-variant <<'YAML'
name: Semgrep
on:
  pull_request:
    branches: ["*"]
permissions:
  contents: read
jobs:
  semgrep:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - name: Semgrep scan
        run: |
          set -euo pipefail
          pip install --user semgrep==1.173.0
          semgrep scan --config p/default --error
YAML
)
expect_exit "accepts a version-pinned pip install running 'semgrep scan'" 0 "$pip_variant"

expect_exit "reports a missing workflow with exit 2" 2 \
  "$WORK_DIR/does-not-exist.yml" "not found"

no_pr=$(write_workflow no-pr <<'YAML'
name: Semgrep
on:
  push:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  semgrep:
    runs-on: ubuntu-latest
    container:
      image: semgrep/semgrep@sha256:67319956da3dcb58baf5b322899c15458e3963e7018a86aeeb5cd224e69cb77a
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - run: |
          set -euo pipefail
          semgrep ci --config p/default --no-suppress-errors
YAML
)
expect_exit "rejects a workflow with no pull_request trigger" 1 "$no_pr" "pull_request"

wrong_branch=$(write_workflow wrong-branch <<'YAML'
name: Semgrep
on:
  pull_request:
    branches:
      - main
permissions:
  contents: read
jobs:
  semgrep:
    runs-on: ubuntu-latest
    container:
      image: semgrep/semgrep@sha256:67319956da3dcb58baf5b322899c15458e3963e7018a86aeeb5cd224e69cb77a
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - run: |
          set -euo pipefail
          semgrep ci --config p/default --no-suppress-errors
YAML
)
expect_exit "rejects a pull_request trigger that skips the default branch" 1 \
  "$wrong_branch" "Develop"

no_scan=$(write_workflow no-scan <<'YAML'
name: Semgrep
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  semgrep:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - name: Semgrep
        run: echo "static analysis would go here"
YAML
)
expect_exit "rejects a workflow that never runs a scan" 1 "$no_scan" "no semgrep scan"

no_config=$(write_workflow no-config <<'YAML'
name: Semgrep
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  semgrep:
    runs-on: ubuntu-latest
    container:
      image: semgrep/semgrep@sha256:67319956da3dcb58baf5b322899c15458e3963e7018a86aeeb5cd224e69cb77a
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - run: |
          set -euo pipefail
          semgrep ci --no-suppress-errors
YAML
)
expect_exit "rejects a scan with no ruleset configured" 1 "$no_config" "ruleset"

swallowed=$(write_workflow swallowed <<'YAML'
name: Semgrep
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  semgrep:
    runs-on: ubuntu-latest
    container:
      image: semgrep/semgrep@sha256:67319956da3dcb58baf5b322899c15458e3963e7018a86aeeb5cd224e69cb77a
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - run: |
          set -euo pipefail
          semgrep ci --config p/default --no-suppress-errors || true
YAML
)
expect_exit "rejects a scan whose exit code is swallowed by '|| true'" 1 \
  "$swallowed" "exit code"

continue_on_error=$(write_workflow continue-on-error <<'YAML'
name: Semgrep
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  semgrep:
    runs-on: ubuntu-latest
    continue-on-error: true
    container:
      image: semgrep/semgrep@sha256:67319956da3dcb58baf5b322899c15458e3963e7018a86aeeb5cd224e69cb77a
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - run: |
          set -euo pipefail
          semgrep ci --config p/default --no-suppress-errors
YAML
)
expect_exit "rejects continue-on-error, which lets a finding merge" 1 \
  "$continue_on_error" "continue-on-error"

suppressed=$(write_workflow suppressed <<'YAML'
name: Semgrep
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  semgrep:
    runs-on: ubuntu-latest
    container:
      image: semgrep/semgrep@sha256:67319956da3dcb58baf5b322899c15458e3963e7018a86aeeb5cd224e69cb77a
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - run: |
          set -euo pipefail
          semgrep ci --config p/default --suppress-errors
YAML
)
expect_exit "rejects an explicit '--suppress-errors', which hides a crashed scan" 1 \
  "$suppressed" "suppress-errors"

default_suppress=$(write_workflow default-suppress <<'YAML'
name: Semgrep
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  semgrep:
    runs-on: ubuntu-latest
    container:
      image: semgrep/semgrep@sha256:67319956da3dcb58baf5b322899c15458e3963e7018a86aeeb5cd224e69cb77a
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - run: |
          set -euo pipefail
          semgrep ci --config p/default
YAML
)
expect_exit "rejects 'semgrep ci' without --no-suppress-errors (its default hides errors)" 1 \
  "$default_suppress" "--no-suppress-errors"

lax_bash=$(write_workflow lax-bash <<'YAML'
name: Semgrep
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  semgrep:
    runs-on: ubuntu-latest
    container:
      image: semgrep/semgrep@sha256:67319956da3dcb58baf5b322899c15458e3963e7018a86aeeb5cd224e69cb77a
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - run: |
          semgrep ci --config p/default --no-suppress-errors
YAML
)
expect_exit "rejects a scan step that does not run under strict bash" 1 \
  "$lax_bash" "set -euo pipefail"

unpinned_action=$(write_workflow unpinned-action <<'YAML'
name: Semgrep
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  semgrep:
    runs-on: ubuntu-latest
    container:
      image: semgrep/semgrep@sha256:67319956da3dcb58baf5b322899c15458e3963e7018a86aeeb5cd224e69cb77a
    steps:
      - uses: actions/checkout@v6
      - run: |
          set -euo pipefail
          semgrep ci --config p/default --no-suppress-errors
YAML
)
expect_exit "rejects an action pinned to a movable tag" 1 "$unpinned_action" \
  "actions/checkout@v6"

floating_image=$(write_workflow floating-image <<'YAML'
name: Semgrep
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  semgrep:
    runs-on: ubuntu-latest
    container:
      image: semgrep/semgrep:latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - run: |
          set -euo pipefail
          semgrep ci --config p/default --no-suppress-errors
YAML
)
expect_exit "rejects a container image pinned to a movable tag" 1 \
  "$floating_image" "digest"

unpinned_pip=$(write_workflow unpinned-pip <<'YAML'
name: Semgrep
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  semgrep:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - run: |
          set -euo pipefail
          pip install --user semgrep
          semgrep scan --config p/default --error
YAML
)
expect_exit "rejects an unpinned 'pip install semgrep'" 1 "$unpinned_pip" \
  "pip install"

echo "check-semgrep-workflow tests: $PASSED passed, $FAILED failed"
if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
