#!/usr/bin/env bash
# Tests for scripts/check-actionlint-gate.sh (Issue #26).
#
# Each case writes a fixture workflow to a temporary directory, runs the real
# checker against it, and asserts the exit code (and, where it matters, that
# the failure names the rule that broke).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKER="$SCRIPT_DIR/check-actionlint-gate.sh"
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
name: CI
on:
  pull_request:
    branches:
      - Develop
jobs:
  workflow-lint:
    name: Workflow Lint
    runs-on: ubuntu-latest
    env:
      ACTIONLINT_VERSION: "1.7.12"
      ACTIONLINT_SHA256: "8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8"
    steps:
      - uses: actions/checkout@93cb6efe18208431cddfb8368fd83d5badbf9bfd  # v5
      - name: Install the linter
        run: |
          set -euo pipefail
          tarball="actionlint_${ACTIONLINT_VERSION}_linux_amd64.tar.gz"
          curl --fail --location --output "$tarball" \
            "https://github.com/rhysd/actionlint/releases/download/v${ACTIONLINT_VERSION}/${tarball}"
          echo "${ACTIONLINT_SHA256}  ${tarball}" | sha256sum --check --strict
          tar -xzf "$tarball" actionlint
      - name: Lint the workflows
        run: |
          set -euo pipefail
          ./actionlint -color
  ci-required:
    name: CI Required Checks
    needs: [workflow-lint]
    runs-on: ubuntu-latest
    steps:
      - name: Verify
        run: echo ok
YAML
)
expect_exit "accepts a checksum-verified, merge-gating actionlint job" 0 "$valid"

pinned_action=$(write_workflow pinned-action <<'YAML'
name: CI
on:
  pull_request:
    branches:
      - Develop
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@93cb6efe18208431cddfb8368fd83d5badbf9bfd  # v5
      - name: Lint the workflows
        uses: raven-actions/actionlint@3a24062651993d40fed1019b58ac6fbdfbf276cc  # v2
      - name: Strict shell
        run: |
          set -euo pipefail
          echo "linted"
  ci-required:
    needs: [lint]
    runs-on: ubuntu-latest
    steps:
      - run: echo ok
YAML
)
expect_exit "accepts a SHA-pinned actionlint action" 0 "$pinned_action"

expect_exit "reports a missing workflow with exit 2" 2 \
  "$WORK_DIR/does-not-exist.yml" "not found"

no_lint=$(write_workflow no-lint <<'YAML'
name: CI
on:
  pull_request:
    branches:
      - Develop
jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      # actionlint would be nice to have here one day
      - name: Run actionlint
        run: |
          set -euo pipefail
          cargo test
  ci-required:
    needs: [quality]
    runs-on: ubuntu-latest
    steps:
      - run: echo ok
YAML
)
expect_exit "rejects a workflow that only mentions actionlint in prose" 1 \
  "$no_lint" "no actionlint invocation"

suppressed=$(write_workflow suppressed <<'YAML'
name: CI
on:
  pull_request:
    branches:
      - Develop
jobs:
  workflow-lint:
    runs-on: ubuntu-latest
    steps:
      - name: Lint the workflows
        run: |
          set -euo pipefail
          actionlint -color || true
  ci-required:
    needs: [workflow-lint]
    runs-on: ubuntu-latest
    steps:
      - run: echo ok
YAML
)
expect_exit "rejects an actionlint invocation whose exit code is swallowed" 1 \
  "$suppressed" "exit code"

continue_on_error=$(write_workflow continue-on-error <<'YAML'
name: CI
on:
  pull_request:
    branches:
      - Develop
jobs:
  workflow-lint:
    runs-on: ubuntu-latest
    continue-on-error: true
    steps:
      - name: Lint the workflows
        run: |
          set -euo pipefail
          actionlint -color
  ci-required:
    needs: [workflow-lint]
    runs-on: ubuntu-latest
    steps:
      - run: echo ok
YAML
)
expect_exit "rejects a lint job marked continue-on-error" 1 \
  "$continue_on_error" "continue-on-error"

lax_shell=$(write_workflow lax-shell <<'YAML'
name: CI
on:
  pull_request:
    branches:
      - Develop
jobs:
  workflow-lint:
    runs-on: ubuntu-latest
    steps:
      - name: Lint the workflows
        run: |
          actionlint -color
  ci-required:
    needs: [workflow-lint]
    runs-on: ubuntu-latest
    steps:
      - run: echo ok
YAML
)
expect_exit "rejects a lint job without strict bash" 1 "$lax_shell" "set -euo pipefail"

orphan=$(write_workflow orphan <<'YAML'
name: CI
on:
  pull_request:
    branches:
      - Develop
jobs:
  workflow-lint:
    runs-on: ubuntu-latest
    steps:
      - name: Lint the workflows
        run: |
          set -euo pipefail
          actionlint -color
  ci-required:
    needs: [quality]
    runs-on: ubuntu-latest
    steps:
      - run: echo ok
YAML
)
expect_exit "rejects a lint job no other job waits on" 1 "$orphan" "needs"

unpinned=$(write_workflow unpinned <<'YAML'
name: CI
on:
  pull_request:
    branches:
      - Develop
jobs:
  workflow-lint:
    runs-on: ubuntu-latest
    steps:
      - name: Lint the workflows
        uses: raven-actions/actionlint@v2
      - name: Strict shell
        run: |
          set -euo pipefail
          echo "linted"
  ci-required:
    needs: [workflow-lint]
    runs-on: ubuntu-latest
    steps:
      - run: echo ok
YAML
)
expect_exit "rejects an actionlint action pinned to a movable tag" 1 \
  "$unpinned" "raven-actions/actionlint@v2"

no_checksum=$(write_workflow no-checksum <<'YAML'
name: CI
on:
  pull_request:
    branches:
      - Develop
jobs:
  workflow-lint:
    runs-on: ubuntu-latest
    steps:
      - name: Lint the workflows
        run: |
          set -euo pipefail
          curl --fail --location --output actionlint.tar.gz \
            "https://github.com/rhysd/actionlint/releases/download/v1.7.12/actionlint_1.7.12_linux_amd64.tar.gz"
          tar -xzf actionlint.tar.gz actionlint
          ./actionlint -color
  ci-required:
    needs: [workflow-lint]
    runs-on: ubuntu-latest
    steps:
      - run: echo ok
YAML
)
expect_exit "rejects an unverified linter download" 1 "$no_checksum" "checksum"

floating_download=$(write_workflow floating-download <<'YAML'
name: CI
on:
  pull_request:
    branches:
      - Develop
jobs:
  workflow-lint:
    runs-on: ubuntu-latest
    steps:
      - name: Lint the workflows
        run: |
          set -euo pipefail
          curl --fail --location --output actionlint.tar.gz \
            "https://github.com/rhysd/actionlint/releases/latest/download/actionlint.tar.gz"
          echo "deadbeef  actionlint.tar.gz" | sha256sum --check --strict
          tar -xzf actionlint.tar.gz actionlint
          ./actionlint -color
  ci-required:
    needs: [workflow-lint]
    runs-on: ubuntu-latest
    steps:
      - run: echo ok
YAML
)
expect_exit "rejects a download pinned to the floating latest release" 1 \
  "$floating_download" "latest"

echo "check-actionlint-gate tests: $PASSED passed, $FAILED failed"
if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
