#!/usr/bin/env bash
# Tests for scripts/check-gitleaks-workflow.sh (Issue #42).
#
# Each case writes a fixture workflow to a temporary directory, runs the real
# checker against it, and asserts the exit code (and, where it matters, that
# the failure names the rule that broke).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKER="$SCRIPT_DIR/check-gitleaks-workflow.sh"
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
name: Gitleaks
on:
  pull_request:
    branches:
      - Develop
      - "milestone/**"
permissions:
  contents: read
jobs:
  gitleaks:
    runs-on: ubuntu-latest
    env:
      GITLEAKS_LICENSE: ${{ secrets.GITLEAKS_LICENSE }}
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
        with:
          fetch-depth: 0
      - name: Fetch base branch
        run: git fetch origin "$BASE_REF:$BASE_REF" || true
      - name: Gitleaks (licensed action)
        if: env.GITLEAKS_LICENSE != ''
        uses: gitleaks/gitleaks-action@e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e  # v3.0.0
      - name: Gitleaks (open-source CLI fallback)
        if: env.GITLEAKS_LICENSE == ''
        env:
          GITLEAKS_VERSION: "8.30.1"
          GITLEAKS_SHA256: "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb"
        run: |
          set -euo pipefail
          archive="gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz"
          curl -sSfL "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/${archive}" -o "$archive"
          echo "${GITLEAKS_SHA256}  ${archive}" | sha256sum -c -
          tar -xzf "$archive" gitleaks
          ./gitleaks git --redact --no-banner --exit-code 1 --log-opts="${BASE_SHA}..${HEAD_SHA}" .
YAML
)
expect_exit "accepts the licensed action plus a licence-less CLI fallback" 0 "$valid"

wildcard=$(write_workflow wildcard <<'YAML'
name: Gitleaks
on:
  pull_request:
    branches: ["*"]
permissions:
  contents: read
jobs:
  gitleaks:
    runs-on: ubuntu-latest
    env:
      GITLEAKS_LICENSE: ${{ secrets.GITLEAKS_LICENSE }}
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
        with:
          fetch-depth: 0
      - if: env.GITLEAKS_LICENSE != ''
        uses: gitleaks/gitleaks-action@e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e  # v3.0.0
      - if: env.GITLEAKS_LICENSE == ''
        run: |
          set -euo pipefail
          curl -sSfL "https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz" -o g.tar.gz
          echo "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb  g.tar.gz" | sha256sum -c -
          tar -xzf g.tar.gz gitleaks
          ./gitleaks git --redact --exit-code 1 .
YAML
)
expect_exit "accepts a branch wildcard covering every base branch" 0 "$wildcard"

expect_exit "reports a missing workflow with exit 2" 2 \
  "$WORK_DIR/does-not-exist.yml" "not found"

no_pr=$(write_workflow no-pr <<'YAML'
name: Gitleaks
on:
  push:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  gitleaks:
    runs-on: ubuntu-latest
    env:
      GITLEAKS_LICENSE: ${{ secrets.GITLEAKS_LICENSE }}
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
        with:
          fetch-depth: 0
      - if: env.GITLEAKS_LICENSE != ''
        uses: gitleaks/gitleaks-action@e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e  # v3.0.0
      - if: env.GITLEAKS_LICENSE == ''
        run: |
          set -euo pipefail
          curl -sSfL "https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz" -o g.tar.gz
          echo "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb  g.tar.gz" | sha256sum -c -
          tar -xzf g.tar.gz gitleaks
          ./gitleaks git --exit-code 1 .
YAML
)
expect_exit "rejects a workflow with no pull_request trigger" 1 "$no_pr" "pull_request"

wrong_branch=$(write_workflow wrong-branch <<'YAML'
name: Gitleaks
on:
  pull_request:
    branches:
      - main
permissions:
  contents: read
jobs:
  gitleaks:
    runs-on: ubuntu-latest
    env:
      GITLEAKS_LICENSE: ${{ secrets.GITLEAKS_LICENSE }}
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
        with:
          fetch-depth: 0
      - if: env.GITLEAKS_LICENSE != ''
        uses: gitleaks/gitleaks-action@e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e  # v3.0.0
      - if: env.GITLEAKS_LICENSE == ''
        run: |
          set -euo pipefail
          curl -sSfL "https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz" -o g.tar.gz
          echo "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb  g.tar.gz" | sha256sum -c -
          tar -xzf g.tar.gz gitleaks
          ./gitleaks git --exit-code 1 .
YAML
)
expect_exit "rejects a pull_request trigger that skips the default branch" 1 \
  "$wrong_branch" "Develop"

no_scan=$(write_workflow no-scan <<'YAML'
name: Gitleaks
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  gitleaks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
        with:
          fetch-depth: 0
      - name: Gitleaks
        run: echo "scanning would go here"
YAML
)
expect_exit "rejects a workflow that never runs a scan" 1 "$no_scan" "no gitleaks scan"

no_fallback=$(write_workflow no-fallback <<'YAML'
name: Gitleaks
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  gitleaks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
        with:
          fetch-depth: 0
      - uses: gitleaks/gitleaks-action@e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e  # v3.0.0
YAML
)
expect_exit "rejects a licensed-only scan that skips licence-less pull requests" 1 \
  "$no_fallback" "GITLEAKS_LICENSE"

swallowed=$(write_workflow swallowed <<'YAML'
name: Gitleaks
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  gitleaks:
    runs-on: ubuntu-latest
    env:
      GITLEAKS_LICENSE: ${{ secrets.GITLEAKS_LICENSE }}
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
        with:
          fetch-depth: 0
      - if: env.GITLEAKS_LICENSE != ''
        uses: gitleaks/gitleaks-action@e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e  # v3.0.0
      - if: env.GITLEAKS_LICENSE == ''
        run: |
          set -euo pipefail
          curl -sSfL "https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz" -o g.tar.gz
          echo "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb  g.tar.gz" | sha256sum -c -
          tar -xzf g.tar.gz gitleaks
          ./gitleaks git --exit-code 1 . || true
YAML
)
expect_exit "rejects a scan whose exit code is swallowed by '|| true'" 1 \
  "$swallowed" "exit code"

exit_zero=$(write_workflow exit-zero <<'YAML'
name: Gitleaks
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  gitleaks:
    runs-on: ubuntu-latest
    env:
      GITLEAKS_LICENSE: ${{ secrets.GITLEAKS_LICENSE }}
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
        with:
          fetch-depth: 0
      - if: env.GITLEAKS_LICENSE != ''
        uses: gitleaks/gitleaks-action@e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e  # v3.0.0
      - if: env.GITLEAKS_LICENSE == ''
        run: |
          set -euo pipefail
          curl -sSfL "https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz" -o g.tar.gz
          echo "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb  g.tar.gz" | sha256sum -c -
          tar -xzf g.tar.gz gitleaks
          ./gitleaks git --exit-code 0 .
YAML
)
expect_exit "rejects '--exit-code 0', which reports a leak as a pass" 1 \
  "$exit_zero" "exit-code 0"

continue_on_error=$(write_workflow continue-on-error <<'YAML'
name: Gitleaks
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  gitleaks:
    runs-on: ubuntu-latest
    continue-on-error: true
    env:
      GITLEAKS_LICENSE: ${{ secrets.GITLEAKS_LICENSE }}
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
        with:
          fetch-depth: 0
      - if: env.GITLEAKS_LICENSE != ''
        uses: gitleaks/gitleaks-action@e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e  # v3.0.0
      - if: env.GITLEAKS_LICENSE == ''
        run: |
          set -euo pipefail
          curl -sSfL "https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz" -o g.tar.gz
          echo "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb  g.tar.gz" | sha256sum -c -
          tar -xzf g.tar.gz gitleaks
          ./gitleaks git --exit-code 1 .
YAML
)
expect_exit "rejects continue-on-error, which lets a leak merge" 1 \
  "$continue_on_error" "continue-on-error"

no_checksum=$(write_workflow no-checksum <<'YAML'
name: Gitleaks
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  gitleaks:
    runs-on: ubuntu-latest
    env:
      GITLEAKS_LICENSE: ${{ secrets.GITLEAKS_LICENSE }}
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
        with:
          fetch-depth: 0
      - if: env.GITLEAKS_LICENSE != ''
        uses: gitleaks/gitleaks-action@e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e  # v3.0.0
      - if: env.GITLEAKS_LICENSE == ''
        run: |
          set -euo pipefail
          curl -sSfL "https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz" -o g.tar.gz
          tar -xzf g.tar.gz gitleaks
          ./gitleaks git --exit-code 1 .
YAML
)
expect_exit "rejects a download with no sha256 verification" 1 \
  "$no_checksum" "checksum"

latest_release=$(write_workflow latest-release <<'YAML'
name: Gitleaks
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  gitleaks:
    runs-on: ubuntu-latest
    env:
      GITLEAKS_LICENSE: ${{ secrets.GITLEAKS_LICENSE }}
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
        with:
          fetch-depth: 0
      - if: env.GITLEAKS_LICENSE != ''
        uses: gitleaks/gitleaks-action@e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e  # v3.0.0
      - if: env.GITLEAKS_LICENSE == ''
        run: |
          set -euo pipefail
          curl -sSfL "https://github.com/gitleaks/gitleaks/releases/latest/download/gitleaks_linux_x64.tar.gz" -o g.tar.gz
          echo "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb  g.tar.gz" | sha256sum -c -
          tar -xzf g.tar.gz gitleaks
          ./gitleaks git --exit-code 1 .
YAML
)
expect_exit "rejects a floating 'releases/latest' download" 1 \
  "$latest_release" "latest release"

lax_bash=$(write_workflow lax-bash <<'YAML'
name: Gitleaks
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  gitleaks:
    runs-on: ubuntu-latest
    env:
      GITLEAKS_LICENSE: ${{ secrets.GITLEAKS_LICENSE }}
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
        with:
          fetch-depth: 0
      - if: env.GITLEAKS_LICENSE != ''
        uses: gitleaks/gitleaks-action@e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e  # v3.0.0
      - if: env.GITLEAKS_LICENSE == ''
        run: |
          curl -sSfL "https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz" -o g.tar.gz
          echo "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb  g.tar.gz" | sha256sum -c -
          tar -xzf g.tar.gz gitleaks
          ./gitleaks git --exit-code 1 .
YAML
)
expect_exit "rejects a fallback that does not run under strict bash" 1 \
  "$lax_bash" "set -euo pipefail"

shallow=$(write_workflow shallow <<'YAML'
name: Gitleaks
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  gitleaks:
    runs-on: ubuntu-latest
    env:
      GITLEAKS_LICENSE: ${{ secrets.GITLEAKS_LICENSE }}
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - if: env.GITLEAKS_LICENSE != ''
        uses: gitleaks/gitleaks-action@e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e  # v3.0.0
      - if: env.GITLEAKS_LICENSE == ''
        run: |
          set -euo pipefail
          curl -sSfL "https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz" -o g.tar.gz
          echo "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb  g.tar.gz" | sha256sum -c -
          tar -xzf g.tar.gz gitleaks
          ./gitleaks git --exit-code 1 .
YAML
)
expect_exit "rejects a shallow checkout that cannot resolve the commit range" 1 \
  "$shallow" "fetch-depth: 0"

unpinned=$(write_workflow unpinned <<'YAML'
name: Gitleaks
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  gitleaks:
    runs-on: ubuntu-latest
    env:
      GITLEAKS_LICENSE: ${{ secrets.GITLEAKS_LICENSE }}
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
        with:
          fetch-depth: 0
      - if: env.GITLEAKS_LICENSE != ''
        uses: gitleaks/gitleaks-action@v2
      - if: env.GITLEAKS_LICENSE == ''
        run: |
          set -euo pipefail
          curl -sSfL "https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz" -o g.tar.gz
          echo "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb  g.tar.gz" | sha256sum -c -
          tar -xzf g.tar.gz gitleaks
          ./gitleaks git --exit-code 1 .
YAML
)
expect_exit "rejects an action pinned to a movable tag" 1 "$unpinned" \
  "gitleaks/gitleaks-action@v2"

echo "check-gitleaks-workflow tests: $PASSED passed, $FAILED failed"
if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
