#!/usr/bin/env bash
# Tests for scripts/check-markdown-lint-workflow.sh (Issue #44).
#
# Each case writes a fixture workflow to a temporary directory, runs the real
# checker against it, and asserts the exit code (and, where it matters, that
# the failure names the rule that broke).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKER="$SCRIPT_DIR/check-markdown-lint-workflow.sh"
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
name: Markdown Lint
on:
  pull_request:
    branches:
      - Develop
      - "milestone/**"
permissions:
  contents: read
jobs:
  markdownlint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - uses: actions/setup-node@48b55a011bda9f5d6aeb4c2d9c7362e8dae4041e  # v6.4.0
        with:
          node-version: "lts/*"
      - name: Lint Markdown
        run: |
          set -euo pipefail
          npm install --global markdownlint-cli2@0.23.2
          markdownlint-cli2
YAML
)
expect_exit "accepts a pinned global install running markdownlint-cli2" 0 "$valid"

# A different shape entirely — the upstream action, a wildcard base-branch list
# — must also pass, so the checker enforces the policy rather than one
# hard-coded file.
action_variant=$(write_workflow action-variant <<'YAML'
name: Markdown Lint
on:
  pull_request:
    branches: ["*"]
permissions:
  contents: read
jobs:
  markdownlint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - uses: DavidAnson/markdownlint-cli2-action@21c1be1b93ad9ed58fa840aacc3f279cde2a72ff  # v24.2.0
YAML
)
expect_exit "accepts the upstream markdownlint-cli2 action" 0 "$action_variant"

npx_variant=$(write_workflow npx-variant <<'YAML'
name: Markdown Lint
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  markdownlint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - name: Lint Markdown
        run: |
          set -euo pipefail
          npx --yes markdownlint-cli2@0.23.2
YAML
)
expect_exit "accepts a version-pinned npx invocation" 0 "$npx_variant"

expect_exit "reports a missing workflow with exit 2" 2 \
  "$WORK_DIR/does-not-exist.yml" "not found"

no_pr=$(write_workflow no-pr <<'YAML'
name: Markdown Lint
on:
  push:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  markdownlint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - run: |
          set -euo pipefail
          npm install --global markdownlint-cli2@0.23.2
          markdownlint-cli2
YAML
)
expect_exit "rejects a workflow with no pull_request trigger" 1 "$no_pr" "pull_request"

wrong_branch=$(write_workflow wrong-branch <<'YAML'
name: Markdown Lint
on:
  pull_request:
    branches:
      - main
permissions:
  contents: read
jobs:
  markdownlint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - run: |
          set -euo pipefail
          npm install --global markdownlint-cli2@0.23.2
          markdownlint-cli2
YAML
)
expect_exit "rejects a pull_request trigger that skips the default branch" 1 \
  "$wrong_branch" "Develop"

no_lint=$(write_workflow no-lint <<'YAML'
name: Markdown Lint
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  markdownlint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - name: Markdown Lint
        run: echo "markdown linting would go here"
YAML
)
expect_exit "rejects a workflow that never runs a lint" 1 "$no_lint" "no markdownlint-cli2 lint"

install_only=$(write_workflow install-only <<'YAML'
name: Markdown Lint
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  markdownlint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - name: Install markdownlint-cli2
        run: |
          set -euo pipefail
          npm install --global markdownlint-cli2@0.23.2
YAML
)
expect_exit "rejects a workflow that installs the linter but never runs it" 1 \
  "$install_only" "no markdownlint-cli2 lint"

fix_mode=$(write_workflow fix-mode <<'YAML'
name: Markdown Lint
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  markdownlint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - run: |
          set -euo pipefail
          npm install --global markdownlint-cli2@0.23.2
          markdownlint-cli2 --fix
YAML
)
expect_exit "rejects '--fix', which rewrites the runner's copy and exits clean" 1 \
  "$fix_mode" "--fix"

action_fix_mode=$(write_workflow action-fix-mode <<'YAML'
name: Markdown Lint
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  markdownlint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - uses: DavidAnson/markdownlint-cli2-action@21c1be1b93ad9ed58fa840aacc3f279cde2a72ff  # v24.2.0
        with:
          fix: true
YAML
)
expect_exit "rejects the upstream action's 'fix: true' input" 1 \
  "$action_fix_mode" "fix"

swallowed=$(write_workflow swallowed <<'YAML'
name: Markdown Lint
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  markdownlint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - run: |
          set -euo pipefail
          npm install --global markdownlint-cli2@0.23.2
          markdownlint-cli2 || true
YAML
)
expect_exit "rejects a lint whose exit code is swallowed by '|| true'" 1 \
  "$swallowed" "exit code"

continue_on_error=$(write_workflow continue-on-error <<'YAML'
name: Markdown Lint
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  markdownlint:
    runs-on: ubuntu-latest
    continue-on-error: true
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - run: |
          set -euo pipefail
          npm install --global markdownlint-cli2@0.23.2
          markdownlint-cli2
YAML
)
expect_exit "rejects continue-on-error, which lets a violation merge" 1 \
  "$continue_on_error" "continue-on-error"

lax_bash=$(write_workflow lax-bash <<'YAML'
name: Markdown Lint
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  markdownlint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - run: |
          npm install --global markdownlint-cli2@0.23.2
          markdownlint-cli2
YAML
)
expect_exit "rejects a lint step that does not run under strict bash" 1 \
  "$lax_bash" "set -euo pipefail"

unpinned_action=$(write_workflow unpinned-action <<'YAML'
name: Markdown Lint
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  markdownlint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - run: |
          set -euo pipefail
          npm install --global markdownlint-cli2@0.23.2
          markdownlint-cli2
YAML
)
expect_exit "rejects an action pinned to a movable tag" 1 "$unpinned_action" \
  "actions/checkout@v6"

unpinned_install=$(write_workflow unpinned-install <<'YAML'
name: Markdown Lint
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  markdownlint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - run: |
          set -euo pipefail
          npm install --global markdownlint-cli2
          markdownlint-cli2
YAML
)
expect_exit "rejects an unpinned 'npm install markdownlint-cli2'" 1 \
  "$unpinned_install" "version-pinned"

unpinned_npx=$(write_workflow unpinned-npx <<'YAML'
name: Markdown Lint
on:
  pull_request:
    branches:
      - Develop
permissions:
  contents: read
jobs:
  markdownlint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2
      - run: |
          set -euo pipefail
          npx --yes markdownlint-cli2
YAML
)
expect_exit "rejects an unpinned 'npx markdownlint-cli2'" 1 \
  "$unpinned_npx" "version-pinned"

# The repository's own workflow must satisfy every rule above.
REPO_WORKFLOW="$(cd "$SCRIPT_DIR/.." && pwd)/.github/workflows/markdown-lint.yml"
expect_exit "the committed markdown-lint workflow satisfies the policy" 0 \
  "$REPO_WORKFLOW"

echo "check-markdown-lint-workflow tests: $PASSED passed, $FAILED failed"
if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
