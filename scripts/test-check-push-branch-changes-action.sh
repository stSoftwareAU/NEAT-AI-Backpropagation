#!/usr/bin/env bash
# Tests for scripts/check-push-branch-changes-action.sh (issue #164).
#
# Every case writes a fixture action (and a fixture workflows directory), runs
# the real checker against it and asserts the exit code — and, where it
# matters, that the failure names the rule that broke. The fixture below
# satisfies every rule; each failing case starts from it and breaks exactly
# one, so a case that goes green for the wrong reason is visible.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKER="$SCRIPT_DIR/check-push-branch-changes-action.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

PASSED=0
FAILED=0

if [[ ! -x "$CHECKER" ]]; then
  echo "FAIL: checker not found or not executable: $CHECKER" >&2
  exit 2
fi

# A composite action satisfying every rule in the checker's header.
write_valid_action() {
  local path="$WORK_DIR/$1-action.yml"
  cat >"$path" <<'YAML'
name: Push branch changes
description: Commit the working tree and push it back onto the PR head branch.
inputs:
  branch:
    required: true
  commit-message:
    required: true
  fallback-token:
    required: true
runs:
  using: composite
  steps:
    - name: Mint repo-scoped push token
      id: push-token
      if: inputs.app-client-id != ''
      uses: actions/create-github-app-token@bcd2ba49218906704ab6c1aa796996da409d3eb1 # v3
      with:
        client-id: ${{ inputs.app-client-id }}
        private-key: ${{ inputs.app-private-key }}
        repositories: ${{ github.event.repository.name }}
        permission-contents: write
    - name: Commit and push
      shell: bash
      env:
        PR_HEAD_REF: ${{ inputs.branch }}
        COMMIT_MESSAGE: ${{ inputs.commit-message }}
        GH_PAT: ${{ steps.push-token.outputs.token || inputs.fallback-token }}
      run: |
        set -euo pipefail
        if [[ -z "$GH_PAT" ]]; then
          echo "::error::no push credential available"
          exit 1
        fi
        git -c core.hooksPath=/dev/null config user.name "github-actions[bot]"
        git -c core.hooksPath=/dev/null commit -am "$COMMIT_MESSAGE"
        if ! git ls-remote --exit-code --heads origin "$PR_HEAD_REF" >/dev/null 2>&1; then
          echo "::notice::head branch is gone; skipping push"
          exit 0
        fi
        git fetch origin "$PR_HEAD_REF"
        git rebase FETCH_HEAD
        AUTH_HEADER="AUTHORIZATION: basic $(printf 'x-access-token:%s' "$GH_PAT" | base64 -w0)"
        git -c core.hooksPath=/dev/null -c http.https://github.com/.extraheader="$AUTH_HEADER" push origin "HEAD:$PR_HEAD_REF"
YAML
  printf '%s' "$path"
}

# A workflows directory whose one workflow delegates to the action.
write_valid_workflows() {
  local dir="$WORK_DIR/$1-workflows"
  mkdir -p "$dir"
  cat >"$dir/auto-format.yml" <<'YAML'
name: Auto Format
on:
  pull_request:
jobs:
  auto-format:
    runs-on: ubuntu-latest
    steps:
      - name: Commit and push the fixes
        if: steps.detect.outputs.changed == 'true'
        uses: ./.github/actions/push-branch-changes
        with:
          branch: ${{ github.event.pull_request.head.ref }}
          commit-message: "style: apply cargo fmt"
          fallback-token: ${{ secrets.ACTIONS_PUSH || secrets.GITHUB_TOKEN }}
YAML
  printf '%s' "$dir"
}

# break_rule NAME SED_EXPRESSION → path to an action with one rule broken.
#
# `sed -i` is written to a new file rather than in place: BSD sed (macOS) reads
# the argument after -i as a backup suffix, so the in-place form fails there.
break_rule() {
  local name="$1" expression="$2" path
  path="$(write_valid_action "$name")"
  sed "$expression" "$path" >"$path.edited"
  mv "$path.edited" "$path"
  printf '%s' "$path"
}

# expect_exit DESCRIPTION EXPECTED_CODE ACTION_PATH WORKFLOWS_DIR [NEEDLE]
expect_exit() {
  local description="$1" expected="$2" action="$3" workflows="$4" needle="${5:-}"
  local output status=0
  output="$("$CHECKER" "$action" "$workflows" 2>&1)" || status=$?
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

VALID_WORKFLOWS="$(write_valid_workflows valid)"

expect_exit "accepts the committed action and workflows" 0 \
  "$REPO_ROOT/.github/actions/push-branch-changes/action.yml" \
  "$REPO_ROOT/.github/workflows"

expect_exit "accepts an action satisfying every rule" 0 \
  "$(write_valid_action valid)" "$VALID_WORKFLOWS"

expect_exit "rejects an action that is not composite" 1 \
  "$(break_rule not-composite 's/using: composite/using: node20/')" \
  "$VALID_WORKFLOWS" "composite"

expect_exit "rejects a token mint pinned to a tag" 1 \
  "$(break_rule tag-pinned 's|create-github-app-token@bcd2ba49218906704ab6c1aa796996da409d3eb1 # v3|create-github-app-token@v3|')" \
  "$VALID_WORKFLOWS" "SHA-pinned"

expect_exit "rejects a mint that does not scope the token to contents: write" 1 \
  "$(break_rule no-permission '/permission-contents: write/d')" \
  "$VALID_WORKFLOWS" "permission-contents"

expect_exit "rejects a missing fallback-token chain" 1 \
  "$(break_rule no-fallback 's/steps.push-token.outputs.token || inputs.fallback-token/steps.push-token.outputs.token/')" \
  "$VALID_WORKFLOWS" "fallback"

# shellcheck disable=SC2016  # the sed address is literal: $GH_PAT must not expand
expect_exit "rejects a push with no empty-credential guard" 1 \
  "$(break_rule no-credential-guard '/-z "\$GH_PAT"/d')" \
  "$VALID_WORKFLOWS" "empty-credential guard"

expect_exit "rejects a commit that leaves repository hooks enabled" 1 \
  "$(break_rule hooks-enabled 's|-c core.hooksPath=/dev/null ||g')" \
  "$VALID_WORKFLOWS" "hooksPath"

expect_exit "rejects a push with no deleted-branch guard" 1 \
  "$(break_rule no-ls-remote '/ls-remote/d')" \
  "$VALID_WORKFLOWS" "ls-remote"

expect_exit "rejects a push with no rebase" 1 \
  "$(break_rule no-rebase '/git rebase FETCH_HEAD/d')" \
  "$VALID_WORKFLOWS" "rebase"

expect_exit "rejects a push with no auth header" 1 \
  "$(break_rule no-auth-header '/AUTHORIZATION: basic/d')" \
  "$VALID_WORKFLOWS" "AUTHORIZATION"

expect_exit "rejects run blocks without strict bash" 1 \
  "$(break_rule no-strict-bash '/set -euo pipefail/d')" \
  "$VALID_WORKFLOWS" "set -euo pipefail"

spliced="$(write_valid_action spliced)"
sed 's|commit -am "\$COMMIT_MESSAGE"|commit -am "${{ inputs.commit-message }}"|' \
  "$spliced" >"$spliced.edited"
mv "$spliced.edited" "$spliced"
expect_exit "rejects an input spliced into the run: block" 1 \
  "$spliced" "$VALID_WORKFLOWS" "interpolates"

inline_mint_dir="$(write_valid_workflows inline-mint)"
cat >"$inline_mint_dir/version-increment.yml" <<'YAML'
name: Version Increment
on:
  pull_request:
jobs:
  version-increment:
    runs-on: ubuntu-latest
    steps:
      - name: Mint repo-scoped push token
        uses: actions/create-github-app-token@bcd2ba49218906704ab6c1aa796996da409d3eb1 # v3
        with:
          permission-contents: write
YAML
expect_exit "rejects a workflow that mints its own push token" 1 \
  "$(write_valid_action inline-mint)" "$inline_mint_dir" "issue #164"

unused_dir="$WORK_DIR/unused-workflows"
mkdir -p "$unused_dir"
cat >"$unused_dir/ci.yml" <<'YAML'
name: CI
on:
  pull_request:
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: cargo test
YAML
expect_exit "rejects an action no workflow calls" 1 \
  "$(write_valid_action unused)" "$unused_dir" "dead code"

expect_exit "reports a missing action with exit 2" 2 \
  "$WORK_DIR/does-not-exist.yml" "$VALID_WORKFLOWS" "not found"

expect_exit "reports a missing workflows directory with exit 2" 2 \
  "$(write_valid_action missing-dir)" "$WORK_DIR/no-such-dir" "not found"

echo "check-push-branch-changes-action tests: $PASSED passed, $FAILED failed"
if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
