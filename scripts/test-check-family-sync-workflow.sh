#!/usr/bin/env bash
# Tests for scripts/check-family-sync-workflow.sh (issues #152 and #153).
#
# Every case writes a fixture workflow, runs the real checker against it and
# asserts the exit code — and, where it matters, that the failure names the
# rule that broke. The fixture below is a valid family-sync workflow; each
# failing case starts from it and breaks exactly one rule, so a case that goes
# green for the wrong reason is visible.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKER="$SCRIPT_DIR/check-family-sync-workflow.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

PASSED=0
FAILED=0

if [[ ! -x "$CHECKER" ]]; then
  echo "FAIL: checker not found or not executable: $CHECKER" >&2
  exit 2
fi

# A workflow satisfying every rule in the checker's header.
write_valid() {
  local path="$WORK_DIR/$1.yml"
  cat >"$path" <<'YAML'
name: Family Sync
on:
  pull_request:
    types: [opened, synchronize, reopened]
    branches:
      - Develop
      - "milestone/**"
permissions:
  contents: read
jobs:
  family-sync:
    runs-on: ubuntu-latest
    if: github.event.pull_request.head.repo.full_name == github.repository
    permissions:
      contents: write
    env:
      CANONICAL_RUNLIB_URL: https://raw.githubusercontent.com/stSoftwareAU/NEAT-AI-core/Develop/scripts/runlib.sh
      CANONICAL_FAMILY_PINS_URL: https://raw.githubusercontent.com/stSoftwareAU/NEAT-AI-core/Develop/scripts/family-pins.sh
    steps:
      - name: Checkout PR branch
        uses: actions/checkout@93cb6efe18208431cddfb8368fd83d5badbf9bfd # v5
        with:
          ref: ${{ github.event.pull_request.head.ref }}
          persist-credentials: false
      - name: Fetch the canonical copies
        run: |
          set -euo pipefail
          curl --fail --silent --show-error --location \
            --output fetched "$CANONICAL_RUNLIB_URL"
          cp fetched scripts/runlib.sh
          curl --fail --silent --show-error --location \
            --output fetched "$CANONICAL_FAMILY_PINS_URL"
          cp fetched scripts/family-pins.sh
      - name: Move the neat-core pin to the latest release
        run: |
          set -euo pipefail
          ./scripts/family-pins.sh
      - name: Detect changes
        id: sync
        run: |
          set -euo pipefail
          if [[ -n "$(git status --porcelain)" ]]; then
            echo "changed=true" >>"$GITHUB_OUTPUT"
          else
            echo "changed=false" >>"$GITHUB_OUTPUT"
          fi
      - name: Commit and push the refreshed copies
        if: steps.sync.outputs.changed == 'true'
        env:
          PR_HEAD_REF: ${{ github.event.pull_request.head.ref }}
          GH_PAT: ${{ steps.push-token.outputs.token || secrets.ACTIONS_PUSH || secrets.GITHUB_TOKEN }}
        run: |
          set -euo pipefail
          git add scripts/runlib.sh scripts/family-pins.sh backpropagation/Cargo.toml Cargo.lock
          git commit -m "chore: sync NEAT-AI-core helpers and move the neat-core pin"
          git fetch origin "$PR_HEAD_REF"
          git rebase FETCH_HEAD
          git push origin "HEAD:$PR_HEAD_REF"
YAML
  printf '%s' "$path"
}

# break_rule NAME SED_EXPRESSION → path to a fixture with one rule broken.
#
# `sed -i` is written to a new file rather than in place: BSD sed (macOS) reads
# the argument after -i as a backup suffix, so the in-place form fails there.
# Multi-line insertions are built with heredocs below for the same reason —
# `\n` in a replacement is a GNU extension.
break_rule() {
  local name="$1" expression="$2" path
  path="$(write_valid "$name")"
  sed "$expression" "$path" >"$path.edited"
  mv "$path.edited" "$path"
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

expect_exit "accepts the committed family-sync workflow" 0 \
  "$REPO_ROOT/.github/workflows/family-sync.yml"

expect_exit "accepts a workflow satisfying every rule" 0 "$(write_valid valid)"

# The committed shape since issue #164: the commit/push step is the shared
# composite action, so staging and rebasing are satisfied by the delegation
# rather than by inline git commands.
write_delegated() {
  local path
  path="$(write_valid "$1")"
  {
    sed '/- name: Commit and push the refreshed copies/,$d' "$path"
    cat <<'YAML'
      - name: Commit and push the refreshed copies
        if: steps.sync.outputs.changed == 'true'
        uses: ./.github/actions/push-branch-changes
        with:
          branch: ${{ github.event.pull_request.head.ref }}
          commit-message: "chore: sync NEAT-AI-core helpers and move the neat-core pin"
          paths: scripts/runlib.sh scripts/family-pins.sh backpropagation/Cargo.toml Cargo.lock
          fallback-token: ${{ secrets.ACTIONS_PUSH || secrets.GITHUB_TOKEN }}
YAML
  } >"$path.edited"
  mv "$path.edited" "$path"
  printf '%s' "$path"
}

expect_exit "accepts a workflow that delegates the push to the shared action" 0 \
  "$(write_delegated delegated)"

delegated_no_lock="$(write_delegated delegated-no-lock)"
sed 's| backpropagation/Cargo.toml Cargo.lock||' "$delegated_no_lock" \
  >"$delegated_no_lock.edited"
mv "$delegated_no_lock.edited" "$delegated_no_lock"
expect_exit "rejects a delegated push whose paths omit Cargo.lock" 1 \
  "$delegated_no_lock" "does not stage Cargo.lock"

expect_exit "rejects a workflow with no pull_request trigger" 1 \
  "$(break_rule no-pr 's/^  pull_request:/  workflow_dispatch:/')" \
  "no pull_request trigger"

push_trigger="$(write_valid push-trigger)"
{
  echo "on:"
  echo "  push:"
  echo "    branches: [Develop]"
  grep -v '^on:$' "$push_trigger"
} >"$push_trigger.edited"
mv "$push_trigger.edited" "$push_trigger"
expect_exit "rejects a push trigger" 1 "$push_trigger" "push trigger present"

expect_exit "rejects a branch filter that skips milestone branches" 1 \
  "$(break_rule no-milestone '/milestone/d')" \
  "milestone"

write_all="$(write_valid write-all)"
sed 's/^permissions:$/permissions: write-all/' "$write_all" >"$write_all.edited"
mv "$write_all.edited" "$write_all"
expect_exit "rejects permissions: write-all" 1 "$write_all" "write-all"

expect_exit "rejects a workflow that never fetches runlib.sh" 1 \
  "$(break_rule no-runlib-source '/runlib.sh/d')" \
  "scripts/runlib.sh source — nothing to sync from"

expect_exit "rejects a workflow that never fetches family-pins.sh" 1 \
  "$(break_rule no-pins-source '\|Develop/scripts/family-pins.sh|d')" \
  "scripts/family-pins.sh source — nothing to sync from"

expect_exit "rejects a workflow that never runs family-pins.sh" 1 \
  "$(break_rule no-pins-run '\|./scripts/family-pins.sh$|d')" \
  "the neat-core pin would never move"

expect_exit "rejects a commit that does not stage Cargo.lock" 1 \
  "$(break_rule no-lock-staged 's|git add .*|git add scripts/runlib.sh|')" \
  "does not stage Cargo.lock"

expect_exit "rejects a fetch that ignores HTTP errors" 1 \
  "$(break_rule no-curl-fail 's/curl --fail/curl/')" \
  "curl --fail"

expect_exit "rejects an unconditional commit/push" 1 \
  "$(break_rule no-guard "/if: steps.sync.outputs.changed/d")" \
  "no conditional"

expect_exit "rejects a push with no rebase" 1 \
  "$(break_rule no-rebase '/git rebase FETCH_HEAD/d')" \
  "rebase"

commented_rebase="$(write_valid commented-rebase)"
{
  echo "# The rebase before the push keeps a moved branch pushable."
  grep -v 'git rebase FETCH_HEAD' "$commented_rebase"
} >"$commented_rebase.edited"
mv "$commented_rebase.edited" "$commented_rebase"
expect_exit "rejects a rebase that exists only in a comment" 1 \
  "$commented_rebase" "rebase"

expect_exit "rejects a missing fork guard" 1 \
  "$(break_rule no-fork-guard '/head.repo.full_name/d')" \
  "head.repo"

expect_exit "rejects a checkout that persists credentials" 1 \
  "$(break_rule persists-credentials 's/persist-credentials: false/persist-credentials: true/')" \
  "persist-credentials"

expect_exit "rejects an action pinned to a tag" 1 \
  "$(break_rule tag-pinned 's|actions/checkout@93cb6efe18208431cddfb8368fd83d5badbf9bfd # v5|actions/checkout@v5|')" \
  "commit SHA"

expect_exit "rejects a push with no ACTIONS_PUSH fallback" 1 \
  "$(break_rule no-push-token 's/secrets.ACTIONS_PUSH || secrets.GITHUB_TOKEN/secrets.GITHUB_TOKEN/')" \
  "ACTIONS_PUSH"

expect_exit "rejects run blocks without strict bash" 1 \
  "$(break_rule no-strict-bash '/set -euo pipefail/d')" \
  "set -euo pipefail"

expect_exit "reports a missing workflow with exit 2" 2 \
  "$WORK_DIR/does-not-exist.yml" "not found"

echo "check-family-sync-workflow tests: $PASSED passed, $FAILED failed"
if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
