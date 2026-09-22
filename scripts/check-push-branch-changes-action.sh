#!/usr/bin/env bash
# Validate the shared push-branch-changes composite action (issue #164).
#
# auto-format.yml, family-sync.yml and version-increment.yml all push a fix
# back onto the pull request's head branch. That push credential — mint a
# repo-scoped App token, fall back to ACTIONS_PUSH, then GITHUB_TOKEN — used to
# be copied into all three, so a fix to one left the other two on the old path.
# `.github/actions/push-branch-changes` is now the single copy, and this
# checker is what keeps it single.
#
# The action must:
#   1. Be a composite action.
#   2. Mint the push token with `actions/create-github-app-token`, pinned to a
#      40-character commit SHA, scoped to this repository with
#      `permission-contents: write`.
#   3. Fall back from the App token to the caller's `fallback-token` input.
#   4. Refuse to push with no credential at all, rather than pushing anonymously.
#   5. Commit as github-actions[bot] with repository hooks disabled.
#   6. Skip the push when the head branch has been deleted.
#   7. Rebase onto the head branch before pushing when asked to.
#   8. Push with the base64 basic-auth AUTHORIZATION header to HEAD:$PR_HEAD_REF.
#   9. Use strict bash (`set -euo pipefail`).
#  10. Pass caller values through `env:` — never interpolate `${{ }}` into a
#      `run:` block, which would splice an input into the shell.
#
# And the workflows must:
#  11. Contain no second copy: no workflow mints `create-github-app-token`
#      itself (issue #164).
#  12. Actually call the action, so it cannot rot as dead code.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ACTION="${1:-$REPO_ROOT/.github/actions/push-branch-changes/action.yml}"
WORKFLOWS_DIR="${2:-$REPO_ROOT/.github/workflows}"
EXIT_CODE=0

usage() {
  cat <<'USAGE'
Usage: check-push-branch-changes-action.sh [ACTION_PATH [WORKFLOWS_DIR]]

Exits 0 when the composite action, and the workflows that call it, satisfy
every rule listed in the script header.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "$ACTION" ]]; then
  echo "FAIL: composite action not found: $ACTION" >&2
  exit 2
fi

if [[ ! -d "$WORKFLOWS_DIR" ]]; then
  echo "FAIL: workflows directory not found: $WORKFLOWS_DIR" >&2
  exit 2
fi

ok() { echo "OK   $ACTION: $*"; }
fail() {
  echo "FAIL $ACTION: $*" >&2
  EXIT_CODE=1
}

# Comment-free view: a rule must be satisfied by real YAML or a real command,
# never by the action's own prose about what it does.
body() { grep -vE '^[[:space:]]*#' "$ACTION"; }

if body | grep -qE '^[[:space:]]*using:[[:space:]]*composite'; then
  ok "declares a composite action"
else
  fail "not a composite action ('using: composite' missing) — the workflows cannot call it"
fi

if body | grep -qE 'uses:[[:space:]]*actions/create-github-app-token@[0-9a-f]{40}([[:space:]]|$)'; then
  ok "mints the push token with a SHA-pinned create-github-app-token"
else
  fail "no SHA-pinned 'actions/create-github-app-token' step — the repo-scoped App token is not minted here"
fi

if body | grep -qE '^[[:space:]]*permission-contents:[[:space:]]*write'; then
  ok "the minted token is scoped to contents: write"
else
  fail "no 'permission-contents: write' — the minted token cannot push, or is over-scoped"
fi

if body | grep -qE 'steps\.push-token\.outputs\.token[[:space:]]*\|\|[[:space:]]*inputs\.fallback-token'; then
  ok "falls back from the App token to the caller's fallback-token"
else
  fail "no 'steps.push-token.outputs.token || inputs.fallback-token' fallback — a repo with no App configured could not push"
fi

# shellcheck disable=SC2016  # the pattern is literal: $GH_PAT must not expand
if body | grep -qE '\-z[[:space:]]+"\$GH_PAT"'; then
  ok "an absent push credential fails loudly"
else
  fail "no empty-credential guard — a missing token would reach 'git push' as an anonymous push"
fi

if body | grep -qF 'core.hooksPath=/dev/null'; then
  ok "repository hooks are disabled for the bot commit"
else
  fail "no 'core.hooksPath=/dev/null' — a repository hook would run against the bot commit"
fi

if body | grep -qF 'github-actions[bot]'; then
  ok "commits under the github-actions[bot] identity"
else
  fail "no github-actions[bot] commit identity"
fi

if body | grep -qE 'ls-remote[^|]*--exit-code' && body | grep -qE 'ls-remote[^|]*--heads'; then
  ok "skips the push when the head branch has been deleted"
else
  fail "no 'git ls-remote --exit-code --heads' guard — a push onto a deleted branch would fail the job"
fi

if body | grep -qE '(^|[[:space:]])rebase([[:space:]]+--[a-z-]+)*[[:space:]]+FETCH_HEAD([^A-Za-z0-9_]|$)'; then
  ok "rebases onto the head branch before pushing"
else
  fail "no 'rebase FETCH_HEAD' — a branch that moved while the job ran would be rejected as non-fast-forward"
fi

if body | grep -qF 'AUTHORIZATION: basic'; then
  ok "pushes with the basic-auth AUTHORIZATION header"
else
  fail "no 'AUTHORIZATION: basic' push header — the credential never reaches git"
fi

# shellcheck disable=SC2016  # the pattern is literal: $PR_HEAD_REF must not expand
if body | grep -qF 'push origin "HEAD:$PR_HEAD_REF"'; then
  ok "pushes HEAD onto the caller's head branch"
else
  fail "no 'push origin \"HEAD:\$PR_HEAD_REF\"' — the fix would never reach the pull request"
fi

if body | grep -qE 'set[[:space:]]+-euo[[:space:]]+pipefail'; then
  ok "strict bash (set -euo pipefail) present"
else
  fail "no 'set -euo pipefail' in the run: block — failures may be swallowed"
fi

# Lines carrying a `${{ }}` expression inside a `run: |` block, which would
# splice a caller-supplied value straight into the shell.
run_block_interpolations() {
  awk '
    { line = $0 }
    line ~ /^[[:space:]]*#/ { next }
    line ~ /^[[:space:]]*$/ { next }
    { match(line, /^[[:space:]]*/); indent = RLENGTH }
    in_run && indent <= run_indent { in_run = 0 }
    in_run && line ~ /\$\{\{/ { print NR ": " line }
    line ~ /^[[:space:]]*run:[[:space:]]*\|/ { in_run = 1; run_indent = indent }
  ' "$ACTION"
}

INTERPOLATED="$(run_block_interpolations)"
if [[ -z "$INTERPOLATED" ]]; then
  ok "no \${{ }} interpolation inside a run: block — caller values arrive through env:"
else
  fail "run: block interpolates a \${{ }} expression: ${INTERPOLATED//$'\n'/; } — pass it through env: and quote the variable"
fi

# Rules 11 and 12 look at the callers, not the action: the point of the action
# is that it is the only copy, and that it is genuinely used.
INLINE_MINTS=()
DELEGATING=()
shopt -s nullglob
for workflow in "$WORKFLOWS_DIR"/*.yml "$WORKFLOWS_DIR"/*.yaml; do
  if grep -vE '^[[:space:]]*#' "$workflow" | grep -qE 'uses:[[:space:]]*actions/create-github-app-token@'; then
    INLINE_MINTS+=("$(basename "$workflow")")
  fi
  if grep -vE '^[[:space:]]*#' "$workflow" | grep -qE 'uses:[[:space:]]*\./\.github/actions/push-branch-changes'; then
    DELEGATING+=("$(basename "$workflow")")
  fi
done
shopt -u nullglob

if [[ "${#INLINE_MINTS[@]}" -eq 0 ]]; then
  ok "no workflow mints its own push token — the credential logic has one home"
else
  fail "workflow(s) mint their own push token instead of calling the action: ${INLINE_MINTS[*]} — a fix here would have to be repeated there (issue #164)"
fi

if [[ "${#DELEGATING[@]}" -gt 0 ]]; then
  ok "called by: ${DELEGATING[*]}"
else
  fail "no workflow calls ./.github/actions/push-branch-changes — the action is dead code"
fi

exit "$EXIT_CODE"
