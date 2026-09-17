#!/usr/bin/env bash
# Validate the family-sync PR workflow (issue #152 / #153, NEAT-AI-core
# #680 / #681).
#
# `scripts/runlib.sh` and `scripts/family-pins.sh` each have one home — the
# same path on NEAT-AI-core `Develop`. This repository carries byte-identical
# copies, and the family-sync workflow is what keeps them that way; it also
# runs `scripts/family-pins.sh`, which is the only way the `neat-core` release
# pin moves (issue #153). A misdeclared job would let a copy rot, or leave the
# pin stale, silently — so CI runs this checker on every PR.
#
# The workflow must:
#   1. Run on `pull_request` events (never on a push to the default branch).
#   2. Cover milestone branches — a filter that omits `milestone/*` leaves
#      every milestone sub-issue PR unsynced (Issue #27).
#   3. Declare minimal permissions (`contents: write`, never `write-all`).
#   4. Fetch both canonical copies from NEAT-AI-core `Develop`.
#   5. Fail non-zero on a fetch error (`curl --fail`) — a stale copy must never
#      be reported as synced.
#   6. Run `scripts/family-pins.sh`, so the neat-core pin is refreshed to the
#      latest release on every PR.
#   7. Stage `Cargo.lock` in the commit, so a moved pin is actually committed.
#   8. Gate the commit/push behind a change-detection output (idempotent).
#   9. Rebase before pushing, so a branch that moved meanwhile is not rejected.
#  10. Refuse to push onto a fork's PR branch.
#  11. Check out with `persist-credentials: false`.
#  12. Pin every action to a 40-character commit SHA.
#  13. Authenticate the push App token -> ACTIONS_PUSH -> GITHUB_TOKEN.
#  14. Use strict bash (`set -euo pipefail`).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKFLOW="${1:-$REPO_ROOT/.github/workflows/family-sync.yml}"
EXIT_CODE=0

usage() {
  cat <<'USAGE'
Usage: check-family-sync-workflow.sh [WORKFLOW_PATH]

Exits 0 when the workflow satisfies every rule listed in the script header.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "$WORKFLOW" ]]; then
  echo "FAIL: workflow file not found: $WORKFLOW" >&2
  exit 2
fi

ok() { echo "OK   $WORKFLOW: $*"; }
fail() {
  echo "FAIL $WORKFLOW: $*" >&2
  EXIT_CODE=1
}

if grep -qE '^[[:space:]]*pull_request:' "$WORKFLOW"; then
  ok "pull_request trigger present"
else
  fail "no pull_request trigger — the copy would never be refreshed on a PR"
fi

if grep -qE '^[[:space:]]*push:' "$WORKFLOW"; then
  fail "push trigger present — the sync belongs on pull_request, not on a push to Develop"
else
  ok "no push trigger"
fi

if ! grep -qE '^[[:space:]]*branches:' "$WORKFLOW"; then
  ok "no branch filter — every PR, milestone branches included, is synced"
elif grep -qE "milestone/[^\"']*\*" "$WORKFLOW"; then
  ok "branch filter covers milestone/* branches"
else
  fail "branch filter matches no milestone/<slug> branch — milestone sub-issue PRs would keep a stale copy (Issue #27)"
fi

if grep -qE '^[[:space:]]*permissions:[[:space:]]*write-all' "$WORKFLOW"; then
  fail "'permissions: write-all' grants more than this job needs (use contents: write)"
elif grep -qE '^[[:space:]]*contents:[[:space:]]*write' "$WORKFLOW"; then
  ok "minimal write permission (contents: write) present"
else
  fail "no 'contents: write' permission — the job cannot push the refreshed copy"
fi

for canonical_script in runlib.sh family-pins.sh; do
  if grep -qF "NEAT-AI-core/Develop/scripts/$canonical_script" "$WORKFLOW" \
    || grep -qF "NEAT-AI-core/contents/scripts/$canonical_script" "$WORKFLOW"; then
    ok "fetches scripts/$canonical_script from NEAT-AI-core Develop"
  else
    fail "no NEAT-AI-core Develop scripts/$canonical_script source — nothing to sync from"
  fi
done

if grep -qE 'curl[^|]*--fail' "$WORKFLOW" || grep -qE 'curl[[:space:]]+-[A-Za-z]*f' "$WORKFLOW"; then
  ok "fetch failures exit non-zero (curl --fail)"
else
  fail "no 'curl --fail' — an HTTP error page would be copied over the script, or a stale copy passed off as synced"
fi

# The invocation, not the fetch: the canonical URL also ends in the script's
# name, so the pattern requires the `./scripts/...` form a run: step uses.
if grep -vE '^[[:space:]]*#' "$WORKFLOW" | grep -qE '\./scripts/family-pins\.sh'; then
  ok "runs scripts/family-pins.sh, so the neat-core pin moves on every PR"
else
  fail "no './scripts/family-pins.sh' run — the neat-core pin would never move (issue #153)"
fi

# Comment-free, with backslash continuations folded onto one line, so a
# `git add` split across lines is read as the single command it is.
joined_lines() {
  grep -vE '^[[:space:]]*#' "$1" | awk '
    {
      line = $0
      sub(/^[[:space:]]+/, "", line)
      buf = (buf == "") ? line : buf " " line
      if (buf ~ /\\$/) { sub(/\\$/, "", buf); next }
      print buf
      buf = ""
    }
    END { if (buf != "") print buf }
  '
}

if joined_lines "$WORKFLOW" | grep -qE '(^|[[:space:]])add([[:space:]]|$)[^#]*Cargo\.lock'; then
  ok "commit stages Cargo.lock, so a moved pin is committed"
else
  fail "the commit does not stage Cargo.lock — a moved pin would be left in the runner's working tree (issue #153)"
fi

if grep -qE '^[[:space:]]*if:[[:space:]]*steps\.[A-Za-z0-9_-]+\.outputs\.' "$WORKFLOW"; then
  ok "commit/push is conditional on change-detection output (idempotent guard)"
else
  fail "no conditional 'if: steps.*.outputs.*' guard — an identical copy would be committed on every run"
fi

# Comment lines are stripped first: the rule has to be satisfied by a real
# `git ... rebase` command, never by the workflow's own prose about rebasing.
if grep -vE '^[[:space:]]*#' "$WORKFLOW" | grep -qE '(^|[[:space:]])rebase([[:space:]]|$)'; then
  ok "rebases before pushing"
else
  fail "no rebase command before the push — a branch that moved meanwhile is rejected as non-fast-forward (a comment mentioning rebase does not count)"
fi

if grep -qE 'github\.event\.pull_request\.head\.repo\.full_name[[:space:]]*==' "$WORKFLOW" \
  || grep -qE 'github\.event\.pull_request\.head\.repo\.fork' "$WORKFLOW"; then
  ok "fork PRs are excluded from the push step"
else
  fail "no head.repo check — pushes onto forks will fail"
fi

if grep -qE '^[[:space:]]*persist-credentials:[[:space:]]*false' "$WORKFLOW"; then
  ok "checkout credential persistence disabled"
else
  fail "no 'persist-credentials: false' on checkout — the job would keep an ambient push credential"
fi

UNPINNED="$(grep -nE '^[[:space:]]*uses:' "$WORKFLOW" |
  grep -vE 'uses:[[:space:]]*\./' |
  grep -vE 'uses:[[:space:]]*[^@]+@[0-9a-f]{40}([[:space:]]|$)' || true)"
if [[ -z "$UNPINNED" ]]; then
  ok "every action is pinned to a 40-character commit SHA"
else
  fail "action reference not pinned to a commit SHA: ${UNPINNED//$'\n'/; }"
fi

if grep -qE 'secrets\.ACTIONS_PUSH[[:space:]]*\|\|[[:space:]]*secrets\.GITHUB_TOKEN' "$WORKFLOW"; then
  ok "push authenticates with ACTIONS_PUSH (GITHUB_TOKEN fallback)"
else
  fail "no 'secrets.ACTIONS_PUSH || secrets.GITHUB_TOKEN' — bot pushes will gate PR checks behind Approve and run"
fi

if grep -qE 'set[[:space:]]+-euo[[:space:]]+pipefail' "$WORKFLOW"; then
  ok "strict bash (set -euo pipefail) present"
else
  fail "no 'set -euo pipefail' in run: blocks — failures may be swallowed"
fi

exit "$EXIT_CODE"
