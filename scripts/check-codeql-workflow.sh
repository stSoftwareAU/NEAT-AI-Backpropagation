#!/usr/bin/env bash
# Validate the CodeQL code-scanning workflow (Issue #20).
#
# The workflow must:
#   1. Run on `pull_request` events targeting the default branch (`Develop`),
#      so a regression is caught before it merges.
#   2. Also run on a `schedule` firing at least weekly — a query pack or
#      advisory published after a merge must not stay invisible until the next
#      pull request happens to be opened.
#   3. Grant `security-events: write`, otherwise the analysis runs and the
#      results are silently dropped instead of reaching the Security tab.
#   4. Analyse `rust` — the only language in this repository.
#   5. Run both `github/codeql-action/init` and `github/codeql-action/analyze`;
#      an init with no analyze uploads nothing.
#   6. Pin every third-party action to a 40-character commit SHA (the repo-wide
#      supply-chain rule; local `./.github/actions/*` composites are exempt).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKFLOW="${1:-$REPO_ROOT/.github/workflows/codeql.yml}"
EXIT_CODE=0

usage() {
  cat <<'EOF'
Usage: check-codeql-workflow.sh [WORKFLOW_PATH]

Exits 0 when the workflow satisfies every rule listed in the script header,
1 when a rule is broken, and 2 when the workflow cannot be read.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "$WORKFLOW" ]]; then
  echo "FAIL: CodeQL workflow not found: $WORKFLOW" >&2
  echo "      Code scanning is unenforced — commit .github/workflows/codeql.yml" >&2
  exit 2
fi

ok() { echo "OK   $WORKFLOW: $*"; }
fail() {
  echo "FAIL $WORKFLOW: $*" >&2
  EXIT_CODE=1
}

# 1. Pull-request trigger on the default branch.
if grep -qE '^[[:space:]]*pull_request:' "$WORKFLOW"; then
  if grep -qE '^[[:space:]]*-[[:space:]]*"?Develop"?[[:space:]]*$' "$WORKFLOW"; then
    ok "pull_request trigger covers the Develop default branch"
  else
    fail "pull_request trigger does not list Develop — PRs onto the default branch would go unscanned"
  fi
else
  fail "no pull_request trigger — a regression would only be found after merge"
fi

# 2. Scheduled run, at least weekly.
if grep -qE '^[[:space:]]*schedule:' "$WORKFLOW"; then
  cron_count=0
  weekly_or_better=0
  while IFS= read -r expression; do
    cron_count=$((cron_count + 1))
    # A here-string splits the fields without letting `*` glob the working
    # directory, which plain word splitting would do.
    read -r _minute _hour day_of_month month day_of_week surplus \
      <<<"$expression"
    if [[ -z "$day_of_week" || -n "$surplus" ]]; then
      fail "cron expression '$expression' does not have five fields"
      continue
    fi
    # A literal day-of-month or month pins the run to one day a month (or a
    # year); only a wildcard or a step fires at least once a week.
    if [[ "$day_of_month" =~ ^(\*|\*/[0-9]+)$ && "$month" =~ ^(\*|\*/[0-9]+)$ ]]; then
      weekly_or_better=$((weekly_or_better + 1))
    else
      fail "cron expression '$expression' fires less often than weekly"
    fi
  done < <(grep -E '^[[:space:]]*-[[:space:]]*cron:' "$WORKFLOW" |
    sed -E 's/^[[:space:]]*-[[:space:]]*cron:[[:space:]]*//; s/^["'\'']//; s/["'\'']$//')

  if [[ "$cron_count" -eq 0 ]]; then
    fail "schedule: block declares no cron expression"
  elif [[ "$weekly_or_better" -gt 0 ]]; then
    ok "scheduled analysis runs at least weekly, independent of PR activity"
  fi
else
  fail "no schedule: trigger — analysis would only run while a PR is open"
fi

# 3. Results upload permission.
if grep -qE '^[[:space:]]*security-events:[[:space:]]*write' "$WORKFLOW"; then
  ok "security-events: write present (results reach the Security tab)"
else
  fail "no 'security-events: write' permission — the analysis would run and its results be discarded"
fi

# 4. Rust is analysed.
if grep -qE '^[[:space:]]*(-[[:space:]]*)?languages?:.*\brust\b' "$WORKFLOW"; then
  ok "rust is analysed"
else
  fail "no 'language: rust' declaration — the repository's own code would not be scanned"
fi

# 5. Both halves of the CodeQL action.
for step in init analyze; do
  if grep -qE 'uses:[[:space:]]*github/codeql-action/'"$step"'@' "$WORKFLOW"; then
    ok "github/codeql-action/$step step present"
  else
    fail "no 'github/codeql-action/$step' step — the analysis is incomplete"
  fi
done

# 6. Third-party actions pinned to a commit SHA.
unpinned=0
while IFS= read -r reference; do
  case "$reference" in
    ./*) continue ;; # local composite action, versioned by this repository
  esac
  if [[ ! "$reference" =~ @[0-9a-f]{40}$ ]]; then
    fail "action '$reference' is not pinned to a 40-character commit SHA"
    unpinned=$((unpinned + 1))
  fi
done < <(grep -E '^[[:space:]]*(-[[:space:]]*)?uses:' "$WORKFLOW" |
  sed -E 's/^[[:space:]]*(-[[:space:]]*)?uses:[[:space:]]*//; s/[[:space:]]*#.*$//; s/[[:space:]]*$//')

if [[ "$unpinned" -eq 0 ]]; then
  ok "every third-party action is pinned to a commit SHA"
fi

exit "$EXIT_CODE"
