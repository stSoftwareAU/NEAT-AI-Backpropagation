#!/usr/bin/env bash
# Validate that dependency review actually runs on every pull request (Issue #28).
#
# `security.yml` commits an `actions/dependency-review-action` step, but a
# committed step that no caller reaches is a defence that never fires — it reads
# as covered in an audit while providing nothing at runtime. The rules below
# check the whole path from trigger to step:
#
#   1. `security.yml` declares a dependency-review step.
#   2. That step is pinned to a 40-character commit SHA (repo-wide supply-chain
#      rule; local `./.github/actions/*` composites are exempt elsewhere).
#   3. The `include-dependency-review` input defaults to `true`, so a caller has
#      to opt out deliberately rather than forget to opt in.
#   4. No caller passes `include-dependency-review: false` — the exact
#      regression this gate exists to stop.
#   5. At least one workflow calls `security.yml`, and at least one such caller
#      is triggered by `pull_request` — the step's own `if:` requires that event,
#      so a schedule-only caller would never run it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKFLOW_DIR="${1:-$REPO_ROOT/.github/workflows}"
SECURITY_WORKFLOW="$WORKFLOW_DIR/security.yml"
EXIT_CODE=0

usage() {
  cat <<'EOF'
Usage: check-dependency-review.sh [WORKFLOW_DIR]

Exits 0 when dependency review is reachable on pull requests, 1 when a rule is
broken, and 2 when the workflows cannot be read.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "$WORKFLOW_DIR" ]]; then
  echo "FAIL: workflow directory not found: $WORKFLOW_DIR" >&2
  exit 2
fi

if [[ ! -f "$SECURITY_WORKFLOW" ]]; then
  echo "FAIL: reusable security workflow not found: $SECURITY_WORKFLOW" >&2
  echo "      Dependency review is unenforced — commit .github/workflows/security.yml" >&2
  exit 2
fi

ok() { echo "OK   dependency review: $*"; }
fail() {
  echo "FAIL dependency review: $*" >&2
  EXIT_CODE=1
}

# 1 + 2. The step exists and is pinned to a commit SHA.
step_reference="$(grep -E '^[[:space:]]*(-[[:space:]]*)?uses:[[:space:]]*actions/dependency-review-action@' \
  "$SECURITY_WORKFLOW" |
  sed -E 's/^[[:space:]]*(-[[:space:]]*)?uses:[[:space:]]*//; s/[[:space:]]*#.*$//; s/[[:space:]]*$//' |
  head -n 1 || true)"

if [[ -z "$step_reference" ]]; then
  fail "security.yml has no actions/dependency-review-action step — a PR can add an advisory-carrying crate with no gate on the diff"
elif [[ ! "$step_reference" =~ @[0-9a-f]{40}$ ]]; then
  fail "action '$step_reference' is not pinned to a 40-character commit SHA"
else
  ok "security.yml runs $step_reference"
fi

# 3. The input defaults to true, so opting out has to be deliberate.
input_default="$(awk '
  /^[[:space:]]*include-dependency-review:[[:space:]]*$/ { in_input = 1; next }
  in_input && /^[[:space:]]*[a-zA-Z0-9_-]+:[[:space:]]*$/ { in_input = 0 }
  in_input && /^[[:space:]]*default:/ {
    sub(/^[[:space:]]*default:[[:space:]]*/, "")
    gsub(/["'\'']/, "")
    print
    exit
  }
' "$SECURITY_WORKFLOW")"

case "$input_default" in
  true) ok "include-dependency-review defaults to true" ;;
  "") fail "the include-dependency-review input declares no default — a caller that omits it gets an empty (falsey) value" ;;
  *) fail "include-dependency-review defaults to '$input_default' — every caller would have to opt in by hand" ;;
esac

# 4. No caller switches the step off.
disabled_at="$(grep -rnE '^[[:space:]]*include-dependency-review:[[:space:]]*("|'\'')?false("|'\'')?[[:space:]]*$' \
  "$WORKFLOW_DIR" || true)"
if [[ -n "$disabled_at" ]]; then
  while IFS= read -r location; do
    fail "include-dependency-review is switched off at ${location} — the committed step never runs"
  done <<<"$disabled_at"
else
  ok "no caller passes include-dependency-review: false"
fi

# 5. Something calls the reusable workflow, and does so on pull requests.
callers=()
while IFS= read -r caller; do
  callers+=("$caller")
done < <(grep -rlE '^[[:space:]]*uses:[[:space:]]*\./\.github/workflows/security\.yml[[:space:]]*$' \
  "$WORKFLOW_DIR" || true)

if [[ ${#callers[@]} -eq 0 ]]; then
  fail "no workflow calls security.yml — the reusable workflow is unreachable in every event"
else
  pull_request_callers=0
  for caller in "${callers[@]}"; do
    if grep -qE '^[[:space:]]*pull_request:' "$caller"; then
      pull_request_callers=$((pull_request_callers + 1))
    fi
  done
  if [[ "$pull_request_callers" -eq 0 ]]; then
    fail "no caller of security.yml is triggered by pull_request — the step's own if: requires that event"
  else
    ok "security.yml is called on pull_request by $pull_request_callers workflow(s)"
  fi
fi

exit "$EXIT_CODE"
