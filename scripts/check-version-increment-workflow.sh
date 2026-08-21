#!/usr/bin/env bash
# Validate the version-increment PR workflow (GRQ-taxation runlib contract).
#
# The workflow must:
#   1. Run on `pull_request` events only.
#   2. Declare minimal permissions (`contents: write`).
#   3. Invoke `scripts/bump-backpropagation-version.sh`.
#   4. Gate commit/push behind a change-detection output (idempotent).
#   5. Refuse to push onto a fork's PR branch.
#   6. Use strict bash (`set -euo pipefail`).
#   7. Trigger on every build-affecting path (issue #95) — a path missing from
#      the filter never starts the job, so remotes keep a stale library.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/build-affecting-paths.sh
source "$SCRIPT_DIR/build-affecting-paths.sh"
WORKFLOW="${1:-$REPO_ROOT/.github/workflows/version-increment.yml}"
EXIT_CODE=0

usage() {
  cat <<'EOF'
Usage: check-version-increment-workflow.sh [WORKFLOW_PATH]

Exits 0 when the workflow satisfies every rule listed in the script header.
EOF
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
  fail "no pull_request trigger — version-increment must only run on PRs"
fi

if grep -qE '^[[:space:]]*permissions:[[:space:]]*write-all' "$WORKFLOW"; then
  fail "'permissions: write-all' grants more than this job needs (use contents: write)"
elif grep -qE '^[[:space:]]*contents:[[:space:]]*write' "$WORKFLOW"; then
  ok "minimal write permission (contents: write) present"
else
  fail "no 'contents: write' permission — the job cannot push bumps"
fi

if grep -qE 'bump-backpropagation-version\.sh' "$WORKFLOW"; then
  ok "bump-backpropagation-version.sh invocation present"
else
  fail "no bump-backpropagation-version.sh invocation"
fi

if grep -qE '^[[:space:]]*if:[[:space:]]*steps\.[A-Za-z0-9_-]+\.outputs\.' "$WORKFLOW"; then
  ok "commit/push is conditional on change-detection output (idempotent guard)"
else
  fail "no conditional 'if: steps.*.outputs.*' guard — commit/push must be conditional"
fi

if grep -qE 'github\.event\.pull_request\.head\.repo\.full_name[[:space:]]*==' "$WORKFLOW" \
  || grep -qE 'github\.event\.pull_request\.head\.repo\.fork' "$WORKFLOW"; then
  ok "fork PRs are excluded from the push step"
else
  fail "no head.repo check — pushes onto forks will fail silently"
fi

if grep -qE 'set -euo pipefail' "$WORKFLOW"; then
  ok "strict bash (set -euo pipefail) present"
else
  fail "no 'set -euo pipefail' in workflow run steps"
fi

# Entries of the `on.pull_request.paths:` list, unquoted, one per line.
workflow_paths() {
  awk '
    /^[[:space:]]*paths:[[:space:]]*$/ { in_block = 1; next }
    in_block && /^[[:space:]]*(#|$)/ { next }
    in_block && /^[[:space:]]*-[[:space:]]*/ {
      line = $0
      sub(/^[[:space:]]*-[[:space:]]*/, "", line)
      gsub(/"/, "", line)
      sub(/[[:space:]]+$/, "", line)
      print line
      next
    }
    in_block { in_block = 0 }
  ' "$WORKFLOW"
}

DECLARED_PATHS="$(workflow_paths)"
MISSING_PATHS=()
for required in "${BUILD_AFFECTING_PATHS[@]}"; do
  if ! printf '%s\n' "$DECLARED_PATHS" | grep -Fxq "$required"; then
    MISSING_PATHS+=("$required")
  fi
done
if [[ "${#MISSING_PATHS[@]}" -eq 0 ]]; then
  ok "paths filter covers every build-affecting path"
else
  fail "paths filter omits build-affecting path(s): ${MISSING_PATHS[*]} — a PR touching them would never run the bump job"
fi

if grep -qE 'chore: auto-increment versions for changed projects' "$WORKFLOW"; then
  ok "auto-increment commit subject present (idempotency grep target)"
else
  fail "missing auto-increment commit subject"
fi

exit "$EXIT_CODE"
