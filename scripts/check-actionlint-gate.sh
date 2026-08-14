#!/usr/bin/env bash
# Validate the actionlint workflow-lint gate in CI (Issue #26).
#
# Workflow YAML is the one part of this repository no other gate reads: an
# invalid expression, an unknown `runs-on`, or a shell bug inside a `run:`
# block only surfaces the next time the workflow runs. `actionlint` is the
# standard linter for it, so CI must invoke it — and the invocation has to be
# able to fail the build.
#
# The workflow must:
#   1. Invoke `actionlint` from a job — a command, a script, or an actionlint
#      action — not merely mention it in a comment or a step name.
#   2. Keep the invocation blocking: no `continue-on-error: true` on the
#      linting job, no `|| true` swallowing the linter's exit code.
#   3. Use strict bash (`set -euo pipefail`) in that job, so a failed install
#      cannot be reconciled as a clean lint.
#   4. Be waited on by another job (the `ci-required` aggregator) — a lint job
#      no job needs runs, reports, and gates nothing.
#   5. Pin its supply chain: an action is pinned to a 40-character commit SHA,
#      and a downloaded release is version-pinned (never `latest`) with its
#      checksum verified.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKFLOW="${1:-$REPO_ROOT/.github/workflows/ci.yml}"
EXIT_CODE=0

usage() {
  cat <<'EOF'
Usage: check-actionlint-gate.sh [WORKFLOW_PATH]

Exits 0 when the workflow satisfies every rule listed in the script header,
1 when a rule is broken, and 2 when the workflow cannot be read.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "$WORKFLOW" ]]; then
  echo "FAIL: workflow not found: $WORKFLOW" >&2
  echo "      Workflow YAML is unlinted — commit a CI job that runs actionlint" >&2
  exit 2
fi

ok() { echo "OK   $WORKFLOW: $*"; }
fail() {
  echo "FAIL $WORKFLOW: $*" >&2
  EXIT_CODE=1
}

# Every line inside `jobs:`, tagged with the job key that owns it, so a rule
# can be applied to the linting job alone rather than the whole file.
tagged_job_lines() {
  awk '
    /^jobs:[[:space:]]*$/ { in_jobs = 1; next }
    in_jobs && /^[^[:space:]#]/ { in_jobs = 0; job = "" }
    in_jobs && /^  [A-Za-z0-9_.-]+:[[:space:]]*$/ {
      job = $1
      sub(/:$/, "", job)
      next
    }
    in_jobs && job != "" { print job "\t" $0 }
  ' "$WORKFLOW"
}

# An `actionlint` word that is a real invocation: comments and step names are
# prose, and `check-actionlint-gate.sh` (this script) is a different command.
ACTIONLINT_WORD='(^|[^A-Za-z0-9_.-])actionlint([^A-Za-z0-9_.-]|$)'

JOB_LINES="$(tagged_job_lines)"

invoking_lines() {
  printf '%s\n' "$JOB_LINES" | awk -F'\t' -v word="$ACTIONLINT_WORD" '
    $2 ~ /^[[:space:]]*#/ { next }
    $2 ~ /^[[:space:]]*(-[[:space:]]*)?name:/ { next }
    $2 ~ word { print }
  '
}

INVOKING_LINES="$(invoking_lines)"
INVOKING_JOBS="$(printf '%s' "$INVOKING_LINES" | cut -f1 | sort -u | sed '/^$/d')"

if [[ -z "$INVOKING_JOBS" ]]; then
  fail "no actionlint invocation — workflow YAML regressions would not fail the build"
  exit "$EXIT_CODE"
fi

for job in $INVOKING_JOBS; do
  block="$(printf '%s\n' "$JOB_LINES" | awk -F'\t' -v j="$job" '$1 == j { print $2 }')"
  job_invocations="$(printf '%s\n' "$INVOKING_LINES" | awk -F'\t' -v j="$job" '$1 == j { print $2 }')"

  ok "job '$job' invokes actionlint"

  # 2a. The linter's exit code must reach the runner.
  if printf '%s\n' "$job_invocations" | grep -qE '\|\|[[:space:]]*(true|:|echo\b)'; then
    fail "job '$job' discards actionlint's exit code with '|| true' — a lint failure would pass"
  fi

  # 2b. A job allowed to fail is not a gate.
  if printf '%s\n' "$block" | grep -qE '^[[:space:]]*continue-on-error:[[:space:]]*true'; then
    fail "job '$job' sets continue-on-error: true — its lint failures would not block the merge"
  fi

  # 3. Strict bash, so an install failure is not read as a clean lint.
  if printf '%s\n' "$block" | grep -qE 'set[[:space:]]+-euo[[:space:]]+pipefail'; then
    ok "job '$job' runs under strict bash (set -euo pipefail)"
  else
    fail "job '$job' has no 'set -euo pipefail' — a failed install would be reported as a clean lint"
  fi

  # 4. Some other job must wait on this one, or nothing gates on the result.
  needle="$(printf '%s' "$job" | sed -E 's/[.]/\\./g')"
  if grep -E '^[[:space:]]*needs:' "$WORKFLOW" |
    grep -qE "(^|[^A-Za-z0-9_.-])$needle([^A-Za-z0-9_.-]|$)"; then
    ok "job '$job' is listed in another job's needs: (it gates the merge)"
  else
    fail "no job lists '$job' in its needs: — the lint would run without gating the merge"
  fi

  # 5a. An actionlint action is pinned to an immutable commit SHA.
  while IFS= read -r reference; do
    [[ -n "$reference" ]] || continue
    case "$reference" in
      ./*) continue ;; # local composite action, versioned by this repository
    esac
    if [[ ! "$reference" =~ @[0-9a-f]{40}$ ]]; then
      fail "action '$reference' is not pinned to a 40-character commit SHA"
    fi
  done < <(printf '%s\n' "$job_invocations" |
    grep -E '^[[:space:]]*(-[[:space:]]*)?uses:' |
    sed -E 's/^[[:space:]]*(-[[:space:]]*)?uses:[[:space:]]*//; s/[[:space:]]*#.*$//; s/[[:space:]]*$//')

  # 5b. A downloaded release is version-pinned and checksum verified.
  if printf '%s\n' "$block" | grep -qE '(^|[^A-Za-z0-9_.-])(curl|wget)([^A-Za-z0-9_.-]|$)'; then
    if printf '%s\n' "$block" | grep -qE 'sha256sum|shasum[[:space:]]+-a[[:space:]]+256'; then
      ok "job '$job' verifies the downloaded linter's checksum"
    else
      fail "job '$job' downloads the linter without a sha256 checksum verification"
    fi
    if printf '%s\n' "$block" | grep -qE 'releases/latest'; then
      fail "job '$job' downloads the latest release — pin the version so a hijacked release cannot land silently"
    fi
  fi
done

exit "$EXIT_CODE"
