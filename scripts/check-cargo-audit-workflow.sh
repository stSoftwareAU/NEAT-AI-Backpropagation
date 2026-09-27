#!/usr/bin/env bash
# Validate that the security workflow's cargo-audit gate uses a prebuilt binary
# (Issue #204).
#
# `rustsec/audit-check` runs `which cargo-audit` and falls back to
# `cargo install cargo-audit` — a ~186s compile on every run — when the binary
# is not already on PATH. Installing the prebuilt binary first skips the
# compile while the audit action still runs, reports and fails the job exactly
# as before. The rules below keep both halves in place:
#
#   1. `security.yml` still runs a SHA-pinned `rustsec/audit-check` step, so
#      the speed-up can never quietly drop the gate itself.
#   2. That step keeps its `token:` input (it posts the advisory check run).
#   3. A SHA-pinned `taiki-e/install-action` step with `tool: cargo-audit`
#      runs *before* the audit step — after it, the compile still happens.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKFLOW_DIR="${1:-$REPO_ROOT/.github/workflows}"
SECURITY_WORKFLOW="$WORKFLOW_DIR/security.yml"
EXIT_CODE=0

usage() {
  cat <<'USAGE'
Usage: check-cargo-audit-workflow.sh [WORKFLOW_DIR]

Exits 0 when security.yml installs a prebuilt cargo-audit ahead of a pinned
rustsec/audit-check step, 1 when a rule is broken, and 2 when the workflow
cannot be read.
USAGE
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
  exit 2
fi

ok() { echo "OK   cargo-audit: $*"; }
fail() {
  echo "FAIL cargo-audit: $*" >&2
  EXIT_CODE=1
}

# One tab-separated row per step: ordinal, uses reference, tool input, and
# whether a token input is present ("-" when absent, since read collapses
# empty tab-separated fields). A step starts at any `- ` list item whose
# indentation matches the first step seen under `steps:`.
steps_table="$(awk '
  function flush() {
    if (n > 0) printf "%d\t%s\t%s\t%s\n", n, uses, tool, token
  }
  /^[[:space:]]*steps:[[:space:]]*$/ { in_steps = 1; step_indent = -1; next }
  in_steps && /^[[:space:]]*-[[:space:]]/ {
    indent = match($0, /[^[:space:]]/) - 1
    if (step_indent < 0) step_indent = indent
    if (indent == step_indent) {
      flush()
      n++; uses = "-"; tool = "-"; token = "no"
      sub(/^[[:space:]]*-[[:space:]]*/, "")
    }
  }
  in_steps && n > 0 {
    line = $0
    sub(/^[[:space:]]*/, "", line)
    if (line ~ /^uses:/) {
      sub(/^uses:[[:space:]]*/, "", line); sub(/[[:space:]]*#.*$/, "", line)
      uses = line
    } else if (line ~ /^tool:/) {
      sub(/^tool:[[:space:]]*/, "", line); sub(/[[:space:]]*#.*$/, "", line)
      gsub(/["'\'']/, "", line)
      tool = line
    } else if (line ~ /^token:[[:space:]]*[^[:space:]]/) {
      token = "yes"
    }
  }
  END { flush() }
' "$SECURITY_WORKFLOW")"

audit_row="$(awk -F'\t' '$2 ~ /^rustsec\/audit-check@/ { print; exit }' <<<"$steps_table")"
# Accept `cargo-audit` alone or as `cargo-audit@<version>` in the tool list.
install_row="$(awk -F'\t' '
  $2 ~ /^taiki-e\/install-action@/ && $3 ~ /(^|,)[[:space:]]*cargo-audit(@[^,]*)?[[:space:]]*(,|$)/ { print; exit }
' <<<"$steps_table")"

if [[ -z "$audit_row" ]]; then
  fail "security.yml has no rustsec/audit-check step — the advisory gate is gone, not faster"
  exit "$EXIT_CODE"
fi

IFS=$'\t' read -r audit_index audit_ref _ audit_token <<<"$audit_row"
if [[ ! "$audit_ref" =~ @[0-9a-f]{40}$ ]]; then
  fail "action '$audit_ref' is not pinned to a 40-character commit SHA"
else
  ok "security.yml runs $audit_ref"
fi

if [[ "$audit_token" != "yes" ]]; then
  fail "the rustsec/audit-check step has no token: input — it can no longer post its advisory check run"
else
  ok "rustsec/audit-check keeps its token input"
fi

if [[ -z "$install_row" ]]; then
  fail "no taiki-e/install-action step installs cargo-audit — rustsec/audit-check compiles cargo-audit from source on every run"
else
  IFS=$'\t' read -r install_index install_ref install_tool _ <<<"$install_row"
  if [[ ! "$install_ref" =~ @[0-9a-f]{40}$ ]]; then
    fail "action '$install_ref' is not pinned to a 40-character commit SHA"
  elif [[ "$install_index" -gt "$audit_index" ]]; then
    fail "the prebuilt cargo-audit install runs after rustsec/audit-check — the audit has already compiled it from source"
  else
    ok "$install_ref installs $install_tool before the audit"
  fi
fi

exit "$EXIT_CODE"
