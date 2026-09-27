#!/usr/bin/env bash
# Tests for scripts/check-cargo-audit-workflow.sh (Issue #204).
#
# Each case writes a fixture security.yml, runs the real checker against it,
# and asserts the exit code and the rule the failure names. The final case runs
# the checker against this repository's own committed workflows.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKER="$SCRIPT_DIR/check-cargo-audit-workflow.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

PASSED=0
FAILED=0

if [[ ! -x "$CHECKER" ]]; then
  echo "FAIL: checker not found or not executable: $CHECKER" >&2
  exit 2
fi

INSTALL_SHA="7a79fe8c3a13344501c80d99cae481c1c9085912"
AUDIT_SHA="858dc40f52ca2b8570b7a997c1c4e35c6fc9a432"

# write_security DIR STEPS → writes DIR/security.yml wrapping STEPS (a block of
# already-indented step YAML) in a minimal reusable workflow.
write_security() {
  local dir="$1" steps="$2"
  mkdir -p "$dir"
  {
    cat <<'YAML'
name: Security Reusable Workflow
on:
  workflow_call:
jobs:
  security:
    runs-on: ubuntu-latest
    steps:
YAML
    printf '%s\n' "$steps"
  } >"$dir/security.yml"
}

INSTALL_STEP="      - name: Install cargo-audit (prebuilt binary)
        uses: taiki-e/install-action@${INSTALL_SHA}  # v2.81.10
        with:
          tool: cargo-audit"
AUDIT_STEP="      - name: Cargo Security Audit
        uses: rustsec/audit-check@${AUDIT_SHA}  # v2
        with:
          token: \${{ secrets.GITHUB_TOKEN }}"

# expect_exit DESCRIPTION EXPECTED_CODE WORKFLOW_DIR [EXPECTED_OUTPUT_SUBSTRING]
expect_exit() {
  local description="$1" expected="$2" workflow_dir="$3" needle="${4:-}"
  local output status=0
  output="$("$CHECKER" "$workflow_dir" 2>&1)" || status=$?
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

write_security "$WORK_DIR/valid" "$INSTALL_STEP
$AUDIT_STEP"
expect_exit "accepts a prebuilt install ahead of the audit" 0 "$WORK_DIR/valid"

write_security "$WORK_DIR/pinned-version" "      - name: Install cargo-audit
        uses: taiki-e/install-action@${INSTALL_SHA}  # v2.81.10
        with:
          tool: cargo-audit@0.21.2
$AUDIT_STEP"
expect_exit "accepts an explicitly versioned cargo-audit tool" 0 \
  "$WORK_DIR/pinned-version"

expect_exit "reports a missing workflow directory with exit 2" 2 \
  "$WORK_DIR/does-not-exist" "not found"

mkdir -p "$WORK_DIR/no-security"
expect_exit "reports a missing security workflow with exit 2" 2 \
  "$WORK_DIR/no-security" "security.yml"

write_security "$WORK_DIR/no-install" "$AUDIT_STEP"
expect_exit "rejects an audit that compiles cargo-audit from source" 1 \
  "$WORK_DIR/no-install" "compiles cargo-audit from source"

write_security "$WORK_DIR/install-after" "$AUDIT_STEP
$INSTALL_STEP"
expect_exit "rejects a prebuilt install that runs after the audit" 1 \
  "$WORK_DIR/install-after" "after rustsec/audit-check"

write_security "$WORK_DIR/other-tool" "      - name: Install cargo-deny
        uses: taiki-e/install-action@${INSTALL_SHA}  # v2.81.10
        with:
          tool: cargo-deny
$AUDIT_STEP"
expect_exit "rejects an install step for a different tool" 1 \
  "$WORK_DIR/other-tool" "compiles cargo-audit from source"

write_security "$WORK_DIR/unpinned-install" "      - name: Install cargo-audit
        uses: taiki-e/install-action@v2
        with:
          tool: cargo-audit
$AUDIT_STEP"
expect_exit "rejects an install action pinned to a movable tag" 1 \
  "$WORK_DIR/unpinned-install" "taiki-e/install-action@v2"

write_security "$WORK_DIR/no-audit" "$INSTALL_STEP"
expect_exit "rejects a workflow whose audit step was dropped" 1 \
  "$WORK_DIR/no-audit" "no rustsec/audit-check step"

write_security "$WORK_DIR/unpinned-audit" "$INSTALL_STEP
      - name: Cargo Security Audit
        uses: rustsec/audit-check@v2
        with:
          token: \${{ secrets.GITHUB_TOKEN }}"
expect_exit "rejects an audit action pinned to a movable tag" 1 \
  "$WORK_DIR/unpinned-audit" "rustsec/audit-check@v2"

write_security "$WORK_DIR/no-token" "$INSTALL_STEP
      - name: Cargo Security Audit
        uses: rustsec/audit-check@${AUDIT_SHA}  # v2"
expect_exit "rejects an audit step that lost its token input" 1 \
  "$WORK_DIR/no-token" "token"

expect_exit "accepts this repository's committed workflows" 0 \
  "$REPO_ROOT/.github/workflows"

echo "check-cargo-audit-workflow tests: $PASSED passed, $FAILED failed"
if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
