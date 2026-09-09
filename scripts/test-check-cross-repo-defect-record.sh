#!/usr/bin/env bash
# Tests for scripts/check-cross-repo-defect-record.sh (Issue #140).
#
# Each case writes a fixture audit doc to a temporary directory, runs the real
# checker against it, and asserts the exit code (and, where it matters, that the
# failure names the rule that broke).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKER="$SCRIPT_DIR/check-cross-repo-defect-record.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

PASSED=0
FAILED=0

if [[ ! -x "$CHECKER" ]]; then
  echo "FAIL: checker not found or not executable: $CHECKER" >&2
  exit 2
fi

# write_doc NAME <<'MD' … MD  → echoes the fixture path.
write_doc() {
  local path="$WORK_DIR/$1.md"
  cat >"$path"
  printf '%s' "$path"
}

# expect_exit DESCRIPTION EXPECTED_CODE DOC_PATH [EXPECTED_OUTPUT_SUBSTRING]
expect_exit() {
  local description="$1" expected="$2" doc="$3" needle="${4:-}"
  local output status=0
  output="$("$CHECKER" "$doc" 2>&1)" || status=$?
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

valid=$(write_doc valid <<'MD'
# Audit

### Finding 3 — something duplicated

Prose about the duplication.

#### Confirmed cross-repo defect — the sibling copy is live (PR #101)

`NEAT-AI-Sibling/src/tags.rs:300-302` re-attaches the source uuid, and
`NEAT-AI-Sibling/src/tags.rs:432` asserts the defective behaviour in the
sibling's own test suite. Folded in from `docs/archive/pr-summaries/pr-summary-101.md`.

**Upstream filing status:** Blocked — needs a human with write access.

## Filing status

| Finding | Home | Status |
| ------- | ---- | ------ |
| 3 (sibling half) | `NEAT-AI-Sibling` | **Blocked** — needs a human to file |
MD
)
expect_exit "accepts a fully recorded confirmed cross-repo defect" 0 "$valid"

# A different shape — `##` heading, a filed upstream issue instead of a block,
# TypeScript citations — must also pass, so the checker enforces the policy
# rather than one hand-written section.
variant=$(write_doc variant <<'MD'
# Audit

## Confirmed cross-repo defect in the twin

`NEAT-AI-Twin/src/tags.ts:300` drops the guard; the regression is asserted at
`NEAT-AI-Twin/src/tags.ts:432`. Recorded first in PR #101.

Filed upstream as https://github.com/stSoftwareAU/NEAT-AI-Twin/issues/7.

## Filing status

| Finding | Home | Status |
| ------- | ---- | ------ |
| 1 | `NEAT-AI-Twin` | Filed — issue 7 |
MD
)
expect_exit "accepts a filed-upstream record with TypeScript citations" 0 "$variant"

expect_exit "reports a missing audit doc with exit 2" 2 \
  "$WORK_DIR/does-not-exist.md" "not found"

no_section=$(write_doc no-section <<'MD'
# Audit

### Finding 3 — something duplicated

The duplication is a maintenance hazard.

## Filing status

| Finding | Home | Status |
| ------- | ---- | ------ |
| 3 | `neat-core` | **Blocked** — needs a human to file |
MD
)
expect_exit "rejects an audit doc with no confirmed-defect section" 1 \
  "$no_section" "no 'Confirmed cross-repo defect' section"

one_citation=$(write_doc one-citation <<'MD'
# Audit

#### Confirmed cross-repo defect — the sibling copy is live (PR #101)

`NEAT-AI-Sibling/src/tags.rs:300-302` re-attaches the source uuid, and the
sibling's tests assert it. Folded in from `pr-summary-101.md`.
Blocked — needs a human with write access.

## Filing status

| Finding | Home | Status |
| ------- | ---- | ------ |
| 3 | `NEAT-AI-Sibling` | **Blocked** — needs a human to file |
MD
)
expect_exit "rejects a record citing the defect but not the test pinning it" 1 \
  "$one_citation" "cites 1 source site"

no_evidence=$(write_doc no-evidence <<'MD'
# Audit

#### Confirmed cross-repo defect — the sibling copy is live (PR #101)

`NEAT-AI-Sibling/src/tags.rs:300-302` re-attaches the source uuid; see also
`NEAT-AI-Sibling/src/candidates.rs:2712`. Folded in from `pr-summary-101.md`.
Blocked — needs a human with write access.

## Filing status

| Finding | Home | Status |
| ------- | ---- | ------ |
| 3 | `NEAT-AI-Sibling` | **Blocked** — needs a human to file |
MD
)
expect_exit "rejects a record that never says how the defect was confirmed" 1 \
  "$no_evidence" "names no test or assertion"

no_provenance=$(write_doc no-provenance <<'MD'
# Audit

#### Confirmed cross-repo defect — the sibling copy is live

`NEAT-AI-Sibling/src/tags.rs:300-302` re-attaches the source uuid, asserted at
`NEAT-AI-Sibling/src/tags.rs:432`.
Blocked — needs a human with write access.

## Filing status

| Finding | Home | Status |
| ------- | ---- | ------ |
| 3 | `NEAT-AI-Sibling` | **Blocked** — needs a human to file |
MD
)
expect_exit "rejects a record with no source PR or PR summary" 1 \
  "$no_provenance" "names no source PR"

no_status=$(write_doc no-status <<'MD'
# Audit

#### Confirmed cross-repo defect — the sibling copy is live (PR #101)

`NEAT-AI-Sibling/src/tags.rs:300-302` re-attaches the source uuid, asserted at
`NEAT-AI-Sibling/src/tags.rs:432`.

## Filing status

| Finding | Home | Status |
| ------- | ---- | ------ |
| 3 | `NEAT-AI-Sibling` | Rejected, reason above |
MD
)
expect_exit "rejects a record with no upstream filing status" 1 \
  "$no_status" "no upstream filing status"

missing_row=$(write_doc missing-row <<'MD'
# Audit

#### Confirmed cross-repo defect — the sibling copy is live (PR #101)

`NEAT-AI-Sibling/src/tags.rs:300-302` re-attaches the source uuid, asserted at
`NEAT-AI-Sibling/src/tags.rs:432`. Folded in from `pr-summary-101.md`.
Blocked — needs a human with write access.

## Filing status

| Finding | Home | Status |
| ------- | ---- | ------ |
| 3 | `neat-core` | **Blocked** — needs a human to file |
MD
)
expect_exit "rejects a named sibling repo with no Filing status row" 1 \
  "$missing_row" "no Filing status row"

no_table=$(write_doc no-table <<'MD'
# Audit

#### Confirmed cross-repo defect — the sibling copy is live (PR #101)

`NEAT-AI-Sibling/src/tags.rs:300-302` re-attaches the source uuid, asserted at
`NEAT-AI-Sibling/src/tags.rs:432`. Folded in from `pr-summary-101.md`.
Blocked — needs a human with write access.
MD
)
expect_exit "rejects an audit doc with no Filing status section" 1 \
  "$no_table" "no 'Filing status' section"

# The repository's own audit doc must satisfy every rule above.
REPO_DOC="$(cd "$SCRIPT_DIR/.." && pwd)/docs/audit/issue-35-neat-ai-core-duplication.md"
expect_exit "the committed audit doc records its confirmed cross-repo defect" 0 \
  "$REPO_DOC"

echo "check-cross-repo-defect-record tests: $PASSED passed, $FAILED failed"
if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
