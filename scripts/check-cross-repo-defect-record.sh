#!/usr/bin/env bash
# Validate that confirmed cross-repo defects are durably recorded (Issue #140).
#
# A defect this repo confirms in a *sibling* repository cannot be filed from
# here — the agent write allowlist refuses `gh issue create` against any repo
# but this one ([SECURITY] [WRITE_REPO_BLOCKED]). The finding therefore survives
# only as prose, and prose that lives in a single archived PR summary is one
# pruning away from being lost: nothing links to it, and its containing PR
# number ages out.
#
# The audit doc is where such a finding has to land. Every "Confirmed cross-repo
# defect" section in it must therefore carry enough to re-file the issue upstream
# from the record alone:
#   1. At least one such section exists.
#   2. Two or more `file.rs:LINE` citations — the defective site, and the test
#      that pins the defective behaviour in place.
#   3. Evidence wording (a test / assertion) saying how the defect was
#      confirmed, so a reader can tell "confirmed" from "suspected".
#   4. Provenance — the PR or PR summary the finding was folded in from, so the
#      source can be summarised away without orphaning the record.
#   5. The upstream filing status: blocked, or the issue it was filed as.
#   6. A row in the "Filing status" table for every sibling repo the sections
#      name, so the outstanding upstream work is visible in one place.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOC="${1:-$REPO_ROOT/docs/audit/issue-35-neat-ai-core-duplication.md}"
EXIT_CODE=0

usage() {
  cat <<'EOF'
Usage: check-cross-repo-defect-record.sh [AUDIT_DOC_PATH]

Exits 0 when every confirmed cross-repo defect is durably recorded, 1 when a
rule listed in the script header is broken, and 2 when the doc cannot be read.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "$DOC" ]]; then
  echo "FAIL: audit doc not found: $DOC" >&2
  echo "      Confirmed cross-repo defects would have no durable home" >&2
  exit 2
fi

ok() { echo "OK   $(basename "$DOC"): $*"; }
fail() {
  echo "FAIL $(basename "$DOC"): $*" >&2
  EXIT_CODE=1
}

SECTION_RE='^#{2,6}[[:space:]].*[Cc]onfirmed cross-repo defect'

# The heading lines opening each confirmed-defect section.
mapfile -t SECTION_STARTS < <(grep -nE "$SECTION_RE" "$DOC" | cut -d: -f1)

if [[ "${#SECTION_STARTS[@]}" -eq 0 ]]; then
  fail "no 'Confirmed cross-repo defect' section — a confirmed sibling-repo defect recorded only in a PR summary is lost when that summary is pruned"
  exit "$EXIT_CODE"
fi
ok "${#SECTION_STARTS[@]} confirmed cross-repo defect section(s) recorded"

TOTAL_LINES="$(wc -l <"$DOC")"

# The "Filing status" table, read once — rule 6 checks every named repo against it.
FILING_START="$(grep -nE '^#{2,6}[[:space:]]+Filing status' "$DOC" | head -n 1 | cut -d: -f1 || true)"
FILING_TABLE=""
if [[ -n "$FILING_START" ]]; then
  FILING_TABLE="$(sed -n "$((FILING_START + 1)),${TOTAL_LINES}p" "$DOC" | sed -n '/^#/q;p')"
fi

for start in "${SECTION_STARTS[@]}"; do
  heading="$(sed -n "${start}p" "$DOC" | sed -E 's/^#+[[:space:]]*//')"
  # Body runs to the next heading of any level, or the end of the doc.
  body="$(sed -n "$((start + 1)),${TOTAL_LINES}p" "$DOC" | sed -n '/^#/q;p')"

  # 2. Concrete citations — the defect site and the test that pins it.
  citations="$(printf '%s\n' "$body" |
    grep -oE '[A-Za-z0-9_./-]+\.(rs|ts):[0-9]+([,-][0-9]+)*' | sort -u || true)"
  citation_count=0
  [[ -n "$citations" ]] && citation_count="$(printf '%s\n' "$citations" | grep -c .)"
  if [[ "$citation_count" -ge 2 ]]; then
    ok "'$heading' cites $citation_count source sites"
  else
    fail "'$heading' cites $citation_count source site(s) — record the defective site *and* the test asserting it, or the issue cannot be re-filed from this doc alone"
  fi

  # 3. How the defect was confirmed.
  if printf '%s\n' "$body" | grep -qiE 'assert|test'; then
    ok "'$heading' says how the defect was confirmed"
  else
    fail "'$heading' names no test or assertion — a reader cannot tell a confirmed defect from a suspected one"
  fi

  # 4. Provenance of the folded-in finding.
  if printf '%s\n' "$body" | grep -qE 'pr-summary-[0-9]+\.md|PR \[?#[0-9]+'; then
    ok "'$heading' names the PR summary it was folded in from"
  else
    fail "'$heading' names no source PR or PR summary — the original record cannot be summarised away without orphaning this one"
  fi

  # 5. Upstream filing status.
  if printf '%s\n' "$body" | grep -qiE 'blocked|needs a human|filed —|issues/[0-9]+'; then
    ok "'$heading' records the upstream filing status"
  else
    fail "'$heading' records no upstream filing status — a reader cannot tell whether the sibling repo has been told"
  fi

  # 6. Every sibling repo named must appear in the Filing status table.
  repos="$(printf '%s\n' "$body" |
    grep -oE 'NEAT-AI-[A-Za-z][A-Za-z0-9-]*' | sort -u || true)"
  while IFS= read -r repo; do
    [[ -n "$repo" ]] || continue
    [[ "$repo" == "NEAT-AI-Backpropagation" ]] && continue
    if [[ -z "$FILING_TABLE" ]]; then
      fail "no 'Filing status' section — '$repo' has no visible outstanding-work row"
      continue
    fi
    if printf '%s\n' "$FILING_TABLE" | grep -qE "^\|.*${repo}.*\|"; then
      ok "'$repo' has a Filing status row"
    else
      fail "'$repo' is named by '$heading' but has no Filing status row — the outstanding upstream work is invisible in the status table"
    fi
  done <<<"$repos"
done

exit "$EXIT_CODE"
