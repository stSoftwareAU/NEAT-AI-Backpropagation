#!/usr/bin/env bash
# Fail when a tracked file references a private stSoftware repository (issue #171).
#
# This repository is public and must be self-contained for a public reader. A
# reference to a private fleet repository — a github.com URL, the `Repo#1234`
# shorthand an issue tracker resolves, or the bare repository name in prose — is
# dead weight to that reader: they cannot open it, and whatever it was cited to
# justify no longer stands on its own. A rule that originated in a private
# repository has to be restated here at concept level ("dev compiles fast,
# release is fully optimised"), citing this repository's own issue number
# instead.
#
# Public stSoftware repositories are untouched — this gates privacy, not the
# organisation name.
#
# The scan reads every git-tracked text file under the tree root, or every
# regular file when the root is not a git work tree (the test fixtures). This
# script and its test companion are skipped: they must spell the private
# repository names out in order to detect them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TREE="${1:-$REPO_ROOT}"

# Private stSoftware repositories a public reader cannot open.
PRIVATE_REPOS=("VibeCoding" "GRQ-taxation")

# The two files that necessarily contain the names above.
SELF_BASENAMES=(
  "check-no-private-repo-references.sh"
  "test-check-no-private-repo-references.sh"
)

usage() {
  cat <<'EOF'
Usage: check-no-private-repo-references.sh [TREE_ROOT]

Exits 0 when no tracked file references a private stSoftware repository, 1 when
one does, and 2 when TREE_ROOT cannot be read.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "$TREE" ]]; then
  echo "FAIL: tree root not found: $TREE" >&2
  echo "      Nothing was scanned — absence of findings is not a pass" >&2
  exit 2
fi
TREE="$(cd "$TREE" && pwd)"

# NUL-separated paths relative to the tree root.
list_files() {
  local top
  if top="$(git -C "$TREE" rev-parse --show-toplevel 2>/dev/null)" &&
    [[ "$top" == "$TREE" ]]; then
    git -C "$TREE" ls-files -z
  else
    (
      cd "$TREE" &&
        find . \
          \( -name .git -o -name target -o -name graft -o -name node_modules \) \
          -prune -o -type f -print0
    )
  fi
}

is_self() {
  local base
  base="$(basename "$1")"
  local self
  for self in "${SELF_BASENAMES[@]}"; do
    [[ "$base" == "$self" ]] && return 0
  done
  return 1
}

# One ERE of the private repository names. Matching the name itself covers
# every citation form at once — the github.com URL, the `Repo#1234` shorthand
# and the bare name in a comment — because all three spell the name out.
PATTERN=""
for repo in "${PRIVATE_REPOS[@]}"; do
  if [[ -z "$PATTERN" ]]; then
    PATTERN="$repo"
  else
    PATTERN="$PATTERN|$repo"
  fi
done

EXIT_CODE=0
SCANNED=0

while IFS= read -r -d '' rel; do
  rel="${rel#./}"
  [[ -n "$rel" ]] || continue
  [[ -f "$TREE/$rel" ]] || continue
  is_self "$rel" && continue
  SCANNED=$((SCANNED + 1))

  hits="$(grep -nIE "$PATTERN" -- "$TREE/$rel" || true)"
  [[ -n "$hits" ]] || continue

  while IFS= read -r hit; do
    [[ -n "$hit" ]] || continue
    echo "FAIL $rel:${hit%%:*}: references a private stSoftware repository" >&2
    echo "     ${hit#*:}" >&2
    EXIT_CODE=1
  done <<<"$hits"
done < <(list_files)

if [[ "$SCANNED" -eq 0 ]]; then
  echo "FAIL: no files scanned under $TREE — an empty scan is not a pass" >&2
  exit 2
fi

if [[ "$EXIT_CODE" -ne 0 ]]; then
  echo "" >&2
  echo "Reword each reference at concept level — state the rule itself and cite" >&2
  echo "this repository's own issue number — so a public reader needs no access" >&2
  echo "to a private repository to understand it (issue #171)." >&2
  exit "$EXIT_CODE"
fi

echo "OK   $SCANNED file(s) scanned: no private stSoftware repository references"
