#!/usr/bin/env bash
# Fail when a canonical NEAT-AI-core helper copied into this repository has
# drifted from its one home (issue #152 / #153, core #680 / #681).
#
# Two scripts are copied byte-for-byte from NEAT-AI-core `Develop`:
#
#   * `scripts/runlib.sh`      — build / install / clean helper.
#   * `scripts/family-pins.sh` — moves the `neat-core` git-tag pin to core's
#                                newest release.
#
# The family-sync workflow re-copies both on same-repo PRs, but it is skipped
# on fork PRs and could be disabled — and a downstream edit is exactly what the
# copy contract forbids. This gate reads the content rather than the workflow's
# shape, so drift fails loud wherever it came from.
#
# Usage: check-canonical-copies.sh [CANONICAL_DIR]
#
# With CANONICAL_DIR the copies in that directory are the authority (CI fetches
# them from NEAT-AI-core `Develop` and passes the directory). With no argument
# the NEAT-AI-core sibling checkout is read at `origin/Develop`, falling back —
# with a warning, never silently — to that checkout's working tree.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COPIES_DIR="${CANONICAL_COPIES_DIR:-$REPO_ROOT/scripts}"
SIBLING="${NEAT_CORE_DIR:-$REPO_ROOT/../NEAT-AI-core}"
CANONICAL_DIR="${1:-}"
WORK_DIR=""
EXIT_CODE=0

# The scripts under the copy contract, by file name.
CANONICAL_SCRIPTS=(runlib.sh family-pins.sh)

usage() {
  cat <<'USAGE'
Usage: check-canonical-copies.sh [CANONICAL_DIR]

Exits 0 when every copied NEAT-AI-core helper is byte-identical to its
canonical copy, 1 when one has drifted, and 2 when a canonical copy cannot
be read.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

trap '[[ -z "$WORK_DIR" ]] || rm -rf "$WORK_DIR"' EXIT

for name in "${CANONICAL_SCRIPTS[@]}"; do
  if [[ ! -f "$COPIES_DIR/$name" ]]; then
    echo "FAIL: no copy to check at $COPIES_DIR/$name" >&2
    exit 2
  fi
done

# No canonical directory given: extract the canonical copies from the sibling
# checkout, preferring the branch that governs them over its working tree.
if [[ -z "$CANONICAL_DIR" ]]; then
  if [[ ! -d "$SIBLING" ]]; then
    echo "FAIL: no NEAT-AI-core sibling at $SIBLING — pass a directory holding the canonical copies as an argument" >&2
    exit 2
  fi
  WORK_DIR="$(mktemp -d)"
  CANONICAL_DIR="$WORK_DIR"
  if git -C "$SIBLING" rev-parse --verify --quiet origin/Develop >/dev/null; then
    echo "INFO canonical copies read from origin/Develop in $SIBLING"
    for name in "${CANONICAL_SCRIPTS[@]}"; do
      if ! git -C "$SIBLING" show "origin/Develop:scripts/$name" >"$WORK_DIR/$name" 2>/dev/null; then
        echo "FAIL: cannot read 'scripts/$name' at origin/Develop in $SIBLING" >&2
        exit 2
      fi
    done
  else
    echo "WARNING no origin/Develop in $SIBLING — falling back to its working tree" >&2
    for name in "${CANONICAL_SCRIPTS[@]}"; do
      if [[ ! -f "$SIBLING/scripts/$name" ]]; then
        echo "FAIL: $SIBLING has no scripts/$name to compare against" >&2
        exit 2
      fi
      cp "$SIBLING/scripts/$name" "$WORK_DIR/$name"
    done
  fi
fi

for name in "${CANONICAL_SCRIPTS[@]}"; do
  canonical="$CANONICAL_DIR/$name"
  copy="$COPIES_DIR/$name"

  if [[ ! -f "$canonical" ]]; then
    echo "FAIL: canonical copy not found: $canonical" >&2
    exit 2
  fi

  if cmp -s "$canonical" "$copy"; then
    echo "OK   $copy is byte-identical to the canonical NEAT-AI-core copy"
    continue
  fi

  echo "FAIL $copy has drifted from the canonical NEAT-AI-core copy" >&2
  echo "      scripts/$name has one home — scripts/$name on NEAT-AI-core" >&2
  echo "      Develop. Make the change there and let family-sync re-copy it;" >&2
  echo "      never edit the copy here (README.md, 'Canonical copies and family sync')." >&2
  diff -u "$canonical" "$copy" >&2 || true
  EXIT_CODE=1
done

exit "$EXIT_CODE"
