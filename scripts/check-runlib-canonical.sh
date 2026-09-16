#!/usr/bin/env bash
# Fail when scripts/runlib.sh has drifted from its canonical NEAT-AI-core copy
# (issue #152 / core #680).
#
# The family-sync workflow re-copies the script on same-repo PRs, but it is
# skipped on fork PRs and could be disabled — and a downstream edit is exactly
# what the copy contract forbids. This gate reads the content rather than the
# workflow's shape, so drift fails loud wherever it came from.
#
# Usage: check-runlib-canonical.sh [CANONICAL_PATH]
#
# With CANONICAL_PATH the file at that path is the authority (CI passes the
# NEAT-AI-core checkout its Rust jobs already make). With no argument the
# NEAT-AI-core sibling checkout is read at `origin/Develop`, falling back — with
# a warning, never silently — to that checkout's working tree.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COPY="${RUNLIB_COPY:-$REPO_ROOT/scripts/runlib.sh}"
SIBLING="${NEAT_CORE_DIR:-$REPO_ROOT/../NEAT-AI-core}"
CANONICAL="${1:-}"
WORK_DIR=""

usage() {
  cat <<'USAGE'
Usage: check-runlib-canonical.sh [CANONICAL_PATH]

Exits 0 when scripts/runlib.sh is byte-identical to the canonical
NEAT-AI-core copy, 1 when it has drifted, and 2 when the canonical copy
cannot be read.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

trap '[[ -z "$WORK_DIR" ]] || rm -rf "$WORK_DIR"' EXIT

if [[ ! -f "$COPY" ]]; then
  echo "FAIL: no copy to check at $COPY" >&2
  exit 2
fi

if [[ -z "$CANONICAL" ]]; then
  if [[ ! -d "$SIBLING" ]]; then
    echo "FAIL: no NEAT-AI-core sibling at $SIBLING — pass the canonical scripts/runlib.sh as an argument" >&2
    exit 2
  fi
  WORK_DIR="$(mktemp -d)"
  CANONICAL="$WORK_DIR/runlib.sh"
  if git -C "$SIBLING" rev-parse --verify --quiet origin/Develop >/dev/null; then
    echo "INFO canonical copy read from origin/Develop in $SIBLING"
    git -C "$SIBLING" show origin/Develop:scripts/runlib.sh >"$CANONICAL"
  else
    echo "WARNING no origin/Develop in $SIBLING — falling back to its working tree" >&2
    if [[ ! -f "$SIBLING/scripts/runlib.sh" ]]; then
      echo "FAIL: $SIBLING has no scripts/runlib.sh to compare against" >&2
      exit 2
    fi
    cp "$SIBLING/scripts/runlib.sh" "$CANONICAL"
  fi
fi

if [[ ! -f "$CANONICAL" ]]; then
  echo "FAIL: canonical copy not found: $CANONICAL" >&2
  exit 2
fi

if cmp -s "$CANONICAL" "$COPY"; then
  echo "OK   $COPY is byte-identical to the canonical NEAT-AI-core copy"
  exit 0
fi

echo "FAIL $COPY has drifted from the canonical NEAT-AI-core copy" >&2
echo "      scripts/runlib.sh has one home — scripts/runlib.sh on NEAT-AI-core" >&2
echo "      Develop. Make the change there and let family-sync re-copy it;" >&2
echo "      never edit the copy here (README.md, 'Canonical copy and family sync')." >&2
diff -u "$CANONICAL" "$COPY" >&2 || true
exit 1
