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
# With CANONICAL_DIR the copies in that directory are the authority. With no
# argument the copies are fetched from NEAT-AI-core `Develop` over https
# (`$CANONICAL_BASE_URL`) — the same source the family-sync job copies from, so
# a local run and CI compare against the same bytes. When that fetch fails the
# NEAT-AI-core sibling checkout is read instead, at `origin/Develop` and then
# its working tree, each announced with a warning: a sibling is only as current
# as its last fetch, so a fallback must never read as the ordinary path. With
# neither the run exits 2 — unverified is not a pass.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COPIES_DIR="${CANONICAL_COPIES_DIR:-$REPO_ROOT/scripts}"
CANONICAL_BASE_URL="${CANONICAL_BASE_URL:-https://raw.githubusercontent.com/stSoftwareAU/NEAT-AI-core/Develop/scripts}"
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
canonical copy, 1 when one has drifted, and 2 when the canonical copies
cannot be read at all.
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

# Fetch the canonical copies from NEAT-AI-core `Develop` into $1. Returns
# non-zero — leaving the caller to say so — when any of them cannot be had.
fetch_canonical() {
  local dir="$1" name
  command -v curl >/dev/null 2>&1 || return 1
  for name in "${CANONICAL_SCRIPTS[@]}"; do
    curl --fail --silent --show-error --location --retry 3 \
      --output "$dir/$name" "$CANONICAL_BASE_URL/$name" >/dev/null 2>&1 || return 1
    [[ -s "$dir/$name" ]] || return 1
  done
  return 0
}

# Extract the canonical copies from the sibling checkout into $1, preferring
# the branch that governs them over its working tree.
sibling_canonical() {
  local dir="$1" name
  if git -C "$SIBLING" rev-parse --verify --quiet origin/Develop >/dev/null; then
    echo "WARNING reading origin/Develop in $SIBLING, which is only as current as its last fetch" >&2
    for name in "${CANONICAL_SCRIPTS[@]}"; do
      if ! git -C "$SIBLING" show "origin/Develop:scripts/$name" >"$dir/$name" 2>/dev/null; then
        echo "FAIL: cannot read 'scripts/$name' at origin/Develop in $SIBLING" >&2
        exit 2
      fi
    done
    return 0
  fi

  echo "WARNING no origin/Develop in $SIBLING — falling back to its working tree" >&2
  for name in "${CANONICAL_SCRIPTS[@]}"; do
    if [[ ! -f "$SIBLING/scripts/$name" ]]; then
      echo "FAIL: $SIBLING has no scripts/$name to compare against" >&2
      exit 2
    fi
    cp "$SIBLING/scripts/$name" "$dir/$name"
  done
  return 0
}

# No canonical directory given: fetch from Develop, and fall back — loudly —
# to the sibling checkout only when that fetch cannot be made.
if [[ -z "$CANONICAL_DIR" ]]; then
  WORK_DIR="$(mktemp -d)"
  CANONICAL_DIR="$WORK_DIR"
  if fetch_canonical "$WORK_DIR"; then
    echo "INFO canonical copies fetched from $CANONICAL_BASE_URL"
  elif [[ -d "$SIBLING" ]]; then
    echo "WARNING could not fetch the canonical copies from $CANONICAL_BASE_URL" >&2
    sibling_canonical "$WORK_DIR"
  else
    echo "FAIL: could not fetch the canonical copies from $CANONICAL_BASE_URL, and no NEAT-AI-core sibling at $SIBLING to fall back to" >&2
    echo "      the copies are UNVERIFIED — this is not a pass" >&2
    exit 2
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
