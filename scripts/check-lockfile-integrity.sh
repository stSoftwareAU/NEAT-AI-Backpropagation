#!/usr/bin/env bash
# Verify Cargo.lock against what crates.io actually published (Issue #148).
#
# A lockfile's `dependencies = [...]` list is meant to mirror the resolved
# crate's own manifest, and nothing reports it when the two diverge. `cargo`
# does not compile a substituted entry — it re-resolves and silently rewrites
# the lockfile back, or under `--locked` refuses with a generic "cannot update
# the lock file" error naming no crate. So a substituted sub-dependency and a
# legitimate upstream rename look identical to a reviewer, which is how issue
# #148 spent a triage cycle on `serde_json` swapping `ryu` for `zmij`.
#
# This gate fetches the crates.io sparse index for every registry package in
# the lockfile and hands the snapshot to `scripts/lockfile_integrity.py`, which
# asserts the checksum and the recorded dependency names against the published
# record. See that script's docstring for the exact rules.
#
# Exit codes: 0 verified, 1 a violation, 2 the lockfile or the index could not
# be read. An unreachable index is exit 2, never a pass — an unverified
# lockfile must not look like a verified one.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VERIFIER="$SCRIPT_DIR/lockfile_integrity.py"
INDEX_BASE_URL="${CRATES_INDEX_BASE_URL:-https://index.crates.io}"

LOCKFILE="$REPO_ROOT/Cargo.lock"
INDEX_DIR=""

usage() {
  cat <<'EOF'
Usage: check-lockfile-integrity.sh [--lockfile PATH] [--index-dir DIR]

  --lockfile PATH   Lockfile to verify (default: the repository's Cargo.lock).
  --index-dir DIR   Verify against an existing sparse-index snapshot instead of
                    fetching one. A snapshot is a directory of files named after
                    each crate, one index JSON object per line.

Exits 0 when every package matches the registry, 1 on a violation, and 2 when
the lockfile or the crates.io index cannot be read.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    --lockfile)
      [[ $# -ge 2 ]] || {
        echo "FAIL: --lockfile needs a path" >&2
        exit 2
      }
      LOCKFILE="$2"
      shift 2
      ;;
    --index-dir)
      [[ $# -ge 2 ]] || {
        echo "FAIL: --index-dir needs a path" >&2
        exit 2
      }
      INDEX_DIR="$2"
      shift 2
      ;;
    *)
      echo "FAIL: unknown argument '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$VERIFIER" ]]; then
  echo "FAIL: verifier not found: $VERIFIER" >&2
  exit 2
fi
if [[ ! -f "$LOCKFILE" ]]; then
  echo "FAIL: lockfile not found: $LOCKFILE" >&2
  exit 2
fi
if ! command -v python3 &>/dev/null; then
  echo "FAIL: python3 is required to verify the lockfile" >&2
  exit 2
fi

# Sparse-index path for a crate: 1/x, 2/xy, 3/x/xyz, else ab/cd/abcdef.
index_path() {
  local name
  name="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "${#name}" in
    1) printf '1/%s' "$name" ;;
    2) printf '2/%s' "$name" ;;
    3) printf '3/%s/%s' "${name:0:1}" "$name" ;;
    *) printf '%s/%s/%s' "${name:0:2}" "${name:2:2}" "$name" ;;
  esac
}

if [[ -z "$INDEX_DIR" ]]; then
  if ! command -v curl &>/dev/null; then
    echo "FAIL: curl is required to fetch the crates.io index" >&2
    exit 2
  fi

  INDEX_DIR="$(mktemp -d)"
  trap 'rm -rf "$INDEX_DIR"' EXIT

  mapfile -t CRATES < <(python3 "$VERIFIER" --lockfile "$LOCKFILE" --list-crates)
  if [[ "${#CRATES[@]}" -eq 0 ]]; then
    echo "FAIL: no registry packages found in $LOCKFILE" >&2
    exit 2
  fi

  echo "Fetching ${#CRATES[@]} crates.io index entries..."
  CURL_ARGS=()
  for crate in "${CRATES[@]}"; do
    CURL_ARGS+=(-o "$INDEX_DIR/$(printf '%s' "$crate" | tr '[:upper:]' '[:lower:]')")
    CURL_ARGS+=(--url "$INDEX_BASE_URL/$(index_path "$crate")")
  done

  # `--parallel` needs curl >= 7.66; fall back to sequential fetches without it.
  PARALLEL_ARGS=()
  if curl --help all 2>/dev/null | grep -q -- '--parallel'; then
    PARALLEL_ARGS=(--parallel --parallel-max 8)
  fi

  if ! curl --fail --silent --show-error --location --retry 2 --max-time 60 \
    ${PARALLEL_ARGS[@]+"${PARALLEL_ARGS[@]}"} "${CURL_ARGS[@]}"; then
    echo "FAIL: could not fetch the crates.io index from $INDEX_BASE_URL" >&2
    echo "      the lockfile is UNVERIFIED — this is not a pass" >&2
    exit 2
  fi
fi

python3 "$VERIFIER" --lockfile "$LOCKFILE" --index-dir "$INDEX_DIR"
