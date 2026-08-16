#!/usr/bin/env bash
# Refuse a neat_ai_backpropagation crate version that is strictly behind base.
#
# Remotes rebuild from the crate version. A merge conflict that silently takes
# Develop's older token used to look like "already different" to the bump
# script and ship a downgrade — the same failure mode that forced NEAT-AI's
# floor restore. `sort -V` is the portable semver order used across the fleet
# (GNU + BSD). See issue #87.
#
# Usage:
#   check-crate-version-no-downgrade.sh [--base-ref REF]
#   check-crate-version-no-downgrade.sh --base-version V --head-version V
#
# Exit codes:
#   0  head >= base (equal or ahead)
#   1  head < base (downgrade)
#   2  usage / parse error
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST="$REPO_ROOT/backpropagation/Cargo.toml"
BASE_REF="origin/Develop"
BASE_VERSION=""
HEAD_VERSION=""
EXPLICIT=0

usage() {
  cat <<'EOF'
Usage: check-crate-version-no-downgrade.sh [--base-ref REF]
       check-crate-version-no-downgrade.sh --base-version V --head-version V

  --base-ref REF         Git ref whose Cargo.toml version is the floor
                         (default: origin/Develop). Reads the working-tree
                         manifest as head.
  --base-version V       Explicit base version (requires --head-version).
  --head-version V       Explicit head version (requires --base-version).

Exit 0 when head >= base, 1 when head is strictly behind base, 2 on error.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-ref)
      BASE_REF="${2:?--base-ref requires a value}"
      shift 2
      ;;
    --base-version)
      BASE_VERSION="${2:?--base-version requires a value}"
      EXPLICIT=1
      shift 2
      ;;
    --head-version)
      HEAD_VERSION="${2:?--head-version requires a value}"
      EXPLICIT=1
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "FAIL: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

read_version_from_text() {
  sed -n 's/^version *= *"\([^"]*\)".*/\1/p' | head -n 1
}

if [[ "$EXPLICIT" -eq 1 ]]; then
  if [[ -z "$BASE_VERSION" || -z "$HEAD_VERSION" ]]; then
    echo "FAIL: --base-version and --head-version must be used together" >&2
    exit 2
  fi
else
  cd "$REPO_ROOT"
  if [[ ! -f "$MANIFEST" ]]; then
    echo "FAIL: missing $MANIFEST" >&2
    exit 2
  fi
  if ! git show-ref --verify --quiet "$BASE_REF" \
    && ! git rev-parse --verify --quiet "$BASE_REF" >/dev/null; then
    echo "FAIL: base ref not found: $BASE_REF" >&2
    exit 2
  fi
  BASE_VERSION="$(git show "${BASE_REF}:backpropagation/Cargo.toml" 2>/dev/null | read_version_from_text || true)"
  HEAD_VERSION="$(read_version_from_text <"$MANIFEST")"
  if [[ -z "$HEAD_VERSION" ]]; then
    echo "FAIL: cannot read version from $MANIFEST" >&2
    exit 2
  fi
  if [[ -z "$BASE_VERSION" ]]; then
    echo "OK   no base version at $BASE_REF — skip downgrade check"
    exit 0
  fi
fi

for v in "$BASE_VERSION" "$HEAD_VERSION"; do
  if [[ ! "$v" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "FAIL: malformed version (expected x.y.z): $v" >&2
    exit 2
  fi
done

# Strictly behind when the version-sort lowest token is head and the two differ.
LOWEST="$(printf '%s\n%s\n' "$HEAD_VERSION" "$BASE_VERSION" | sort -V | head -n 1)"
if [[ "$LOWEST" == "$HEAD_VERSION" && "$HEAD_VERSION" != "$BASE_VERSION" ]]; then
  echo "FAIL: neat_ai_backpropagation ${HEAD_VERSION} is behind base ${BASE_VERSION}; crate versions must never go backwards" >&2
  exit 1
fi

echo "OK   crate version ${HEAD_VERSION} is not behind base ${BASE_VERSION}"
exit 0
