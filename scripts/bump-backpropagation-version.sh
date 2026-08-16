#!/usr/bin/env bash
# Bump backpropagation/Cargo.toml patch version when backpropagation/src/ changed vs a base ref.
#
# Mirrors GRQ-taxation's version-increment job / runlib.sh contract: remotes
# rebuild when Cargo.toml version changes. Idempotent — skips when the PR
# branch is already *ahead* of base or an auto-increment commit exists.
# A head version strictly *behind* base fails (exit 2); treating a downgrade
# as "already different" used to ship silent trainDir / FFI rebuilds of an
# older crate (issue #87).
#
# Exit codes:
#   0  version bumped (or --check would bump)
#   1  no bump needed
#   2  usage / parse error / version downgrade vs base
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST=""
LOCKFILE=""
SRC_PATH="backpropagation/src"
BASE_REF="origin/Develop"
CHECK_ONLY=0
COMMIT_SUBJECT="chore: auto-increment versions for changed projects"

usage() {
  cat <<'EOF'
Usage: bump-backpropagation-version.sh [--base-ref REF] [--repo-root DIR] [--check]

  --base-ref REF   Git ref to diff against (default: origin/Develop).
  --repo-root DIR  Repository root (default: parent of scripts/).
  --check          Report whether a bump is needed; do not modify files.

Exit 0 when a bump is applied (or would be with --check), 1 when skipped
(equal-or-ahead with nothing to do), 2 on error or when head is strictly
behind base.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-ref)
      BASE_REF="${2:?--base-ref requires a value}"
      shift 2
      ;;
    --repo-root)
      REPO_ROOT="$(cd "${2:?--repo-root requires a value}" && pwd)"
      shift 2
      ;;
    --check)
      CHECK_ONLY=1
      shift
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

MANIFEST="$REPO_ROOT/backpropagation/Cargo.toml"
LOCKFILE="$REPO_ROOT/Cargo.lock"

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

read_version() {
  local file="$1"
  sed -n 's/^version *= *"\([^"]*\)".*/\1/p' "$file" | head -n 1
}

BASE_VERSION="$(git show "${BASE_REF}:backpropagation/Cargo.toml" 2>/dev/null | sed -n 's/^version *= *"\([^"]*\)".*/\1/p' | head -n 1 || true)"
CURRENT_VERSION="$(read_version "$MANIFEST")"

if [[ -z "$CURRENT_VERSION" ]]; then
  echo "FAIL: cannot read version from $MANIFEST" >&2
  exit 2
fi

# Refuse downgrades before any skip path (auto-increment commit included).
if [[ -n "$BASE_VERSION" ]]; then
  downgrade_status=0
  "$SCRIPT_DIR/check-crate-version-no-downgrade.sh" \
    --base-version "$BASE_VERSION" --head-version "$CURRENT_VERSION" \
    || downgrade_status=$?
  if [[ "$downgrade_status" -eq 1 ]]; then
    exit 2
  fi
  if [[ "$downgrade_status" -ne 0 ]]; then
    exit "$downgrade_status"
  fi
fi

# Skip when an auto-increment commit is already on this branch.
if git log --oneline "${BASE_REF}..HEAD" --grep="$COMMIT_SUBJECT" | grep -q .; then
  echo "OK   auto-increment commit already present vs $BASE_REF — skip"
  exit 1
fi

# Ahead of base: accept without forcing another bump. Equal may still patch-bump.
if [[ -n "$BASE_VERSION" && "$CURRENT_VERSION" != "$BASE_VERSION" ]]; then
  echo "OK   version already ahead of base ($BASE_VERSION -> $CURRENT_VERSION) — skip"
  exit 1
fi

if git diff --quiet "${BASE_REF}...HEAD" -- "$SRC_PATH"; then
  echo "OK   no changes under $SRC_PATH vs $BASE_REF — skip"
  exit 1
fi

IFS='.' read -r major minor patch <<<"$CURRENT_VERSION"
major=${major:-0}
minor=${minor:-0}
patch=${patch:-0}
if ! [[ "$patch" =~ ^[0-9]+$ ]]; then
  echo "FAIL: malformed patch in version '$CURRENT_VERSION'" >&2
  exit 2
fi
NEW_VERSION="$major.$minor.$((patch + 1))"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  echo "WOULD bump $CURRENT_VERSION -> $NEW_VERSION (src changes vs $BASE_REF)"
  exit 0
fi

ESCAPED_CURRENT="$(printf '%s' "$CURRENT_VERSION" | sed 's/[\/.^$*+?()[\]{}|]/\\&/g')"
sed -i.bak "s/^version = \"$ESCAPED_CURRENT\"/version = \"$NEW_VERSION\"/" "$MANIFEST"
rm -f "${MANIFEST}.bak"

if [[ -f "$LOCKFILE" ]]; then
  # Keep Cargo.lock's workspace package stanza in sync (path crate version).
  python3 - "$LOCKFILE" "$NEW_VERSION" <<'PY'
import pathlib
import re
import sys

lock = pathlib.Path(sys.argv[1])
new = sys.argv[2]
text = lock.read_text()
updated, n = re.subn(
    r'(name = "neat_ai_backpropagation"\nversion = ")[^"]+(")',
    rf"\g<1>{new}\2",
    text,
    count=1,
)
if n != 1:
    sys.stderr.write(f"FAIL: expected one neat_ai_backpropagation version in {lock}, found {n}\n")
    sys.exit(2)
lock.write_text(updated)
PY
fi

echo "OK   bumped neat_ai_backpropagation $CURRENT_VERSION -> $NEW_VERSION"
exit 0
