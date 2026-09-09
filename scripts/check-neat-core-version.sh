#!/usr/bin/env bash
# Gate against an unhandled breaking neat-core bump.
#
# this crate consumes neat-core through an UNPINNED `path` dependency that always
# tracks head (see `backpropagation/Cargo.toml`). The safeguard is this CI gate: it
# fails when the sibling neat-core presents a breaking bump this crate has not yet
# acknowledged, forcing a deliberate upgrade instead of tracking head blindly.
#
# Mechanism — version-baseline check:
#   * this crate records the last-handled neat-core version in the checked-in
#     `neat-core.expected-version` file.
#   * the gate reads neat-core's actual version from the sibling checkout at
#     `../NEAT-AI-core` ([workspace.package] version in its Cargo.toml).
#   * That version is taken from the branch that GOVERNS neat-core — its
#     default branch, `Develop` (`--core-ref`) — not from whatever branch the
#     sibling working tree happens to be parked on (issue #141). CI clones
#     neat-core at `Develop`, so the two agree there; locally the sibling is a
#     shared developer checkout that may sit on any unmerged branch, and an
#     unmerged bump is not a bump neat-core has presented.
#   * Divergence is never silent. When the working tree carries a DIFFERENT
#     version from the governing branch the gate WARNs, naming both — the
#     unpinned `path` dependency compiles the working tree, so a developer
#     building locally against an unmerged neat-core is told so even though
#     that unmerged bump does not fail the gate. Likewise, when the ref cannot
#     be resolved (the sibling is not a git checkout, or was cloned
#     `--single-branch` off another branch) the gate WARNs that it fell back to
#     the working tree rather than passing the fallback off as the ordinary
#     path. Either way it says which source it read.
#   * Known limitation: `origin/REF` is read as of the last fetch — this gate
#     does no network I/O. A sibling checkout that has not fetched for a while
#     is compared against a stale `Develop`. CI clones neat-core fresh on every
#     run, so the enforcing copy of this gate is never stale; a local pass is
#     advisory to that extent.
#   * The "breaking component" is the major for >= 1.0 releases and the minor
#     for pre-1.0 (0.x) releases, per SemVer. The gate FAILS when neat-core's
#     breaking component is greater than the recorded baseline; it PASSES on
#     patch-level drift (policy) and when the two match.
#
# "Handling" a breaking bump = a deliberate this crate PR that makes the
# corresponding code change AND bumps the recorded baseline to the new
# neat-core version.
#
# Usage:
#   check-neat-core-version.sh [--baseline PATH] [--core-manifest PATH] \
#                              [--core-ref REF]
#
# Defaults resolve the same paths Cargo uses: the baseline at the repo root and
# the neat-core workspace manifest at the sibling `../NEAT-AI-core/Cargo.toml`
# (the symlink CI creates makes this resolve on the runner too — see
# `.github/workflows/ci.yml`).
#
# Exit codes:
#   0  versions are compatible (match, patch drift, or core behind baseline)
#   1  breaking neat-core bump above the recorded baseline — gate fails
#   2  usage / parse error (missing file, malformed version)
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: check-neat-core-version.sh [--baseline PATH] [--core-manifest PATH]
                                  [--core-ref REF]

Options:
  --baseline PATH        File recording the last-handled neat-core version
                         (default: neat-core.expected-version at the repo root).
  --core-manifest PATH   neat-core workspace Cargo.toml carrying
                         [workspace.package] version (default: the sibling
                         ../NEAT-AI-core/Cargo.toml).
  --core-ref REF         Branch of the neat-core checkout that governs the
                         comparison (default: Develop). "origin/REF" wins over
                         a local "REF". Pass an empty value to read the
                         sibling working tree as it stands instead.
  -h, --help             Show this message.

Exits 0 when compatible, 1 on an unhandled breaking bump, 2 on a usage error.
EOF
}

BASELINE=""
CORE_MANIFEST=""
CORE_REF="Develop"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline)
      [[ $# -ge 2 ]] || { echo "Missing value for --baseline" >&2; usage >&2; exit 2; }
      BASELINE="$2"
      shift 2
      ;;
    --core-manifest)
      [[ $# -ge 2 ]] || { echo "Missing value for --core-manifest" >&2; usage >&2; exit 2; }
      CORE_MANIFEST="$2"
      shift 2
      ;;
    --core-ref)
      [[ $# -ge 2 ]] || { echo "Missing value for --core-ref" >&2; usage >&2; exit 2; }
      CORE_REF="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "$BASELINE" ]]; then
  BASELINE="$REPO_ROOT/neat-core.expected-version"
fi
if [[ -z "$CORE_MANIFEST" ]]; then
  CORE_MANIFEST="$REPO_ROOT/../NEAT-AI-core/Cargo.toml"
fi

if [[ ! -f "$BASELINE" ]]; then
  echo "FAIL: baseline file not found: $BASELINE" >&2
  exit 2
fi
if [[ ! -f "$CORE_MANIFEST" ]]; then
  echo "FAIL: neat-core manifest not found: $CORE_MANIFEST" >&2
  echo "      Is the sibling NEAT-AI-core clone present? CI clones + symlinks it." >&2
  exit 2
fi

# First non-comment, non-blank line of the baseline file is the version.
read_baseline_version() {
  awk '
    { sub(/#.*/, "") }            # strip inline comments
    { gsub(/[[:space:]]+/, "") }  # trim all whitespace
    NF { print; exit }            # first non-empty line wins
  ' "$BASELINE"
}

# Resolve which copy of the neat-core manifest the comparison reads.
#
# Default: the copy on the governing branch ($CORE_REF) of the sibling
# checkout, so a shared working tree parked on an unmerged branch cannot fail
# this repo's gate (issue #141). Falls back to the working-tree file when the
# sibling is not a git checkout or carries no branch of that name, and WARNs
# when it does, so the fallback is loud rather than passed off as the ordinary
# path.
CORE_SOURCE="working tree at $CORE_MANIFEST"
RESOLVED_MANIFEST="$CORE_MANIFEST"
TEMP_MANIFEST=""
# Invoked indirectly by the EXIT trap below.
# shellcheck disable=SC2329
cleanup() {
  if [[ -n "$TEMP_MANIFEST" ]]; then
    rm -f "$TEMP_MANIFEST"
  fi
}
trap cleanup EXIT

resolve_governing_manifest() {
  local core_dir base candidate ref prefix
  [[ -n "$CORE_REF" ]] || return 0

  core_dir="$(cd "$(dirname "$CORE_MANIFEST")" && pwd -P)"
  base="$(basename "$CORE_MANIFEST")"

  git -C "$core_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1 || return 0

  ref=""
  for candidate in "origin/$CORE_REF" "$CORE_REF"; do
    if git -C "$core_dir" rev-parse --verify --quiet "$candidate^{commit}" >/dev/null 2>&1; then
      ref="$candidate"
      break
    fi
  done
  if [[ -z "$ref" ]]; then
    echo "WARN neither 'origin/$CORE_REF' nor '$CORE_REF' resolves in $core_dir;" >&2
    echo "     falling back to the working tree, which may sit on an unmerged branch." >&2
    return 0
  fi

  prefix="$(git -C "$core_dir" rev-parse --show-prefix)"
  TEMP_MANIFEST="$(mktemp "${TMPDIR:-/tmp}/neat-core-manifest.XXXXXX")"
  # The ref resolves, so the manifest must be readable at it. A failure here is
  # a real fault (the file does not exist on that branch), not a fallback.
  if ! git -C "$core_dir" show "$ref:$prefix$base" >"$TEMP_MANIFEST" 2>/dev/null; then
    echo "FAIL: cannot read '$prefix$base' at ref '$ref' in $core_dir" >&2
    exit 2
  fi
  RESOLVED_MANIFEST="$TEMP_MANIFEST"
  CORE_SOURCE="ref '$ref' in $core_dir"
}

# Extract [workspace.package] version from the neat-core manifest. Only the
# version key that lives under the [workspace.package] table counts — a bare
# scan would also match dependency versions.
read_core_version() {
  awk '
    /^\[/ { in_wp = ($0 ~ /^\[workspace\.package\]/) }
    in_wp && /^[[:space:]]*version[[:space:]]*=/ {
      if (match($0, /"[^"]*"/)) {
        v = substr($0, RSTART + 1, RLENGTH - 2)
        print v
        exit
      }
    }
  ' "$1"
}

# Validate X.Y.Z (optionally with a -prerelease/+build suffix we ignore) and
# echo the bare "major minor patch" triple. Returns non-zero when malformed.
parse_semver() {
  local raw="$1" core
  core="${raw%%[-+]*}"   # drop pre-release / build metadata
  if [[ ! "$core" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    return 1
  fi
  local major minor patch
  # Scope IFS to this single read rather than tampering with it globally.
  IFS='.' read -r major minor patch <<<"$core"
  echo "$major $minor $patch"
}

baseline_raw="$(read_baseline_version)"
if [[ -z "$baseline_raw" ]]; then
  echo "FAIL: baseline file is empty: $BASELINE" >&2
  exit 2
fi

resolve_governing_manifest
echo "INFO neat-core version read from $CORE_SOURCE"

core_raw="$(read_core_version "$RESOLVED_MANIFEST")"
if [[ -z "$core_raw" ]]; then
  echo "FAIL: no [workspace.package] version found in $CORE_SOURCE" >&2
  exit 2
fi

# The `path` dependency compiles the WORKING TREE, not the governing branch.
# When the two differ, say so: the gate deliberately does not fail on an
# unmerged bump, but it must not let a local build against one look like a
# build against the version it just reported OK.
if [[ -n "$TEMP_MANIFEST" ]]; then
  worktree_raw="$(read_core_version "$CORE_MANIFEST")"
  if [[ -n "$worktree_raw" && "$worktree_raw" != "$core_raw" ]]; then
    echo "WARN the sibling working tree is at $worktree_raw, not $core_raw." >&2
    echo "     The unpinned path dependency compiles the working tree, so a local" >&2
    echo "     build here does NOT match the version this gate just checked." >&2
  fi
fi

if ! baseline_parts="$(parse_semver "$baseline_raw")"; then
  echo "FAIL: malformed baseline version '$baseline_raw' in $BASELINE (expected X.Y.Z)" >&2
  exit 2
fi
if ! core_parts="$(parse_semver "$core_raw")"; then
  echo "FAIL: malformed neat-core version '$core_raw' in $CORE_SOURCE (expected X.Y.Z)" >&2
  exit 2
fi

read -r b_major b_minor b_patch <<<"$baseline_parts"
read -r c_major c_minor c_patch <<<"$core_parts"
: "$b_patch" "$c_patch"  # patch components are informational only

remediation() {
  cat >&2 <<EOF
       neat-core has presented a breaking bump this crate has not handled.
       To clear this gate, in a single deliberate PR:
         1. Update backpropagation for the breaking neat-core change.
         2. Bump the recorded baseline in neat-core.expected-version to $core_raw.
EOF
}

# Breaking-bump decision (SemVer): the major signals breaking changes once a
# crate reaches 1.0; before that, the minor carries that role.
if (( c_major > b_major )); then
  echo "FAIL: breaking neat-core bump: $core_raw exceeds handled baseline $baseline_raw (major increased)" >&2
  remediation
  exit 1
fi

if (( c_major == b_major )); then
  if (( b_major == 0 )); then
    # Pre-1.0: the minor is the breaking component.
    if (( c_minor > b_minor )); then
      echo "FAIL: breaking neat-core bump: $core_raw exceeds handled baseline $baseline_raw (pre-1.0 minor increased)" >&2
      remediation
      exit 1
    fi
    if (( c_minor < b_minor )); then
      echo "OK   neat-core $core_raw is behind handled baseline $baseline_raw (no breaking bump)"
      exit 0
    fi
    echo "OK   neat-core $core_raw matches handled baseline $baseline_raw (patch-level drift allowed)"
    exit 0
  fi
  # >= 1.0: same major — minor/patch drift is additive, never breaking.
  echo "OK   neat-core $core_raw within handled baseline $baseline_raw major line (patch-level drift allowed)"
  exit 0
fi

# c_major < b_major: neat-core sits below the baseline major — not a breaking
# bump scorer needs to act on.
echo "OK   neat-core $core_raw is behind handled baseline $baseline_raw (no breaking bump)"
exit 0
