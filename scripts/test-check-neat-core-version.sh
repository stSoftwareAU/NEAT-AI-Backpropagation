#!/usr/bin/env bash
# WHAT-tests for the neat-core breaking-bump gate (Issue #141).
#
# Covers the version-comparison contract:
#   * equal / patch drift / core behind → pass
#   * pre-1.0 minor bump, major bump    → fail (unhandled breaking bump)
#   * malformed or missing inputs       → usage error
# And the ref-resolution contract added for Issue #141: the version is read
# from the branch that *governs* neat-core (default `Develop`), not from
# whatever branch a shared sibling checkout happens to be parked on.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKER="$SCRIPT_DIR/check-neat-core-version.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

PASSED=0
FAILED=0

if [[ ! -x "$CHECKER" ]]; then
  echo "FAIL: checker not found or not executable: $CHECKER" >&2
  exit 2
fi

LAST_OUTPUT=""

# expect_exit DESCRIPTION EXPECTED_CODE COMMAND...
expect_exit() {
  local description="$1" expected="$2"
  shift 2
  local status=0
  LAST_OUTPUT="$("$@" 2>&1)" || status=$?
  if [[ "$status" -ne "$expected" ]]; then
    echo "FAIL $description: expected exit $expected, got $status" >&2
    printf '%s\n' "$LAST_OUTPUT" >&2
    FAILED=$((FAILED + 1))
    return 1
  fi
  echo "OK   $description"
  PASSED=$((PASSED + 1))
}

# expect_output DESCRIPTION PATTERN — asserts on the last captured output.
expect_output() {
  local description="$1" pattern="$2"
  if ! printf '%s\n' "$LAST_OUTPUT" | grep -qE "$pattern"; then
    echo "FAIL $description: output did not match /$pattern/" >&2
    printf '%s\n' "$LAST_OUTPUT" >&2
    FAILED=$((FAILED + 1))
    return 1
  fi
  echo "OK   $description"
  PASSED=$((PASSED + 1))
}

# write_baseline NAME VERSION → echoes the baseline file path.
write_baseline() {
  local path="$WORK_DIR/baseline-$1"
  cat >"$path" <<EOF
# Last-handled neat-core version.
#
# Comment block the real file carries; the first bare version wins.
$2
EOF
  printf '%s\n' "$path"
}

# write_manifest PATH VERSION — a neat-core workspace manifest.
write_manifest() {
  mkdir -p "$(dirname "$1")"
  cat >"$1" <<EOF
[workspace]
members = ["neat-core"]

[workspace.package]
version = "$2"
edition = "2024"

[workspace.dependencies]
serde = { version = "9.9.9" }
EOF
}

BASE_0_11_2="$(write_baseline 0-11-2 0.11.2)"
BASE_1_2_3="$(write_baseline 1-2-3 1.2.3)"

# ---------------------------------------------------------------------------
# Version-comparison contract (--core-ref '' pins the read to the given file).
# ---------------------------------------------------------------------------
MANIFESTS="$WORK_DIR/manifests"
write_manifest "$MANIFESTS/equal/Cargo.toml" 0.11.2
write_manifest "$MANIFESTS/patch/Cargo.toml" 0.11.9
write_manifest "$MANIFESTS/minor/Cargo.toml" 0.12.0
write_manifest "$MANIFESTS/major/Cargo.toml" 1.0.0
write_manifest "$MANIFESTS/behind/Cargo.toml" 0.10.7
write_manifest "$MANIFESTS/prerelease/Cargo.toml" 0.11.4-rc.1
write_manifest "$MANIFESTS/post-1-0-minor/Cargo.toml" 1.3.0
write_manifest "$MANIFESTS/post-1-0-major/Cargo.toml" 2.0.0
printf '[workspace]\nmembers = ["neat-core"]\n' >"$MANIFESTS/no-version.toml"
printf '[package]\nversion = "0.11.2"\n' >"$MANIFESTS/wrong-table.toml"
write_manifest "$MANIFESTS/malformed/Cargo.toml" 0.11

# Invoked indirectly, as the command expect_exit runs.
# shellcheck disable=SC2329
check() { "$CHECKER" --core-ref "" --baseline "$1" --core-manifest "$2"; }

expect_exit "equal versions pass" 0 check "$BASE_0_11_2" "$MANIFESTS/equal/Cargo.toml" || true
expect_exit "patch drift passes" 0 check "$BASE_0_11_2" "$MANIFESTS/patch/Cargo.toml" || true
expect_exit "core behind baseline passes" 0 check "$BASE_0_11_2" "$MANIFESTS/behind/Cargo.toml" || true
expect_exit "pre-release suffix ignored on patch drift" 0 check "$BASE_0_11_2" "$MANIFESTS/prerelease/Cargo.toml" || true

expect_exit "pre-1.0 minor bump fails" 1 check "$BASE_0_11_2" "$MANIFESTS/minor/Cargo.toml" || true
expect_output "minor-bump failure names the remediation version" 'Bump the recorded baseline in neat-core\.expected-version to 0\.12\.0' || true
expect_exit "major bump fails" 1 check "$BASE_0_11_2" "$MANIFESTS/major/Cargo.toml" || true

expect_exit "post-1.0 minor bump passes (additive)" 0 check "$BASE_1_2_3" "$MANIFESTS/post-1-0-minor/Cargo.toml" || true
expect_exit "post-1.0 major bump fails" 1 check "$BASE_1_2_3" "$MANIFESTS/post-1-0-major/Cargo.toml" || true

expect_exit "missing manifest is a usage error" 2 check "$BASE_0_11_2" "$WORK_DIR/absent.toml" || true
expect_exit "missing baseline is a usage error" 2 check "$WORK_DIR/absent-baseline" "$MANIFESTS/equal/Cargo.toml" || true
expect_exit "manifest without a workspace version is a usage error" 2 check "$BASE_0_11_2" "$MANIFESTS/no-version.toml" || true
expect_exit "version outside [workspace.package] is not accepted" 2 check "$BASE_0_11_2" "$MANIFESTS/wrong-table.toml" || true
expect_exit "malformed core version is a usage error" 2 check "$BASE_0_11_2" "$MANIFESTS/malformed/Cargo.toml" || true
expect_exit "malformed baseline version is a usage error" 2 \
  check "$(write_baseline malformed 0.11)" "$MANIFESTS/equal/Cargo.toml" || true
expect_exit "empty baseline is a usage error" 2 \
  check "$(: >"$WORK_DIR/baseline-empty"; printf '%s\n' "$WORK_DIR/baseline-empty")" "$MANIFESTS/equal/Cargo.toml" || true
expect_exit "unknown argument is a usage error" 2 "$CHECKER" --nope || true
expect_exit "--core-ref without a value is a usage error" 2 "$CHECKER" --core-ref || true
expect_exit "--baseline without a value is a usage error" 2 "$CHECKER" --baseline || true
expect_exit "--core-manifest without a value is a usage error" 2 "$CHECKER" --core-manifest || true
expect_exit "--help succeeds" 0 "$CHECKER" --help || true

# ---------------------------------------------------------------------------
# Ref resolution (Issue #141). A shared sibling checkout parked on an unmerged
# feature branch must not fail this repo's gate: the governing branch is what
# CI evaluates, so it is what the gate reads.
# ---------------------------------------------------------------------------
UPSTREAM="$WORK_DIR/core-upstream"
mkdir -p "$UPSTREAM"
write_manifest "$UPSTREAM/Cargo.toml" 0.11.3
git -C "$UPSTREAM" init -q -b Develop
git -C "$UPSTREAM" config user.name test
git -C "$UPSTREAM" config user.email test@example.com
git -C "$UPSTREAM" add Cargo.toml
git -C "$UPSTREAM" commit -q -m "Develop at 0.11.3"

# Clone, then park the working tree on an unmerged branch carrying 0.12.0.
SIBLING="$WORK_DIR/core-sibling"
git clone -q "$UPSTREAM" "$SIBLING"
git -C "$SIBLING" config user.name test
git -C "$SIBLING" config user.email test@example.com
git -C "$SIBLING" checkout -q -b milestone/breaking
write_manifest "$SIBLING/Cargo.toml" 0.12.0
git -C "$SIBLING" add Cargo.toml
git -C "$SIBLING" commit -q -m "unmerged breaking bump to 0.12.0"

expect_exit "parked feature branch does not fail the gate" 0 \
  "$CHECKER" --baseline "$BASE_0_11_2" --core-manifest "$SIBLING/Cargo.toml" || true
expect_output "gate reports the governing ref it read" 'origin/Develop' || true
expect_output "gate reports the governing branch version" '0\.11\.3' || true

expect_output "divergence between the ref and the working tree is announced" \
  'working tree is at 0\.12\.0, not 0\.11\.3' || true

expect_exit "--core-ref '' still reads the parked working tree" 1 \
  "$CHECKER" --core-ref "" --baseline "$BASE_0_11_2" --core-manifest "$SIBLING/Cargo.toml" || true

# A non-default --core-ref selects that branch instead.
expect_exit "a non-default --core-ref selects that branch" 1 \
  "$CHECKER" --core-ref milestone/breaking --baseline "$BASE_0_11_2" \
  --core-manifest "$SIBLING/Cargo.toml" || true
expect_output "non-default ref is named in the output" "milestone/breaking" || true

# A real breaking bump on the governing branch must still fail.
write_manifest "$UPSTREAM/Cargo.toml" 0.13.0
git -C "$UPSTREAM" add Cargo.toml
git -C "$UPSTREAM" commit -q -m "Develop breaking bump to 0.13.0"
git -C "$SIBLING" fetch -q origin
expect_exit "breaking bump on the governing branch still fails" 1 \
  "$CHECKER" --baseline "$BASE_0_11_2" --core-manifest "$SIBLING/Cargo.toml" || true
expect_output "failure names the governing-branch version" '0\.13\.0' || true

# No remote: a local branch of that name governs.
LOCAL_ONLY="$WORK_DIR/core-local"
mkdir -p "$LOCAL_ONLY"
write_manifest "$LOCAL_ONLY/Cargo.toml" 0.11.3
git -C "$LOCAL_ONLY" init -q -b Develop
git -C "$LOCAL_ONLY" config user.name test
git -C "$LOCAL_ONLY" config user.email test@example.com
git -C "$LOCAL_ONLY" add Cargo.toml
git -C "$LOCAL_ONLY" commit -q -m "Develop at 0.11.3"
git -C "$LOCAL_ONLY" checkout -q -b spike
write_manifest "$LOCAL_ONLY/Cargo.toml" 0.12.0
git -C "$LOCAL_ONLY" add Cargo.toml
git -C "$LOCAL_ONLY" commit -q -m "spike 0.12.0"
expect_exit "local governing branch is used when there is no remote" 0 \
  "$CHECKER" --baseline "$BASE_0_11_2" --core-manifest "$LOCAL_ONLY/Cargo.toml" || true

# Manifest nested below the repository root resolves via its git-relative path.
NESTED="$WORK_DIR/core-nested"
mkdir -p "$NESTED/workspace"
write_manifest "$NESTED/workspace/Cargo.toml" 0.11.3
git -C "$NESTED" init -q -b Develop
git -C "$NESTED" config user.name test
git -C "$NESTED" config user.email test@example.com
git -C "$NESTED" add workspace/Cargo.toml
git -C "$NESTED" commit -q -m "Develop at 0.11.3"
git -C "$NESTED" checkout -q -b spike
write_manifest "$NESTED/workspace/Cargo.toml" 0.12.0
git -C "$NESTED" add workspace/Cargo.toml
git -C "$NESTED" commit -q -m "spike 0.12.0"
expect_exit "nested manifest resolves against the governing branch" 0 \
  "$CHECKER" --baseline "$BASE_0_11_2" --core-manifest "$NESTED/workspace/Cargo.toml" || true

# CI shape: a detached checkout of Develop with no branch of that name. The
# gate falls back to the working tree and says so, rather than failing.
DETACHED="$WORK_DIR/core-detached"
git clone -q "$UPSTREAM" "$DETACHED"
git -C "$DETACHED" checkout -q --detach HEAD
git -C "$DETACHED" remote remove origin
git -C "$DETACHED" branch -q -D Develop
write_manifest "$DETACHED/Cargo.toml" 0.11.3
expect_exit "unresolvable ref falls back to the working tree" 0 \
  "$CHECKER" --baseline "$BASE_0_11_2" --core-manifest "$DETACHED/Cargo.toml" || true
expect_output "fallback announces the working-tree source" 'working tree' || true
expect_output "fallback warns rather than passing itself off as normal" \
  'WARN neither .origin/Develop. nor .Develop. resolves' || true

# A manifest that exists in the working tree but not on the governing branch is
# a real fault, not another fallback: the ref resolved, the file did not.
ABSENT_ON_REF="$WORK_DIR/core-absent-on-ref"
mkdir -p "$ABSENT_ON_REF"
printf 'placeholder\n' >"$ABSENT_ON_REF/README.md"
git -C "$ABSENT_ON_REF" init -q -b Develop
git -C "$ABSENT_ON_REF" config user.name test
git -C "$ABSENT_ON_REF" config user.email test@example.com
git -C "$ABSENT_ON_REF" add README.md
git -C "$ABSENT_ON_REF" commit -q -m "Develop without a manifest"
write_manifest "$ABSENT_ON_REF/Cargo.toml" 0.12.0
expect_exit "manifest missing on the governing branch fails loudly" 2 \
  "$CHECKER" --baseline "$BASE_0_11_2" --core-manifest "$ABSENT_ON_REF/Cargo.toml" || true
expect_output "the loud failure names the ref it could not read" \
  "cannot read 'Cargo.toml' at ref 'Develop'" || true

# Not a git checkout at all: the manifest on disk is all there is.
expect_exit "non-git manifest falls back to the working tree" 1 \
  "$CHECKER" --baseline "$BASE_0_11_2" --core-manifest "$MANIFESTS/minor/Cargo.toml" || true

echo
echo "Passed: $PASSED  Failed: $FAILED"
if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
exit 0
