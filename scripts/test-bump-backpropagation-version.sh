#!/usr/bin/env bash
# WHAT-tests for the auto version bump (issue #95).
#
# Unattended machines rebuild `neat_ai_backpropagation` only when the crate
# version in `backpropagation/Cargo.toml` changes (runlib.sh `ensure_lib_built`
# compares an installed version marker against `cargo metadata`). Every change
# that alters the built artefact must therefore bump the version — not just
# changes under `backpropagation/src/`.
#
# Covers: each build-affecting path triggers a bump, non-build-affecting paths
# do not, and a real (non `--check`) run rewrites both the manifest and the
# lockfile exactly once.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUMPER="$SCRIPT_DIR/bump-backpropagation-version.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

PASSED=0
FAILED=0

if [[ ! -x "$BUMPER" ]]; then
  echo "FAIL: bumper not found or not executable: $BUMPER" >&2
  exit 2
fi

# expect_exit DESCRIPTION EXPECTED_CODE COMMAND...
expect_exit() {
  local description="$1" expected="$2"
  shift 2
  local output status=0
  output="$("$@" 2>&1)" || status=$?
  if [[ "$status" -ne "$expected" ]]; then
    echo "FAIL $description: expected exit $expected, got $status" >&2
    printf '%s\n' "$output" >&2
    FAILED=$((FAILED + 1))
    return
  fi
  echo "OK   $description"
  PASSED=$((PASSED + 1))
}

# expect_file_contains DESCRIPTION FILE NEEDLE
expect_file_contains() {
  local description="$1" file="$2" needle="$3"
  if grep -Fq "$needle" "$file"; then
    echo "OK   $description"
    PASSED=$((PASSED + 1))
  else
    echo "FAIL $description: '$needle' not found in $file" >&2
    FAILED=$((FAILED + 1))
  fi
}

FIXTURE="$WORK_DIR/repo"
mkdir -p "$FIXTURE/backpropagation/src" \
  "$FIXTURE/backpropagation/tests" \
  "$FIXTURE/include" \
  "$FIXTURE/.cargo"
cd "$FIXTURE"
git init -q -b Develop
git config user.name "test"
git config user.email "test@example.com"

cat >backpropagation/Cargo.toml <<'EOF'
[package]
name = "neat_ai_backpropagation"
version = "0.1.10"
edition = "2024"

[dependencies]
serde_json = "1"
EOF
cat >Cargo.toml <<'EOF'
[workspace]
members = ["backpropagation"]
resolver = "2"

[profile.release]
opt-level = 3
EOF
cat >Cargo.lock <<'EOF'
version = 4

[[package]]
name = "neat_ai_backpropagation"
version = "0.1.10"
dependencies = ["serde_json"]
EOF
cat >.cargo/config.toml <<'EOF'
[target.'cfg(not(target_arch = "wasm32"))']
rustflags = ["-C", "target-cpu=native"]
EOF
cat >rust-toolchain.toml <<'EOF'
[toolchain]
channel = "stable"
EOF
echo '// stub' >backpropagation/src/lib.rs
echo '// header stub' >include/neat_ai_backpropagation.h
echo '// test stub' >backpropagation/tests/smoke.rs
echo '# Fixture' >README.md
git add -A
git commit -q -m "base Develop 0.1.10"

# branch_with NAME FILE APPENDED_LINE — fresh branch off Develop touching FILE.
branch_with() {
  local name="$1" file="$2" line="$3"
  git checkout -q Develop
  git branch -q -D "$name" 2>/dev/null || true
  git checkout -q -b "$name"
  printf '%s\n' "$line" >>"$file"
  git add -A
  git commit -q -m "touch $file"
}

# One `--check` case per row: NAME|FILE|APPENDED LINE|EXPECTED EXIT|DESCRIPTION.
# `|` separates because the appended lines carry spaces, quotes, colons and
# equals signs. Expected exit 0 = the change must bump the version (a
# build-affecting path), 1 = it must not.
CHECK_CASES=$(
  cat <<'EOF'
lib-src|backpropagation/src/lib.rs|// changed|0|src change bumps
crate-manifest|backpropagation/Cargo.toml|rand = "0.9"|0|crate Cargo.toml dependency change bumps
workspace-manifest|Cargo.toml|lto = "fat"|0|workspace Cargo.toml profile change bumps
lockfile|Cargo.lock|# dependency graph moved|0|Cargo.lock dependency change bumps
cargo-config|.cargo/config.toml|# rustflags moved|0|.cargo/config.toml rustflags change bumps
toolchain|rust-toolchain.toml|profile = "minimal"|0|rust-toolchain.toml change bumps
ffi-header|include/neat_ai_backpropagation.h|// new export|0|include/ FFI header change bumps
docs-only|README.md|Docs only.|1|docs-only change skips
tests-only|backpropagation/tests/smoke.rs|// more coverage|1|integration-test-only change skips
EOF
)

CHECKED_FILES=""
while IFS= read -r row; do
  IFS='|' read -r name file line expected description <<<"$row"
  if [[ -z "$name" || -z "$file" || -z "$line" || -z "$expected" || -z "$description" ]]; then
    echo "FAIL: malformed case row: '$row'" >&2
    exit 2
  fi
  branch_with "$name" "$file" "$line"
  expect_exit "$description" "$expected" \
    "$BUMPER" --repo-root "$FIXTURE" --base-ref Develop --check
  if [[ "$expected" -eq 0 ]]; then
    CHECKED_FILES="$CHECKED_FILES$file"$'\n'
  fi
done <<<"$CHECK_CASES"

# The table must mirror the canonical list, so a path added to
# build-affecting-paths.sh fails here until a case covers it.
# shellcheck source=scripts/build-affecting-paths.sh
source "$SCRIPT_DIR/build-affecting-paths.sh"
UNCOVERED=""
while IFS= read -r pathspec; do
  covered=0
  while IFS= read -r checked; do
    if [[ "$checked" == "$pathspec" || "$checked" == "$pathspec"/* ]]; then
      covered=1
      break
    fi
  done <<<"$CHECKED_FILES"
  if [[ "$covered" -eq 0 ]]; then
    UNCOVERED="$UNCOVERED $pathspec"
  fi
done < <(build_affecting_pathspecs)

if [[ -z "$UNCOVERED" ]]; then
  echo "OK   every build-affecting path has a bump case"
  PASSED=$((PASSED + 1))
else
  echo "FAIL: build-affecting paths with no bump case:$UNCOVERED" >&2
  FAILED=$((FAILED + 1))
fi

# A real run rewrites manifest and lockfile, and is idempotent afterwards.
branch_with real-bump Cargo.lock '# real dependency move'
expect_exit "real run bumps" 0 \
  "$BUMPER" --repo-root "$FIXTURE" --base-ref Develop
expect_file_contains "manifest patched to 0.1.11" \
  "$FIXTURE/backpropagation/Cargo.toml" 'version = "0.1.11"'
expect_file_contains "lockfile patched to 0.1.11" \
  "$FIXTURE/Cargo.lock" 'version = "0.1.11"'

git add -A
git commit -q -m "chore: auto-increment versions for changed projects"
expect_exit "re-run after a bump is a no-op" 1 \
  "$BUMPER" --repo-root "$FIXTURE" --base-ref Develop --check

echo
echo "Passed: $PASSED  Failed: $FAILED"
if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
exit 0
