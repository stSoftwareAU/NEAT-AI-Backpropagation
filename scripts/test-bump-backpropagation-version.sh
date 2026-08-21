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

# Every path that changes the produced artefact must bump the version.
branch_with lib-src backpropagation/src/lib.rs '// changed'
expect_exit "src change bumps" 0 \
  "$BUMPER" --repo-root "$FIXTURE" --base-ref Develop --check

branch_with crate-manifest backpropagation/Cargo.toml 'rand = "0.9"'
expect_exit "crate Cargo.toml dependency change bumps" 0 \
  "$BUMPER" --repo-root "$FIXTURE" --base-ref Develop --check

branch_with workspace-manifest Cargo.toml 'lto = "fat"'
expect_exit "workspace Cargo.toml profile change bumps" 0 \
  "$BUMPER" --repo-root "$FIXTURE" --base-ref Develop --check

branch_with lockfile Cargo.lock '# dependency graph moved'
expect_exit "Cargo.lock dependency change bumps" 0 \
  "$BUMPER" --repo-root "$FIXTURE" --base-ref Develop --check

branch_with cargo-config .cargo/config.toml '# rustflags moved'
expect_exit ".cargo/config.toml rustflags change bumps" 0 \
  "$BUMPER" --repo-root "$FIXTURE" --base-ref Develop --check

branch_with toolchain rust-toolchain.toml 'profile = "minimal"'
expect_exit "rust-toolchain.toml change bumps" 0 \
  "$BUMPER" --repo-root "$FIXTURE" --base-ref Develop --check

branch_with ffi-header include/neat_ai_backpropagation.h '// new export'
expect_exit "include/ FFI header change bumps" 0 \
  "$BUMPER" --repo-root "$FIXTURE" --base-ref Develop --check

# Paths outside the artefact must not force a bump.
branch_with docs-only README.md 'Docs only.'
expect_exit "docs-only change skips" 1 \
  "$BUMPER" --repo-root "$FIXTURE" --base-ref Develop --check

branch_with tests-only backpropagation/tests/smoke.rs '// more coverage'
expect_exit "integration-test-only change skips" 1 \
  "$BUMPER" --repo-root "$FIXTURE" --base-ref Develop --check

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
