#!/usr/bin/env bash
# Single source of truth for the paths that change the built artefact.
#
# Unattended machines rebuild `neat_ai_backpropagation` only when the crate
# version moves — `ensure_lib_built()` in the consumers' `runlib.sh` compares
# an installed version marker against `cargo metadata` and skips the build
# when they match. So a change to any path listed here must bump the version,
# or remotes keep running a stale library (issue #95).
#
# Entries are GitHub Actions `paths:` globs. `build_affecting_pathspecs`
# converts them to git pathspecs for diffing.
#
# Deliberately excluded: `backpropagation/tests/` (integration tests are not
# linked into the cdylib/rlib), docs, and CI config.

# shellcheck disable=SC2034  # sourced by the bump and workflow-check scripts
BUILD_AFFECTING_PATHS=(
  "backpropagation/src/**"
  "backpropagation/Cargo.toml"
  "Cargo.toml"
  "Cargo.lock"
  ".cargo/config.toml"
  "rust-toolchain.toml"
  "include/**"
)

# Print one git pathspec per line (the `paths:` globs without the `/**` suffix).
build_affecting_pathspecs() {
  local path
  for path in "${BUILD_AFFECTING_PATHS[@]}"; do
    printf '%s\n' "${path%/\*\*}"
  done
}
