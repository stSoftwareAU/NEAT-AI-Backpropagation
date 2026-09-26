#!/usr/bin/env bash
# Single source of truth for reading the neat_ai_backpropagation crate version
# (issue #177). Sourced by the bump and no-downgrade scripts; the TOML parsing
# itself is runlib.sh's `_runlib_crate_field`, scoped to `[package]`.

# shellcheck source=scripts/runlib.sh
source "$(dirname "${BASH_SOURCE[0]}")/runlib.sh"

# `[package] version` of manifest $1; empty when absent.
read_version() {
  _runlib_crate_field "$1" "$1" version
}

# `[package] version` of backpropagation/Cargo.toml at git ref $1; empty when
# the ref has no manifest.
read_version_at_ref() {
  local tmp
  tmp="$(mktemp)"
  git show "$1:backpropagation/Cargo.toml" >"$tmp" 2>/dev/null || true
  read_version "$tmp"
  rm -f "$tmp"
}
