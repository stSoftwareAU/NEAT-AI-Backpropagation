#!/usr/bin/env bash
# Tests for check-no-private-repo-references.sh (issues #171, #191).
#
# Every case runs the real checker against a fixture tree and asserts on its
# exit code and message. The final case runs it against this repository's own
# tree — the regression test for the four build-profile references that named
# the first private fleet repository, and now for the second one as well
# (issue #191).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKER="$SCRIPT_DIR/check-no-private-repo-references.sh"

if [[ ! -x "$CHECKER" ]]; then
  echo "FAIL: checker not found or not executable: $CHECKER" >&2
  exit 2
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

PASSED=0
FAILED=0

# write_tree NAME FILENAME CONTENT -> echoes the fixture tree root
write_tree() {
  local name="$1" filename="$2" content="$3"
  local dir="$WORK_DIR/$name"
  mkdir -p "$dir/$(dirname "$filename")"
  printf '%s\n' "$content" >"$dir/$filename"
  echo "$dir"
}

# expect_exit DESCRIPTION EXPECTED_CODE TREE [EXPECTED_OUTPUT_SUBSTRING]
expect_exit() {
  local description="$1" expected="$2" tree="$3" substring="${4:-}"
  local output actual=0
  output="$("$CHECKER" "$tree" 2>&1)" || actual=$?

  if [[ "$actual" -ne "$expected" ]]; then
    echo "FAIL: $description (expected exit $expected, got $actual)" >&2
    echo "      output: $output" >&2
    FAILED=$((FAILED + 1))
    return
  fi

  if [[ -n "$substring" && "$output" != *"$substring"* ]]; then
    echo "FAIL: $description (output missing '$substring')" >&2
    echo "      output: $output" >&2
    FAILED=$((FAILED + 1))
    return
  fi

  echo "PASS: $description"
  PASSED=$((PASSED + 1))
}

tree="$(write_tree clean README.md 'Build profiles follow the fleet rule (issue #88).')"
expect_exit "a tree with no private-repo reference passes" 0 "$tree" "no private"

tree="$(write_tree url README.md \
  'See https://github.com/stSoftwareAU/VibeCoding/issues/4159 for the rule.')"
expect_exit "a full URL to the private repo is rejected" 1 "$tree" "README.md:1"

tree="$(write_tree shorthand CHANGELOG.md \
  '- Build profiles follow VibeCoding#4159 / issue #88.')"
expect_exit "the Repo#1234 shorthand is rejected" 1 "$tree" "CHANGELOG.md:1"

tree="$(write_tree owner_shorthand .cargo/config.toml \
  '# Fleet rule lives in stSoftwareAU/VibeCoding.')"
expect_exit "the owner/repo shorthand is rejected in a nested file" 1 "$tree" \
  ".cargo/config.toml:1"

tree="$(write_tree concept Cargo.toml \
  '# Fastest practical compile (issue #88): keep panic file:line.')"
expect_exit "concept-level wording citing a local issue passes" 0 "$tree"

tree="$(write_tree public_repo README.md \
  'Depends on https://github.com/stSoftwareAU/NEAT-AI-core for the core crate.')"
expect_exit "a public stSoftware repo reference is not flagged" 0 "$tree"

tree="$(write_tree multi README.md 'VibeCoding#4159')"
printf '%s\n' 'Also stSoftwareAU/VibeCoding here.' >"$tree/CHANGELOG.md"
expect_exit "every offending file is reported, not just the first" 1 "$tree" \
  "CHANGELOG.md:1"

tree="$(write_tree second_repo_url CONTRIBUTING.md \
  'Further reading: https://github.com/stSoftwareAU/GRQ-taxation/blob/Develop/scripts/runlib.sh')"
expect_exit "a full URL to the second private repo is rejected" 1 "$tree" \
  "CONTRIBUTING.md:1"

tree="$(write_tree second_repo_shorthand CHANGELOG.md \
  '- Mirrors the remote contract (GRQ-taxation#4774).')"
expect_exit "the second private repo's Repo#1234 shorthand is rejected" 1 \
  "$tree" "CHANGELOG.md:1"

tree="$(write_tree second_repo_owner scripts/bump-backpropagation-version.sh \
  '# Mirrors stSoftwareAU/GRQ-taxation version-increment job.')"
expect_exit "the second private repo's owner/repo shorthand is rejected" 1 \
  "$tree" "scripts/bump-backpropagation-version.sh:1"

tree="$(write_tree second_repo_concept .github/workflows/version-increment.yml \
  '# Auto-increment the patch (shared runlib.sh version-marker contract).')"
expect_exit "concept-level wording for the second private repo passes" 0 "$tree"

tree="$(write_tree both_repos README.md 'Fleet rule: VibeCoding#4159.')"
printf '%s\n' 'Remote contract: stSoftwareAU/GRQ-taxation.' >"$tree/CONTRIBUTING.md"
expect_exit "both private repos are matched by the one pattern" 1 "$tree" \
  "CONTRIBUTING.md:1"

tree="$(write_tree self_exclusion scripts/check-no-private-repo-references.sh \
  'PRIVATE_REPOS=("VibeCoding")')"
printf '%s\n' 'expect_exit "the shorthand is rejected" 1 fixture # VibeCoding#4159' \
  >"$tree/scripts/test-check-no-private-repo-references.sh"
printf '%s\n' 'Nothing to see here.' >"$tree/README.md"
expect_exit "the checker and its test companion are skipped" 0 "$tree"

mkdir -p "$WORK_DIR/empty"
expect_exit "an empty tree fails loudly rather than passing vacuously" 2 \
  "$WORK_DIR/empty" "not a pass"

expect_exit "an unreadable tree root exits 2" 2 "$WORK_DIR/does-not-exist" \
  "tree root not found"

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
expect_exit "this repository references no private stSoftware repository" 0 \
  "$REPO_ROOT"

echo "check-no-private-repo-references tests: $PASSED passed, $FAILED failed"
if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
