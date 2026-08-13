#!/usr/bin/env bash
# Tests for scripts/check-renovate-config.sh (Issue #19).
#
# Each case writes a fixture Renovate config to a temporary directory, runs
# the real checker against it, and asserts the exit code (and, where it
# matters, that the failure names the rule that broke).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKER="$SCRIPT_DIR/check-renovate-config.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

PASSED=0
FAILED=0

if [[ ! -x "$CHECKER" ]]; then
  echo "FAIL: checker not found or not executable: $CHECKER" >&2
  exit 2
fi

# write_config NAME <<'JSON' … JSON  → echoes the fixture path.
write_config() {
  local path="$WORK_DIR/$1.json"
  cat >"$path"
  printf '%s' "$path"
}

# expect_exit DESCRIPTION EXPECTED_CODE CONFIG_PATH [EXPECTED_OUTPUT_SUBSTRING]
expect_exit() {
  local description="$1" expected="$2" config="$3" needle="${4:-}"
  local output status=0
  output="$("$CHECKER" "$config" 2>&1)" || status=$?
  if [[ "$status" -ne "$expected" ]]; then
    echo "FAIL $description: expected exit $expected, got $status" >&2
    printf '%s\n' "$output" >&2
    FAILED=$((FAILED + 1))
    return
  fi
  if [[ -n "$needle" && "$output" != *"$needle"* ]]; then
    echo "FAIL $description: output did not mention '$needle'" >&2
    printf '%s\n' "$output" >&2
    FAILED=$((FAILED + 1))
    return
  fi
  echo "OK   $description"
  PASSED=$((PASSED + 1))
}

valid=$(write_config valid <<'JSON'
{
  "extends": ["config:recommended"],
  "minimumReleaseAge": "24 hours",
  "packageRules": [
    { "matchPackageNames": ["neat-core"], "enabled": false }
  ]
}
JSON
)
expect_exit "accepts a 24-hour quarantine with neat-core disabled" 0 "$valid"

longer=$(write_config longer <<'JSON'
{
  "extends": ["config:best-practices"],
  "minimumReleaseAge": "3 days",
  "packageRules": [
    { "matchPackageNames": ["neat-core"], "enabled": false }
  ]
}
JSON
)
expect_exit "accepts a window longer than 24 hours expressed in days" 0 "$longer"

internal_exempt=$(write_config internal-exempt <<'JSON'
{
  "extends": ["config:recommended"],
  "minimumReleaseAge": "24 hours",
  "packageRules": [
    { "matchPackageNames": ["neat-core"], "enabled": false },
    {
      "matchPackageNames": ["/^stsoftware-/"],
      "minimumReleaseAge": "0 hours"
    }
  ]
}
JSON
)
expect_exit "accepts an internal-only exemption from the quarantine" 0 "$internal_exempt"

expect_exit "reports a missing config file with exit 2" 2 \
  "$WORK_DIR/does-not-exist.json" "not found"

malformed=$(write_config malformed <<'JSON'
{ "extends": ["config:recommended"],
JSON
)
expect_exit "rejects malformed JSON" 1 "$malformed" "not valid JSON"

no_age=$(write_config no-age <<'JSON'
{
  "extends": ["config:recommended"],
  "packageRules": [
    { "matchPackageNames": ["neat-core"], "enabled": false }
  ]
}
JSON
)
expect_exit "rejects a config with no minimumReleaseAge" 1 "$no_age" "minimumReleaseAge"

short_age=$(write_config short-age <<'JSON'
{
  "extends": ["config:recommended"],
  "minimumReleaseAge": "6 hours",
  "packageRules": [
    { "matchPackageNames": ["neat-core"], "enabled": false }
  ]
}
JSON
)
expect_exit "rejects a quarantine window shorter than 24 hours" 1 "$short_age" "24"

unparsable_age=$(write_config unparsable-age <<'JSON'
{
  "extends": ["config:recommended"],
  "minimumReleaseAge": "soon",
  "packageRules": [
    { "matchPackageNames": ["neat-core"], "enabled": false }
  ]
}
JSON
)
expect_exit "rejects an unparsable duration rather than assuming it passes" 1 \
  "$unparsable_age" "soon"

external_override=$(write_config external-override <<'JSON'
{
  "extends": ["config:recommended"],
  "minimumReleaseAge": "24 hours",
  "packageRules": [
    { "matchPackageNames": ["neat-core"], "enabled": false },
    { "matchPackageNames": ["clap"], "minimumReleaseAge": "1 hour" }
  ]
}
JSON
)
expect_exit "rejects a package rule that shortens the window for an external crate" 1 \
  "$external_override" "clap"

unscoped_override=$(write_config unscoped-override <<'JSON'
{
  "extends": ["config:recommended"],
  "minimumReleaseAge": "24 hours",
  "packageRules": [
    { "matchPackageNames": ["neat-core"], "enabled": false },
    { "matchManagers": ["cargo"], "minimumReleaseAge": "0 hours" }
  ]
}
JSON
)
expect_exit "rejects an unscoped rule that shortens the window for every crate" 1 \
  "$unscoped_override" "minimumReleaseAge"

no_neat_core=$(write_config no-neat-core <<'JSON'
{
  "extends": ["config:recommended"],
  "minimumReleaseAge": "24 hours"
}
JSON
)
expect_exit "rejects a config that leaves the neat-core path dependency enabled" 1 \
  "$no_neat_core" "neat-core"

cargo_disabled=$(write_config cargo-disabled <<'JSON'
{
  "extends": ["config:recommended"],
  "minimumReleaseAge": "24 hours",
  "enabledManagers": ["github-actions"],
  "packageRules": [
    { "matchPackageNames": ["neat-core"], "enabled": false }
  ]
}
JSON
)
expect_exit "rejects a config where the cargo manager is switched off" 1 \
  "$cargo_disabled" "cargo"

cargo_rule_disabled=$(write_config cargo-rule-disabled <<'JSON'
{
  "extends": ["config:recommended"],
  "minimumReleaseAge": "24 hours",
  "packageRules": [
    { "matchPackageNames": ["neat-core"], "enabled": false },
    { "matchManagers": ["cargo"], "enabled": false }
  ]
}
JSON
)
expect_exit "rejects a package rule that disables every cargo update" 1 \
  "$cargo_rule_disabled" "cargo"

no_extends=$(write_config no-extends <<'JSON'
{
  "minimumReleaseAge": "24 hours",
  "packageRules": [
    { "matchPackageNames": ["neat-core"], "enabled": false }
  ]
}
JSON
)
expect_exit "rejects a config with no shared preset in extends" 1 "$no_extends" "extends"

echo "check-renovate-config tests: $PASSED passed, $FAILED failed"
if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
