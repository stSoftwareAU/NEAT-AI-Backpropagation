#!/usr/bin/env bash
# Tests for scripts/check-branch-protection.sh (Issue #21).
#
# Each case writes a fixture rules payload — the shape returned by
# `gh api repos/OWNER/REPO/rules/branches/BRANCH` — to a temporary directory,
# runs the real checker against it, and asserts the exit code (and, where it
# matters, that the failure names the rule that broke). No network access and
# no `gh` authentication are needed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKER="$SCRIPT_DIR/check-branch-protection.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

PASSED=0
FAILED=0

if [[ ! -x "$CHECKER" ]]; then
  echo "FAIL: checker not found or not executable: $CHECKER" >&2
  exit 2
fi

# write_rules NAME <<'JSON' … JSON  → echoes the fixture path.
write_rules() {
  local path="$WORK_DIR/$1.json"
  cat >"$path"
  printf '%s' "$path"
}

# expect_exit DESCRIPTION EXPECTED_CODE RULES_PATH [EXPECTED_OUTPUT_SUBSTRING]
expect_exit() {
  local description="$1" expected="$2" rules="$3" needle="${4:-}"
  local output status=0
  output="$("$CHECKER" "$rules" 2>&1)" || status=$?
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

compliant=$(write_rules compliant <<'JSON'
[
  {
    "type": "required_status_checks",
    "parameters": {
      "strict_required_status_checks_policy": true,
      "required_status_checks": [
        {"context": "CI Required Checks", "integration_id": 15368},
        {"context": "Quality Checks", "integration_id": 15368}
      ]
    }
  },
  {
    "type": "pull_request",
    "parameters": {
      "required_approving_review_count": 1,
      "require_code_owner_review": true
    }
  },
  {"type": "non_fast_forward"}
]
JSON
)
expect_exit "accepts a compliant Develop ruleset" 0 "$compliant"

two_approvals=$(write_rules two_approvals <<'JSON'
[
  {
    "type": "required_status_checks",
    "parameters": {
      "required_status_checks": [{"context": "CI Required Checks"}]
    }
  },
  {
    "type": "pull_request",
    "parameters": {
      "required_approving_review_count": 2,
      "require_code_owner_review": true
    }
  },
  {"type": "non_fast_forward"},
  {"type": "deletion"}
]
JSON
)
expect_exit "accepts a stricter ruleset (two approvals, deletion blocked)" 0 \
  "$two_approvals"

no_approvals=$(write_rules no_approvals <<'JSON'
[
  {
    "type": "required_status_checks",
    "parameters": {
      "required_status_checks": [{"context": "CI Required Checks"}]
    }
  },
  {
    "type": "pull_request",
    "parameters": {
      "required_approving_review_count": 0,
      "require_code_owner_review": true
    }
  },
  {"type": "non_fast_forward"}
]
JSON
)
expect_exit "rejects zero required approvals" 1 "$no_approvals" \
  "required_approving_review_count"

no_code_owner=$(write_rules no_code_owner <<'JSON'
[
  {
    "type": "required_status_checks",
    "parameters": {
      "required_status_checks": [{"context": "CI Required Checks"}]
    }
  },
  {
    "type": "pull_request",
    "parameters": {
      "required_approving_review_count": 1,
      "require_code_owner_review": false
    }
  },
  {"type": "non_fast_forward"}
]
JSON
)
expect_exit "rejects a ruleset that never asks CODEOWNERS to review" 1 \
  "$no_code_owner" "CODEOWNERS"

no_pull_request=$(write_rules no_pull_request <<'JSON'
[
  {
    "type": "required_status_checks",
    "parameters": {
      "required_status_checks": [{"context": "CI Required Checks"}]
    }
  },
  {"type": "non_fast_forward"}
]
JSON
)
expect_exit "rejects a ruleset with no pull_request rule (direct pushes)" 1 \
  "$no_pull_request" "pushed directly"

wrong_check=$(write_rules wrong_check <<'JSON'
[
  {
    "type": "required_status_checks",
    "parameters": {
      "required_status_checks": [{"context": "Quality Checks"}]
    }
  },
  {
    "type": "pull_request",
    "parameters": {
      "required_approving_review_count": 1,
      "require_code_owner_review": true
    }
  },
  {"type": "non_fast_forward"}
]
JSON
)
expect_exit "rejects a ruleset that does not require the CI aggregator" 1 \
  "$wrong_check" "CI Required Checks"

no_status_checks=$(write_rules no_status_checks <<'JSON'
[
  {
    "type": "pull_request",
    "parameters": {
      "required_approving_review_count": 1,
      "require_code_owner_review": true
    }
  },
  {"type": "non_fast_forward"}
]
JSON
)
expect_exit "rejects a ruleset with no required status checks at all" 1 \
  "$no_status_checks" "required_status_checks"

no_force_push_block=$(write_rules no_force_push_block <<'JSON'
[
  {
    "type": "required_status_checks",
    "parameters": {
      "required_status_checks": [{"context": "CI Required Checks"}]
    }
  },
  {
    "type": "pull_request",
    "parameters": {
      "required_approving_review_count": 1,
      "require_code_owner_review": true
    }
  }
]
JSON
)
expect_exit "rejects a ruleset that allows force-pushes" 1 \
  "$no_force_push_block" "non_fast_forward"

unprotected=$(write_rules unprotected <<'JSON'
[]
JSON
)
expect_exit "rejects an entirely unprotected branch" 1 "$unprotected" \
  "no branch-protection rules"

not_an_array=$(write_rules not_an_array <<'JSON'
{"type": "pull_request"}
JSON
)
expect_exit "rejects a payload that is not a rules array" 2 "$not_an_array" \
  "JSON array"

malformed=$(write_rules malformed <<'JSON'
{ not json
JSON
)
expect_exit "reports an unparsable payload with exit 2" 2 "$malformed" \
  "not valid JSON"

expect_exit "reports a missing payload with exit 2" 2 \
  "$WORK_DIR/does-not-exist.json" "not found"

echo "check-branch-protection tests: $PASSED passed, $FAILED failed"
if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
