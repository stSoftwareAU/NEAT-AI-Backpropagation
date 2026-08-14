#!/usr/bin/env bash
# Validate the branch-protection ruleset on the default branch (Issue #21).
#
# Branch protection is a repository *setting*, not a committed file, so this
# checker reads the live rules through the GitHub API (or a saved payload) and
# compares them with the policy recorded in CONTRIBUTING.md:
#
#   1. A `pull_request` rule — without it commits can be pushed straight to
#      `Develop`, bypassing CI and review entirely.
#   2. At least one required approving review, so no single account can merge
#      its own change.
#   3. `require_code_owner_review` — `.github/CODEOWNERS` only has teeth when
#      the ruleset asks the owners of `/.github/workflows/` to review. The
#      Auto Format and Version Increment workflows mint GitHub App push tokens,
#      so an unreviewed workflow edit is an unreviewed secret grab.
#   4. The `CI Required Checks` aggregator (`ci-required` in ci.yml) registered
#      as a required status check — the aggregator only gates merges when it is
#      required.
#   5. A `non_fast_forward` rule, so merged history on `Develop` cannot be
#      rewritten by a force-push.
#
# Required signed commits are deliberately *not* checked: the Auto Format and
# Version Increment workflows push unsigned bot commits back to PR branches,
# so requiring signatures would block the repository's own automation.
#
# Usage:
#   ./scripts/check-branch-protection.sh              # query the live ruleset
#   ./scripts/check-branch-protection.sh RULES_JSON   # check a saved payload
#
# RULES_JSON is the array returned by
# `gh api repos/OWNER/REPO/rules/branches/BRANCH`.
#
# Exit codes: 0 policy satisfied, 1 policy broken, 2 rules could not be read.
set -euo pipefail

REPO="${BRANCH_PROTECTION_REPO:-stSoftwareAU/NEAT-AI-Backpropagation}"
BRANCH="${BRANCH_PROTECTION_BRANCH:-Develop}"

usage() {
  cat <<'EOF'
Usage: check-branch-protection.sh [RULES_JSON]

With no argument the live ruleset is fetched with `gh`. With a path, that
saved `rules/branches` payload is checked instead.

Exits 0 when the branch-protection policy in the script header is satisfied,
1 when a rule is broken, and 2 when the rules cannot be read.

Environment:
  BRANCH_PROTECTION_REPO    owner/repo to query   (default stSoftwareAU/NEAT-AI-Backpropagation)
  BRANCH_PROTECTION_BRANCH  branch to query       (default Develop)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v python3 &>/dev/null; then
  echo "FAIL: python3 is required to parse the branch-protection rules" >&2
  exit 2
fi

RULES_FILE="${1:-}"
SOURCE=""

if [[ -n "$RULES_FILE" ]]; then
  if [[ ! -f "$RULES_FILE" ]]; then
    echo "FAIL: branch-protection rules payload not found: $RULES_FILE" >&2
    exit 2
  fi
  SOURCE="$RULES_FILE"
else
  if ! command -v gh &>/dev/null; then
    echo "FAIL: gh is required to read the live ruleset — install: https://cli.github.com" >&2
    exit 2
  fi
  RULES_FILE="$(mktemp)"
  trap 'rm -f "$RULES_FILE"' EXIT
  SOURCE="$REPO@$BRANCH"
  if ! gh api "repos/$REPO/rules/branches/$BRANCH" >"$RULES_FILE" 2>/dev/null; then
    echo "FAIL: could not read the branch-protection rules for $SOURCE" >&2
    echo "      authenticate with 'gh auth login' (or set GH_TOKEN) and retry" >&2
    exit 2
  fi
fi

python3 - "$RULES_FILE" "$SOURCE" <<'PY'
"""Compare a branch's live rules with the committed protection policy."""
import json
import sys

rules_path, source = sys.argv[1], sys.argv[2]
# The ci.yml aggregator that gates every merge (job id `ci-required`).
REQUIRED_CHECK = "CI Required Checks"
MINIMUM_APPROVALS = 1

exit_code = 0


def ok(message):
    print("OK   {}: {}".format(source, message))


def fail(message):
    global exit_code
    print("FAIL {}: {}".format(source, message), file=sys.stderr)
    exit_code = 1


try:
    with open(rules_path, encoding="utf-8") as handle:
        rules = json.load(handle)
except OSError as error:
    print("FAIL {}: could not read — {}".format(source, error), file=sys.stderr)
    sys.exit(2)
except ValueError as error:
    print("FAIL {}: not valid JSON — {}".format(source, error), file=sys.stderr)
    sys.exit(2)

if not isinstance(rules, list):
    print(
        "FAIL {}: expected a JSON array of branch rules".format(source),
        file=sys.stderr,
    )
    sys.exit(2)

rules = [rule for rule in rules if isinstance(rule, dict)]
if not rules:
    print(
        "FAIL {}: no branch-protection rules apply — the branch is "
        "unprotected".format(source),
        file=sys.stderr,
    )
    sys.exit(1)


def rules_of(rule_type):
    return [rule for rule in rules if rule.get("type") == rule_type]


def parameters(rule):
    params = rule.get("parameters")
    return params if isinstance(params, dict) else {}


# 1-3. Pull requests, approvals, and code-owner review.
pull_request_rules = rules_of("pull_request")
if not pull_request_rules:
    fail(
        "no pull_request rule — commits can be pushed directly to the branch, "
        "skipping CI and review"
    )
else:
    ok("pull_request rule present — every change arrives through a PR")

    approvals = max(
        parameters(rule).get("required_approving_review_count") or 0
        for rule in pull_request_rules
    )
    if approvals < MINIMUM_APPROVALS:
        fail(
            "required_approving_review_count is {} — a single account can "
            "merge its own change; require at least {}".format(
                approvals, MINIMUM_APPROVALS
            )
        )
    else:
        ok("{} approving review(s) required before merge".format(approvals))

    if any(
        parameters(rule).get("require_code_owner_review") is True
        for rule in pull_request_rules
    ):
        ok("code-owner review required — .github/CODEOWNERS is enforced")
    else:
        fail(
            "require_code_owner_review is off — .github/CODEOWNERS is advisory, "
            "so a workflow edit can merge without an owner's review"
        )

# 4. The CI aggregator must be a required status check.
status_check_rules = rules_of("required_status_checks")
if not status_check_rules:
    fail(
        "no required_status_checks rule — CI results do not gate merges at all"
    )
else:
    contexts = set()
    for rule in status_check_rules:
        for check in parameters(rule).get("required_status_checks") or []:
            if isinstance(check, dict) and isinstance(check.get("context"), str):
                contexts.add(check["context"])
    if REQUIRED_CHECK in contexts:
        ok("'{}' is a required status check".format(REQUIRED_CHECK))
    else:
        fail(
            "'{}' is not a required status check (found: {}) — the ci-required "
            "aggregator does not block merges".format(
                REQUIRED_CHECK, ", ".join(sorted(contexts)) or "none"
            )
        )

# 5. Force-pushes must be blocked.
if rules_of("non_fast_forward"):
    ok("non_fast_forward rule present — force-pushes are blocked")
else:
    fail(
        "no non_fast_forward rule — merged history can be rewritten by a "
        "force-push"
    )

sys.exit(exit_code)
PY
