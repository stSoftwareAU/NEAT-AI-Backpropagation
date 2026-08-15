#!/usr/bin/env bash
# Validate the Gitleaks secrets-detection workflow (Issue #42).
#
# A committed credential is the one defect this repository cannot fix by
# reverting — every leaked secret has to be rotated. The workflow must
# therefore catch it in the pull request, and must fail loudly when it does.
#
# The workflow must:
#   1. Run on `pull_request` events covering the default branch (`Develop`),
#      so a secret is caught before it reaches a branch anyone else pulls.
#   2. Actually run a scan — the `gitleaks/gitleaks-action` action or the
#      open-source `gitleaks` CLI, not merely a step named after it.
#   3. Keep the scan blocking: no `|| true` swallowing its exit code, no
#      `continue-on-error: true`, no `--exit-code 0` reporting a leak as a pass.
#   4. Cover licence-less runs. `gitleaks-action@v2` needs an organisation
#      licence, and a Dependabot-authored PR receives no Actions secrets, so
#      without a `GITLEAKS_LICENSE == ''` fallback those diffs go unscanned
#      while the job still reports green.
#   5. Pin its supply chain: third-party actions on a 40-character commit SHA,
#      any downloaded release version-pinned (never `latest`), checksum
#      verified, and installed under strict bash so a failed download cannot be
#      reconciled as a clean scan.
#   6. Check out full history (`fetch-depth: 0`) — the base..head commit range
#      does not resolve in a shallow clone, and gitleaks then scans nothing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKFLOW="${1:-$REPO_ROOT/.github/workflows/gitleaks.yml}"
EXIT_CODE=0

usage() {
  cat <<'EOF'
Usage: check-gitleaks-workflow.sh [WORKFLOW_PATH]

Exits 0 when the workflow satisfies every rule listed in the script header,
1 when a rule is broken, and 2 when the workflow cannot be read.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "$WORKFLOW" ]]; then
  echo "FAIL: Gitleaks workflow not found: $WORKFLOW" >&2
  echo "      Pull request diffs are unscanned — commit .github/workflows/gitleaks.yml" >&2
  exit 2
fi

ok() { echo "OK   $WORKFLOW: $*"; }
fail() {
  echo "FAIL $WORKFLOW: $*" >&2
  EXIT_CODE=1
}

# Lines that really run a scan: the upstream action, or the CLI with one of its
# scan subcommands. A step *named* "Gitleaks" that echoes a message is prose.
SCAN_LINES="$(grep -vE '^[[:space:]]*#' "$WORKFLOW" |
  grep -E 'uses:[[:space:]]*gitleaks/gitleaks-action@|(^|[^A-Za-z0-9_./-])(\./)?gitleaks[[:space:]]+(git|detect|dir|protect)([^A-Za-z0-9_-]|$)' || true)"

# 1. Pull-request trigger covering the default branch.
if grep -qE '^[[:space:]]*pull_request:' "$WORKFLOW"; then
  # Either an explicit `Develop` entry or a `*` wildcard covers the default
  # branch; a list naming only other branches does not.
  if grep -qE '^[[:space:]]*branches:.*(Develop|\*)' "$WORKFLOW" ||
    grep -qE '^[[:space:]]*-[[:space:]]*["'\'']?(Develop|\*)["'\'']?[[:space:]]*$' "$WORKFLOW"; then
    ok "pull_request trigger covers the Develop default branch"
  else
    fail "pull_request trigger does not cover Develop — PRs onto the default branch would go unscanned"
  fi
else
  fail "no pull_request trigger — a leaked secret would only be found after merge"
fi

# 2. A scan really runs.
if [[ -n "$SCAN_LINES" ]]; then
  ok "a gitleaks scan is invoked (action or CLI)"
else
  fail "no gitleaks scan is invoked — the job would report green without reading the diff"
fi

# 3. The scan's verdict reaches the runner.
if printf '%s\n' "$SCAN_LINES" | grep -qE '\|\|[[:space:]]*(true|:|echo\b)'; then
  fail "the scan's exit code is discarded with '|| true' — a detected secret would pass"
fi
if printf '%s\n' "$SCAN_LINES" | grep -qE -- '--exit-code[[:space:]=]+0'; then
  fail "the scan runs with '--exit-code 0' — a detected secret would be reported as a pass"
fi
if grep -qE '^[[:space:]]*continue-on-error:[[:space:]]*true' "$WORKFLOW"; then
  fail "continue-on-error: true — the job's failures would not block the merge"
fi

# 4. Licence-less pull requests (e.g. Dependabot) are still scanned.
if grep -qE "GITLEAKS_LICENSE[[:space:]]*!=[[:space:]]*''" "$WORKFLOW" &&
  grep -qE "GITLEAKS_LICENSE[[:space:]]*==[[:space:]]*''" "$WORKFLOW"; then
  ok "licensed and licence-less paths both scan (Dependabot PRs are covered)"
else
  fail "no GITLEAKS_LICENSE fallback — a licence-less PR (Dependabot) would go unscanned while reporting green"
fi

# 5a. Third-party actions pinned to an immutable commit SHA.
unpinned=0
while IFS= read -r reference; do
  [[ -n "$reference" ]] || continue
  case "$reference" in
    ./*) continue ;; # local composite action, versioned by this repository
  esac
  if [[ ! "$reference" =~ @[0-9a-f]{40}$ ]]; then
    fail "action '$reference' is not pinned to a 40-character commit SHA"
    unpinned=$((unpinned + 1))
  fi
done < <(grep -E '^[[:space:]]*(-[[:space:]]*)?uses:' "$WORKFLOW" |
  sed -E 's/^[[:space:]]*(-[[:space:]]*)?uses:[[:space:]]*//; s/[[:space:]]*#.*$//; s/[[:space:]]*$//')
if [[ "$unpinned" -eq 0 ]]; then
  ok "every third-party action is pinned to a commit SHA"
fi

# 5b. A downloaded gitleaks release is pinned, verified, and installed strictly.
if grep -qE '(^|[^A-Za-z0-9_.-])(curl|wget)([^A-Za-z0-9_.-]|$)' "$WORKFLOW"; then
  if grep -qE 'sha256sum|shasum[[:space:]]+-a[[:space:]]+256' "$WORKFLOW"; then
    ok "the downloaded gitleaks release has its checksum verified"
  else
    fail "the gitleaks release is downloaded without a sha256 checksum verification"
  fi
  if grep -qE 'releases/latest' "$WORKFLOW"; then
    fail "downloads the latest release — pin the version so a hijacked release cannot land silently"
  fi
  if grep -qE 'set[[:space:]]+-euo[[:space:]]+pipefail' "$WORKFLOW"; then
    ok "the fallback runs under strict bash (set -euo pipefail)"
  else
    fail "the download step has no 'set -euo pipefail' — a failed download would be reported as a clean scan"
  fi
fi

# 6. Full history, so the PR commit range resolves on the runner.
if grep -qE '^[[:space:]]*fetch-depth:[[:space:]]*0[[:space:]]*$' "$WORKFLOW"; then
  ok "checkout uses fetch-depth: 0 (the base..head range resolves)"
else
  fail "checkout does not set 'fetch-depth: 0' — the commit range would not resolve and nothing would be scanned"
fi

exit "$EXIT_CODE"
