#!/usr/bin/env bash
# Validate the Semgrep SAST scanning workflow (Issue #43).
#
# CodeQL reads this crate's Rust with GitHub's own query packs; Semgrep reads
# the same tree with a second, independently maintained rule set — including the
# shell and workflow YAML CodeQL's Rust analysis never looks at. A SAST job is
# only worth having if a finding actually blocks the merge, so the workflow
# must:
#   1. Run on `pull_request` events covering the default branch (`Develop`),
#      so a finding is raised before the code reaches a branch anyone pulls.
#   2. Actually run a scan — `semgrep ci` / `semgrep scan`, or the upstream
#      semgrep action — not merely a step named after it.
#   3. Configure a rule set (`--config`). Without one an unauthenticated scan
#      has no rules to run and reports a clean tree.
#   4. Keep the verdict blocking: no `|| true` swallowing the exit code and no
#      `continue-on-error: true`.
#   5. Fail loud on a broken scan. `semgrep ci` defaults to `--suppress-errors`,
#      which exits 0 when Semgrep itself errors — a crashed scan then reads as
#      a clean one — so `--no-suppress-errors` is required, and a CLI scan must
#      run under `set -euo pipefail`.
#   6. Pin its supply chain: third-party actions on a 40-character commit SHA,
#      any container image on a `@sha256:` digest rather than a movable tag, and
#      a `pip install semgrep` on an exact version.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKFLOW="${1:-$REPO_ROOT/.github/workflows/semgrep.yml}"
EXIT_CODE=0

usage() {
  cat <<'EOF'
Usage: check-semgrep-workflow.sh [WORKFLOW_PATH]

Exits 0 when the workflow satisfies every rule listed in the script header,
1 when a rule is broken, and 2 when the workflow cannot be read.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "$WORKFLOW" ]]; then
  echo "FAIL: Semgrep workflow not found: $WORKFLOW" >&2
  echo "      Pull requests get no SAST scan — commit .github/workflows/semgrep.yml" >&2
  exit 2
fi

ok() { echo "OK   $WORKFLOW: $*"; }
fail() {
  echo "FAIL $WORKFLOW: $*" >&2
  EXIT_CODE=1
}

UNCOMMENTED="$(grep -vE '^[[:space:]]*#' "$WORKFLOW" || true)"

# Lines that really run a scan: the upstream action, or the CLI with one of its
# scan subcommands. A step *named* "Semgrep" that echoes a message is prose.
SCAN_LINES="$(printf '%s\n' "$UNCOMMENTED" |
  grep -E 'uses:[[:space:]]*(semgrep|returntocorp)/semgrep-action@|(^|[^A-Za-z0-9_./-])(\./)?semgrep[[:space:]]+(ci|scan)([^A-Za-z0-9_-]|$)' || true)"
# CLI invocations only — these are the ones that need strict bash around them.
CLI_SCAN_LINES="$(printf '%s\n' "$SCAN_LINES" | grep -vE '^[[:space:]]*-?[[:space:]]*uses:' || true)"

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
  fail "no pull_request trigger — a finding would only surface after merge"
fi

# 2. A scan really runs.
if [[ -n "$SCAN_LINES" ]]; then
  ok "a semgrep scan is invoked (action or CLI)"
else
  fail "no semgrep scan is invoked — the job would report green without reading the code"
fi

# 3. A rule set is configured.
if printf '%s\n' "$UNCOMMENTED" | grep -qE -- '(--config[[:space:]=]|^[[:space:]]*config:)'; then
  ok "a ruleset is configured (--config)"
else
  fail "no ruleset is configured — an unauthenticated scan has no rules and reports a clean tree"
fi

# 4. The scan's verdict reaches the runner.
if printf '%s\n' "$SCAN_LINES" | grep -qE '\|\|[[:space:]]*(true|:|echo\b)'; then
  fail "the scan's exit code is discarded with '|| true' — a finding would pass"
fi
if grep -qE '^[[:space:]]*continue-on-error:[[:space:]]*true' "$WORKFLOW"; then
  fail "continue-on-error: true — the job's failures would not block the merge"
fi

# 5. A broken scan fails loud rather than reading as a clean one.
if printf '%s\n' "$UNCOMMENTED" | grep -qE -- '--suppress-errors'; then
  fail "'--suppress-errors' exits 0 when Semgrep itself errors — a crashed scan would be reported as clean"
fi
if printf '%s\n' "$SCAN_LINES" | grep -qE '(^|[^A-Za-z0-9_./-])(\./)?semgrep[[:space:]]+ci([^A-Za-z0-9_-]|$)'; then
  if printf '%s\n' "$UNCOMMENTED" | grep -qE -- '--no-suppress-errors'; then
    ok "'semgrep ci' runs with --no-suppress-errors (an internal error fails the job)"
  else
    fail "'semgrep ci' defaults to suppressing errors — pass '--no-suppress-errors' so a crashed scan fails the job"
  fi
fi
if [[ -n "$CLI_SCAN_LINES" ]]; then
  if grep -qE 'set[[:space:]]+-euo[[:space:]]+pipefail' "$WORKFLOW"; then
    ok "the scan step runs under strict bash (set -euo pipefail)"
  else
    fail "the scan step has no 'set -euo pipefail' — a failed install or setup command would be reported as a clean scan"
  fi
fi

# 6a. Third-party actions pinned to an immutable commit SHA.
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

# 6b. Container images pinned to a digest, not a movable tag.
floating_images=0
while IFS= read -r image; do
  [[ -n "$image" ]] || continue
  if [[ ! "$image" =~ @sha256:[0-9a-f]{64}$ ]]; then
    fail "container image '$image' is not pinned to a @sha256: digest — a republished tag would run unreviewed code"
    floating_images=$((floating_images + 1))
  fi
done < <(printf '%s\n' "$UNCOMMENTED" | grep -E '^[[:space:]]*image:' |
  sed -E 's/^[[:space:]]*image:[[:space:]]*//; s/["'\'']//g; s/[[:space:]]*#.*$//; s/[[:space:]]*$//')
if [[ "$floating_images" -eq 0 ]]; then
  ok "every container image is pinned to a digest"
fi

# 6c. A pip-installed CLI pinned to an exact version.
while IFS= read -r install_line; do
  [[ -n "$install_line" ]] || continue
  if [[ ! "$install_line" =~ semgrep==[0-9] ]]; then
    fail "'pip install' of semgrep is not version-pinned — pin it as 'semgrep==<version>' so a hijacked release cannot land silently"
  fi
done < <(printf '%s\n' "$UNCOMMENTED" |
  grep -E 'pip[0-9]*[[:space:]]+install[^|]*semgrep' || true)

exit "$EXIT_CODE"
