#!/usr/bin/env bash
# Validate the Markdown Lint workflow (Issue #44).
#
# `.markdownlint-cli2.yaml` has been committed since Issue #39, but nothing in
# CI ever read it — the rules were advisory and drifted with every hand-written
# README, CHANGELOG and PR summary. A lint job is only worth having if a
# violation actually blocks the merge, so the workflow must:
#   1. Run on `pull_request` events covering the default branch (`Develop`), so
#      a violation is raised before the prose reaches a branch anyone pulls.
#   2. Actually run markdownlint-cli2 — the CLI or the upstream action, not
#      merely a step named after it, and not an install with no invocation.
#   3. Report rather than rewrite. `--fix` (or the action's `fix: true`) edits
#      the runner's throwaway checkout and exits 0, so a fixable violation
#      merges unfixed while the job reads green.
#   4. Keep the verdict blocking: no `|| true` swallowing the exit code and no
#      `continue-on-error: true`.
#   5. Run its CLI steps under `set -euo pipefail`, so a failed install cannot
#      be reported as a clean lint.
#   6. Pin its supply chain: third-party actions on a 40-character commit SHA,
#      and any npm/npx install of markdownlint-cli2 on an exact version.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKFLOW="${1:-$REPO_ROOT/.github/workflows/markdown-lint.yml}"
EXIT_CODE=0

usage() {
  cat <<'EOF'
Usage: check-markdown-lint-workflow.sh [WORKFLOW_PATH]

Exits 0 when the workflow satisfies every rule listed in the script header,
1 when a rule is broken, and 2 when the workflow cannot be read.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "$WORKFLOW" ]]; then
  echo "FAIL: Markdown Lint workflow not found: $WORKFLOW" >&2
  echo "      Pull requests get no markdown lint — commit .github/workflows/markdown-lint.yml" >&2
  exit 2
fi

ok() { echo "OK   $WORKFLOW: $*"; }
fail() {
  echo "FAIL $WORKFLOW: $*" >&2
  EXIT_CODE=1
}

UNCOMMENTED="$(grep -vE '^[[:space:]]*#' "$WORKFLOW" || true)"

# Every mention of the linter, split into the lines that install it and the
# lines that run it. A step *named* "Markdown Lint" that echoes a message is
# prose, and an install on its own lints nothing — so `name:` labels are
# dropped before either split.
TOOL_LINES="$(printf '%s\n' "$UNCOMMENTED" | grep -E 'markdownlint-cli2' |
  grep -vE '^[[:space:]]*-?[[:space:]]*name:' || true)"
INSTALL_LINES="$(printf '%s\n' "$TOOL_LINES" |
  grep -E '(npm|pnpm|yarn|bun)[[:space:]]+(install|add|i|ci)([[:space:]]|$)' || true)"
LINT_LINES="$(printf '%s\n' "$TOOL_LINES" |
  grep -vE '(npm|pnpm|yarn|bun)[[:space:]]+(install|add|i|ci)([[:space:]]|$)' || true)"
# CLI invocations only — these are the ones that need strict bash around them.
CLI_LINT_LINES="$(printf '%s\n' "$LINT_LINES" | grep -vE '^[[:space:]]*-?[[:space:]]*uses:' || true)"

# 1. Pull-request trigger covering the default branch.
if grep -qE '^[[:space:]]*pull_request:' "$WORKFLOW"; then
  # Either an explicit `Develop` entry or a `*` wildcard covers the default
  # branch; a list naming only other branches does not.
  if grep -qE '^[[:space:]]*branches:.*(Develop|\*)' "$WORKFLOW" ||
    grep -qE '^[[:space:]]*-[[:space:]]*["'\'']?(Develop|\*)["'\'']?[[:space:]]*$' "$WORKFLOW"; then
    ok "pull_request trigger covers the Develop default branch"
  else
    fail "pull_request trigger does not cover Develop — PRs onto the default branch would go unlinted"
  fi
else
  fail "no pull_request trigger — a violation would only surface after merge"
fi

# 2. A lint really runs.
if [[ -n "$LINT_LINES" ]]; then
  ok "a markdownlint-cli2 lint is invoked (action or CLI)"
else
  fail "no markdownlint-cli2 lint is invoked — the job would report green without reading the markdown"
fi

# 3. The lint reports rather than rewrites.
if printf '%s\n' "$LINT_LINES" | grep -qE '(^|[[:space:]])--fix([[:space:]]|$)'; then
  fail "'--fix' rewrites the runner's throwaway checkout and exits 0 — a fixable violation would merge unfixed"
fi
if printf '%s\n' "$UNCOMMENTED" | grep -qE '^[[:space:]]*fix:[[:space:]]*(true|"true"|yes)[[:space:]]*$'; then
  fail "the action's 'fix: true' input rewrites the runner's checkout and exits 0 — a violation would merge unfixed"
fi

# 4. The lint's verdict reaches the runner.
if printf '%s\n' "$LINT_LINES" | grep -qE '\|\|[[:space:]]*(true|:|echo\b)'; then
  fail "the lint's exit code is discarded with '|| true' — a violation would pass"
fi
if grep -qE '^[[:space:]]*continue-on-error:[[:space:]]*true' "$WORKFLOW"; then
  fail "continue-on-error: true — the job's failures would not block the merge"
fi

# 5. A CLI lint runs under strict bash, so a failed install fails loud.
if [[ -n "$CLI_LINT_LINES" ]]; then
  if grep -qE 'set[[:space:]]+-euo[[:space:]]+pipefail' "$WORKFLOW"; then
    ok "the lint step runs under strict bash (set -euo pipefail)"
  else
    fail "the lint step has no 'set -euo pipefail' — a failed install or setup command would be reported as a clean lint"
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

# 6b. The linter itself pinned to an exact version — an install or an `npx`
# resolving `latest` runs whatever was published minutes ago.
floating_installs=0
while IFS= read -r install_line; do
  [[ -n "$install_line" ]] || continue
  if [[ ! "$install_line" =~ markdownlint-cli2@[0-9] ]]; then
    fail "markdownlint-cli2 is not version-pinned in '$(printf '%s' "$install_line" | sed -E 's/^[[:space:]]+//')' — pin it as 'markdownlint-cli2@<version>' so a hijacked release cannot land silently"
    floating_installs=$((floating_installs + 1))
  fi
done < <(printf '%s\n' "$INSTALL_LINES"
  printf '%s\n' "$LINT_LINES" | grep -E '(^|[^A-Za-z0-9_./-])npx([[:space:]]|$)' || true)
if [[ "$floating_installs" -eq 0 ]]; then
  ok "markdownlint-cli2 is version-pinned wherever it is fetched"
fi

exit "$EXIT_CODE"
