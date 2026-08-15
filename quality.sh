#!/usr/bin/env bash
# Local gate — mirrors `.github/workflows/ci.yml` quality checks.
set -euo pipefail

if [ -f "$HOME/.cargo/env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi

export RUSTFLAGS="-D warnings"
echo "Pre-deployment Quality Check (NEAT-AI-Backpropagation)"
echo "====================================================="

echo "Checking bash script syntax..."
find . -name "*.sh" -type f -not -path "./target/*" -not -path "./.git/*" -exec bash -n {} \;

echo "Running shellcheck on bash scripts..."
if ! command -v shellcheck &>/dev/null; then
  echo "shellcheck is required — install: https://github.com/koalaman/shellcheck#installing"
  exit 1
fi
SHELLCHECK_FAILED=0
while IFS= read -r script; do
  echo "  shellcheck: $script"
  if ! shellcheck -x -s bash "$script"; then
    SHELLCHECK_FAILED=1
  fi
done < <(find . -name "*.sh" -type f -not -path "./target/*" -not -path "./.git/*")
if [[ "$SHELLCHECK_FAILED" -ne 0 ]]; then
  echo "shellcheck: FAILED"
  exit 1
fi
echo "shellcheck: all scripts passed"

if [ -f "./../NEAT-AI-core/Cargo.toml" ]; then
  echo "Gating on unhandled breaking neat-core bump..."
  ./scripts/check-neat-core-version.sh
else
  echo "sibling ../NEAT-AI-core not found — skipping neat-core version gate (CI runs this for real)"
fi

echo "Validating auto-format PR workflow..."
./scripts/test-check-auto-format-workflow.sh
./scripts/check-auto-format-workflow.sh

echo "Validating version-increment PR workflow (runlib / GRQ-taxation)..."
./scripts/check-version-increment-workflow.sh

echo "Linting GitHub Actions workflows (actionlint)..."
if ! command -v actionlint &>/dev/null; then
  echo "actionlint is required — install: https://github.com/rhysd/actionlint/blob/main/docs/install.md"
  exit 1
fi
actionlint -no-color

echo "Validating the actionlint workflow-lint gate..."
./scripts/test-check-actionlint-gate.sh
./scripts/check-actionlint-gate.sh

echo "Validating CodeQL code-scanning workflow..."
./scripts/test-check-codeql-workflow.sh
./scripts/check-codeql-workflow.sh

echo "Validating Gitleaks secrets-detection workflow..."
./scripts/test-check-gitleaks-workflow.sh
./scripts/check-gitleaks-workflow.sh

echo "Validating Semgrep SAST scanning workflow..."
./scripts/test-check-semgrep-workflow.sh
./scripts/check-semgrep-workflow.sh

echo "Validating Renovate dependency-update config..."
./scripts/test-check-renovate-config.sh
./scripts/check-renovate-config.sh

echo "Validating dependency review is enabled on pull requests..."
./scripts/test-check-dependency-review.sh
./scripts/check-dependency-review.sh

echo "Validating default-branch protection policy..."
./scripts/test-check-branch-protection.sh
if command -v gh &>/dev/null && gh auth status &>/dev/null; then
  # Advisory: branch protection is a repository setting outside the checkout,
  # so only an administrator can repair drift — see CONTRIBUTING.md.
  if ! ./scripts/check-branch-protection.sh; then
    echo "WARNING: Develop branch protection does not satisfy the committed policy"
    echo "         — a repository administrator must apply it (CONTRIBUTING.md)"
  fi
else
  echo "gh unavailable or unauthenticated — skipping the live branch-protection check"
fi

echo "Running codespell preflight..."
if ! ./scripts/spell-check.sh; then
  echo "spell-check: FAILED — fix the typos above or update .codespellrc"
  exit 1
fi

echo "Running licence and dependency audit (cargo-deny)..."
if ! command -v cargo-deny &>/dev/null; then
  echo "cargo-deny is required — install: cargo install cargo-deny --locked"
  exit 1
fi
cargo deny check

echo "Checking formatting..."
cargo fmt --all -- --check

echo "Running linter..."
cargo clippy --workspace --all-targets --all-features -- \
  -D warnings \
  -D clippy::filter_next \
  -D clippy::collapsible_if

echo "Running tests..."
cargo test --workspace --all-features -- --test-threads=2

echo "Building documentation..."
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

echo "All quality checks passed!"
