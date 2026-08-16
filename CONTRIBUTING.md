# Contributing to NEAT-AI-Backpropagation

Thanks for improving **NEAT-AI-Backpropagation** — an experimental standalone
backpropagation trainer for already-fit NEAT-AI creatures. This guide
summarises how to build, test, and submit changes.

## Repository layout

Clone **NEAT-AI-core** and **NEAT-AI-Backpropagation** as siblings:

```text
parent/
  NEAT-AI-core/
  NEAT-AI-Backpropagation/
  NEAT-AI/                 # optional, TypeScript dual-run
  NEAT-AI-scorer/          # optional, rust_scorer
```

The `neat-core` path dependency in
[`backpropagation/Cargo.toml`](./backpropagation/Cargo.toml) resolves to
`../../NEAT-AI-core/neat-core`.

## Prerequisites

- **Rust** — pinned in [`rust-toolchain.toml`](./rust-toolchain.toml).
  Workspace profiles: fast `dev` (`debug = "line-tables-only"`), fully
  optimised `release` (`opt-level = 3`, fat LTO, `codegen-units = 1`),
  plus same-host `-C target-cpu=native` in [`.cargo/config.toml`](./.cargo/config.toml)
  (see README **Build profiles**, issue #88).
- **shellcheck** — lints bash scripts
- **actionlint** — lints GitHub Actions workflow YAML (`brew install actionlint`)
- **cargo-deny** — `cargo install cargo-deny --locked`
- **codespell** — `pip install --user codespell`

## Local gate

```bash
./quality.sh < /dev/null
```

This mirrors CI: shellcheck, actionlint, the auto-format workflow validator, the
version-increment workflow validator, the CodeQL workflow validator, the
Gitleaks workflow validator, the Semgrep workflow validator, the
Markdown Lint workflow validator, the Renovate config validator, the
branch-protection policy checker, codespell,
cargo-deny, fmt `--check`, clippy with warnings denied, tests, and rustdoc.

## Branch protection

`Develop` is protected by a repository **ruleset**. The policy is deliberate,
and `./scripts/check-branch-protection.sh` verifies it against the live API:

| Rule | Setting | Why |
| ---- | ------- | --- |
| `pull_request` | required | No direct pushes to `Develop`; every change is reviewable before it merges. |
| `required_approving_review_count` | ≥ 1 | A single account cannot merge its own change. |
| `require_code_owner_review` | on | [`.github/CODEOWNERS`](./.github/CODEOWNERS) is advisory without it. `auto-format.yml` and `version-increment.yml` mint GitHub App push tokens, so an unreviewed workflow edit is an unreviewed secret grab. |
| `required_status_checks` | `CI Required Checks` | The `ci-required` aggregator in [`ci.yml`](./.github/workflows/ci.yml) only gates merges when it is registered as required. |
| `non_fast_forward` | required | Merged history on `Develop` cannot be rewritten by a force-push. |

**Signed commits are not required.** Auto Format and Version Increment push
unsigned bot commits back to PR branches, so requiring signatures would block
the repository's own automation.

Rulesets are a repository *setting*, not a committed file — the same situation
as Dependabot alerts in [SECURITY.md](./SECURITY.md#automated-scanning). Only an
administrator can change one, so the checker is **advisory** in `quality.sh` and
in CI: drift is printed as a loud `FAIL` (and a CI warning annotation) rather
than blocking every merge behind a setting contributors cannot fix. Run it any
time:

```bash
./scripts/check-branch-protection.sh < /dev/null
```

An administrator repairs drift by patching the ruleset (id from
`gh api repos/stSoftwareAU/NEAT-AI-Backpropagation/rulesets`):

```bash
gh api --method PUT \
  repos/stSoftwareAU/NEAT-AI-Backpropagation/rulesets/RULESET_ID \
  --input ruleset.json   # existing rules, plus the table above
```

Code scanning runs in GitHub Actions, not locally — see
[Code scanning](./README.md#code-scanning). Changing
[`.github/workflows/codeql.yml`](./.github/workflows/codeql.yml) must keep
`./scripts/check-codeql-workflow.sh` green.

Every pull request diff is scanned for committed secrets — see
[Secrets detection](./README.md#secrets-detection). Changing
[`.github/workflows/gitleaks.yml`](./.github/workflows/gitleaks.yml) must keep
`./scripts/check-gitleaks-workflow.sh` green.

Every pull request is also scanned by Semgrep — see
[Static analysis](./README.md#static-analysis-semgrep). Changing
[`.github/workflows/semgrep.yml`](./.github/workflows/semgrep.yml) must keep
`./scripts/check-semgrep-workflow.sh` green.

Every pull request's Markdown is linted against
[`.markdownlint-cli2.yaml`](./.markdownlint-cli2.yaml) — see
[Markdown linting](./README.md#markdown-linting). Changing
[`.github/workflows/markdown-lint.yml`](./.github/workflows/markdown-lint.yml)
must keep `./scripts/check-markdown-lint-workflow.sh` green.

External crate bumps arrive as Renovate PRs under a 24-hour quarantine — see
[Dependency updates](./README.md#dependency-updates). Changing
[`renovate.json`](./renovate.json) must keep
`./scripts/check-renovate-config.sh` green.

On each PR the **Auto Format** workflow
([`.github/workflows/auto-format.yml`](./.github/workflows/auto-format.yml))
runs `cargo fmt --all` and `cargo update -p neat-core`, then pushes any
tracked-tree changes back to the PR branch. It does **not** bump
`neat-core.expected-version` — acknowledge breaking neat-core bumps
deliberately in the same PR that updates this crate for them.

## Version bumping

**Every binary-affecting change must bump the patch version in
[`backpropagation/Cargo.toml`](./backpropagation/Cargo.toml)** (and keep
`Cargo.lock` in sync). Remote GRQ runners use the same pattern as
[`runlib.sh`](https://github.com/stSoftwareAU/GRQ-taxation/blob/Develop/scripts/runlib.sh):
they compare the installed `neat_ai_backpropagation` version marker against
`Cargo.toml` and skip rebuilding when they match. Forgetting to bump leaves
stale binaries on remote machines.

CI also runs a **Version Increment** workflow
([`.github/workflows/version-increment.yml`](./.github/workflows/version-increment.yml))
that auto-increments the patch on a pull request when `backpropagation/src/`
has changed — but only if the PR branch is not already *ahead* of Develop
(same approach as GRQ-taxation). Bumping locally when your change touches
`backpropagation/src/` keeps the version correct and avoids an extra bot
commit.

**Never ship a crate version behind `origin/Develop`.** A merge conflict that
silently takes Develop's older `version` used to look like “already bumped”
and skip the bot — remotes would then rebuild an older `trainDir` / FFI
binary. `scripts/check-crate-version-no-downgrade.sh` (and the bump script)
refuse that with `sort -V`; `quality.sh` runs the gate on every PR.

Docs-only or CI-config-only changes do not need a bump. Record notable changes
under **[Unreleased]** in [`CHANGELOG.md`](./CHANGELOG.md).

## Production proof

Tiny identity-chain fixtures are unit regression only. A claimed win must use
the production creature and real `.bin` records — see the README production
win protocol.
