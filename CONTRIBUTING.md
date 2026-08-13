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

- **Rust** — pinned in [`rust-toolchain.toml`](./rust-toolchain.toml)
- **shellcheck** — lints bash scripts
- **cargo-deny** — `cargo install cargo-deny --locked`
- **codespell** — `pip install --user codespell`

## Local gate

```bash
./quality.sh < /dev/null
```

This mirrors CI: shellcheck, the auto-format workflow validator, the
version-increment workflow validator, the Renovate config validator, codespell,
cargo-deny, fmt `--check`, clippy with warnings denied, tests, and rustdoc.

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
has changed — but only if the PR branch has not already bumped it (same
approach as GRQ-taxation). Bumping locally when your change touches
`backpropagation/src/` keeps the version correct and avoids an extra bot
commit.

Docs-only or CI-config-only changes do not need a bump. Record notable changes
under **[Unreleased]** in [`CHANGELOG.md`](./CHANGELOG.md).

## Production proof

Tiny identity-chain fixtures are unit regression only. A claimed win must use
the production creature and real `.bin` records — see the README production
win protocol.
