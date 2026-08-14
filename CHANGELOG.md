# Changelog

All notable changes to NEAT-AI-Backpropagation are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Removed

- `train::MIN_SCORE_IMPROVEMENT` — a `pub const` no code ever read. Its doc
  comment described a score-improvement accept gate, but `run_train` accepts on
  `accept_always || after_mse < best_mse` alone and never applied a score
  threshold, so the constant documented behaviour that was never wired in.
  Being `pub` it was invisible to the workspace `dead_code` lint. The 1e-6
  production margin itself is unchanged — it is enforced by the GRQ consumer
  (`grq_backprop_score_improves`), not by this crate (issue #36).

### Changed

- The accumulate pass reduces each record's squared error through
  `neat_core::mse_record` instead of its own fused loop, so this crate holds no
  loss arithmetic (issue #33). `AccumulateReport::mse` is unchanged — core
  applies the same `1/outputs` mean over the same activations.
- `train --step-scale` now defaults to `0.01` (the top of `sweep`'s own grid)
  instead of `1.0`. A full coordinated jump moves every gene to a value
  proposed as if the others stayed put, which overshoots on large creatures —
  on the GRQ network it raised train-slice MSE 0.6515 → 0.7505 (issue #39).
- `train` resolves the learning rate **per epoch** from the configured
  strategy and journals it as `learningRate` on each epoch record. Previously
  only iteration 0 was ever evaluated (issue #39).
- The `backpropagation` tag — used verbatim as the GRQ check-in commit subject
  — is now marked `🌀` and drops the word "Backprop":
  `🌀 · 2 accepts / 4 epochs · score: … improved by …` (issue #31).

### Added

- A `workflow-lint` CI job runs `actionlint` over `.github/workflows` and feeds
  the `ci-required` aggregator, so a workflow YAML regression fails the build
  instead of surfacing on the next run. The linter is installed from a
  version-pinned, checksum-verified release.
  `scripts/check-actionlint-gate.sh` (tested by
  `scripts/test-check-actionlint-gate.sh`) gates the policy in `quality.sh` and
  CI: no invocation, a suppressed exit code, a non-strict shell, a lint job no
  other job needs, or an unpinned/unverified linter all fail (issue #26).
- `backpropagation/tests/scorer_boundary.rs` — process-boundary coverage for
  `score_creature`, the accept gate `run_train` uses when `--scorer` is set. A
  stub `rust_scorer` executable exercises the map and single-object stdout
  forms, the candidate-directory layout (`scorer-candidate/trained.json` plus
  the training-data argument), the non-zero-exit error, an empty score map, and
  unparsable stdout (issue #24).
- Parity dump round-trip coverage in `backpropagation/src/compare.rs`:
  `run_compare` writes a dump that `load_compare_dump` reloads to an equal
  value, the on-disk field names (`camelCase` plus `fromUUID` / `toUUID`) are
  asserted on the serialised JSON so a rename cannot break Rust ↔ TypeScript
  parity silently, `diff_compare_dumps` is checked against a perturbed
  `proposedBias`, and `load_compare_dump` fails loud on a missing, truncated,
  or wrongly-named dump (issue #23).
- `scripts/check-branch-protection.sh` — verifies the live `Develop` ruleset
  against the branch-protection policy now recorded in CONTRIBUTING.md
  (pull request required, ≥ 1 approving review, code-owner review, the
  `CI Required Checks` aggregator required, force-pushes blocked). Advisory in
  `quality.sh` and CI because only an administrator can repair a ruleset
  (issue #21).
- `.github/workflows/codeql.yml` — CodeQL `security-and-quality` analysis of
  this crate's own Rust on pull requests, pushes to `Develop`, and a weekly
  cron, so an advisory or query pack published after a merge is applied without
  waiting for the next PR. `scripts/check-codeql-workflow.sh` gates the policy
  in `quality.sh` and CI (issue #20).
- `backpropagation/tests/mse_surface_agreement.rs` — cross-surface guard that
  `compute_mse` (eval path) and `AccumulateReport::mse` (accumulate pass) agree
  within `nearly_equal` on the same records, for a feed-forward creature and
  for MINIMUM / MAXIMUM / IF aggregates, uncapped and under `max_records`
  (issue #34).
- `train --learning-rate-strategy` (`fixed`, `decay`, `adaptive`,
  `warm-restart`), `--learning-rate-decay`, and `--normalise-gradients` —
  `BackpropConfig` already supported all three but the trainer only ever ran a
  fixed rate with multi-path gradients un-normalised (issue #39).
- `gradient-check` CLI: accumulate once, sample genes, and compare each
  proposal Δ against a central finite-difference ∂MSE/∂θ — sign-agreement
  rates by gene class (issue #40). Confirms whether the aggregated learning
  direction is a descent direction before trusting full-network apply.
- `train` stamps `score`, `error`, and a run-summary `backpropagation` tag on
  `best.json` when `--scorer` is set, so GRQ workers can gate check-in on the
  full-corpus rust_scorer score without reading another program's tag
  (GRQ #3991 / #3952).
- Experimental standalone Rust trainer (`neat_ai_backpropagation`) with
  `compare`, `diff`, `train`, and `sweep` subcommands. Reverse-topo math
  stays in sibling `neat-core`; this crate ports the Lamarck config /
  apply surface and adds a trainDir-style apply-if-improved loop.
- Lamarck-identical build contract: Rust 1.95.0, unpinned `neat-core` path
  dependency gated by `neat-core.expected-version`, patch auto-increment, and
  auto-format lock sync.
- Production proof against GRQ `network.json` and the full
  `.trainData-binary_116` corpus (2,262,277 records). Rust continues
  blame through IF/MIN/MAX. A 12-gene hidden apply (11 inbound aggregate
  weights + 1 MAXIMUM bias) raised `rust_scorer` from 0.347586415202 to
  0.347614794359 (Δ +2.84e-5). Slice-only and full-net saturated applies
  overfit and are not a win (see `docs/production-win.json`).
- Train refuses re-entrant creatures (`forwardOnly: false`). Optional
  `--hidden-only`, `--accept-always`, and `sweep --skip-mse`.
