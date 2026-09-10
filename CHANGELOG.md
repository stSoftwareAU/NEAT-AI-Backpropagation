# Changelog

All notable changes to NEAT-AI-Backpropagation are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Lockfile integrity gate (`scripts/check-lockfile-integrity.sh`,
  `scripts/lockfile_integrity.py`), wired into `quality.sh` and CI. It fetches
  the crates.io sparse index for every registry package in `Cargo.lock` and
  fails unless the recorded sha256 matches the registry's `cksum` for that
  exact version and every recorded dependency is genuinely declared by that
  version's published manifest — the comparison `cargo` never makes, so a
  substituted sub-dependency can no longer compile silently. Dangling
  dependency references, sources outside the one registry `deny.toml` allows,
  and registry packages with no checksum fail too; an unreachable index exits 2
  rather than passing. Raised by issue #148, which reported `serde_json`
  1.0.151 depending on `zmij` instead of `ryu`: `zmij` is `ryu`'s legitimate
  successor from the same author and the lockfile is correct — see
  [`docs/audit/issue-148-serde-json-zmij-dependency.md`](./docs/audit/issue-148-serde-json-zmij-dependency.md)
  for the registry evidence.
- Trust-region update budget for the whole-creature apply: `--step-scale` is a
  *per-gene* factor, so the aggregate move grows with the number and magnitude
  of the genes that move — the `0.01` default was justified against a
  ~16.6k-parameter GRQ network that has since grown to thousands of neurons
  and tens of thousands of synapses. `train` now measures the update it is
  about to apply (changed genes, L1 / L2 / RMS, relative L2 and relative RMS,
  reported for the whole update and split by bias / weight and hidden /
  output) and rescales the whole proposal to fit a configured budget:
  `--trust-region-l2`, `--trust-region-rms`, `--trust-region-relative-rms`,
  `--trust-region-bias-l2`, `--trust-region-weight-l2` and
  `--trust-region-max-genes` (an L0 budget that keeps the largest moves and
  holds the rest). The tightest budget decides the rescale, and the trust
  region only ever shrinks the step — never amplifies it. Every budget is off
  by default, which is the historical fixed-step apply byte for byte. The run
  header records the configured `trustRegion`; every epoch and candidate line
  records the requested `stepScale`, the `realisedStepScale` actually applied,
  the `updateScale` the budget imposed, the `trimmedGenes` it held back and the
  realised `update` norms, so a scorer-guided rung's move size sits
  beside the `scoreDelta` it produced, and `blocks.json` records the same
  norms per block candidate. An unusable budget (zero, negative, non-finite,
  or `maxChangedGenes: 0`) is refused before the corpus is read, and no budget
  is allowed to "bound" a non-finite update norm, an unmeasurable
  `relativeRms` fails rather than being read as satisfied, and a budget that
  underflows the step to zero is refused instead of inverting into a full step.
  Because a budget clips every step above it to the same update, the realised
  step is snapped down to 12 significant digits: a ladder's clipped rungs are
  scored once instead of once per rung, and the backtracking line search halves
  the step it realised rather than re-applying an identical candidate.
  `scripts/run-trust-region-experiment.sh` sweeps several budgets against an
  unbudgeted control arm and prints accepted epochs, scorer gain, realised
  update norm and wins/hour per arm. The budget is reachable over the C ABI as
  `trustRegion` (issue #109).

- Evidence-driven sparse target selection: `blocks` no longer spreads its
  scarce scorer runs over focus neurons picked without regard to the learning
  signal. `--target-selection evidence` (the new default) ranks candidate
  targets from the **same single accumulation pass** on accumulated absolute
  error mass, proposal magnitude relative to the parameter it moves,
  activation coverage / range, per-record direction consistency and, as a
  secondary term, fan-in / fan-out; longest-path depth is recorded as a rank
  feature but deliberately not scored, and the relative-proposal term is gated
  by the absolute move so an infinitesimal proposal against a near-zero
  parameter cannot outrank a real one. `--target-selection random` retains the
  uniform draw as the control arm, and `--random-control-fraction` reserves a
  configurable share of every draw for a uniform draw over the targets
  exploitation did not take — so a run measures its own heuristic and still
  finds accidental wins. Every focus candidate in `blocks.json` carries
  `selection` (`source`, `rank`, `score` and each rank feature), every scorer
  run is timed into `scorerSeconds`, and a scored run writes
  `selectionComparison` — `winsPerHour`, `scoreGainPerHour`, `wins`,
  `candidatesScored`, `totalScoreGain` and `bestScoreDelta` per arm, with
  `null` rates rather than a rate divided out of no scorer time.
  `scripts/run-target-selection-benchmark.sh` runs the same creature, corpus
  and seed through both arms and prints wins/hour and score gain/hour for
  each; its targets are env vars, so no stock-market logic lands in this
  public library. Two boundaries are stated rather than left to be
  discovered: the in-run control arm draws from what exploitation did not
  take, so `selectionComparison` compares against the ranking's *tail* and
  the two-run script is the unbiased measurement; and a `sparse_ratio` below
  `1.0` accumulates for its own random subset alone, which `blocks` now warns
  about because the ranking can then only rank that subset. A control
  fraction on a `--target-selection random` run is refused rather than
  ignored, as `train` refuses scorer settings on an MSE run (issue #108).

- Gradient diagnostics by gene class and squash: `gradient-check` now labels
  every probed gene with the facets a NEAT creature actually varies over —
  bias vs weight, output vs hidden, squash name, aggregate vs ordinary
  (`SquashType::is_aggregate()`), longest-path depth, fan-in / fan-out,
  activation health read from the accumulate trace (`active`, `saturated`,
  `lowActivity`, `unobserved`) and proposal-magnitude decade — and aggregates
  sign agreement, applied-proposal improvement and gradient error across all
  of them. Gradient error is the proposal judged against the finite
  difference in its own units: a descent step is inverted back through
  `lr × step` to the gradient it implies (`proposalGrad`), giving
  `gradAbsError` / `gradRelError`, so a clamped or mis-scaled proposal is
  visible. Each sampled gene is also **applied on its own**, so a row records
  whether the proposal really lowered slice MSE (`improved`) beside the
  first-order prediction it is compared against (`predictedDeltaMse` vs
  `actualDeltaMse`); run cost is `(2 + 3 × sampled genes)` MSE passes, bounded
  by the existing sample caps. `gradient-check.json` gains `schemaVersion`
  (now `2`), `neatCoreBaseline`, the `seed`, a creature fingerprint, `byFacet`
  and ranked, non-overlapping `bestClasses` / `worstClasses`
  (`--facet-min-scored`, `--rank-limit`), and
  the run leaves a concise `summary.txt` for unattended readers. The same
  seed over the same creature, corpus and caps writes byte-identical
  artefacts. `scripts/run-gradient-diagnostics.sh` is the documented GRQ
  integration-testing command — every target is an env var, so no
  stock-market logic lands in this public library (issue #107).

- Blockwise candidate generation: `blocks` accumulates the corpus **once**
  and applies that one learning signal to a small region at a time instead
  of moving every gene together. Five block strategies sit beside the
  whole-creature `global` apply — `neuron` (a neuron's bias plus every
  incident synapse), `neighbourhood` (a neuron plus its neighbours out to
  `--radius` hops), `output-head`, `subgraph` (a seeded random connected
  walk) and `top-genes` (the loudest genes by proposal magnitude) — with
  focus neurons ranked by how much learning wants to move them. Every
  candidate is written as a standalone creature, gated by
  `neat_core::creature_validate`, and scored independently when `--scorer`
  is supplied; `blocks.json` records the strategy, focus, selected neuron
  UUIDs, each selected synapse's export index and endpoints, the block's
  `geneCount`, moved-gene counts, MSE and scorer deltas, and `scoreWin`
  against `--min-score-improvement`. Nothing is dropped silently: empty and
  duplicate blocks are reported as `droppedEmptyBlocks` /
  `droppedDuplicateBlocks`, a block whose genes all held still is counted in
  `unmovedBlocks` instead of writing a candidate identical to the incumbent,
  and `--radius 0` is refused because it would turn every neighbourhood block
  into a duplicate of its `neuron` block.
  `scripts/run-blockwise-benchmark.sh` runs `global` and the blockwise
  strategies over the same creature, corpus and step scale and prints
  candidates, scorer wins, elapsed seconds and wins/hour for each. On its
  generated corpus, one accumulation pass yielded 1 candidate / 1 scorer win
  for `global` against 14 candidates / 14 wins for the blockwise strategies
  (best `scoreDelta` `+4.29e-4` global vs `+3.87e-4` blockwise) — more
  independently judged candidates for the same corpus cost, on a creature too
  small for the whole-creature apply to overshoot. Point it at the production
  creature with `CREATURE=` / `DATA_DIR=` to measure the comparison there
  (issue #105).

- Scored step-scale ladder: `train --acceptance scorer --step-scale-ladder`
  applies the epoch's single accumulation at every rung of a configurable grid
  (default `0.0001,0.00025,0.0005,0.001,0.0025,0.005,0.01`), scores the whole
  grid in **one** `rust_scorer` invocation — the scorer takes a directory of
  creatures, so results are matched back by file stem and a candidate it did
  not report fails the run — and keeps the best scorer improvement rather than
  the first candidate to clear the epsilon. Every rung is journalled with its
  own `stepScale`, `candidateMse`, `mseDelta`, `candidateScore` and
  `scoreDelta`; a rung that improved but lost to a better one is
  `scoreNotBest`; ties keep the smaller step; no winner leaves the incumbent
  unchanged. The epoch line gains `ladderRungs` and the run header records the
  grid. `--mse-pre-screen` still drops a rung whose slice MSE did not fall
  before the batch is scored. The ladder supersedes `--max-backtracks`,
  requires `--acceptance scorer`, and refuses a rung outside `(0, 1]` rather
  than letting the applier silently rewrite it. `stepScaleLadder` crosses the C
  ABI. `scripts/run-step-scale-ladder-experiment.sh` runs one corpus through
  both searches and prints accepted epochs, scorer gain, wall clock and
  wins/hour. **Not yet measured against `rust_scorer` or GRQ history**: on the
  script's generated corpus, run with `STEP_SCALE=1.0` (an overshooting start,
  not the script's `0.01` default) and a throwaway stand-in scorer whose
  fitness is the negative full-corpus MSE, the line search accepted 1 epoch for
  `+1.225e-2` while the ladder accepted 4 for `+1.742e-2`; at the default step
  both searches found the same winner. Point the script at the production
  creature and corpus for the real comparison (issue #106).

- Scorer-guided acceptance: `train --acceptance scorer` puts `NEAT-AI-scorer`
  in the accept/rollback loop instead of training-slice MSE. The baseline is
  scored before epoch 1, every attempted candidate (each backtracking step
  included) is scored and journalled as a `"kind":"candidate"` line carrying
  the MSE delta beside the scorer delta, and a candidate is kept only when
  fitness rises by `--min-score-improvement` (default `1e-6`, the production
  win margin). `--mse-pre-screen` optionally drops a candidate whose slice MSE
  did not fall before paying for a scorer run — it is a gate in front of the
  scorer, so a scorer win MSE disagreed with is lost. `--accept-always`, a
  missing `--scorer`, and a scorer knob on an `--acceptance mse` run are all
  refused rather than silently degrading to MSE. The run header carries the
  run's own `baselineScore` (a candidate line's is the incumbent, which moves
  with every accept), every epoch line now carries an `acceptReason`, and
  `acceptance` / `minScoreImprovement` / `msePreScreen` cross the C ABI. MSE
  remains the default, so existing runs are unchanged. `scripts/run-scorer-guided-experiment.sh`
  runs one corpus through both modes: on its generated corpus the MSE loop
  accepted a candidate that cut slice MSE `14.839 → 3.490` while real
  `rust_scorer` fitness fell `0.7032 → −0.3324`, which scorer-guided
  acceptance rejected (issue #104).

- Every trained creature is validated by `neat_core::creature_validate`
  before it is scored, written or returned (`validate::TrainedTopology`).
  `train` gates the creature it finishes with, once per completed run;
  `sweep` gates each candidate as it is produced. The pinned source neuron /
  synapse counts and `forwardOnly` are passed as `ValidateOptions`, so the
  gate also proves training preserved the topology. A diverged run that
  produced a non-finite bias now fails loudly, naming neat-core's reason,
  message and the offending index, instead of writing a creature whose
  biases serialise as `null` (issue #94).

- The auto version bump now fires on **every** build-affecting path, not just
  `backpropagation/src/`. `scripts/build-affecting-paths.sh` is the single
  source of truth (sources, both manifests, `Cargo.lock`, `.cargo/config.toml`,
  `rust-toolchain.toml`, `include/`); the bump script diffs it and
  `scripts/check-version-increment-workflow.sh` fails the PR when the
  workflow's `paths:` filter drops one. Unattended machines rebuild only when
  the crate version moves, so a dependency, profile or toolchain change that
  skipped the bump left them running a stale library (issue #95).

- Root `README.md` opens with a full-width banner hot-linking the hub social
  preview `neat-ai-backpropagation.png` from NEAT-AI `Develop`, so crate and
  GitHub README branding stay in sync with the single source of artwork
  (issue #83).
- Refuse a `neat_ai_backpropagation` crate version strictly behind
  `origin/Develop` (`scripts/check-crate-version-no-downgrade.sh`, wired into
  `quality.sh` and the version-increment bump script). A merge conflict that
  took Develop's older token used to look like “already different” and skip
  the bot — remotes would rebuild a downgraded trainDir / FFI binary. Equal
  versions may still auto-patch-bump; ahead versions are accepted without a
  further bump (issue #87).

- A `cdylib` C ABI for an in-process `trainDir`, so NEAT-AI can `dlopen` the
  library instead of spawning the CLI on every memetic run. `neat_backprop_train`
  takes UTF-8 JSON in and hands back an owned buffer with an explicit length
  (`NeatBackpropBuffer`), released through `neat_backprop_buffer_free`;
  `neat_backprop_abi_version` / `neat_backprop_version` are the probes. The
  creature crosses the boundary as JSON text, not a path, and every CLI `train`
  flag NEAT-AI forwards — including `--max-records` sampling (#77) and
  `--trace-store` (#78) — is a request field. Null pointers, non-UTF-8 bytes,
  malformed JSON, trainer errors and caught panics all return a non-zero status
  *and* a message in the same buffer. Declarations live in
  `include/neat_ai_backpropagation.h` (issue #84).

- `train --trace-store DIR` writes NEAT-AI `CreatureTrace` artifacts. An epoch
  that lowered the best MSE writes `best-trace.json` beside `best.json`; one
  that did not writes `<store>/failed/epoch-<N>.json`, the same failed-candidate
  store `TrainOptions.traceStore` fills in TypeScript. The payload is the
  UUID-only creature export with NEAT-AI `NeuronState` / `SynapseState` objects
  on every gene the epoch accumulated, so a Rust epoch is as debuggable as a
  TypeScript one and the NEAT-AI bridge can attach it to `TrainingResult.trace`
  (issue #78).
- `train --max-records N` now draws a **seeded random** sample instead of the
  first *N* records in directory scan order. The cap resolves to a rate of
  `N / total_records`, and every `.bin` file contributes
  `ceil(file_records × rate)` shuffled-then-sorted indexes — a port of NEAT-AI's
  TypeScript `selectFileSampleIndexes`, which NEAT-AI feeds through
  `TrainOptions.trainingSampleRate`. A capped epoch therefore spans the whole
  corpus rather than over-fitting its earliest files. `--seed` makes the draw
  reproducible and the new `--disable-random-samples` reduces it to each file's
  leading prefix (NEAT-AI `disableRandomSamples`). The sample is planned once
  per run, so baseline MSE, every epoch's accumulate and every candidate's
  post-apply MSE score the identical records and accept / rollback stays a
  like-for-like comparison. `journal.jsonl` gains `sampledRecords`,
  `totalRecords` and `disableRandomSamples` (issue #77).
- **Markdown linting** — `.github/workflows/markdown-lint.yml` runs
  `markdownlint-cli2` against the `.markdownlint-cli2.yaml` config that had been
  committed since issue #39 but that nothing in CI ever read, so a heading,
  indentation or fencing violation now blocks the merge instead of drifting. The
  job reports rather than rewrites: `--fix` would edit the runner's throwaway
  checkout and exit 0, merging the violation unfixed. The linter is installed at
  an exact version, and `scripts/check-markdown-lint-workflow.sh` gates the
  policy in `quality.sh` and CI (issue #44).
- **Semgrep SAST scanning** — `.github/workflows/semgrep.yml` runs `semgrep ci
  --config p/default` over every pull request in the digest-pinned official
  Semgrep image, a second opinion alongside CodeQL that also reads the shell and
  workflow YAML the Rust analysis never sees. `--no-suppress-errors` overrides
  the `semgrep ci` default that exits 0 when Semgrep itself errors, so a crashed
  scan fails the job instead of reading as a clean one. The scan is
  unauthenticated when `SEMGREP_APP_TOKEN` is unset, so bot pull requests are
  covered identically. One rule is excluded and justified in the workflow —
  `renovate-missing-minimum-release-age` demands a ≥ 7-day embargo, which
  contradicts the 24-hour quarantine committed in `renovate.json`.
  `scripts/check-semgrep-workflow.sh` gates the policy in `quality.sh` and CI
  (issue #43).
- **Gitleaks secrets detection** — `.github/workflows/gitleaks.yml` scans every
  pull request diff for committed credentials. Licensed runs use
  `gitleaks-action@v2`; licence-less runs (Renovate, Dependabot — bot PRs get no
  Actions secrets) fall back to the version-pinned, checksum-verified
  open-source CLI, so those diffs are scanned rather than silently skipped.
  `scripts/check-gitleaks-workflow.sh` gates the policy in `quality.sh` and CI
  (issue #42).
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
- The confirmed `uuid`-dedup defect in the sibling `NEAT-AI-Lamarck` copy of
  `tags.rs`, found while fixing this repo's own copy in PR #101, is recorded in
  `docs/audit/issue-35-neat-ai-core-duplication.md` beside Finding 3 instead of
  only in that PR's summary, with a `NEAT-AI-Lamarck` row in the audit's Filing
  status table (issue #140). `scripts/check-cross-repo-defect-record.sh` keeps
  every such record complete enough to re-file upstream — source citations, how
  it was confirmed, its provenance and its upstream filing status — and
  `quality.sh` runs it on every PR.

### Changed

- The CLI argument groups repeated across `train`, `sweep`, `blocks` and
  `gradient-check` are declared once and flattened into each subcommand
  (issue #137): `CorpusArgs` (`--max-records`, `--seed`), `RateArgs`
  (`--learning-rate`, `--maximum-bias-adjustment-scale`,
  `--maximum-weight-adjustment-scale`) and `GeneScopeArgs` (`--outputs-only`,
  `--hidden-only`). Flag names and defaults are unchanged; only the `--help`
  descriptions were reworded to cover every subcommand that carries the flag.

- CodeQL analyses this repository's own code only.
  `.github/codeql/codeql-config.yml` excludes `NEAT-AI-core/**`: the
  `setup-rust-workspace` composite checks the sibling out so the `neat-core`
  path dependency resolves, and Rust extraction (`build-mode: none`) then read
  it as source and filed NEAT-AI-core's findings against this repository's
  Security tab, where nobody can fix them. NEAT-AI-core runs its own CodeQL.

- GitHub Actions audit follow-ups (issues #117, #118, #125, #126, #128):
  - `.github/actions/setup-rust-workspace` replaces `setup-neat-core` and now
    installs the pinned Rust toolchain as well as the NEAT-AI-core sibling,
    taking the rustup `components` list as an input. The
    `dtolnay/rust-toolchain` pin lived in five call sites across `ci.yml`
    (two jobs), `codeql.yml`, `security.yml` and `auto-format.yml`; it lives
    in one now. Each workflow keeps its own `actions/checkout`, because
    GitHub has to find a local composite action in the workspace before it
    can run any step in it (issue #126).
  - `gitleaks/gitleaks-action` moves from v2.3.9 to v3.0.0 — the same inputs,
    outputs and behaviour on the Node 24 runtime, which GitHub makes
    mandatory when Node 20 leaves the hosted runners (issue #118).
  - The Markdown Lint workflow gates pull requests only; its `push:` trigger
    on `Develop` re-ran, after every merge, the run that had already gated
    the pull request. `workflow_dispatch` replaces it for manual runs
    (issue #117).
  - The Semgrep container image carries an explicit `:1.173.0` tag beside its
    digest. The digest still pins it, but Renovate's docker manager resolves
    bumps from the tag, so a bare digest never received one (issue #125).
  - `SECURITY.md` documents the fast lane past Renovate's 24-hour
    `minimumReleaseAge` quarantine for an actively exploited advisory — who
    approves it, how to raise the bump, and how to remove the carve-out
    afterwards (issue #128).

- `TrainEpochRecord` requires the new `acceptReason` field, so a `journal.jsonl`
  line written before 0.1.26 no longer deserialises into it. An epoch's verdict
  cannot be inferred after the fact, and inventing a default would report a
  guess as a record — the read fails loudly instead (issue #104).

- Workspace build profiles follow VibeCoding#4159 / issue #88: `dev` uses
  `debug = "line-tables-only"` for faster rebuilds; `release` is
  workspace-wide `opt-level = 3`, `lto = "fat"`, `codegen-units = 1`
  (no longer scoped only to `neat_ai_backpropagation`); non-`wasm32`
  builds get `-C target-cpu=native` from `.cargo/config.toml` for
  same-host GRQ / local release artefacts.
- `train` no longer prints a per-epoch progress line to stderr. Epoch detail
  remains in `journal.jsonl`; the CLI still prints the one-line
  `train: baseline_mse=… best_mse=…` summary. Quiets Deno FFI / parallel
  memetic hosts that previously drowned in `epoch 1:` spam.
- `TrainRequest.creature` is now a `TrainCreature` — `Path(&Path)` for the CLI
  or `Json(&str)` for the C ABI — so an in-process caller does not have to write
  the creature to a temporary file first. `TrainResult` gained `best_json`, the
  exact bytes written to `best.json`, so the ABI returns the trained creature
  without re-reading it (issue #84).

- The step-scale sanitising rule (finite and positive, capped at `1.0`,
  otherwise `1.0`) now lives in one place — `backprop::effective_step_scale`.
  `apply_learnings_with`, `run_gradient_check` and `run_train` each held their
  own copy, so a policy change had to land identically on all three or the step
  the journal reports and backtracking halves would silently disagree with the
  step actually applied. Behaviour is unchanged (issue #55).
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

### Fixed

- **Builds against neat-core 0.13.0 — `CompiledNetwork`'s fields went private
  (neat-core #625 / #633; GRQ #4724).** neat-core 0.12.0 made every
  `CompiledNetwork` field private behind borrow-only accessors (neat-core #625 /
  #633) and 0.13.0 followed the same day. Every GRQ host builds the Rust
  consumers from the sibling neat-core at head, so `rust_scorer` failed to
  compile fleet-wide within minutes and, with no fallback engine, the fleet
  stopped scoring ([GRQ
  #4724](https://github.com/stSoftwareAU/GRQ/issues/4724)).
  `propagate_layout.rs` read `activations` and `hint_values_buffer` directly;
  those reads now go through `activations()` and `hint_values()`.
  `neat-core.expected-version` acknowledges 0.13.0 (the earlier deliberate hold
  on 0.12.0 is lifted now that it has merged).

- The neat-core breaking-bump gate
  ([`scripts/check-neat-core-version.sh`](./scripts/check-neat-core-version.sh))
  no longer fails every local `./quality.sh` run because of a branch nobody in
  this repo chose. It compared `neat-core.expected-version` against the sibling
  `../NEAT-AI-core` **working tree** — a shared developer checkout that may sit
  on any unmerged branch — so a local branch carrying `0.12.0` failed the gate
  while neat-core's `Develop` was `0.11.3`, inside the recorded baseline. The
  version is now read from the branch that *governs* neat-core (`--core-ref`,
  default `Develop`; `origin/REF` wins over a local `REF`), which is also what
  CI clones and builds. Divergence is never silent: the gate warns when the
  working tree carries a different version from the governing branch (the
  unpinned `path` dependency compiles the working tree, so a local build
  against an unmerged neat-core is reported), and warns again when it falls
  back to the working tree because no such branch resolves. `--core-ref ''`
  restores the previous working-tree comparison. Adds
  `scripts/test-check-neat-core-version.sh` — the gate had no test companion —
  and wires it into `quality.sh` and CI beside the checker, as every other gate
  already is. The recorded baseline moves `0.11.2 -> 0.11.3` with the review of
  that (source-free) bump written into the file; it is deliberately **not**
  moved to `0.12.0`, which exists only on an unmerged neat-core branch
  (issue #141).

- `best.json` — and the identical bytes returned over the C ABI as
  `bestCreatureJson` — no longer re-attach the **source** creature's
  `uuid`. That uuid is a content-derived v5 hash over the creature's
  neurons, synapses and `input`, and training moves every bias and weight,
  so the inherited value described content that no longer existed. NEAT-AI's
  `makeUUID` short-circuits on a present uuid and `Fitness` deduplicates its
  evaluation queue by uuid, so a trained creature wearing its parent's
  identity could be handed a score it never earned without ever being
  evaluated. `CreatureMeta` now keeps only `tags` — which are excluded from
  the uuid hash — and per-neuron `uuid` (an *input* to the hash, not the
  hash) is still preserved verbatim (issue #101).
- `observation_width` no longer builds its widthless fixture through
  `parse_creature_json`: neat-core now rejects `input < 1` inside the loader
  itself (NEAT-AI-core#550), so the parse panicked before the write-guard
  assertions ran. The test asserts that loader rejection explicitly and
  exercises the local guard against a zero-width struct; the handled neat-core
  baseline moves to 0.9.10 (issue #96).
- Every creature load (`train`, `sweep`, `compare`, `gradient-check`, and the
  C ABI `trainDir`) now rejects `input < 1` / `output < 1` with the NEAT-AI
  wording (`Must have at least one input neurons was: 0`) before any epoch
  runs, and no `best.json` / `journal.jsonl` / `candidate.json` is written.
  The top-level `input` / `output` integers are the observation width and
  cannot be re-derived — `neurons` lists only non-input neurons — so a zeroed
  count used to train silently on a mis-shaped corpus. `train` also pins the
  source width (`creature_io::ObservationWidth`) and refuses to write
  `best.json`, the per-epoch `candidate.json`, the scorer copy, `sweep`
  candidates, or a `CreatureTrace` whose struct or serialised bytes do not
  carry exactly that width. `tags::serialize_creature_with_meta` takes the
  source width as a third argument. Local guard at the binary boundary; it
  stays once `neat-core` validates the same rule (NEAT-AI-core#550)
  (issue #92).
- `compare` now refuses a recurrent / re-entrant creature instead of running the
  unsupported accumulate path and writing a parity dump. The read → parse →
  forward-only-guard preamble had been copy-pasted across `gradient-check`,
  `sweep`, and `train`, and `compare` was the copy that never got the guard. All
  four now call the new `creature_io::load_forward_only_creature`
  (`parse_forward_only_creature` for `train`, which also mines the raw text for
  tags), so the supported-graph rule and its wording have a single owner
  (issue #54).

### Removed

- `scorer::default_scorer_path()` — a `pub fn` no code ever called. It returned
  a bare `rust_scorer` PATH lookup for a fallback the CLI never adopted:
  `--scorer` is `Option<PathBuf>` with no default and `run_train` simply skips
  scoring when it is omitted. Being `pub` it was invisible to the workspace
  `dead_code` lint (issue #37).
- `train::MIN_SCORE_IMPROVEMENT` — a `pub const` no code ever read. Its doc
  comment described a score-improvement accept gate, but `run_train` accepts on
  `accept_always || after_mse < best_mse` alone and never applied a score
  threshold, so the constant documented behaviour that was never wired in.
  Being `pub` it was invisible to the workspace `dead_code` lint. The 1e-6
  production margin itself is unchanged — it is enforced by the GRQ consumer
  (`grq_backprop_score_improves`), not by this crate (issue #36).
