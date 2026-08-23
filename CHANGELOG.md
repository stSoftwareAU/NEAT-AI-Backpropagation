# Changelog

All notable changes to NEAT-AI-Backpropagation are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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

### Changed

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
