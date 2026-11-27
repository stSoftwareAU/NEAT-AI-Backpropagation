# Compaction audit — issue #217

## Summary

Audited backpropagation, structural edit and candidate export for whole-creature
compaction or duplicated structural rewrites. None exists: every path changes
bias and weight values only, and the `TrainedTopology` output gate refuses any
creature whose neuron or synapse count differs from its source. Per the issue's
"if no duplicate exists, record evidence and close with no unnecessary code",
the evidence is recorded in `docs/audit/issue-217-shared-compaction.md`. Closes #217.

## Spec

### Intent and Rationale

- This crate is a value-only trainer, so there is no compaction caller to switch
  to the shared core API; adding one would breach the tracker's "repositories
  that do not compact must not acquire new runtime compaction".

### Essential Design Decisions

- Audit-only compaction outcome, matching the sibling Refinery audit (stSoftwareAU/NEAT-AI-Refinery#89, merged as PR #90).
- The `Depends on` core #779 release gates switching a caller; with no caller,
  the audit completes while #779 is still open. Any future structure-changing
  path must call the shared neat-core API (recorded in the audit's "When to
  revisit").
- The PR bumps `neat_ai_backpropagation` from `0.1.50` to `0.1.51` and moves the neat-core pin in `backpropagation/Cargo.toml` from `v0.23.4`. The pin moved in four steps: `v0.23.11` (`9ef3668def917ffa3266c212d2990f1c77cab27d`), then `v0.23.13` (`847a3e38a17704104ca05ef1efa08481ff042a17`) via the automated family-sync commit `672ea9c`, then the released `v0.23.17` (`939ccf5f561277763eb7a473238a86074ca8a8cc`) via the family-sync commit `9f111ea`, then the released `v0.23.18` via the family-sync commit `ffb5d1c`. The earlier pins are historical. At commit `ffb5d1c`, `backpropagation/Cargo.toml:31` pins `v0.23.18` and `Cargo.lock:196-197` resolves it to `4bc275da4d683dcb559ee46874ba3a8ea9fc3060`; a later family-sync commit can move the pin again.

### Undiscoverable Facts

- On 2026-11-26, core #779 and NEAT-AI#4196 were open; the newest core release
  was `v0.23.11` (`9ef3668`), resolved with `git ls-remote --tags`; the
  family-sync commit `672ea9c` later moved the pin to the released `v0.23.13`
  (`847a3e3`), `9f111ea` then moved it to the released `v0.23.17`
  (`939ccf5`), and `ffb5d1c` then moved it to the released `v0.23.18`
  (`4bc275d`).

## Evidence

- `docs/audit/issue-217-shared-compaction.md` cites each path:
  `backpropagation/src/backprop.rs` (`apply_learnings_with`),
  `backpropagation/src/validate.rs` (`TrainedTopology`),
  the `assert_valid` gates in `train.rs`, `sweep.rs` and `blockwise.rs`, the
  serialisers in `creature_io.rs`, and `scripts/ts-compare.ts`. It also
  classifies every hit of a
  `compact|prune|cleanup|simplif|retain|dedup` search over
  `backpropagation/src` and `scripts/`.
- Existing coverage of the "training never changes topology" invariant:
  `backpropagation/src/validate.rs::a_dropped_neuron_is_rejected_by_the_pinned_count`.
- Issues cited, checked with `gh issue view`:
  - #217: Compaction migration 4: audit and converge on shared Rust compaction
  - stSoftwareAU/NEAT-AI-core#775: Family-wide DRY: unified Rust creature compaction — staged migration tracker
  - stSoftwareAU/NEAT-AI-core#779: Compaction migration 3: publish one native/WASM compaction API with parity gates
  - stSoftwareAU/NEAT-AI#4196: Compaction migration 4A: switch TypeScript compact callers to shared Rust engine

## Test Plan

The eight existing cases in `backpropagation/tests/scorer_boundary.rs` now hold a shared mutex while writing and executing their scorer stubs. This prevents concurrent tests from racing on an open executable (`ETXTBSY`). All eight cases retain their original assertions.

## Merge verification

At the earlier commit `051d2df` (pin `v0.23.11`), CI compiled the pinned neat-core and ran `cargo test --workspace --all-features -- --test-threads=2 -q`. All 383 Rust tests passed, including the eight scorer-boundary cases. [Quality Checks job 139978579433](https://github.com/stSoftwareAU/NEAT-AI-Backpropagation/runs/139978579433)

At the intermediate commit `672ea9c` (pin `v0.23.13`, historical), every check passed, including [Quality Checks](https://github.com/stSoftwareAU/NEAT-AI-Backpropagation/actions/runs/45902028014/job/140190523906).

At the historical commit `9f111ea` (pin `v0.23.17`), `gh pr checks 218` reported every check passing, including [Quality Checks job 140275441181](https://github.com/stSoftwareAU/NEAT-AI-Backpropagation/actions/runs/45928350249/job/140275441181), which compiled `v0.23.17` and ran the same workspace test command.

At commit `ffb5d1c` (pin `v0.23.18`, resolved to `4bc275da`), [Quality Checks job 140293295391](https://github.com/stSoftwareAU/NEAT-AI-Backpropagation/actions/runs/45933953037/job/140293295391) compiled neat-core `v0.23.18` at `4bc275da` and passed `cargo test --workspace --all-features -- --test-threads=2 -q`, including the eight scorer-boundary cases. `672ea9c`, `9f111ea` and `ffb5d1c` changed only `Cargo.lock` and `backpropagation/Cargo.toml`; no source or test file differs from `051d2df`. This audit did not re-run the suite locally.

## Not checked

The audit did not benchmark performance or run a separate serialised-creature parity comparison.
