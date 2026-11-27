# Issue #217 — audit for whole-creature compaction (shared Rust compaction migration)

Issue #217 is stage 4H of the family compaction migration (tracker stSoftwareAU/NEAT-AI-core#775). It asks this repo to audit backpropagation, structural edit and candidate export for existing whole-creature compaction or duplicated structural rewrites, replace any with the shared Rust core API, and, where none exists, record the evidence and add no code. Repositories that do not compact must not acquire new compaction.

## Verdict — no compaction, nothing to migrate

This crate neither compacts a creature nor rewrites its structure on any path. Training moves bias and weight values only, and every creature the crate writes is checked to have the same neuron and synapse counts as its source. The audit found no compaction caller to migrate.

## Evidence

| Path | What it does to a creature | Source |
| --- | --- | --- |
| Backpropagation apply | Clones the creature and changes only `neuron.bias` and `synapse.weight`; no gene added, removed or reordered | `backpropagation/src/backprop.rs:346` (`apply_learnings_with`) |
| Topology gate | `TrainedTopology` pins `input + neurons.len()` and `synapses.len()` from the source and passes them to `neat_core::creature_validate`, so any changed gene count is refused | `backpropagation/src/validate.rs:27` (struct), `:86` (`creature_validate` call); test `a_dropped_neuron_is_rejected_by_the_pinned_count` at `backpropagation/src/validate.rs:178` |
| Output gate on a completed train run | Runs once per run, before scoring and before `best.json` is written; the CLI `train` command and the C ABI both reach `run_train_until` | `backpropagation/src/train.rs:1127` (`assert_valid`), `backpropagation/src/main.rs:596`, `backpropagation/src/ffi.rs:465` |
| Step-scale sweep candidates | Each candidate is gated with `assert_valid` as it is produced, then written | `backpropagation/src/sweep.rs:154` |
| Blockwise candidates | Each candidate is gated with `assert_valid` as it is produced, then written | `backpropagation/src/blockwise.rs:384` |
| Ladder rungs | Serialised value-only candidates via `checked_json_pretty` | `backpropagation/src/ladder.rs:261` |
| Candidate export and serialisation | `checked_json` and `checked_json_pretty` call `neat_core::creature_to_json` and `creature_to_json_pretty` after a width check; no rewrite | `backpropagation/src/creature_io.rs:104`, `:112` |
| Block planning | `dedup` de-duplicates index lists of planned blocks, never creature genes | `backpropagation/src/blocks.rs:206` |
| Parity probe | Loads via NEAT-AI `Creature.fromJSON` and runs propagation (`:58`); no compaction call | `scripts/ts-compare.ts:23` |

The variable `compact_best` at `backpropagation/src/train.rs:1142` names the untagged JSON serialisation used for scoring, not compaction.

The parity probe is a parity script, not a production path, so no NEAT-AI TypeScript adapter (stSoftwareAU/NEAT-AI#4196) is needed.

### Search

The search was `grep -rniE 'compact|prune|cleanup|simplif|retain|dedup'` over `backpropagation/src` and `scripts/`. Every hit is classified below.

- `dedup` hits at `backpropagation/src/blocks.rs:206`, `:818` and `:824`: block planning at `:206`; `:818` and `:824` sit in the unit test `adjacency_holds_each_synapse_and_neighbour_once` and check index lists for duplicates. None touch creature genes.
- `compact` hits in `backpropagation/src/creature_io.rs:101` (doc comment, "serialise compactly"), `backpropagation/src/ffi.rs:583` (the "compact capability probe document", a JSON capability probe rather than compaction) and `backpropagation/src/train.rs:1142`, `:1157`, `:1164` (`compact_best`, see above).
- `backpropagation/src/ladder.rs:363` is a comment saying the winning rung is rebuilt rather than retained. No creature is compacted.
- `retain` and `remove` hits in `#[cfg(test)]` fixtures: `backpropagation/src/trust_region.rs:569` (test module from `:513`) and `backpropagation/src/validate.rs:180`–`:182` (test module from `:100`).
- `backpropagation/src/train.rs:1185` (`meta.tags.retain(...)`) retains *tags*, not genes. This was checked against the source.
- `scripts/check-cross-repo-defect-record.sh:63` is message text about pruned PR summaries. `scripts/check-no-private-repo-references.sh:70` is a `find -prune` directory walk. The cleanup traps and helper in `scripts/runlib.sh` (`:564`, `:581`, `:979`, `:1163`) reap temporary files. None touch a creature.

## Dependency state on 2026-11-26

The PR bumps `neat_ai_backpropagation` from `0.1.50` to `0.1.51` and moves the neat-core pin in `backpropagation/Cargo.toml` from `v0.23.4`. The pin moved in four steps: `v0.23.11` (`9ef3668def917ffa3266c212d2990f1c77cab27d`), then `v0.23.13` (`847a3e38a17704104ca05ef1efa08481ff042a17`) via the automated family-sync commit `672ea9c`, then the released `v0.23.17` (`939ccf5f561277763eb7a473238a86074ca8a8cc`) via the family-sync commit `9f111ea`, then the released `v0.23.18` via the family-sync commit `ffb5d1c`. The earlier pins are historical. At commit `ffb5d1c`, `backpropagation/Cargo.toml:31` pins `v0.23.18` and `Cargo.lock:196-197` resolves it to `4bc275da4d683dcb559ee46874ba3a8ea9fc3060`; a later family-sync commit can move the pin again.

Core #779 and NEAT-AI#4196 were open on the audit date. This crate has no caller that depends on the shared compaction API.

## Compatibility evidence

The eight existing cases in `backpropagation/tests/scorer_boundary.rs` now hold a shared mutex while writing and executing their scorer stubs. This prevents concurrent tests from racing on an open executable (`ETXTBSY`). All eight cases retain their original assertions.

At the earlier commit `051d2df` (pin `v0.23.11`), CI compiled the pinned neat-core and ran `cargo test --workspace --all-features -- --test-threads=2 -q`. All 383 Rust tests passed, including the eight scorer-boundary cases. [Quality Checks job 139978579433](https://github.com/stSoftwareAU/NEAT-AI-Backpropagation/runs/139978579433)

At the intermediate commit `672ea9c` (pin `v0.23.13`, historical), every check passed, including [Quality Checks](https://github.com/stSoftwareAU/NEAT-AI-Backpropagation/actions/runs/45902028014/job/140190523906).

At the historical commit `9f111ea` (pin `v0.23.17`), `gh pr checks 218` reported every check passing, including [Quality Checks job 140275441181](https://github.com/stSoftwareAU/NEAT-AI-Backpropagation/actions/runs/45928350249/job/140275441181), which compiled `v0.23.17` and ran the same workspace test command.

At commit `ffb5d1c` (pin `v0.23.18`, resolved to `4bc275da`), [Quality Checks job 140293295391](https://github.com/stSoftwareAU/NEAT-AI-Backpropagation/actions/runs/45933953037/job/140293295391) compiled neat-core `v0.23.18` at `4bc275da` and passed `cargo test --workspace --all-features -- --test-threads=2 -q`, including the eight scorer-boundary cases. `672ea9c`, `9f111ea` and `ffb5d1c` changed only `Cargo.lock` and `backpropagation/Cargo.toml`; no source or test file differs from `051d2df`. This audit did not re-run the suite locally.

## When to revisit

Revisit only if a future change adds a path that alters creature structure. That path must call the shared neat-core compaction API rather than a local rewrite.
