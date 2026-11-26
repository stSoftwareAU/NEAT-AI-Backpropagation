# Compaction audit — issue #217

## Summary

Audited backpropagation, structural edit and candidate export for whole-creature
compaction or duplicated structural rewrites. None exists: every path changes
bias and weight values only, and the `TrainedTopology` output gate refuses any
creature whose neuron or synapse count differs from its source. Per the issue's
"if no duplicate exists, record evidence and close with no unnecessary code",
the evidence is recorded in `docs/audit/issue-217-shared-compaction.md` and no
runtime code, test or pin changed. Closes #217.

## Spec

### Intent and Rationale

- This crate is a value-only trainer, so there is no compaction caller to switch
  to the shared core API; adding one would breach the tracker's "repositories
  that do not compact must not acquire new runtime compaction".

### Essential Design Decisions

- Audit-only outcome, matching the sibling Refinery audit (stSoftwareAU/NEAT-AI-Refinery#89, merged as PR #90).
- The `Depends on` core #779 release gates switching a caller; with no caller,
  the audit completes while #779 is still open. Any future structure-changing
  path must call the shared neat-core API (recorded in the audit's "When to
  revisit").
- The neat-core pin (`v0.23.4`) is left for the normal family-sync job
  (`scripts/family-pins.sh`); this audit does not move it.

### Undiscoverable Facts

- On 2026-11-26, core #779 and NEAT-AI#4196 were open; the newest core release
  was `v0.23.11` (`9ef3668`), resolved with `git ls-remote --tags`.

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
- Documentation-only change; no visual surface.

## Test Plan

- No tests added, modified or removed.
- `./quality.sh < /dev/null` in the worker environment: four
  `scripts/test-check-push-branch-changes-action.sh` cases failed with
  `[PRE_PUSH_BLOCKED]` (for example "FAIL commits every tracked change when no paths are given").
  Their throwaway fixture repos inherit the worker's injected `core.hooksPath`
  pre-push hook (`GIT_CONFIG_KEY_0`). This is a worker environment fault,
  unrelated to this diff.
- The same `./quality.sh < /dev/null` with only that injected config unset
  passed: "All quality checks passed!". That covers shell checks, markdownlint,
  codespell, cargo fmt, clippy, cargo-deny and docs, plus 383 Rust tests passed
  and 0 failed.

## Merge verification

- CI and code review pending.

## Not checked

- No scorer, serialised-creature parity or performance run was needed or done.
  No code path changed, so existing tests are the compatibility evidence.
  Compatibility with neat-core `v0.23.11` or a later #779 release was not
  tested here. The family-sync job's CI covers it when the pin moves.
