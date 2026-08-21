# Validate every trained creature with `neat_core::creature_validate`

## Summary

Backpropagation hands back creatures with rewritten biases and weights, so it
is one of the three consumers that must gate its own output against the shared
definition of a valid creature (NEAT-AI#3800). Until now a diverged run
escaped as a "trained" creature: a non-finite bias serialises to `null`, so
`best.json` was written and only broke later, in whatever loaded it.

`validate::TrainedTopology` is that gate. It pins the **source** creature's
neuron and synapse counts — the same source-pinning pattern
`creature_io::ObservationWidth` already uses for the observation width — and
calls `neat_core::creature_validate`. No rule is re-implemented here.

- **`train`** validates the creature it finishes with, **once per completed
  run**, before the scorer copy, `best.json` or `TrainResult` can see it. The
  per-epoch `candidate.json` dumps stay ungated: they are working state, not a
  returned creature, and gating them would put the check inside the hot loop.
  The C ABI / FFI `trainDir` runs through `run_train`, so it is gated too.
- **`sweep`** validates each candidate as it is produced, before it reaches
  `candidates/`.
- A failure reports neat-core's class, `reason` and message plus the offending
  `neuron_index` / `synapse_index`, and names the run that produced it. Nothing
  is written, scored or returned.

`ValidateOptions` are chosen deliberately and justified in
`validate.rs::TrainedTopology::options`:

| Field | Value | Why |
| ----- | ----- | --- |
| `neurons` / `connections` | pinned from the source | training moves values, never genes — the counts are always available, so the gate also proves the topology survived |
| `forward_only` | `true` | every creature this trainer accepts is forward-only (`FORWARD_ONLY_REQUIRED`); its output is held to the same feed-forward contract |
| `feedback_loop` | `None` | `forward_only` already resolves it to `Some(false)`; setting both invites drift |

`neat-core.expected-version` is unchanged: `creature_validate` is reachable on
the handled 0.9.10 baseline, and the port that fills in its rule bodies is a
patch-level change, so `scripts/check-neat-core-version.sh` does not gate.

Closes #94.

## Blocked on the dependency — read before merging

`creature_validate` on **neat-core `Develop` returns an unconditional
failure**: only the neuron half of the rules is ported
(`neat-core/src/creature_validate.rs:550-561`, Issue
stSoftwareAU/NEAT-AI-core#560), and the synapse / forward-only / memetic half
is stSoftwareAU/NEAT-AI-core#561 — open, in review as PR
stSoftwareAU/NEAT-AI-core#565. Against that build **no** creature can be
certified, so this crate's tests (including the ten pre-existing training tests)
fail with core's own "rule bodies are not ported yet" failure.

**Do not merge this PR until stSoftwareAU/NEAT-AI-core#565 lands on core
`Develop`**; this repo consumes `neat-core` by path against head, so CI here
goes green the moment it does — no version pin, no bump. This PR is deliberately
*not* weakened to work around the gap: swallowing core's failure, or
downgrading it to a warning so a run continues, is exactly the silent failure
the issue exists to stop.

## Evidence

Backend/CLI change only — no web interface to screenshot. Verified by tests.

The gate's placement, once per completed run:

```mermaid
flowchart TD
    A[epoch loop: accumulate → apply → accept/rollback] --> B[epochs finish]
    B --> C[TrainedTopology::assert_valid<br/>neat_core::creature_validate]
    C -- Ok --> D[scorer → best.json → TrainResult]
    C -- ValidationFailure --> E[run fails loudly:<br/>reason, message, index, which run<br/>nothing written]
    A -. candidate.json per epoch<br/>working state, not gated .-> A
```

**The bug reproduced.** With the gate removed, the regression test's diverged
run returns a creature whose biases are `-inf` and writes it to `best.json` as:

```json
"neurons": [
  { "bias": null, "squash": "IDENTITY", "type": "hidden", "uuid": "h1" },
  { "bias": null, "squash": "IDENTITY", "type": "output", "uuid": "o1" }
]
```

With the gate in place the same run fails loudly and writes no `best.json`.

**Test run** (against a `neat-core` build whose `creature_validate` entry point
is complete — see *Blocked on the dependency* above): `cargo test --workspace
--all-features -- --test-threads=2` → **144 passed, 0 failed** across every
target, of which 8 are new (5 unit, 3 integration). `cargo fmt
--check`, `cargo clippy --workspace --all-targets --all-features -D warnings`,
`cargo deny check` and `RUSTDOCFLAGS="-D warnings" cargo doc` all pass against
the current sibling checkout. `./quality.sh` stops at its `codespell`
preflight on this host — the tool is not installed and there is no `pip` to
install it — and at `cargo test` for the dependency reason above; every other
gate it runs passes.

## Test Plan

`backpropagation/src/validate.rs` (unit):

- `pins_the_source_topology` — the counts come from the source creature.
- `options_pin_the_counts_and_demand_forward_only` — the documented
  `ValidateOptions`, including `forwardOnly` resolving the feedback-loop flag.
- `a_healthy_trained_creature_passes` — a value-only change stays valid.
- `a_non_finite_bias_is_reported_with_its_neuron_index` — reason, message and
  `[neuron index 1]` all surface, with the producing run named.
- `a_dropped_neuron_is_rejected_by_the_pinned_count` — topology drift fails.

`backpropagation/tests/trained_creature_validation.rs` (integration):

- `train_refuses_to_return_a_creature_with_a_non_finite_bias` — the regression
  test: a diverged run under `--accept-always` fails loudly and leaves no
  `best.json`. It fails against the unfixed code (the creature is returned and
  written with `"bias": null`) and passes with the gate.
- `a_healthy_train_run_passes_the_gate_unchanged` — a normal run still
  succeeds, its `best.json` bytes match `TrainResult::best_json`, topology and
  finiteness hold, MSE improved, and a repeat run is byte-identical.
- `sweep_refuses_to_write_a_candidate_with_a_non_finite_bias` — the diverged
  candidate never reaches `candidates/`.
