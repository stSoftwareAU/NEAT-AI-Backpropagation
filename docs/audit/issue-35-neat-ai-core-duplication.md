# Issue #35 — cross-repo duplication audit

Audit of `backpropagation/src/**` against `neat-core/src/**`, per issue
[#35](https://github.com/stSoftwareAU/NEAT-AI-Backpropagation/issues/35)
(sub-issue of [#30](https://github.com/stSoftwareAU/NEAT-AI-Backpropagation/issues/30)).

**Audit and file only — no code changes.** Sources read at
`backpropagation` `cdd652f`, sibling `neat-core` and `NEAT-AI-Lamarck` clones,
and `NEAT-AI-scorer` `rust_scorer/src/stream_score.rs` fetched via the GitHub
API (that repo is not cloned beside this one).

## Method

For every `backpropagation/src/*.rs` module: locate the rule it encodes, search
`neat-core/src/**` for an existing owner, and decide whether the local copy is a
**thin adapter** to a core API (leave it) or a **restatement** that can silently
drift (file it). Where the sensible home is core, the finding names the
core-side API addition, because a `neat-core` issue is worth more than a
`backpropagation` one that cannot be actioned alone.

Two sibling repos were pulled in beyond the stated scope, deliberately — see
[Scope note](#scope-note).

## Audit table

Every module in `backpropagation/src/` appears, including the rejections.

| # | Candidate (local site) | Core / sibling equivalent | Diverged? | Verdict | Home |
| - | ---------------------- | ------------------------- | --------- | ------- | ---- |
| 1 | `propagate_layout.rs:76-83` `aggregate_kind_for` | `SquashType::is_aggregate()` — `neat-core/src/squash.rs:137-158` | **Yes** — 3 of 6 aggregates | **File** | `neat-core` + this repo |
| 2 | `backprop.rs:20-449` config / signals / apply / tolerances | `NEAT-AI-Lamarck/lamarck/src/backprop.rs:1-380` — verbatim twin | **Yes** — each side has features the other lacks | **File** | `neat-core` |
| 3 | `tags.rs:12-196` `uuid` / `tags` round-trip + `%g` formatting | `NEAT-AI-Lamarck/lamarck/src/tags.rs:1-196` — verbatim twin | No (identical) | **File** | `neat-core` |
| 4 | `scorer.rs:10-76` `ScoreResult` / `parse_scorer_stdout` | `NEAT-AI-Lamarck/lamarck/src/scorer.rs:18-26,482` — twin | **Yes** — Lamarck has a typed `ScorerError` | **File** | `NEAT-AI-scorer` |
| 5 | (criterion) scorer `stream_score.rs` → `mse_mean_streaming` | `neat-core/src/loss.rs:1542-1641` | n/a | **Reject** | — |
| 6 | (spun off from 5) packed-record framing loop | `loss.rs:1577-1636` vs scorer `run_io_loop` | **Yes** — sampling / cap / error text | **File** | `neat-core` |
| 7 | `mse.rs:24-42` `compute_mse` | `mse_mean_streaming` | No | Reject — thin adapter (#32) | — |
| 8 | `propagate_layout.rs:446-455` training-data iteration | `TrainingDataConfig` / `TrainingDataIterator` | No | Reject — see below | — |
| 9 | `propagate_layout.rs:140,156` `apply_get_range`, `parse_squash_name` | `neat-core/src/range.rs:33`, `creature.rs` | No | Reject — direct core calls | — |
| 10 | `propagate_layout.rs:157-161` neuron-type string → `NEURON_TYPE_*` | none in core | No | Reject — no core owner | — |
| 11 | `backprop.rs:442-449` `nearly_equal`, `FLOAT_*_TOL` | none in core | No | Reject vs core — folded into finding 2 | — |
| 12 | `compare.rs:1-437` parity dump / diff | none in core | No | Reject — TypeScript-harness-specific | — |
| 13 | `gradient_check.rs:1-626` FD probe, `percentile`, stratified sample | none in core or Lamarck | No | Reject — no second copy | — |
| 14 | `train.rs:1-771` epoch loop / journal | none in core | No | Reject — host-only orchestration | — |
| 15 | `sweep.rs:1-249` step-scale sweep | none in core | No | Reject — host-only orchestration | — |
| 16 | `main.rs:1-588` clap CLI, `parse_step_scales` | none in core | No | Reject — CLI surface | — |
| 17 | `lib.rs:1-39` re-exports | n/a | No | Reject — module declarations | — |
| 18 | `backprop.rs:343-348`, `411-415`, `gradient_check.rs:188-193` output-UUID set | intra-repo, 3 sites | No | Reject **here** — intra-repo, #17's remit | — |

### Rejections worth stating

- **MSE (7).** After [#32](https://github.com/stSoftwareAU/NEAT-AI-Backpropagation/issues/32)
  and [#33](https://github.com/stSoftwareAU/NEAT-AI-Backpropagation/issues/33),
  `compute_mse` is a six-line call into `mse_mean_streaming` plus the fail-loud
  empty-corpus guard, and the accumulate pass reduces through `mse_record`. The
  audit confirms **no other module in this crate touches loss maths** —
  `grep` for arithmetic on `record.outputs` finds only `propagate_layout.rs:463`,
  which is the `mse_record` call itself. Excluded from the findings as #35 asks.
- **Training-data iteration (8).** #35 asked whether a shared record-scan
  wrapper is warranted once #32/#33 land. It is not: `mse.rs` no longer iterates
  at all, so `propagate_layout.rs:446-455` is now the crate's **only**
  `TrainingDataConfig` + `TrainingDataIterator` + cap loop. One site is not
  duplication, and the loop cannot delegate to `mse_mean_streaming` anyway —
  it needs `activate_and_trace` per record to feed the propagate step. Ten lines
  of adapter over a core iterator; leave it.
- **Squash / range (9).** `parse_squash_name` and `apply_get_range` are called
  straight out of core. Only the *aggregate-kind mapping* beside them restates
  something core owns — that is finding 1, not these.
- **Tolerances vs core (11).** Core has **no** float-comparison helper: a grep
  for `nearly_equal|ABS_TOL|REL_TOL|approx|tolerance` across `neat-core/src`
  returns nothing. So `nearly_equal` does not duplicate core. It *is* duplicated
  with Lamarck, which is why it rides along in finding 2 rather than standing
  alone.
- **Output-UUID set (18).** A genuine three-site intra-repo repetition, but the
  sensible home is this crate, not core, and #35 is explicitly the *cross-repo*
  audit — [#17](https://github.com/stSoftwareAU/NEAT-AI-Backpropagation/issues/17)
  owns intra-repo duplication. Recorded so it is not lost; not filed here, and
  #17 is closed having filed nothing, so no twin issue exists.

### Scope note

#35 scopes the comparison to `backpropagation` versus `neat-core`. Findings 2,
3 and 4 involve **NEAT-AI-Lamarck**, which #35 does not name. They are included
because #30's stated end goal is wider than one repo pair — *"remove duplicate
code across NEAT-AI\* repos, everything in its sensible location, high
performance, no silently different implementations"* — and because in all three
the sensible home is still `neat-core` or `NEAT-AI-scorer`, which are in scope.
Leaving a verbatim 300-line twin of `backprop.rs` unrecorded would have been the
audit's largest miss.

## Findings

### Finding 1 — aggregate-squash membership restated (severity: high)

`neat-core` documents `SquashType::is_aggregate()` (`squash.rs:137-158`) as
**the single home** of the aggregate-squash membership rule:

> This predicate is the single home of that membership rule — a site that
> restated the list and missed a type would silently route it down the
> weighted-sum path.

Four sites restate it anyway. Two are inside core:

- **`neat-core/src/topological_backprop.rs:348-357`** —
  `matches!(squash, SquashType::If | SquashType::Maximum | SquashType::Minimum)`
  gates `PropagateOutcome::Special`, covering **3 of the 6** aggregates.
  `Hypotenuse`, `HypotenuseV2` and `Mean` fall through to the generic
  weighted-sum error-distribution path below it, which models the neuron as
  `activation = squash(bias + Σ w·a)`. That is not how those three compute —
  `MEAN` is `(Σ w·a)/n + bias`, `HYPOT` is `hypot(w·a…) + bias` — so the
  per-synapse error split is taken under the wrong model. `apply_unsquash` does
  not repair it: its aggregate arm (`unsquash.rs:621-630`) correctly returns the
  hint, but that supplies only the target pre-squash value, not the split across
  inward links.
- **`neat-core/src/derivative.rs:301-302`** — restates all six across two arms
  returning `0.0`. Exhaustive today, but still a copy a seventh aggregate must
  remember.

Two are downstream, because core exposes no typed accessor for the *kind*:

- `backpropagation/src/propagate_layout.rs:76-83` (`aggregate_kind_for`)
- `NEAT-AI-Lamarck/lamarck/src/propagate_layout.rs:76-83` (byte-identical)

`parse_squash_name` accepts `"MEAN"`, `"HYPOT"` and `"HYPOTv2"`
(`creature.rs:198-200`), so a creature reaches the gap through the public API.
`unsquash.rs:624` and `batch_scoring.rs:389` already do it correctly, via
`aggregate_squash_patterns!()` and `is_aggregate()` — the codebase holds both
the right pattern and the copies that ignore it.

Production GRQ creatures are not known to carry the three deprecated
aggregates, so this is filed as a latent-drift and maintenance hazard rather
than a reported production defect.

**Core-side API addition.** Expose the kind, not just the boolean:

```rust
/// Which aggregate rule selects a neuron's carrying inward links.
pub enum AggregateKind { Minimum, Maximum, If, Hypotenuse, HypotenuseV2, Mean }

impl SquashType {
    pub const fn aggregate_kind(self) -> Option<AggregateKind> { … }
}
```

`is_aggregate()` becomes `self.aggregate_kind().is_some()`, both downstream
copies collapse to one call, and a test asserting
`aggregate_kind().is_some() == is_aggregate()` for every `SquashType` sits
beside the existing `tests/aggregate_squash_set.rs`.

Split across two issues, one per repo that owns a fix:

- **`neat-core`** — route `topological_backprop.rs:351` and
  `derivative.rs:301-302` through `aggregate_squash_patterns!()`, decide
  deliberately what the three deprecated aggregates should do, and add
  `aggregate_kind()`.
- **`NEAT-AI-Backpropagation`** — replace `aggregate_kind_for` with the core
  accessor. Filed as
  [#52](https://github.com/stSoftwareAU/NEAT-AI-Backpropagation/issues/52).
  The crate needs this fix either way: even once core routes all six to
  `Special`, `aggregate_kind_for` returning `None` for `MEAN`/`HYPOT` means
  those neurons are never linearised onto their carrying links, so
  `LearningSignal::accumulate_propagate_output` drops them entirely (only
  `PropagateOutcome::Standard` accumulates) — no bias learning, no upstream
  propagation.

### Finding 2 — backprop config / learning-signal / apply surface duplicated (severity: high)

`backpropagation/backpropagation/src/backprop.rs:1-449` and
`NEAT-AI-Lamarck/lamarck/src/backprop.rs:1-380` hold the **same** surface, and a
`diff` shows roughly 300 lines that are byte-identical:

`BackpropConfig` (20 fields) and its `Default`, `LearningRateStrategy`,
`calculate_learning_rate` (all four strategies, including the adaptive
`1.1`/`1.3`/`0.5` constants and the `clamp(1e-8, 1.0)`), `BiasSignal` /
`WeightSignal` with `accumulate_standard` / `accumulate_delta` / `propose`,
`LearningSignal::new` / `accumulate_propagate_output`, `apply_learnings`,
`nearly_equal`, `FLOAT_ABS_TOL`, `FLOAT_REL_TOL`.

**The copies have already diverged**, in both directions:

| Feature | `backpropagation` | Lamarck |
| ------- | ----------------- | ------- |
| `BiasSignal::merge` / `WeightSignal::merge` / `LearningSignal::merge` (parallel chunk accumulation, Lamarck #107) | absent | `backprop.rs:170-176,232-242,311-337` |
| `ApplyOptions` (`step_scale` / `outputs_only` / `hidden_only`), `apply_learnings_with` | `backprop.rs:290-390` | absent |
| `ApplyDeltaCounts`, `count_apply_deltas` | `backprop.rs:392-439` | absent |

That is the latent-bug signature this audit exists to catch: the next change to
`calculate_learning_rate` or to `propose`'s clamping must land on both copies,
and the last two feature additions each landed on one.

**Core-side API addition.** `neat-core` already owns the arithmetic both
`propose` methods call — `calculate_bias` / `calculate_weight` in
`accumulate.rs`. What is duplicated is the *configuration and accumulation
shell* around them. Add a `neat_core::backprop_config` module owning
`BackpropConfig`, `LearningRateStrategy`, `calculate_learning_rate`,
`BiasSignal`, `WeightSignal`, `LearningSignal` (with both `merge` and
`accumulate_propagate_output`), `ApplyOptions`, `ApplyDeltaCounts`,
`apply_learnings_with`, `count_apply_deltas`, and the `nearly_equal` /
`FLOAT_*_TOL` pair. It is pure arithmetic over `CreatureExport` and
`PropagateOutput` — both already core types — so nothing host-only crosses into
core's wasm surface. Both consumers then delete their copy and import.

Home: **`neat-core`**.

### Finding 3 — creature `uuid` / `tags` round-tripping duplicated (severity: medium)

`neat_core::CreatureExport` (`creature.rs:36-52`) carries `input`, `output`,
`neurons`, `synapses`, `semanticVersion` and `forwardOnly` — and **drops `uuid`
and `tags`**. Every consumer that must write a creature back for GRQ check-in
therefore re-implements the same rescue: parse the raw JSON a second time, keep
`uuid` + `tags`, re-attach them after `creature_to_json`.

Two verbatim implementations:

- `backpropagation/backpropagation/src/tags.rs:12-196`
- `NEAT-AI-Lamarck/lamarck/src/tags.rs:12-196`

Identical bar comments and the program-specific tag name: `CreatureTag`,
`CreatureMeta`, `from_creature_json`, `upsert`, `creature_value_with_meta`,
`serialize_creature_with_meta`, plus the `%g` formatting trio `format_g` /
`trim_g_scientific` / `trim_trailing_zeros_and_dot` and `format_score_improved`.

**Core-side API addition.** Either preserve `uuid` and `tags` through
`parse_creature_json` / `creature_to_json` (a `#[serde(flatten)]`-style
passthrough so unknown top-level fields survive a round trip), or export a
`CreatureMeta` beside `CreatureExport` that owns the split explicitly. The
round-trip rule is core's — it is core's own type that drops the fields.

The `%g` formatter is a presentation concern, not core's; it should follow
whichever crate ends up owning the check-in tag layer rather than being pushed
into core. Noted here because it sits in the same file and the same root cause,
not as a separate finding.

Home: **`neat-core`**.

### Finding 4 — `rust_scorer` stdout contract re-implemented by both consumers (severity: medium)

`rust_scorer` prints a JSON map of `stem → { score, error, complexityPenalty }`.
Two consumers each restate that shape and re-implement the parser, and neither
is tested against the producer:

- `backpropagation/backpropagation/src/scorer.rs:10-26` (`ScoreResult`) and
  `55-76` (`parse_scorer_stdout` — last non-empty line wins, map-or-single
  fallback, whole-text fallback)
- `NEAT-AI-Lamarck/lamarck/src/scorer.rs:18-26` and `482`

**Already diverged**: Lamarck returns a typed `ScorerError`, this crate returns
`String`. A change to the scorer's stdout shape needs two edits in two repos,
and the compiler catches neither.

**Owner-side API addition.** `NEAT-AI-scorer` is the *producer* of that
contract, so it is the sensible home — not core, which has no notion of the
scorer binary's CLI. Publish the result schema from `NEAT-AI-scorer` (a small
`rust_scorer_types` crate, or a public `ScoreResult` + `parse_scorer_stdout` on
the existing crate) and have both consumers depend on it. The producer's own
tests then pin the shape.

Home: **`NEAT-AI-scorer`**.

### Finding 5 — scorer `stream_score.rs` → `mse_mean_streaming`: **rejected**

#35 requires this convergence to be filed in `NEAT-AI-scorer` referencing
`NEAT-AI-core#538`, **or explicitly rejected with a reason**. It is rejected.

`NEAT-AI-core#538` is closed and `mse_mean_streaming` landed
(`neat-core/src/loss.rs:1542-1641`). But it is MSE-only, single-threaded and
unsampled, whereas `NEAT-AI-scorer/rust_scorer/src/stream_score.rs` is none of
those:

| Capability | scorer `stream_score.rs` | core `mse_mean_streaming` |
| ---------- | ------------------------ | ------------------------- |
| Loss functions | generic over `CostKind` via `accumulate_cost_sum` (`stream_score.rs:20,482`) | MSE only |
| Corpus sampling | `SampleSpec` + a stateful sampler threaded across chunks (Issue #310) | none |
| Parallel activation | Rayon, one `CompiledNetwork` per worker (Issue #42) | none |
| Parallel file reads | dynamic work queue across readers (Issue #529) | sequential |
| Read tuning | `NEAT_SCORER_READ_BYTES`, per-reader buffers (Issue #549) | `training_read_tuning_from_env` only |

Converging the scorer onto `mse_mean_streaming` would delete cost-kind
genericity and sampling and drop the scorer to single-threaded sequential reads
— a functional and performance regression, not a de-duplication. The premise
that the scorer carries "a third streaming `.bin` → packed-buffer →
`mse_sum_batch_packed` loop" was right in 2024 shape, but the scorer's loop has
since grown past what the core helper covers.

The *genuine* duplication underneath it is narrower, and is filed as finding 6.

### Finding 6 — packed-record framing loop duplicated (severity: medium)

Both sides frame a byte stream into whole `f32` records with a carry buffer,
and the rule is the same: accumulate a partial record across chunk (and file)
boundaries, emit whole records to a batched kernel, and fail loud on a trailing
partial record.

- `neat-core/src/loss.rs:1577-1636` — inline `pending` + `packed`, `append_le_f32`,
  `max_records` capping, trailing-bytes error.
- `NEAT-AI-scorer/rust_scorer/src/stream_io.rs` `run_io_loop`, driven from
  `stream_score.rs:536-556` — `pending` + `head` + compact, plus the sampler.

Already diverged: core caps by `max_records`, the scorer samples by
`SampleSpec`; the two trailing-byte error messages differ. A change to the
framing rule — a new endianness guard, a different partial-record policy —
needs both edits.

**Core-side API addition.** `neat-core` already owns the layer below
(`training_bin_stream::for_each_read_chunk`, which both call). Add the framing
layer above it — `for_each_packed_record_batch(files, buf_len, mode, |packed,
n_records| …)` — owning the carry buffer and the trailing-byte contract, with
the record *policy* (cap, sample) staying in the caller's closure.
`mse_mean_streaming` becomes a thin user of it, and the scorer's `run_io_loop`
keeps only its sampler.

Home: **`neat-core`**.

## Filing status

Cross-repo issue creation is blocked in this run: the agent's write allowlist is
`stsoftwareau/neat-ai-backpropagation` only, so
`gh issue create --repo stSoftwareAU/NEAT-AI-core` (and `…/NEAT-AI-scorer`)
is refused with `[SECURITY] [WRITE_REPO_BLOCKED]`. Reads across repos work,
which is how this audit was carried out.

| Finding | Home | Status |
| ------- | ---- | ------ |
| 1 (consumer half) | `NEAT-AI-Backpropagation` | Filed — [#52](https://github.com/stSoftwareAU/NEAT-AI-Backpropagation/issues/52) |
| 1 (core half) | `neat-core` | **Blocked** — needs a human to file |
| 2 | `neat-core` | **Blocked** — needs a human to file |
| 3 | `neat-core` | **Blocked** — needs a human to file |
| 4 | `NEAT-AI-scorer` | **Blocked** — needs a human to file |
| 5 | — | Rejected, reason above |
| 6 | `neat-core` | **Blocked** — needs a human to file |

Each blocked finding's section above is written to be pasted into
`gh issue create` as-is. Issue #35 carries `needs-human` and a comment naming
the exact commands.
