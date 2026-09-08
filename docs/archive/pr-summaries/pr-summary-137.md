## Summary

`backpropagation/src/main.rs` redeclared the same CLI argument groups in four
`Commands` variants — `--max-records` / `--seed` in `train`, `sweep`, `blocks`
and `gradient-check`; `--learning-rate` plus the two adjustment-scale clamps in
the same four; `--outputs-only` / `--hidden-only` in `train`, `sweep` and
`gradient-check`. Each block repeated the same defaults and near-identical doc
comments hundreds of lines apart, so a new default or a reworded help string had
to be applied by hand four times.

Each group is now declared once as a `#[derive(clap::Args)]` struct and
flattened into every variant that carries it:

- `CorpusArgs` — `--max-records`, `--seed`
- `RateArgs` — `--learning-rate`, `--maximum-bias-adjustment-scale`,
  `--maximum-weight-adjustment-scale`
- `GeneScopeArgs` — `--outputs-only`, `--hidden-only`

`RateArgs::to_config()` also replaces the identical fixed-strategy
`BackpropConfig` literal that `sweep`, `blocks` and `gradient-check` each built
by hand, and `train_backprop_config` now takes the group rather than five loose
scalars. Mechanical refactor: **flag names and defaults are unchanged**.

Closes #137.

## Evidence

Backend/CLI change — there is no web interface, so the rendered evidence below
is the CLI surface itself, captured with the headless browser from the
before/after `--help` capture. The test suite is the second half of the
evidence.

![Before/after flag and default counts for train, sweep, blocks and gradient-check are identical, with sweep's full flag list shown side by side](docs/evidence/issue-137-help-parity.png)

**The flag/default surface is byte-identical before and after.** Every
subcommand's `--help` was captured from the pre-change binary and again after
the refactor, then reduced to `<subcommand> <flag>` and `<subcommand>
[default: …]` lines and diffed:

```console
$ diff <(surface help-before.txt) <(surface help-after.txt) \
    && echo "CLI SURFACE IDENTICAL (flags + defaults)"
CLI SURFACE IDENTICAL (flags + defaults)
```

The only `--help` change is prose: a shared field carries one description, so
wording that used to differ per subcommand was widened to cover all of them —
e.g. `--maximum-bias-adjustment-scale` reads "per apply / propose" instead of
"per apply" (`train`) and "per propose" (`gradient-check`). The shared
`--max-records` text keeps `train`'s rate-sampling paragraph and now states
explicitly that the other subcommands read each file's leading records, which
is what `compute_mse`'s `RecordSelection::Prefix` does. Because that shared
description spans two paragraphs, clap renders `sweep`, `blocks` and
`gradient-check` help in the same expanded layout `train` already used.

```mermaid
classDiagram
    class CorpusArgs {
        --max-records
        --seed
    }
    class RateArgs {
        --learning-rate
        --maximum-bias-adjustment-scale
        --maximum-weight-adjustment-scale
        to_config() BackpropConfig
    }
    class GeneScopeArgs {
        --outputs-only
        --hidden-only
    }
    Train --> CorpusArgs : flatten
    Train --> RateArgs : flatten
    Train --> GeneScopeArgs : flatten
    Sweep --> CorpusArgs : flatten
    Sweep --> RateArgs : flatten
    Sweep --> GeneScopeArgs : flatten
    Blocks --> CorpusArgs : flatten
    Blocks --> RateArgs : flatten
    GradientCheck --> CorpusArgs : flatten
    GradientCheck --> RateArgs : flatten
    GradientCheck --> GeneScopeArgs : flatten
```

**Quality gate.** `./quality.sh` stops at the `check-neat-core-version.sh`
stage, which fails for a pre-existing reason unrelated to this diff: the
recorded baseline in `neat-core.expected-version` is `0.10.0` while sibling
`NEAT-AI-core` is `0.11.2` on its `Develop`. That stage reads only those two
files and never looks at the PR diff, so it fails on every PR until the bump is
handled in its own deliberate PR — filed as
[#141](https://github.com/stSoftwareAU/NEAT-AI-Backpropagation/issues/141).
Every other stage was then run individually and passed: bash syntax,
shellcheck, the workflow validators, `actionlint`, the version-increment and
crate-version-no-downgrade gates, codespell, `cargo deny check`,
`cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets
--all-features -- -D warnings`, `cargo test --workspace --all-features --
--test-threads=2` (all suites green, 21 in the CLI binary) and
`RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`.

The crate version is bumped `0.1.31` → `0.1.32` (`backpropagation/src/**` is a
build-affecting path, issue #95) and `Cargo.lock` re-synced.

## Test Plan

Added to `backpropagation/src/main.rs` (`mod tests`) — each drives the real
clap parser and asserts on parsed values, and each failed to compile against
the pre-refactor enum:

- `the_corpus_group_parses_identically_in_every_subcommand` — `--max-records` /
  `--seed` defaults (`None`, `1`) and explicit values (`7`, `42`) across
  `train`, `sweep`, `blocks` and `gradient-check`.
- `the_rate_group_parses_identically_in_every_subcommand` — `--learning-rate`
  (`0.01`) and both adjustment clamps (`1.0`) default and parse the same way in
  the same four subcommands.
- `the_gene_scope_group_parses_identically_in_every_subcommand` —
  `--outputs-only` / `--hidden-only` default to `false` and set independently
  in `train`, `sweep` and `gradient-check`.
- `the_rate_group_builds_the_fixed_strategy_config` — `RateArgs::to_config()`
  carries every field onto `BackpropConfig`, keeps `initial_learning_rate` in
  step with `learning_rate`, and leaves the `Fixed` strategy and
  `normalise_gradients: false` untouched.

Modified (no test was removed or weakened — the assertions are unchanged, only
the field path through the new groups):

- `train_clamps_default_to_one_not_ten`, `train_learning_rate_defaults_to_fixed_schedule`
  and `record_sampling_is_random_unless_disabled` now read `rate.*` / `corpus.*`
  instead of the flattened-away variant fields.
- `train_config_carries_schedule_and_normalisation` builds a `RateArgs` and
  passes it to `train_backprop_config`, whose signature dropped the three loose
  rate scalars.
