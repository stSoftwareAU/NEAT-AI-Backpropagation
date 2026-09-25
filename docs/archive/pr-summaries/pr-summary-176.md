# PR Summary — #176

## Summary

Closes #176.

The output-neuron UUID set was built by the same five-line filter in three places. `trust_region::output_uuids` is now `pub(crate)` and is the only place that set gets built:

- `backprop.rs` — `apply_learnings_with` and `count_apply_deltas`
- `gradient_check.rs` — `run_gradient_check`

Behaviour does not change. The crate version goes from 0.1.44 to 0.1.45 because `backpropagation/src/**` changed.

- [x] Add unit tests for `output_uuids`
- [x] Replace the three inline copies with calls to the helper
- [x] Bump the crate version
- [x] Pass `./quality.sh`

## Evidence

This is a backend Rust refactor with no web interface, so there is no screenshot. The evidence is the test output:

- `./quality.sh < /dev/null` exits 0 with `All quality checks passed!`. That run covers fmt, clippy `-D warnings`, the full workspace tests (312 passed, 0 failed), rustdoc, codespell, markdownlint, shellcheck and the policy guards.
- Two new tests, in `trust_region::tests`, pass:
  - `output_uuids_selects_only_output_neurons`
  - `output_uuids_of_a_creature_without_outputs_is_empty`
- The existing tests for `apply_learnings_with`, `count_apply_deltas`, `measure_update` and gradient check still pass unchanged. That confirms the substitution kept behaviour the same.

## Test Plan

- `cargo test --workspace --all-features -- --test-threads=2`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `./quality.sh < /dev/null`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
