## Summary

`journal_header_carries_crate_version` built a `TrainJournalHeader` literal and
asserted the field it had just assigned — a Rust language guarantee that can
never fail for a reason this project can fix. Meanwhile the behaviour the name
promises (a train run's `journal.jsonl` actually carries the crate version,
which remote GRQ runners read for stale-binary detection) had no test at all.

Rewrote the test to option (a) of the issue: run a one-epoch train on the
identity-chain fixture, deserialise the first line of the produced
`journal.jsonl` back into `TrainJournalHeader`, and assert `kind == "runHeader"`,
`version == env!("CARGO_PKG_VERSION")` and that the seed round-trips. It now
exercises `run_train`'s journalling at `backpropagation/src/train.rs:150-162`
rather than the compiler. No production code changed. Closes #25.

## Evidence

Backend/CLI change with no web interface, so no screenshot applies. Verified by
temporarily replacing `version: env!("CARGO_PKG_VERSION").to_string()` with
`version: String::new()` in `run_train` and confirming the rewritten test fails
— proving it is now wired to the project's own logic:

```text
---- train::tests::journal_header_carries_crate_version stdout ----
assertion `left == right` failed
  left: ""
 right: "0.1.9"
test result: FAILED. 0 passed; 1 failed
```

With the implementation restored:

```text
test train::tests::journal_header_carries_crate_version ... ok
```

`./quality.sh` passes cleanly (fmt, clippy, full test suite, docs).

## Test Plan

- Rewrote `backpropagation/src/train.rs::tests::journal_header_carries_crate_version`
  to assert against a real `run_train` journal instead of a struct literal.
- No tests removed or disabled; no other test modified.
