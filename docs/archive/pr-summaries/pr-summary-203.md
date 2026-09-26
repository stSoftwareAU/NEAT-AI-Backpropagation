# PR Summary — Issue #203

## Summary

Closes #203

Quietens green test output in the local gate and in CI:

- `.github/workflows/ci.yml`: drop cargo's `--verbose` and pass libtest's `-q`
  (`cargo test --workspace --all-features -- --test-threads=2 -q`).
- `quality.sh`: pass `-q` to the libtest harness (same invocation as CI).

Passing tests now print one `.` each rather than one line each. Failures still
print their full name, panic message and captured output, so no diagnostic
detail is lost.

## Evidence

This change affects CLI and CI output only, so there is nothing to screenshot.
A `./quality.sh < /dev/null` run exited with 0 and ended with `All quality checks passed!`. Its
test section now reads like this:

```text
running 14 tests
..............
test result: ok. 14 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

No `test <name> ... ok` lines appear in the log.

## Test Plan

- [x] `./quality.sh < /dev/null` passes (fmt, clippy, tests, docs, workflow
      checks).
- [x] The test output is quiet (dots only) and every suite still reports `test result: ok`.
- [ ] CI `quality` job passes with the new `cargo test` invocation.
