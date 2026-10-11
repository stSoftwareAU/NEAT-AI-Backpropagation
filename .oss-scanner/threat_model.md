# Threat model

Disclosure channels, automated scanning and scope are in
[SECURITY.md](../SECURITY.md). The C ABI is declared in
[include/neat_ai_backpropagation.h](../include/neat_ai_backpropagation.h).
This file is the short brief for the scanner.

## What this project does and where untrusted input enters

NEAT-AI-Backpropagation is an experimental, offline Rust trainer for NEAT-AI
creatures. It ships as a CLI (`neat_ai_backpropagation`: `compare`, `diff`,
`train`, `sweep`, `blocks`, `gradient-check`) and as a `cdylib` whose C ABI
(`src/ffi.rs`) lets NEAT-AI run `trainDir` in-process through Deno FFI. It
opens no network connections and holds no credentials. Treat as untrusted:

- creature JSON (UUID-only export), which in production is pulled from a
  repository shared by many machines (`src/creature_io.rs`);
- training and holdout directories of little-endian `f32` `.bin` records
  (`--training-data`, `--eval-dir`);
- the JSON train request passed to `neat_backprop_train`, and the creature and
  corpus paths it names;
- compare dumps read by `diff`, and the scorer's output when `--scorer` is
  given.

The scorer path and output directories are chosen by the operator and are
trusted. The FFI caller is trusted to honour the pointer contract documented
on each exported function; the bytes it passes are not.

## Components that matter most / least

Most important: `src/ffi.rs`, the only `unsafe` code in the crate (raw request
slice, owned output buffers, `neat_backprop_buffer_free`, panic capture across
the ABI); creature loading and validation (`src/creature_io.rs`,
`src/validate.rs`); record sampling and corpus reads (`src/sampling.rs`,
`src/train.rs`); and the trace store, which writes per-epoch files. Lower
priority: `src/compare.rs`, `diff` and the research scripts under `scripts/`.
`docs/` and `.github/` are documentation and CI.

## How to exercise it

From `/src`: `cargo test --workspace --all-features` runs the suite;
`backpropagation/tests/ffi_abi.rs` drives the exported symbols as a Deno FFI
caller does, and `target/debug/libneat_ai_backpropagation.so` is the built
`cdylib`. `target/debug/neat_ai_backpropagation --help` lists the subcommands.
`rust_scorer` is on `PATH` when its optional install succeeded.

## How you rate severity

This is a research tool run by its own maintainers on their own hosts, but the
`cdylib` runs inside the NEAT-AI process.

- High or above: any memory-safety bug in `src/ffi.rs` (out-of-bounds read,
  double free, use-after-free, unwinding across the ABI) reachable with
  request bytes that respect the documented contract.
- High: writing outside the requested output or trace directory (path
  traversal), or running a command other than the operator's scorer.
- Medium: panics, unbounded memory or CPU, or hangs on malformed creature
  JSON, `.bin` records, train requests or dumps.
- Low: numerical disagreement with the TypeScript trainer, or issues that need
  the operator's own arguments to be hostile.

## Anything to leave alone

- Bugs in `neat-core` (NEAT-AI-core, which owns the backprop loop) or
  `rust_scorer` belong in those repositories.
- A caller that breaks the documented pointer contract (wrong length, freeing
  a buffer twice itself) is not a library bug.
