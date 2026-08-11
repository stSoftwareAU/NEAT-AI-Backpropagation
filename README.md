# NEAT-AI-Backpropagation

Experimental standalone Rust backpropagation for **production** NEAT-AI
creatures. The reverse-topological loop lives in sibling
[NEAT-AI-core](https://github.com/stSoftwareAU/NEAT-AI-core). This crate
owns the creature / `.bin` bridge, apply step, and a `trainDir`-style
epoch loop.

TypeScript backpropagation orchestration in
[NEAT-AI](https://github.com/stSoftwareAU/NEAT-AI) stays until this
program proves numerical parity **and** a measured learning win on the
GRQ production creature and corpus. Tiny identity-chain fixtures are
unit regression only — they are not a win.

## Sibling layout

```text
parent/
  NEAT-AI-core/
  NEAT-AI-Backpropagation/
  NEAT-AI/            # optional TypeScript dual-run
  NEAT-AI-scorer/     # optional rust_scorer
```

`neat-core` is an unpinned path dependency
(`../../NEAT-AI-core/neat-core`). Breaking SemVer bumps are gated by
[`neat-core.expected-version`](./neat-core.expected-version).

Toolchain: [`rust-toolchain.toml`](./rust-toolchain.toml) (`1.95.0`).

## CLI

```bash
cargo run -p neat_ai_backpropagation --release -- compare \
  ~/src/GRQ-cluster/network.json \
  /tmp/grq-slice \
  --max-records 256 --seed 1 --out rust-compare.json

cargo run -p neat_ai_backpropagation --release -- diff \
  rust-compare.json ts-compare.json

cargo run -p neat_ai_backpropagation --release -- train \
  ~/src/GRQ-cluster/network.json \
  /tmp/grq-train-slice \
  --epochs 4 --max-records 2048 --seed 1 \
  --scorer ../NEAT-AI-scorer/target/release/rust_scorer \
  --output-dir .backprop

cargo run -p neat_ai_backpropagation --release -- sweep \
  ~/src/GRQ-cluster/network.json \
  ~/src/GRQ/.trainData-binary_116 \
  --skip-mse --step-scales 0.002,0.01 \
  --output-dir .backprop/sweep
```

`--version` reports `CARGO_PKG_VERSION`. Train journals that version in
`journal.jsonl`.

`train` measures MSE on the **applied** creature and keeps the apply
only when post-apply MSE is strictly lower than the best so far
(rollback otherwise). `--accept-always` keeps the candidate anyway
(for a later full-corpus `rust_scorer` check). `sweep` accumulates
once and writes one candidate per `--step-scales` entry. Recurrent /
re-entrant creatures are refused.

## Production win protocol

Locked targets:

| Item | Path | Shape |
| ---- | ---- | ----- |
| Creature | `~/src/GRQ-cluster/network.json` | 2511→1, 1605 neurons, 22011 synapses |
| Corpus | `~/src/GRQ/.trainData-binary_116` | ~2.26M records, 10048 bytes/record |

1. Slice the first *N* records of a real production `.bin` into `0.bin`
   (`scripts/extract-bin-slice.sh`) so Rust and TypeScript read identical
   bytes.
2. `compare` on that slice; Deno `scripts/ts-compare.ts` on the same
   slice; `diff` must report no field mismatches (abs `1e-9` / rel
   `1e-6`).
3. Accumulate on the **full** production directory (not one year file)
   **without** `--outputs-only`, so IF/MIN/MAX linearisation can move
   hidden genes. A win is `rust_scorer` on all 2,262,277 records up by
   more than `1e-6`. Saturated full-net applies overfit a slice and
   **lower** the full-corpus score — treat slice MSE as a hint only.

```bash
./scripts/run-production-win.sh
```

Recorded result (see [`docs/production-win.json`](./docs/production-win.json)):

- Parity on 256 records of `A-2007.bin`: forward MSE agrees; the 3
  overlap neurons match at `1e-9`. Rust continues through IF/MIN/MAX
  (TypeScript/WASM does not).
- Full-corpus `rust_scorer` win: 11 hidden weights into IF/MINIMUM
  plus one MAXIMUM bias, signed from a 2,262,277-record accumulate.
  Score `0.347586415202` → `0.347614794359` (Δ `+2.84e-5`). Topology
  and complexity penalty unchanged.

## Local quality

```bash
./quality.sh < /dev/null
```

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the version-bump contract
(same as NEAT-AI-Lamarck / GRQ `runlib.sh`).
