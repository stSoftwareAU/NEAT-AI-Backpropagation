# Issue #148 — `serde_json` → `zmij` in `Cargo.lock`

Issue #148 (`security-scan`, severity high, confidence medium) reported that
`Cargo.lock` records `serde_json` 1.0.151 depending on a crate named `zmij`
where `ryu` — `serde_json`'s long-standing float-formatting dependency — was
expected, and flagged it as a possible supply-chain substitution. The scan was
static-only and said so: it had no registry access to resolve `zmij`.

## Verdict — not a substitution

`zmij` is the legitimate successor to `ryu`, published by the same author, and
the lockfile matches what crates.io actually serves. Checked against the
crates.io API and sparse index on 2026-09-10:

| Evidence | Value |
| --- | --- |
| Owner of `zmij` | `dtolnay` (David Tolnay) — the author of `ryu`, `serde` and `serde_json` |
| Repository | `https://github.com/dtolnay/zmij` |
| Description | "A double-to-string conversion algorithm based on Schubfach and xjb" |
| First published | 2025-12-18; 37 versions; ~409M downloads |
| `zmij` 1.0.23 `cksum` | `29666d0abbfad1e3dc4dcf6144730dd3a3ab225bbbdac83319345b1b44ccfc1b` — identical to the lockfile |
| `serde_json` 1.0.151 `cksum` | `c841b55ecdae098c80dcae9cf767f6f8a0c2cdb3416bbef72181df4d0fe73f14` — identical to the lockfile |
| `serde_json` 1.0.151 built dependencies, per its published manifest | `indexmap` (optional), `itoa`, `memchr`, `serde`, `serde_core`, `zmij` |

`ryu` is absent from the lockfile because `serde_json` 1.0.151 no longer
declares it — the crate swapped its float formatter upstream. Both names are a
dragon (`ryu`, Japanese; `zmij`, Slavic), which is the naming continuity, not a
coincidence to be read as camouflage. No change to `Cargo.lock` was warranted,
and none was made; `zmij` was **not** added to `deny.toml`'s ban list, because
banning it would ban `serde_json`.

## What did change

The finding was undecidable from the lockfile alone, which is the real gap: a
lockfile's `dependencies = [...]` list is *never* re-checked against the
published manifest during a build, so a genuine substitution and a legitimate
upstream rename look identical to a reader and to `cargo`. That comparison is
now a gate — `scripts/check-lockfile-integrity.sh`, run by `quality.sh` and CI
— which fetches the crates.io sparse index for every registry package in
`Cargo.lock` and fails loudly when a checksum or a recorded dependency name
disagrees with what was published. A future substitution fails CI; a future
`zmij` verifies clean and needs no human adjudication.

```mermaid
flowchart TD
    A["Cargo.lock<br/>serde_json → zmij"] --> B{"scripts/check-lockfile-integrity.sh"}
    C["index.crates.io<br/>published manifest + cksum"] --> B
    B -->|"checksum and every dependency name match"| D["verified — build proceeds"]
    B -->|"substituted or altered entry"| E["exit 1 — quality.sh and CI fail loudly"]
```

## Provenance

Recorded from `docs/archive/pr-summaries/pr-summary-148.md`. Filing status: no
upstream filing is needed — the finding was a false positive about an external
crate, and nothing in `stSoftwareAU/*` is defective.
