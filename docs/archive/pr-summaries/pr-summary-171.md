# Reword private-repo references to concept level (issue #171)

## Summary

Four build-profile references named a private fleet repository and one of its
issues — a full `github.com` URL in `README.md` and the `Repo#1234` shorthand in
`CHANGELOG.md`, `Cargo.toml` and `.cargo/config.toml`. This repository is
public, so a reader who follows either one hits something they cannot see: the
rule the citation was meant to justify no longer stands on its own.

Each reference is now stated at concept level — **dev compiles as fast as
practical; release is fully optimised** (compile time irrelevant) — citing this
repository's own tracking issue #88. No private repository is named or linked.

A regression guard keeps the class from returning:
`scripts/check-no-private-repo-references.sh`, with
`scripts/test-check-no-private-repo-references.sh`, wired into `quality.sh`
beside the other policy guards. It scans every git-tracked text file and fails
on either citation form (`stSoftwareAU/<Repo>` or `<Repo>#<digits>`) for the
repositories in its `PRIVATE_REPOS` list. Public stSoftware repositories are not
flagged — the guard gates privacy, not the organisation name. It fails loud:
an empty scan or an unreadable tree root exits 2 rather than reporting a vacuous
pass, and every offending file is reported, not just the first. The guard and
its test companion must spell the private names out in order to detect them, so
both basenames are excluded by design and a fixture case proves it.

`Cargo.toml` and `.cargo/config.toml` are build-affecting paths (issue #95), so
the crate version moves with them: `neat_ai_backpropagation` 0.1.42 → 0.1.43,
bumped by `scripts/bump-backpropagation-version.sh` with `Cargo.lock` refreshed.

Closes #171

## Evidence

This is a documentation and build-configuration change with no web interface, so
there is no rendered surface to screenshot. The evidence is the guard's own
before/after behaviour against the real repository tree.

**Before the reword** — the guard's final case runs against this repository and
named exactly the four cited locations:

```text
FAIL: this repository references no private stSoftware repository (expected exit 0, got 1)
      output: FAIL .cargo/config.toml:5: references a private stSoftware repository
     FAIL CHANGELOG.md:476: references a private stSoftware repository
     FAIL Cargo.toml:5: references a private stSoftware repository
     FAIL README.md:135: references a private stSoftware repository
check-no-private-repo-references tests: 10 passed, 1 failed
```

**After the reword** — same suite, same tree:

```text
PASS: this repository references no private stSoftware repository
check-no-private-repo-references tests: 11 passed, 0 failed
```

The four locations now read:

- `README.md` — "Workspace root `Cargo.toml` follows the fleet rule (issue #88):
  **dev compiles as fast as practical; release is fully optimised** (compile
  time irrelevant)."
- `CHANGELOG.md:476` — "Workspace build profiles follow the fleet rule — dev
  compiles as fast as practical, release is fully optimised (issue #88)".
- `Cargo.toml:5` — "Fleet rule (issue #88): dev compiles as fast as practical."
  The `[profile.release]` comment two blocks below already reads "Fully
  optimised artefact (compile time irrelevant)", so the whole rule stays stated
  in the file it governs.
- `.cargo/config.toml:5` — "an exported RUSTFLAGS replaces this list entirely
  (issue #88)".

## Test Plan

- `./scripts/test-check-no-private-repo-references.sh` — 11 cases, each running
  the real checker against a fixture tree and asserting on its exit code and
  message: a clean tree passes; the full URL form is rejected; the `Repo#1234`
  shorthand is rejected; the `owner/repo` form is rejected in a nested file;
  concept-level wording citing a local issue passes; a **public** stSoftware URL
  is not flagged; every offending file is reported, not just the first; the
  checker and its test companion are skipped; an empty tree exits 2; a missing
  tree root exits 2; and this repository's own tree passes. **11 passed, 0
  failed.**
- The last of those is the regression test for this issue — it fails against the
  unfixed tree (output above) and passes after the reword.
- `./scripts/bump-backpropagation-version.sh --base-ref origin/Develop` —
  detected `.cargo/config.toml` and `Cargo.toml` as build-affecting and bumped
  0.1.42 → 0.1.43.
- `markdownlint-cli2@0.23.2` on `README.md` and `CHANGELOG.md` — 0 issues.
- `./scripts/spell-check.sh` — no typos found.
- `bash -n quality.sh`, `shellcheck -x -s bash` on both new scripts — clean.
- `./quality.sh` — full local gate.
