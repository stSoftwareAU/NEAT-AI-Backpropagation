# Reword the private runlib.sh link to concept level (issue #172)

## Summary

`CONTRIBUTING.md`'s "Version bumping" section explained the rebuild-skip rule
by linking a copy of `runlib.sh` inside a **private** sibling repository. Every
public reader who followed that "further reading" link hit a 404 — a public
repository has to be self-contained, so the link was dead weight to everyone
outside the organisation.

The section now states the convention itself and links this repository's **own
public** [`scripts/runlib.sh`](../../../scripts/runlib.sh), which implements
it: remote runners compare the installed `neat_ai_backpropagation` version
marker against `Cargo.toml` and skip rebuilding when the two match. A second
sentence in the same section named the same private repository for the
version-increment rule and is now stated at concept level as well, so no reader
needs access to a private repository to follow the section.

No behaviour changes: this is documentation only. No build-affecting path is
touched (`scripts/build-affecting-paths.sh` lists `backpropagation/src/**`,
the two `Cargo.toml`s, `Cargo.lock`, `.cargo/config.toml`, `rust-toolchain.toml`
and `include/**`), so no crate version bump is required and none was made.

Closes #172

## Evidence

This is a documentation change with no web interface, so there is no rendered
surface to screenshot. The evidence is the private-repository reference
disappearing from the tracked tree. The removed URL is elided below rather than
quoted, so this summary does not reintroduce the very reference it removes.

Before — `CONTRIBUTING.md:165-169`:

```text
`Cargo.lock` in sync). Remote GRQ runners use the same pattern as
[`runlib.sh`](https://github.com/<private-repo>/blob/Develop/scripts/runlib.sh):
they compare the installed `neat_ai_backpropagation` version marker against
`Cargo.toml` and skip rebuilding when they match. Forgetting to bump leaves
stale binaries on remote machines.
```

After:

```text
`Cargo.lock` in sync). Remote runners follow the shared version-marker
convention that this repository's own
[`scripts/runlib.sh`](./scripts/runlib.sh) implements: they compare the
installed `neat_ai_backpropagation` version marker against `Cargo.toml` and
skip rebuilding when the two match. Forgetting to bump leaves stale binaries
on remote machines.
```

Before — `CONTRIBUTING.md:175`, which named the private repository in prose:

```text
Develop (same approach as <private-repo>). Bumping locally keeps the version
correct and avoids an extra bot commit.
```

After:

```text
Develop (the same approach the sibling repositories in this family take).
Bumping locally keeps the version correct and avoids an extra bot commit.
```

`CONTRIBUTING.md` now carries no reference a public reader cannot open:

```text
$ grep -cn "github.com/stSoftwareAU" CONTRIBUTING.md
0
```

The replacement link resolves inside this repository — `scripts/runlib.sh` is
the public canonical NEAT-AI-core copy this repository already ships and
`README.md` already links.

## Test Plan

- `./quality.sh` — full gate, **exit 0** ("All quality checks passed!"),
  including `markdownlint-cli2`, `./scripts/spell-check.sh`, the workflow
  policy guards, `cargo clippy`, `cargo test` and `cargo doc`.
- `grep` for the private repository name and for any `github.com/stSoftwareAU`
  URL in `CONTRIBUTING.md` — no matches remain.
- Relative link check — `scripts/runlib.sh` exists in the tracked tree, so the
  replacement link resolves for a public reader.
- No build-affecting path is touched, so
  `scripts/check-crate-version-no-downgrade.sh` and the Version Increment
  workflow require no bump; the crate version is unchanged.
