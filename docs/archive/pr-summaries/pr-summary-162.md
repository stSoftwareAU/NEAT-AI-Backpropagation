## Summary

Moved both `github/codeql-action` call sites in `.github/workflows/codeql.yml`
off the `c16c0f3f…` pin (codeql-bundle-v2.26.3, published 2026-08-12) to the
current `v4` release at `1c5b6756…` (v4.38.1, published 2026-09-18). The old
pin was a v4-tagged bundle — and therefore never inside the advisory's
vulnerable `>= 3.26.11, <= 3.28.2` range — but it predated the fix, and its
trailing comment named a bundle tag rather than an exact release. Pinning to a
tag at or after the first patched version (3.28.3) and naming that tag in the
comment keeps the finding's "move every call site to a SHA at or after 3.28.3"
requirement satisfied on both counts. Closes #162.

## Evidence

Backend/CI-config change with no web interface, so no screenshot. Evidence is
the resolved pin and the local gate output.

The new SHA resolves to `v4.38.1` (verified with
`gh api repos/github/codeql-action/commits/1c5b675653bb5c22dbe9b12b556ec555138e09fd`,
published 2026-09-18T13:09:51Z — clear of the repo's 24-hour supply-chain
quarantine), and the GHSA affected ranges confirm the fix applies only to the
`3.x` line (`introduced: 3.26.11`, `fixed: 3.28.3`, plus `introduced: 2.26.11`
with `last_known_affected < 3.0.0`).

Targeted checks for the changed workflow all pass:

```text
actionlint -no-color .github/workflows/codeql.yml     → PASS
scripts/check-codeql-workflow.sh                      → 7/7 OK
scripts/test-check-codeql-workflow.sh                 → 13 passed, 0 failed
```

Full `./quality.sh < /dev/null` was run and reaches the CodeQL and workflow
gates cleanly, but fails at the *canonical-copies drift* gate: the base branch's
`scripts/runlib.sh` has drifted from NEAT-AI-core `Develop` (core added an MSRV
toolchain gate after the last sync). That failure is pre-existing on the base
commit (`b44a7e7`) — `git status` shows the only file this change touches is
`.github/workflows/codeql.yml` — and is out of scope for this issue: the
`family-sync` workflow re-copies `scripts/runlib.sh` from NEAT-AI-core `Develop`
on every same-repo PR, so the drift heals when this PR opens.

## Test Plan

- `scripts/test-check-codeql-workflow.sh` — 13 cases running the real checker
  against fixture workflows; all pass (the checker asserts SHA pinning, so the
  updated pin is exercised by the "every third-party action is pinned" rule).
- `scripts/check-codeql-workflow.sh` against the committed workflow — passes,
  including "every third-party action is pinned to a commit SHA".
- `actionlint -no-color` on `.github/workflows/codeql.yml` — no findings.
