## Summary

Finishes #191. PR #194 had already added the second private sibling repository to
`PRIVATE_REPOS` (step 1) and added its red/green tests (step 2). No archived PR
summary contains the name, so step 3b needed no change. What was still open was
step 3a: the bare name in the comment on line 2 of
`.github/workflows/version-increment.yml`. That comment now describes the shared
fleet version-increment job and the `runlib.sh` contract without naming the
repository. It matches the wording already used in
`scripts/bump-backpropagation-version.sh:10`. Closes #193.

## Evidence

This change affects CI configuration and a comment only, so there is no UI to
screenshot.

- `git grep GRQ-taxation` now finds the name only in the guard script and its
  test companion, which the guard deliberately skips.
- `scripts/check-no-private-repo-references.sh` passes: 186 files scanned.
- `scripts/test-check-no-private-repo-references.sh` passes: 16 passed, 0
  failed.
- `scripts/check-version-increment-workflow.sh` passes.
- `./quality.sh` passes after the change.

The guard still matches only URL and `Repo#N` citations, so it would not catch
the bare name if it came back. Matching bare names is tracked as #192.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — The body of #191 checked against what was delivered: step 1, `PRIVATE_REPOS` covers the second name — evidence: `scripts/check-no-private-repo-references.sh:29` (landed in #194) — reviewer: met
- **met** — #191 step 2, a red/green test case for the new name — evidence: `scripts/test-check-no-private-repo-references.sh:87-108`, 16 passed (landed in #194) — reviewer: met
- **met** — #191 step 3a, reword `version-increment.yml:2` at concept level — evidence: `.github/workflows/version-increment.yml:2` in this diff — reviewer: met
- **met** — #191 step 3b, decide how to handle the archived pr-summaries — evidence: `git grep` finds no archived summary containing the name, and `docs/archive/pr-summaries/pr-summary-191.md` records that neither rewording nor an exclusion is needed — reviewer: met

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **clean** — The reviewer found no violations. It checked for private-repo references (the guard passes), Australian English (codespell passes), the workflow validators (`check-version-increment-workflow.sh` and actionlint pass), and version bump and CHANGELOG policy (a CI-config comment needs neither). The repository has no `CODING-STANDARDS.md`, so the reviewer used `CONTRIBUTING.md`. Optional note: the comment could cite an issue number.

## Test Plan

- Ran `scripts/check-no-private-repo-references.sh`: it passes.
- Ran `scripts/test-check-no-private-repo-references.sh`: 16 passed.
- Ran `scripts/check-version-increment-workflow.sh`: it passes.
- Ran `./quality.sh`: all checks pass.
- No new tests: the change is a comment reword that the existing guard and workflow validator already cover.
