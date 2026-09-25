## Summary

Reworded the two remaining direct references to the private GRQ repository to
concept level, so a public reader is not pointed at a repository they cannot
open. Closes #173.

- `CHANGELOG.md` — the `[GRQ #4724](https://github.com/stSoftwareAU/GRQ/issues/4724)`
  link is now the bare `GRQ #4724`, matching the `GRQ #NNNN` form used
  elsewhere in the repo.
- `docs/archive/pr-summaries/pr-summary-31.md` — `` `stSoftwareAU/GRQ` `Develop` ``
  is now "GRQ's `Develop` branch".

Bare "GRQ" mentions elsewhere are concept-level and left alone, as the issue
scopes them out.

## Evidence

Documentation-only change; no UI to screenshot.

- `git grep -nE "stSoftwareAU/GRQ([^-]|$)|github\.com/stSoftwareAU/GRQ/"`
  returns nothing after the change (it matched both lines before).
- `./quality.sh < /dev/null` passes, including
  `scripts/test-check-no-private-repo-references.sh`.

## Test Plan

- No code changed, so no tests were added. The existing private-repo reference
  check still passes. GRQ is not on its list of private repositories, so the
  grep above is the check for this change.
