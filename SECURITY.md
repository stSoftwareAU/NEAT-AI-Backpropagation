# Security Policy

This document describes how to report a security vulnerability in
**NEAT-AI-Backpropagation** and what to expect once you do. It follows
[GitHub's guidance on adding a security policy](https://docs.github.com/en/code-security/getting-started/adding-a-security-policy-to-your-repository),
so GitHub surfaces it in the repository's **Security** tab.

## Reporting a vulnerability

**Please do not open a public issue for security vulnerabilities.** A public
issue discloses the problem before a fix exists and puts every consumer of the
tool at risk.

Use one of the private channels below instead:

1. **GitHub private vulnerability reporting (preferred).** Open the
   repository's **Security** tab and choose **Report a vulnerability** to start
   a private advisory visible only to you and the maintainers. See
   [Privately reporting a security vulnerability](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability).
2. **Email.** If you cannot use GitHub private reporting, email
   **`security@stsoftware.com.au`** with the details. Use the subject line
   `NEAT-AI-Backpropagation security report`.

Whichever channel you choose, please include as much of the following as you
can so we can reproduce and triage quickly:

- a description of the vulnerability and its impact;
- the affected component, file, or command (for example the
  `neat_ai_backpropagation` CLI);
- step-by-step reproduction instructions, including a creature JSON and data
  directory where relevant;
- any proof-of-concept input, logs, or stack traces;
- the commit SHA or branch you tested against.

## Response targets

We aim to honour the following timeline. These are targets, not contractual
guarantees, and they are measured in business days.

| Stage                          | Target                              |
| ------------------------------ | ----------------------------------- |
| Acknowledge your report        | Within 3 business days              |
| Initial assessment / triage    | Within 10 business days             |
| Fix or mitigation plan         | Communicated after triage           |
| Public disclosure              | Coordinated with you once a fix lands |

We will keep you informed of progress and coordinate the timing of any public
disclosure with you. Please give us a reasonable opportunity to remediate
before disclosing publicly.

## Supported versions

NEAT-AI-Backpropagation is developed as a single-consumer internal experiment
and is not published to a registry (`neat-core` is consumed as a local `path`
dependency). There is no semantic-version release line, so security fixes are
applied to the active development branch only.

| Version            | Supported          |
| ------------------ | ------------------ |
| `Develop` (latest) | yes                |
| Older commits      | no                 |

Always update to the latest commit on `Develop` to receive security fixes.

## Automated scanning

| Gate | Where | Covers |
| ---- | ----- | ------ |
| CodeQL (`security-and-quality`) | [`.github/workflows/codeql.yml`](./.github/workflows/codeql.yml) — PRs, pushes to `Develop`, weekly cron | this repository's own Rust |
| `rustsec/audit-check`, `cargo-deny` | [`.github/workflows/security.yml`](./.github/workflows/security.yml), `quality.sh` | advisories and licences in dependencies |
| `actions/dependency-review-action` | [`.github/workflows/security.yml`](./.github/workflows/security.yml) — every pull request | the crates a PR adds or upgrades, summarised as a PR comment |
| Renovate (`osvVulnerabilityAlerts`) | [`renovate.json`](./renovate.json) | advisory-driven crate bumps, no PR needed to trigger |

Dependabot alerts and Dependabot security updates are repository settings
rather than committed files. A repository administrator enables them under
**Settings → Advanced Security**; nothing in the checkout can turn them on.
Once enabled they complement the gates above by raising a PR the moment an
advisory lands.

### Emergency bypass of the quarantine window

[`renovate.json`](./renovate.json) holds every external bump for
`minimumReleaseAge: "24 hours"`, so a malicious or broken release has a day to
be caught upstream before this repository pulls it in. That default is the
wrong one exactly once: when an advisory lands against a version this
repository already ships, the fix is published upstream, and waiting out the
window leaves the exploitable version in `Cargo.lock` for another day.

The fast lane, so it is not improvised under pressure:

1. **Who approves.** A repository administrator (a
   [`.github/CODEOWNERS`](./.github/CODEOWNERS) owner) decides that an
   advisory is being actively exploited, or is severe enough, to justify
   skipping the wait. Record that decision in the pull request — the
   advisory identifier (GHSA/RUSTSEC/CVE) and why it cannot wait.
2. **How to raise the bump.** Renovate honours the window when it *creates* a
   branch, so do not wait for it. Bump the dependency by hand
   (`cargo update -p <crate> --precise <version>`, or edit the pin and run
   `cargo update -p <crate>`) on a branch and open a pull request that cites
   the advisory. A hand-raised bump is subject to the same required checks as
   any other pull request, so nothing merges unreviewed or untested.
   Alternatively, scope a `packageRules` entry to the single affected package
   with `"minimumReleaseAge": null`, merge that, and let Renovate raise the
   bump immediately.
3. **Afterwards.** If a `packageRules` carve-out was used, remove it in the
   same pull request as the bump or the one straight after — a carve-out left
   behind quietly drops the quarantine for that package for ever. The blanket
   24-hour default stays untouched for everything else.

Nothing here bypasses a review or a status check: the quarantine window is the
only thing waived, and only for the named advisory.

## Branch protection

`Develop` is protected by a repository ruleset requiring a pull request, at
least one approving review, code-owner review of
[`.github/CODEOWNERS`](./.github/CODEOWNERS) paths, the `CI Required Checks`
aggregator, and no force-pushes. The rules and the reasoning — including why
signed commits are *not* required — are in
[CONTRIBUTING.md](./CONTRIBUTING.md#branch-protection), and
`./scripts/check-branch-protection.sh` checks the live ruleset against them.
Like Dependabot above, a ruleset is a repository setting: only an administrator
can change one, so the checker reports drift rather than enforcing it.

## Scope

This policy covers the code in this repository. Vulnerabilities in the
upstream [`NEAT-AI-core`](https://github.com/stSoftwareAU/NEAT-AI-core)
dependency should be reported against that repository. Authoritative scoring
behaviour belongs to [`NEAT-AI-scorer`](https://github.com/stSoftwareAU/NEAT-AI-scorer).
