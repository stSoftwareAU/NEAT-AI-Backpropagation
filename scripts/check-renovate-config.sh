#!/usr/bin/env bash
# Validate the Renovate dependency-update configuration (Issue #19).
#
# The config must:
#   1. Be a valid JSON object extending a shared `config:` preset.
#   2. Declare a top-level `minimumReleaseAge` of at least 24 hours, so a
#      freshly-hijacked crates.io release cannot be merged on publish day.
#   3. Only shorten that window in rules scoped to internal stSoftwareAU
#      packages (external crates keep the full quarantine).
#   4. Disable `neat-core` — it is a sibling path dependency whose lockfile
#      entry is already synced by the Auto Format workflow.
#   5. Leave the `cargo` manager enabled, otherwise nothing is updated at all.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${1:-$REPO_ROOT/renovate.json}"

usage() {
  cat <<'EOF'
Usage: check-renovate-config.sh [CONFIG_PATH]

Exits 0 when the Renovate config satisfies every rule listed in the script
header, 1 when a rule is broken, and 2 when the config cannot be read.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "FAIL: renovate config not found: $CONFIG" >&2
  exit 2
fi

if ! command -v python3 &>/dev/null; then
  echo "FAIL: python3 is required to parse $CONFIG" >&2
  exit 2
fi

python3 - "$CONFIG" <<'PY'
"""Validate a Renovate config against this repository's quarantine policy."""
import json
import re
import sys

config_path = sys.argv[1]
# 24h external quarantine — the organisation's standard posture (Issue #19).
MINIMUM_HOURS = 24.0
UNIT_HOURS = {
    "minute": 1.0 / 60.0,
    "hour": 1.0,
    "day": 24.0,
    "week": 24.0 * 7.0,
    "month": 24.0 * 30.0,
    "year": 24.0 * 365.0,
}
# A dependency is internal when it comes from an stSoftwareAU repository.
INTERNAL_PATTERN = re.compile(r"stsoftware|neat-core|neat-ai|neat_ai")

exit_code = 0


def ok(message):
    print("OK   {}: {}".format(config_path, message))


def fail(message):
    global exit_code
    print("FAIL {}: {}".format(config_path, message), file=sys.stderr)
    exit_code = 1


def parse_hours(value):
    """Return the duration in hours, or None when it cannot be parsed."""
    if not isinstance(value, str):
        return None
    match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*(minute|hour|day|week|month|year)s?\s*",
        value.strip(),
        re.IGNORECASE,
    )
    if match is None:
        return None
    return float(match.group(1)) * UNIT_HOURS[match.group(2).lower()]


def match_names(rule):
    """Every package/dependency selector a rule narrows on."""
    names = []
    for key in ("matchPackageNames", "matchDepNames", "matchPackagePatterns"):
        value = rule.get(key)
        if isinstance(value, str):
            names.append(value)
        elif isinstance(value, list):
            names.extend(entry for entry in value if isinstance(entry, str))
    return names


def is_internal_only(rule):
    """True when the rule targets internal packages and nothing else."""
    names = match_names(rule)
    if not names:
        return False
    return all(INTERNAL_PATTERN.search(name.lower()) for name in names)


try:
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
except (OSError, ValueError) as error:
    print(
        "FAIL {}: not valid JSON — {}".format(config_path, error),
        file=sys.stderr,
    )
    sys.exit(1)

if not isinstance(config, dict):
    print(
        "FAIL {}: top level must be a JSON object".format(config_path),
        file=sys.stderr,
    )
    sys.exit(1)

extends = config.get("extends")
if isinstance(extends, list) and any(
    isinstance(preset, str) and preset.startswith("config:") for preset in extends
):
    ok("extends a shared config: preset")
else:
    fail("no 'config:' preset in extends — the repo would inherit no defaults")

age = config.get("minimumReleaseAge")
if age is None:
    fail(
        "no top-level minimumReleaseAge — external crates would be updatable "
        "the moment they are published"
    )
else:
    hours = parse_hours(age)
    if hours is None:
        fail(
            "minimumReleaseAge {!r} is not a parseable duration "
            "(expected e.g. '24 hours' or '3 days')".format(age)
        )
    elif hours < MINIMUM_HOURS:
        fail(
            "minimumReleaseAge {!r} is shorter than the {:.0f}-hour external "
            "quarantine".format(age, MINIMUM_HOURS)
        )
    else:
        ok("minimumReleaseAge {!r} meets the 24-hour external quarantine".format(age))

rules = config.get("packageRules", [])
if not isinstance(rules, list):
    fail("packageRules must be an array")
    rules = []
rules = [rule for rule in rules if isinstance(rule, dict)]

enabled_managers = config.get("enabledManagers")
if enabled_managers is None:
    ok("cargo manager enabled (no enabledManagers allowlist)")
elif isinstance(enabled_managers, list) and "cargo" in enabled_managers:
    ok("cargo manager present in enabledManagers")
else:
    fail(
        "enabledManagers {!r} excludes cargo — crate updates would never be "
        "raised".format(enabled_managers)
    )

short_window_offenders = []
cargo_disablers = []
neat_core_disabled = False

for index, rule in enumerate(rules):
    names = match_names(rule)
    managers = rule.get("matchManagers")
    managers = managers if isinstance(managers, list) else []

    rule_age = rule.get("minimumReleaseAge")
    if rule_age is not None:
        rule_hours = parse_hours(rule_age)
        too_short = rule_hours is None or rule_hours < MINIMUM_HOURS
        if too_short and not is_internal_only(rule):
            target = ", ".join(names) if names else "all packages"
            short_window_offenders.append(
                "rule {} ({}) sets minimumReleaseAge {!r}".format(
                    index, target, rule_age
                )
            )

    if rule.get("enabled") is False:
        if any(INTERNAL_PATTERN.search(name.lower()) for name in names):
            neat_core_disabled = neat_core_disabled or any(
                "neat-core" in name.lower() for name in names
            )
        elif "cargo" in managers and not names:
            cargo_disablers.append(
                "rule {} disables the cargo manager outright".format(index)
            )

if short_window_offenders:
    for offender in short_window_offenders:
        fail(
            "{} — only internal stSoftwareAU packages may skip the "
            "quarantine".format(offender)
        )
else:
    ok("no package rule shortens the quarantine for an external crate")

if cargo_disablers:
    for disabler in cargo_disablers:
        fail("{} — no crate would ever be updated".format(disabler))
else:
    ok("no package rule disables the cargo manager")

if neat_core_disabled:
    ok("neat-core is disabled (path dependency, synced by Auto Format)")
else:
    fail(
        "no package rule disables neat-core — Renovate would fight the Auto "
        "Format workflow over the sibling path dependency"
    )

sys.exit(exit_code)
PY
