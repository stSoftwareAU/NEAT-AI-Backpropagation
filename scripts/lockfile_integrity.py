#!/usr/bin/env python3
"""Verify `Cargo.lock` against a snapshot of the crates.io sparse index (Issue #148).

A lockfile's `dependencies = [...]` list is meant to mirror what the resolved
crate's own manifest declares, and nothing reports it when the two diverge.
`cargo` itself does not compile a substituted entry — it re-resolves against
the real manifest and silently rewrites the lockfile back, or under `--locked`
refuses with a generic "cannot update the lock file" error that names no crate.
Neither response distinguishes tampering from ordinary drift, so a reviewer
reading a lockfile has no way to tell a substituted sub-dependency from a
legitimate upstream rename (issue #148, where `serde_json` swapping `ryu` for
`zmij` read as a compromise indicator).

This module makes that comparison explicit, checking each `[[package]]` block
against the crates.io index entry for that exact version:

1. Every registry package is sourced from the one registry `deny.toml` allows
   and carries a 64-hex sha256 checksum.
2. Every name in a `dependencies` list resolves to a `[[package]]` block in the
   same lockfile — no dangling references.
3. The recorded checksum equals the registry's `cksum` for that version.
4. Every recorded dependency is genuinely declared (normal or build kind,
   honouring `package = ` renames) by that version's published manifest.

Path packages (workspace members and sibling path dependencies) have no
registry entry, so only rules 1 and 2 apply to them.

The index snapshot is a directory of sparse-index files named after the crate,
one JSON object per line — the format `index.crates.io` serves. Keeping the
fetch out of this module makes the verification deterministic and testable
offline.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

CRATES_IO_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"
CHECKSUM_RE = re.compile(r"^[0-9a-f]{64}$")
FIELD_RE = re.compile(r'^(name|version|source|checksum) = "([^"]*)"$')
DEP_RE = re.compile(r'^"([^"]+)",?$')

# Dependency kinds that end up in the built artefact. `dev` dependencies of a
# registry crate are never compiled for a downstream consumer and so never
# appear in the consumer's lockfile dependency list.
BUILT_KINDS = (None, "normal", "build")


class Package:
    """One `[[package]]` block of a lockfile."""

    def __init__(self) -> None:
        self.name: str | None = None
        self.version: str | None = None
        self.source: str | None = None
        self.checksum: str | None = None
        self.dependencies: list[str] = []

    @property
    def label(self) -> str:
        return f"{self.name} {self.version}"

    @property
    def is_registry(self) -> bool:
        return self.source is not None


def parse_lockfile(text: str) -> list[Package]:
    """Parse the `[[package]]` blocks of a `Cargo.lock`.

    Cargo writes a fixed, machine-generated subset of TOML, so a line reader is
    enough and keeps this script free of a TOML dependency.
    """
    packages: list[Package] = []
    current: Package | None = None
    in_dependencies = False

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if line == "[[package]]":
            current = Package()
            packages.append(current)
            in_dependencies = False
            continue

        if current is None:
            continue

        if in_dependencies:
            if line.startswith("]"):
                in_dependencies = False
                continue
            match = DEP_RE.match(line)
            if match:
                current.dependencies.append(match.group(1))
            continue

        if line.startswith("["):
            # Any other table (`[metadata]`, `[[patch…]]`) ends this block.
            current = None
            continue

        if line.startswith("dependencies = ["):
            if not line.endswith("]"):
                in_dependencies = True
            continue

        match = FIELD_RE.match(line)
        if match:
            setattr(current, match.group(1), match.group(2))

    return packages


def split_dependency(reference: str) -> tuple[str, str | None]:
    """Split a lockfile dependency reference into its name and optional version.

    Cargo qualifies a reference with the version (`"syn 3.0.3"`) only when the
    lockfile holds more than one version of that crate.
    """
    parts = reference.split(" ")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return reference, None


def load_index_entry(index_dir: str, name: str, version: str) -> dict | None:
    """Return the sparse-index record for `name@version`, or None if unpublished."""
    path = os.path.join(index_dir, name.lower())
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("vers") == version:
                return record
    return None


def declared_dependencies(entry: dict) -> set[str]:
    """The crate names a published manifest declares as built dependencies.

    A renamed dependency carries the alias in `name` and the real crate in
    `package`; the lockfile records the real crate, so that is what we compare.
    """
    declared: set[str] = set()
    for dependency in entry.get("deps", []):
        if dependency.get("kind") not in BUILT_KINDS:
            continue
        declared.add(dependency.get("package") or dependency["name"])
    return declared


def verify(packages: list[Package], index_dir: str) -> tuple[list[str], list[str]]:
    """Check every package against the index snapshot.

    Returns `(ok_messages, failures)`; an empty `failures` means the lockfile
    reflects what crates.io actually published.
    """
    ok: list[str] = []
    failures: list[str] = []

    resolvable: set[str] = set()
    for package in packages:
        resolvable.add(str(package.name))
        resolvable.add(package.label)

    for package in packages:
        # Rule 1 — source and checksum shape.
        if package.is_registry:
            if package.source != CRATES_IO_SOURCE:
                failures.append(
                    f"{package.label}: source '{package.source}' is not the "
                    f"crates.io registry allowed by deny.toml"
                )
                continue
            if not package.checksum or not CHECKSUM_RE.match(package.checksum):
                failures.append(
                    f"{package.label}: registry package has no valid sha256 "
                    f"checksum — cargo cannot verify what it downloads"
                )
                continue
        elif package.checksum:
            failures.append(
                f"{package.label}: has a checksum but no source — a path "
                f"package's contents are not registry-verifiable"
            )

        # Rule 2 — every dependency reference resolves inside the lockfile.
        for reference in package.dependencies:
            if reference not in resolvable:
                failures.append(
                    f"{package.label}: depends on '{reference}', which has no "
                    f"[[package]] entry in this lockfile"
                )

        if not package.is_registry:
            ok.append(f"{package.label}: path package, no registry entry to verify")
            continue

        # Rules 3 and 4 — the published record for this exact version.
        entry = load_index_entry(index_dir, str(package.name), str(package.version))
        if entry is None:
            failures.append(
                f"{package.label}: no such version in the crates.io index — "
                f"the lockfile pins a release that was never published"
            )
            continue

        if entry.get("cksum") != package.checksum:
            failures.append(
                f"{package.label}: checksum {package.checksum} does not match "
                f"the registry's {entry.get('cksum')} — the lockfile has been "
                f"altered or points at different content"
            )
            continue

        declared = declared_dependencies(entry)
        phantoms = []
        for reference in package.dependencies:
            dependency_name, _ = split_dependency(reference)
            if dependency_name not in declared:
                phantoms.append(dependency_name)
        if phantoms:
            failures.append(
                f"{package.label}: records dependencies "
                f"{sorted(phantoms)} that the published manifest never "
                f"declares (it declares {sorted(declared)}) — the lockfile "
                f"does not describe the crate crates.io published"
            )
            continue

        ok.append(
            f"{package.label}: checksum and {len(package.dependencies)} "
            f"dependencies match the registry"
        )

    return ok, failures


def registry_crate_names(packages: list[Package]) -> list[str]:
    """The distinct registry crate names needing an index fetch."""
    return sorted({str(p.name) for p in packages if p.is_registry})


def read_lockfile(path: str) -> list[Package]:
    with open(path, encoding="utf-8") as handle:
        packages = parse_lockfile(handle.read())
    if not packages:
        raise ValueError(f"no [[package]] blocks found in {path}")
    for package in packages:
        if not package.name or not package.version:
            raise ValueError(f"malformed [[package]] block in {path}: {package.label}")
    return packages


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lockfile", required=True, help="path to Cargo.lock")
    parser.add_argument("--index-dir", help="directory holding sparse-index files")
    parser.add_argument(
        "--list-crates",
        action="store_true",
        help="print the registry crate names needing an index fetch, then exit",
    )
    arguments = parser.parse_args(argv)

    try:
        packages = read_lockfile(arguments.lockfile)
    except (OSError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2

    if arguments.list_crates:
        for name in registry_crate_names(packages):
            print(name)
        return 0

    if not arguments.index_dir:
        print("FAIL: --index-dir is required to verify the lockfile", file=sys.stderr)
        return 2
    if not os.path.isdir(arguments.index_dir):
        print(
            f"FAIL: index snapshot not found: {arguments.index_dir}", file=sys.stderr
        )
        return 2

    ok, failures = verify(packages, arguments.index_dir)
    for message in ok:
        print(f"OK   {message}")
    for message in failures:
        print(f"FAIL {message}", file=sys.stderr)

    if failures:
        print(
            f"lockfile integrity: {len(failures)} violation(s) across "
            f"{len(packages)} packages",
            file=sys.stderr,
        )
        return 1

    print(f"lockfile integrity: {len(packages)} packages verified")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
