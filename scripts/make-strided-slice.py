#!/usr/bin/env python3
"""Copy evenly spaced records from every production .bin into numbered files."""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--records", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument(
        "--phase",
        type=int,
        default=0,
        help="Shift the stride start (0..records-1) for a disjoint holdout",
    )
    args = parser.parse_args()
    if args.records <= 0 or args.width <= 0:
        print("records and width must be positive", file=sys.stderr)
        return 2

    rec_bytes = args.width * 4
    os.makedirs(args.dest, exist_ok=True)
    files = sorted(
        name
        for name in os.listdir(args.src_dir)
        if name.endswith(".bin") and os.path.isfile(os.path.join(args.src_dir, name))
    )
    idx = 0
    copied = 0
    for name in files:
        src = os.path.join(args.src_dir, name)
        size = os.path.getsize(src)
        n = size // rec_bytes
        if n < args.records:
            print(f"WARN: skip {name} (only {n} records)", file=sys.stderr)
            continue
        dest = os.path.join(args.dest, f"{idx}.bin")
        with open(src, "rb") as inf, open(dest, "wb") as outf:
            for i in range(args.records):
                src_i = (i * n + args.phase) // args.records
                if src_i >= n:
                    src_i = n - 1
                inf.seek(src_i * rec_bytes)
                chunk = inf.read(rec_bytes)
                if len(chunk) != rec_bytes:
                    print(f"FAIL: short read {name} record {src_i}", file=sys.stderr)
                    return 1
                outf.write(chunk)
        idx += 1
        copied += args.records
    if idx == 0:
        print(f"FAIL: no usable .bin files in {args.src_dir}", file=sys.stderr)
        return 1
    print(f"OK   wrote {idx} files under {args.dest} ({copied} records, phase={args.phase})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
