#!/usr/bin/env bash
# Single source of truth for the synthetic corpus the experiment scripts train
# and score on (issue #178).
#
# Usage: generate-synthetic-corpus.sh DATA_DIR TARGET
#
# Writes `0.bin`..`3.bin` into DATA_DIR (created if absent), 250 records per
# file. Record `i` of each file is `struct.pack("<ff", x, y)` with
# `x = i / 250 * 2 - 1` and `y = TARGET`, where TARGET is an arithmetic
# expression over `x` and `i` — e.g. `0.5 * x + 0.25`, or
# `-2.0 * x + 1.0 if i < 5 else 1.0 * x + 0.1` for a corpus whose leading
# records contradict the rest.
#
# TARGET is parsed, not trusted: anything beyond numbers, `x`, `i`,
# arithmetic, comparisons and `if`/`else` is refused before a byte is written.
# Every record is computed before any file is opened, so a failure never
# leaves a partial corpus that a caller's `0.bin` gate would mistake for data.
#
# Exit codes: 0 written, 1 generation failed, 2 usage error or refused TARGET.
set -euo pipefail

if [[ $# -ne 2 || -z "$1" || -z "$2" ]]; then
  echo "usage: $(basename "$0") DATA_DIR TARGET" >&2
  exit 2
fi
if ! command -v python3 &>/dev/null; then
  echo "FAIL: python3 is required to generate the synthetic corpus" >&2
  exit 1
fi

python3 - "$1" "$2" <<'PY'
import ast
import struct
import sys
from pathlib import Path

FILES = 4
RECORDS_PER_FILE = 250

data, target = Path(sys.argv[1]), sys.argv[2]

ALLOWED = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.IfExp, ast.Compare, ast.BoolOp,
    ast.Load, ast.Name, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.And, ast.Or, ast.Not,
)

def refuse(reason):
    print(f"FAIL: TARGET {target!r} refused: {reason}", file=sys.stderr)
    sys.exit(2)

try:
    tree = ast.parse(target, mode="eval")
except SyntaxError as error:
    refuse(f"not an expression ({error.msg})")
for node in ast.walk(tree):
    if not isinstance(node, ALLOWED):
        refuse(f"{type(node).__name__} is not allowed")
    if isinstance(node, ast.Name) and node.id not in ("x", "i"):
        refuse(f"unknown name {node.id!r}; only x and i are defined")
    if isinstance(node, ast.Constant) and (
        isinstance(node.value, bool) or not isinstance(node.value, (int, float))
    ):
        refuse(f"constant {node.value!r} is not a number")
formula = compile(tree, "<TARGET>", "eval")

try:
    blobs = []
    for _ in range(FILES):
        records = bytearray()
        for i in range(RECORDS_PER_FILE):
            x = i / RECORDS_PER_FILE * 2 - 1
            y = eval(formula, {"__builtins__": {}}, {"x": x, "i": i})
            records += struct.pack("<ff", x, y)
        blobs.append(bytes(records))
    data.mkdir(parents=True, exist_ok=True)
    for file_index, blob in enumerate(blobs):
        (data / f"{file_index}.bin").write_bytes(blob)
except Exception as error:  # re-raised as a loud, non-zero exit
    print(f"FAIL: corpus generation into {data} failed: {error}", file=sys.stderr)
    sys.exit(1)
PY
