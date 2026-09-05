#!/usr/bin/env bash
# Issue #104: does scorer-guided acceptance find more usable wins than the
# global MSE loop?
#
# Runs the same creature, corpus, seed and epoch budget twice — once with
# `--acceptance mse` (training-slice MSE decides) and once with
# `--acceptance scorer` (NEAT-AI-scorer decides) — and prints both verdicts.
# A usable win is a run whose final full-corpus `rust_scorer` fitness beat its
# own baseline; a slice-MSE "win" that lowered fitness is not.
#
# The generated corpus reproduces the production mismatch: the trainer accepts
# on a small unrepresentative slice while `rust_scorer` judges every record, so
# a candidate can cut slice MSE and still destroy real fitness.
#
# Point it at the production targets to answer the same question there:
#   CREATURE=~/src/GRQ-cluster/network.json \
#   DATA_DIR=~/src/GRQ/.trainData-binary_116 \
#   scripts/run-scorer-guided-experiment.sh path/to/rust_scorer
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCORER="${1:-${SCORER:-$ROOT/../NEAT-AI-scorer/target/release/rust_scorer}}"
OUT="${OUT:-$ROOT/.backprop/scorer-guided-experiment}"
EPOCHS="${EPOCHS:-8}"
SLICE_RECORDS="${SLICE_RECORDS:-20}"
SEED="${SEED:-1}"
LEARNING_RATE="${LEARNING_RATE:-1.0}"
STEP_SCALE="${STEP_SCALE:-1.0}"
MAX_BACKTRACKS="${MAX_BACKTRACKS:-0}"

if [[ ! -x "$SCORER" ]]; then
  echo "FAIL: rust_scorer not found or not executable: $SCORER" >&2
  echo "      build NEAT-AI-scorer, or pass the binary path as \$1" >&2
  exit 2
fi
if ! command -v python3 &>/dev/null; then
  echo "FAIL: python3 is required to generate the synthetic corpus" >&2
  exit 2
fi

CREATURE="${CREATURE:-$OUT/creature.json}"
DATA_DIR="${DATA_DIR:-$OUT/data}"
# Only this run's own outputs are cleared — never a caller-supplied CREATURE or
# DATA_DIR, which may live inside an overridden $OUT.
rm -rf "$OUT/mse" "$OUT/scorer"
mkdir -p "$OUT"

# Gate on a record file, never on the directory: a directory left behind by an
# interrupted generator would otherwise look like a corpus and both modes would
# "compare" on no records at all.
if [[ ! -f "$DATA_DIR/0.bin" ]]; then
  mkdir -p "$DATA_DIR"
  # 1000 records of `y = x + 0.1`, except the 5 leading records of each file
  # (20 of 1000), which follow `y = -2x + 1`. A leading-prefix slice of 20
  # therefore sees only the 2% of the corpus that contradicts the other 98%.
  python3 - "$DATA_DIR" <<'PY'
import struct
import sys
from pathlib import Path

data = Path(sys.argv[1])
for file_index in range(4):
    with (data / f"{file_index}.bin").open("wb") as handle:
        for i in range(250):
            x = i / 250 * 2 - 1
            y = -2.0 * x + 1.0 if i < 5 else 1.0 * x + 0.1
            handle.write(struct.pack("<ff", x, y))
PY
  if [[ ! -s "$DATA_DIR/0.bin" ]]; then
    echo "FAIL: corpus generation wrote no records to $DATA_DIR" >&2
    exit 2
  fi
fi

if [[ ! -s "$CREATURE" ]]; then
  # Already fitted to the 98% majority — the shape of an evolved production
  # creature, where a slice-driven move is far more likely to hurt than help.
  cat >"$CREATURE" <<'JSON'
{
  "semanticVersion": "4.0.0",
  "forwardOnly": true,
  "input": 1,
  "output": 1,
  "neurons": [
    { "type": "hidden", "uuid": "a0", "bias": 0.0, "squash": "IDENTITY" },
    { "type": "hidden", "uuid": "b0", "bias": 0.0, "squash": "IDENTITY" },
    { "type": "output", "uuid": "o1", "bias": 0.1, "squash": "IDENTITY" }
  ],
  "synapses": [
    { "fromUUID": "input-0", "toUUID": "a0", "weight": 1.0 },
    { "fromUUID": "a0", "toUUID": "b0", "weight": 1.0 },
    { "fromUUID": "b0", "toUUID": "o1", "weight": 1.0 }
  ]
}
JSON
fi

echo "Building neat_ai_backpropagation (release)..."
(
  cd "$ROOT"
  cargo build -p neat_ai_backpropagation --release
)
BIN="$ROOT/target/release/neat_ai_backpropagation"

run_mode() {
  local mode="$1"
  echo
  echo "=== --acceptance $mode ==="
  "$BIN" train "$CREATURE" "$DATA_DIR" \
    --epochs "$EPOCHS" \
    --max-records "$SLICE_RECORDS" \
    --seed "$SEED" \
    --disable-random-samples \
    --learning-rate "$LEARNING_RATE" \
    --step-scale "$STEP_SCALE" \
    --max-backtracks "$MAX_BACKTRACKS" \
    --acceptance "$mode" \
    --scorer "$SCORER" \
    --output-dir "$OUT/$mode"
  # Epoch lines only: scorer mode also journals a line per attempted
  # candidate, so counting every acceptReason would not compare like for like.
  echo "epoch verdicts:"
  local verdicts
  if ! verdicts="$(grep '"kind":"epoch"' "$OUT/$mode/journal.jsonl")"; then
    echo "FAIL: $OUT/$mode/journal.jsonl has no epoch lines" >&2
    exit 2
  fi
  printf '%s\n' "$verdicts" | grep -o '"acceptReason":"[a-zA-Z]*"' | sort | uniq -c
}

run_mode mse
run_mode scorer

echo
echo "Journals: $OUT/mse/journal.jsonl and $OUT/scorer/journal.jsonl"
