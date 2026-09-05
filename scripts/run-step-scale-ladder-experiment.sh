#!/usr/bin/env bash
# Issue #106: does a scored step-scale ladder win more per hour than the
# MSE-halving line search?
#
# Runs the same creature, corpus, seed and epoch budget twice under
# `--acceptance scorer` — once with the backtracking line search
# (`--max-backtracks`) and once with `--step-scale-ladder` — then prints each
# run's accepted epochs, scorer gain, wall-clock seconds and wins/hour.
#
# Point it at the production targets to answer the same question on real GRQ
# history:
#   CREATURE=~/src/GRQ-cluster/network.json \
#   DATA_DIR=~/src/GRQ/.trainData-binary_116 \
#   scripts/run-step-scale-ladder-experiment.sh path/to/rust_scorer
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCORER="${1:-${SCORER:-$ROOT/../NEAT-AI-scorer/target/release/rust_scorer}}"
OUT="${OUT:-$ROOT/.backprop/step-scale-ladder-experiment}"
EPOCHS="${EPOCHS:-8}"
SLICE_RECORDS="${SLICE_RECORDS:-20}"
SEED="${SEED:-1}"
LEARNING_RATE="${LEARNING_RATE:-1.0}"
STEP_SCALE="${STEP_SCALE:-0.01}"
MAX_BACKTRACKS="${MAX_BACKTRACKS:-6}"
LADDER="${LADDER:-0.0001,0.00025,0.0005,0.001,0.0025,0.005,0.01}"

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
rm -rf "$OUT/line-search" "$OUT/ladder"
mkdir -p "$OUT"

# Gate on a record file, never on the directory: a directory left behind by an
# interrupted generator would otherwise look like a corpus and both modes would
# "compare" on no records at all.
if [[ ! -f "$DATA_DIR/0.bin" ]]; then
  mkdir -p "$DATA_DIR"
  # 1000 records of `y = x + 0.1`, except the 5 leading records of each file
  # (20 of 1000), which follow `y = -2x + 1` — the same slice-versus-corpus
  # mismatch the #104 experiment reproduces.
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
  # creature, where the useful step is far smaller than the default.
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

# Accepted epochs, scorer gain and wins/hour for one search strategy.
run_mode() {
  local mode="$1"
  shift
  echo
  echo "=== $mode ==="
  local started ended elapsed
  # Fractional seconds: `date +%s` would round a fast synthetic run to 0 and
  # make wins/hour undefined, and BSD date has no %N.
  started="$(python3 -c 'import time; print(time.time())')"
  "$BIN" train "$CREATURE" "$DATA_DIR" \
    --epochs "$EPOCHS" \
    --max-records "$SLICE_RECORDS" \
    --seed "$SEED" \
    --disable-random-samples \
    --learning-rate "$LEARNING_RATE" \
    --step-scale "$STEP_SCALE" \
    --acceptance scorer \
    --scorer "$SCORER" \
    --output-dir "$OUT/$mode" \
    "$@"
  ended="$(python3 -c 'import time; print(time.time())')"
  elapsed="$(python3 -c "print(max($ended - $started, 1e-9))")"

  local journal="$OUT/$mode/journal.jsonl"
  local epochs_line
  if ! epochs_line="$(grep '"kind":"epoch"' "$journal")"; then
    echo "FAIL: $journal has no epoch lines" >&2
    exit 2
  fi
  local wins
  wins="$(printf '%s\n' "$epochs_line" | grep -c '"accepted":true' || true)"
  echo "accepted epochs: $wins"
  echo "epoch verdicts:"
  printf '%s\n' "$epochs_line" | grep -o '"acceptReason":"[a-zA-Z]*"' | sort | uniq -c
  echo "candidates scored: $(grep -c '"kind":"candidate"' "$journal" || true)"
  python3 -c "print(f'wall clock: {$elapsed:.2f}s'); print(f'wins/hour: {$wins * 3600 / $elapsed:.1f}')"
}

run_mode line-search --max-backtracks "$MAX_BACKTRACKS"
run_mode ladder --step-scale-ladder "$LADDER"

echo
echo "Journals: $OUT/line-search/journal.jsonl and $OUT/ladder/journal.jsonl"
