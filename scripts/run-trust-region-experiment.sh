#!/usr/bin/env bash
# Issue #109: which whole-creature update budget wins most per hour?
#
# Runs the same creature, corpus, seed and epoch budget through
# `--acceptance scorer` once per `--trust-region-l2` value in $BUDGETS, plus a
# `none` arm with no budget at all (the historical fixed-step apply), and
# prints each arm's accepted epochs, scorer gain, realised update norm, wall
# clock and wins/hour.
#
# Point it at the production targets to answer the same question on a
# current production-size creature:
#   CREATURE=~/src/GRQ-cluster/network.json \
#   DATA_DIR=~/src/GRQ/.trainData-binary_116 \
#   BUDGETS=0.05,0.5,5 \
#   scripts/run-trust-region-experiment.sh path/to/rust_scorer
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCORER="${1:-${SCORER:-$ROOT/../NEAT-AI-scorer/target/release/rust_scorer}}"
OUT="${OUT:-$ROOT/.backprop/trust-region-experiment}"
EPOCHS="${EPOCHS:-8}"
SLICE_RECORDS="${SLICE_RECORDS:-20}"
SEED="${SEED:-1}"
LEARNING_RATE="${LEARNING_RATE:-1.0}"
STEP_SCALE="${STEP_SCALE:-0.01}"
MAX_BACKTRACKS="${MAX_BACKTRACKS:-6}"
# L2 budgets for the whole-creature update, smallest first.
BUDGETS="${BUDGETS:-0.001,0.01,0.1}"

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
# Only this run's own arm directories are cleared — never a caller-supplied
# CREATURE or DATA_DIR, which may live inside an overridden $OUT.
rm -rf "$OUT/arm-"*
mkdir -p "$OUT"

# Gate on a record file, never on the directory: a directory left behind by an
# interrupted generator would otherwise look like a corpus and every arm would
# "compare" on no records at all.
if [[ ! -f "$DATA_DIR/0.bin" ]]; then
  mkdir -p "$DATA_DIR"
  # 1000 records of `y = 0.5x + 0.25`.
  python3 - "$DATA_DIR" <<'PY'
import struct
import sys
from pathlib import Path

data = Path(sys.argv[1])
for file_index in range(4):
    with (data / f"{file_index}.bin").open("wb") as handle:
        for i in range(250):
            x = i / 250 * 2 - 1
            handle.write(struct.pack("<ff", x, 0.5 * x + 0.25))
PY
  if [[ ! -s "$DATA_DIR/0.bin" ]]; then
    echo "FAIL: corpus generation wrote no records to $DATA_DIR" >&2
    exit 2
  fi
fi

if [[ ! -s "$CREATURE" ]]; then
  # Two dense hidden layers: many genes move together, which is the shape that
  # makes a fixed per-gene step scale grow into a large aggregate move.
  python3 - "$CREATURE" <<'PY'
import json
import sys
from pathlib import Path

LAYER = 8
neurons = [
    {"type": "hidden", "uuid": f"a{i}", "bias": 0.01, "squash": "IDENTITY"}
    for i in range(LAYER)
]
neurons += [
    {"type": "hidden", "uuid": f"b{j}", "bias": 0.01, "squash": "IDENTITY"}
    for j in range(LAYER)
]
neurons.append({"type": "output", "uuid": "o1", "bias": 0.0, "squash": "IDENTITY"})

# Emitted in (from, to) neuron-index order — the sort neat-core enforces.
synapses = [
    {"fromUUID": "input-0", "toUUID": f"a{i}", "weight": 0.1} for i in range(LAYER)
]
synapses += [
    {"fromUUID": f"a{i}", "toUUID": f"b{j}", "weight": 0.05}
    for i in range(LAYER)
    for j in range(LAYER)
]
synapses += [
    {"fromUUID": f"b{j}", "toUUID": "o1", "weight": 0.05} for j in range(LAYER)
]

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "semanticVersion": "4.0.0",
            "forwardOnly": True,
            "input": 1,
            "output": 1,
            "neurons": neurons,
            "synapses": synapses,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
fi

echo "Building neat_ai_backpropagation (release)..."
(
  cd "$ROOT"
  cargo build -p neat_ai_backpropagation --release
)
BIN="$ROOT/target/release/neat_ai_backpropagation"

# Accepted epochs, scorer gain, realised update norm and wins/hour for one arm.
run_arm() {
  local arm="$1"
  shift
  echo
  echo "=== $arm ==="
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
    --max-backtracks "$MAX_BACKTRACKS" \
    --acceptance scorer \
    --scorer "$SCORER" \
    --output-dir "$OUT/$arm" \
    "$@"
  ended="$(python3 -c 'import time; print(time.time())')"
  elapsed="$(python3 -c "print(max($ended - $started, 1e-9))")"

  local journal="$OUT/$arm/journal.jsonl"
  local epoch_lines
  if ! epoch_lines="$(grep '"kind":"epoch"' "$journal")"; then
    echo "FAIL: $journal has no epoch lines" >&2
    exit 2
  fi
  local wins
  wins="$(printf '%s\n' "$epoch_lines" | grep -c '"accepted":true' || true)"
  echo "accepted epochs: $wins"
  echo "epoch verdicts:"
  printf '%s\n' "$epoch_lines" | grep -o '"acceptReason":"[a-zA-Z]*"' | sort | uniq -c
  # Gain and realised update size are read back out of the journal rather than
  # scraped from stderr, so a run whose journal disagrees with its own log
  # fails here instead of being reported as a win.
  python3 - "$journal" <<'PY'
import json
import sys

baseline = None
best = None
norms = []
genes = []
steps = []
for line in open(sys.argv[1], encoding="utf-8"):
    record = json.loads(line)
    if record["kind"] == "runHeader":
        baseline = best = record.get("baselineScore")
    elif record["kind"] == "epoch":
        update = record.get("update", {}).get("total", {})
        if update.get("l2") is not None:
            norms.append(update["l2"])
            genes.append(update.get("changed", 0))
            steps.append(record.get("realisedStepScale", 0.0))
        if record.get("accepted") and record.get("candidateScore") is not None:
            best = record["candidateScore"]
if baseline is None:
    sys.exit(f"FAIL: {sys.argv[1]} has no baseline score — was --acceptance scorer used?")
print(f"scorer gain: baseline={baseline:.12f} best={best:.12f} Δ={best - baseline:+.6e}")
if norms:
    mean = sum(norms) / len(norms)
    print(
        f"realised update: mean L2={mean:.6e} max L2={max(norms):.6e} "
        f"mean genes={sum(genes) / len(genes):.1f} "
        f"mean step={sum(steps) / len(steps):.6e}"
    )
else:
    print("realised update: no epoch reported an update norm")
PY
  python3 -c "print(f'wall clock: {$elapsed:.2f}s'); print(f'wins/hour: {$wins * 3600 / $elapsed:.1f}')"
}

# The control arm: no budget at all, which is the fixed-step apply.
run_arm arm-none

IFS=',' read -r -a BUDGET_LIST <<<"$BUDGETS"
for budget in "${BUDGET_LIST[@]+"${BUDGET_LIST[@]}"}"; do
  trimmed="$(printf '%s' "$budget" | tr -d '[:space:]')"
  [[ -n "$trimmed" ]] || continue
  run_arm "arm-l2-$trimmed" --trust-region-l2 "$trimmed"
done

echo
echo "Journals: $OUT/arm-*/journal.jsonl"
