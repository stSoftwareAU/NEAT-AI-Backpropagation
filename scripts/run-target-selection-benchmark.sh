#!/usr/bin/env bash
# Issue #108: does ranking sparse backprop targets on the accumulated evidence
# find more scorer wins per hour than drawing them uniformly at random?
#
# Runs the same creature, corpus, seed and step scale twice through the
# `blocks` subcommand — once with `--target-selection evidence` and once with
# `--target-selection random`, the uniform draw this crate shipped before —
# and prints candidates, scorer wins, best score delta, elapsed seconds,
# wins/hour and score gain/hour for each arm. Those two rates are the measure
# the issue asks for: a strategy that spends its scorer runs where the
# evidence points should win more often per hour than one that spreads them
# evenly.
#
# The bundled creature and corpus are a smoke run only — a tiny synthetic
# network is not a highly evolved creature. Point it at the production targets
# to answer the same question there:
#   CREATURE=~/src/GRQ-cluster/network.json \
#   DATA_DIR=~/src/GRQ/.trainData-binary_116 \
#   scripts/run-target-selection-benchmark.sh path/to/rust_scorer
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCORER="${1:-${SCORER:-$ROOT/../NEAT-AI-scorer/target/release/rust_scorer}}"
OUT="${OUT:-$ROOT/.backprop/target-selection-benchmark}"
SEED="${SEED:-1}"
STEP_SCALE="${STEP_SCALE:-0.01}"
LEARNING_RATE="${LEARNING_RATE:-0.1}"
BLOCKS_PER_STRATEGY="${BLOCKS_PER_STRATEGY:-4}"
RADIUS="${RADIUS:-1}"
MAX_RECORDS="${MAX_RECORDS:-}"
# Focus strategies only — `global`, `output-head` and `top-genes` select a
# region rather than a target, so they say nothing about target selection.
BLOCK_STRATEGIES="${BLOCK_STRATEGIES:-neuron,neighbourhood}"

if [[ ! -x "$SCORER" ]]; then
  echo "FAIL: rust_scorer not found or not executable: $SCORER" >&2
  echo "      build NEAT-AI-scorer, or pass the binary path as \$1" >&2
  exit 2
fi
if ! command -v python3 &>/dev/null; then
  echo "FAIL: python3 is required to generate the corpus and read blocks.json" >&2
  exit 2
fi

CREATURE="${CREATURE:-$OUT/creature.json}"
DATA_DIR="${DATA_DIR:-$OUT/data}"
# Only this run's own outputs are cleared — never a caller-supplied CREATURE or
# DATA_DIR, which may live inside an overridden $OUT.
rm -rf "$OUT/evidence" "$OUT/random"
mkdir -p "$OUT"

# Gate on a record file, never on the directory: a directory left behind by an
# interrupted generator would otherwise look like a corpus and both arms would
# "compare" on no records at all.
if [[ ! -f "$DATA_DIR/0.bin" ]]; then
  mkdir -p "$DATA_DIR"
  # 1000 records of `y = x + 0.1`, except the 5 leading records of each file,
  # which follow `y = -2x + 1` — the same majority/minority corpus the earlier
  # blockwise and scorer-guided experiments generated, so the runs compare.
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
  # One live path (`live` carries the whole signal) beside eleven dormant
  # branches wired in and out with zero weight: the dormant units never
  # activate and never see error, so a uniform draw over twelve targets spends
  # most of its scorer runs on them and a ranked draw does not. The live path
  # computes `o = x` exactly, so the only gain available is the missing `+0.1`
  # on the output bias. The pool is deliberately larger than
  # `--blocks-per-strategy`, or both arms would simply select everything.
  python3 - "$CREATURE" <<'PY'
import json
import sys
from pathlib import Path

hidden = ["live"] + [f"dormant{i}" for i in range(11)]
neurons = [
    {"type": "hidden", "uuid": uuid, "bias": 0.0, "squash": "IDENTITY"} for uuid in hidden
]
neurons.append({"type": "output", "uuid": "o1", "bias": 0.0, "squash": "IDENTITY"})
# neat-core requires the synapse list sorted by source then target, so every
# `input-0 →` edge precedes the hidden `→ o1` edges.
synapses = [
    {"fromUUID": "input-0", "toUUID": uuid, "weight": 1.0 if uuid == "live" else 0.0}
    for uuid in hidden
]
synapses += [
    {"fromUUID": uuid, "toUUID": "o1", "weight": 1.0 if uuid == "live" else 0.0}
    for uuid in hidden
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
    + "\n"
)
PY
  if [[ ! -s "$CREATURE" ]]; then
    echo "FAIL: creature generation wrote nothing to $CREATURE" >&2
    exit 2
  fi
fi

echo "Building neat_ai_backpropagation (release)..."
(
  cd "$ROOT"
  cargo build -p neat_ai_backpropagation --release
)
BIN="$ROOT/target/release/neat_ai_backpropagation"

# Read one arm's blocks.json and print
# "candidates wins best_delta wins_per_hour gain_per_hour".
# Arguments reach python through argv, never through interpolated source — an
# empty value must fail loudly here, not become a python syntax error.
summarise() {
  python3 - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
elapsed = max(int(sys.argv[2]), 1)
# Only candidates that reached disk were scored; an unmoved block writes no
# creature and must not inflate the count.
candidates = [c for c in summary["candidates"] if c.get("candidate")]
wins = [c for c in candidates if c.get("scoreWin")]
deltas = [c.get("scoreDelta") or 0.0 for c in candidates]
best = max(deltas) if deltas else 0.0
# A rejected candidate is rolled back, so only the gains that survived count.
gain = sum(d for d in deltas if d > 0.0)
hours = elapsed / 3600
print(
    f"{len(candidates)} {len(wins)} {best:+.6e} "
    f"{len(wins) / hours:.1f} {gain / hours:+.6e}"
)
PY
}

run_arm() {
  local arm="$1"
  echo
  echo "=== $arm target selection ==="
  local started ended elapsed
  local -a limit=()
  [[ -n "$MAX_RECORDS" ]] && limit=(--max-records "$MAX_RECORDS")
  started="$(date +%s)"
  "$BIN" blocks "$CREATURE" "$DATA_DIR" \
    --seed "$SEED" \
    --learning-rate "$LEARNING_RATE" \
    --step-scale "$STEP_SCALE" \
    --strategies "$BLOCK_STRATEGIES" \
    --blocks-per-strategy "$BLOCKS_PER_STRATEGY" \
    --radius "$RADIUS" \
    --target-selection "$arm" \
    ${limit[@]+"${limit[@]}"} \
    --skip-mse \
    --scorer "$SCORER" \
    --output-dir "$OUT/$arm"
  ended="$(date +%s)"
  elapsed=$((ended - started))
  # A sub-second arm still divides by a whole second, which understates its
  # rate — never overstates it.
  [[ "$elapsed" -lt 1 ]] && elapsed=1
  local stats
  stats="$(summarise "$OUT/$arm/blocks.json" "$elapsed")"
  local candidates wins best rate gain
  read -r candidates wins best rate gain <<<"$stats"
  printf '%-8s candidates=%-3s wins=%-3s best_score_delta=%s elapsed=%ss wins/hour=%s score_gain/hour=%s\n' \
    "$arm" "$candidates" "$wins" "$best" "$elapsed" "$rate" "$gain"
}

run_arm evidence
run_arm random

echo
echo "Candidates and per-target rank features: $OUT/evidence/blocks.json and $OUT/random/blocks.json"
echo "A single run can also carry both arms: add --random-control-fraction 0.25"
echo "to the evidence run and read selectionComparison from its blocks.json."
