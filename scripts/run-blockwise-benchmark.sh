#!/usr/bin/env bash
# Issue #105: does moving one region at a time find more scorer wins than
# moving the whole creature at once?
#
# Runs the same creature, corpus, seed and step scale twice through the
# `blocks` subcommand — once with `--strategies global` (the whole-creature
# apply this crate already shipped) and once with the blockwise strategies —
# and prints candidates, scorer wins, elapsed seconds and wins/hour for each.
# Wins/hour is the measure the issue asks for: a mode that produces ten cheap
# candidates and one win beats a mode that produces one expensive candidate and
# none.
#
# Point it at the production targets to answer the same question there:
#   CREATURE=~/src/GRQ-cluster/network.json \
#   DATA_DIR=~/src/GRQ/.trainData-binary_116 \
#   scripts/run-blockwise-benchmark.sh path/to/rust_scorer
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCORER="${1:-${SCORER:-$ROOT/../NEAT-AI-scorer/target/release/rust_scorer}}"
OUT="${OUT:-$ROOT/.backprop/blockwise-benchmark}"
SEED="${SEED:-1}"
STEP_SCALE="${STEP_SCALE:-0.01}"
LEARNING_RATE="${LEARNING_RATE:-0.1}"
BLOCKS_PER_STRATEGY="${BLOCKS_PER_STRATEGY:-4}"
RADIUS="${RADIUS:-1}"
SUBGRAPH_SIZE="${SUBGRAPH_SIZE:-4}"
TOP_GENES="${TOP_GENES:-4}"
BLOCK_STRATEGIES="${BLOCK_STRATEGIES:-neuron,neighbourhood,output-head,subgraph,top-genes}"

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
rm -rf "$OUT/global" "$OUT/blockwise"
mkdir -p "$OUT"

# Gate on a record file, never on the directory: a directory left behind by an
# interrupted generator would otherwise look like a corpus and both modes would
# "compare" on no records at all.
if [[ ! -f "$DATA_DIR/0.bin" ]]; then
  mkdir -p "$DATA_DIR"
  # 1000 records of `y = x + 0.1`, except the 5 leading records of each file,
  # which follow `y = -2x + 1` — the same majority/minority corpus issue #104
  # generated, so the two experiments are comparable.
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
  # Two dense hidden layers feeding one output: `o = x` exactly, so the only
  # gain available is the missing `+0.1` on the output bias. Every hidden gene
  # is already where it should be, which is the shape of an evolved creature —
  # moving all 19 genes at once is what the blockwise mode has to beat.
  cat >"$CREATURE" <<'JSON'
{
  "semanticVersion": "4.0.0",
  "forwardOnly": true,
  "input": 1,
  "output": 1,
  "neurons": [
    { "type": "hidden", "uuid": "a0", "bias": 0.0, "squash": "IDENTITY" },
    { "type": "hidden", "uuid": "a1", "bias": 0.0, "squash": "IDENTITY" },
    { "type": "hidden", "uuid": "a2", "bias": 0.0, "squash": "IDENTITY" },
    { "type": "hidden", "uuid": "b0", "bias": 0.0, "squash": "IDENTITY" },
    { "type": "hidden", "uuid": "b1", "bias": 0.0, "squash": "IDENTITY" },
    { "type": "hidden", "uuid": "b2", "bias": 0.0, "squash": "IDENTITY" },
    { "type": "output", "uuid": "o1", "bias": 0.0, "squash": "IDENTITY" }
  ],
  "synapses": [
    { "fromUUID": "input-0", "toUUID": "a0", "weight": 1.0 },
    { "fromUUID": "input-0", "toUUID": "a1", "weight": 1.0 },
    { "fromUUID": "input-0", "toUUID": "a2", "weight": 1.0 },
    { "fromUUID": "a0", "toUUID": "b0", "weight": 0.1111111111111111 },
    { "fromUUID": "a0", "toUUID": "b1", "weight": 0.1111111111111111 },
    { "fromUUID": "a0", "toUUID": "b2", "weight": 0.1111111111111111 },
    { "fromUUID": "a1", "toUUID": "b0", "weight": 0.1111111111111111 },
    { "fromUUID": "a1", "toUUID": "b1", "weight": 0.1111111111111111 },
    { "fromUUID": "a1", "toUUID": "b2", "weight": 0.1111111111111111 },
    { "fromUUID": "a2", "toUUID": "b0", "weight": 0.1111111111111111 },
    { "fromUUID": "a2", "toUUID": "b1", "weight": 0.1111111111111111 },
    { "fromUUID": "a2", "toUUID": "b2", "weight": 0.1111111111111111 },
    { "fromUUID": "b0", "toUUID": "o1", "weight": 1.0 },
    { "fromUUID": "b1", "toUUID": "o1", "weight": 1.0 },
    { "fromUUID": "b2", "toUUID": "o1", "weight": 1.0 }
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

# Read one mode's blocks.json and print "candidates wins best_delta wins_per_hour".
# Arguments reach python through argv, never through interpolated source — an
# empty value must fail loudly here, not become a python syntax error.
summarise() {
  python3 - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
elapsed = max(int(sys.argv[2]), 1)
# Only candidates that reached disk were measured; an unmoved block writes no
# creature and must not inflate the count.
candidates = [c for c in summary["candidates"] if c.get("candidate")]
wins = [c for c in candidates if c.get("scoreWin")]
best = max((c.get("scoreDelta") or 0.0) for c in candidates) if candidates else 0.0
print(f"{len(candidates)} {len(wins)} {best:+.6e} {len(wins) * 3600 / elapsed:.1f}")
PY
}

run_mode() {
  local mode="$1"
  local strategies="$2"
  echo
  echo "=== $mode ($strategies) ==="
  local started ended elapsed
  started="$(date +%s)"
  "$BIN" blocks "$CREATURE" "$DATA_DIR" \
    --seed "$SEED" \
    --learning-rate "$LEARNING_RATE" \
    --step-scale "$STEP_SCALE" \
    --strategies "$strategies" \
    --blocks-per-strategy "$BLOCKS_PER_STRATEGY" \
    --radius "$RADIUS" \
    --subgraph-size "$SUBGRAPH_SIZE" \
    --top-genes "$TOP_GENES" \
    --skip-mse \
    --scorer "$SCORER" \
    --output-dir "$OUT/$mode"
  ended="$(date +%s)"
  elapsed=$((ended - started))
  # A sub-second mode still divides by a whole second, which understates its
  # rate — never overstates it.
  [[ "$elapsed" -lt 1 ]] && elapsed=1
  local stats
  stats="$(summarise "$OUT/$mode/blocks.json" "$elapsed")"
  local candidates wins best rate
  read -r candidates wins best rate <<<"$stats"
  printf '%-10s candidates=%-3s wins=%-3s best_score_delta=%s elapsed=%ss wins/hour=%s\n' \
    "$mode" "$candidates" "$wins" "$best" "$elapsed" "$rate"
}

run_mode global "global"
run_mode blockwise "$BLOCK_STRATEGIES"

echo
echo "Candidates and metadata: $OUT/global/blocks.json and $OUT/blockwise/blocks.json"
