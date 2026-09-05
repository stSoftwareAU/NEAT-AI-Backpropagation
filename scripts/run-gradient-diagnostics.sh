#!/usr/bin/env bash
# Gradient diagnostics on a production creature (issue #107).
#
# Runs `gradient-check` over a real evolved creature and a real record slice,
# then prints the per-facet evidence — sign agreement, applied-proposal
# improvement and relative gradient error by gene class, squash, aggregate vs
# ordinary, depth, fan-in / fan-out, activation health and proposal magnitude.
#
# Every target is an env var — no stock-market logic belongs in this public
# library. The defaults below are the same locked integration paths
# `scripts/run-production-win.sh` already carries; override any of them.
#
#   CREATURE=... DATA_DIR=... OUT=... ./scripts/run-gradient-diagnostics.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CREATURE="${CREATURE:-$HOME/src/GRQ-cluster/network.json}"
DATA_DIR="${DATA_DIR:-$HOME/src/GRQ/.trainData-binary_116}"
MAX_RECORDS="${MAX_RECORDS:-512}"
SAMPLE_BIASES="${SAMPLE_BIASES:-60}"
SAMPLE_WEIGHTS="${SAMPLE_WEIGHTS:-120}"
FACET_MIN_SCORED="${FACET_MIN_SCORED:-5}"
RANK_LIMIT="${RANK_LIMIT:-5}"
STEP_SCALE="${STEP_SCALE:-0.01}"
LEARNING_RATE="${LEARNING_RATE:-0.01}"
FD_EPS="${FD_EPS:-1e-4}"
SEED="${SEED:-1}"
OUT="${OUT:-$ROOT/.backprop/gradient-diagnostics}"

if [[ ! -f "$CREATURE" ]]; then
  echo "FAIL: creature not found: $CREATURE (set CREATURE=...)" >&2
  exit 2
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "FAIL: training data directory not found: $DATA_DIR (set DATA_DIR=...)" >&2
  exit 2
fi

mkdir -p "$OUT"

echo "Building neat_ai_backpropagation (release)..."
(
  cd "$ROOT"
  cargo build -p neat_ai_backpropagation --release
)
BIN="$ROOT/target/release/neat_ai_backpropagation"

# Cost is (2 + 3 x sampled genes) MSE passes over MAX_RECORDS — the sample
# caps below are what bounds the run, so raise them deliberately.
echo "Probing up to $((SAMPLE_BIASES + SAMPLE_WEIGHTS)) genes over $MAX_RECORDS records..."
"$BIN" gradient-check "$CREATURE" "$DATA_DIR" \
  --max-records "$MAX_RECORDS" \
  --seed "$SEED" \
  --learning-rate "$LEARNING_RATE" \
  --step-scale "$STEP_SCALE" \
  --fd-eps "$FD_EPS" \
  --sample-biases "$SAMPLE_BIASES" \
  --sample-weights "$SAMPLE_WEIGHTS" \
  --facet-min-scored "$FACET_MIN_SCORED" \
  --rank-limit "$RANK_LIMIT" \
  --output-dir "$OUT"

echo
echo "Artefacts:"
echo "  $OUT/gradient-check.json  (machine-readable: byFacet, bestClasses, worstClasses)"
echo "  $OUT/genes.jsonl          (one row per probed gene, with its facets)"
echo "  $OUT/summary.txt          (the concise report printed above)"
