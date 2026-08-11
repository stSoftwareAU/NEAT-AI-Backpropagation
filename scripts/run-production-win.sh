#!/usr/bin/env bash
# Production proof: parity dump + trainDir-style apply on GRQ creature/data.
#
# Defaults match the locked production targets. Override with env vars.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CREATURE="${CREATURE:-$HOME/src/GRQ-cluster/network.json}"
DATA_DIR="${DATA_DIR:-$HOME/src/GRQ/.trainData-binary_116}"
SRC_BIN="${SRC_BIN:-$DATA_DIR/A-2007.bin}"
SLICE_RECORDS="${SLICE_RECORDS:-256}"
TRAIN_RECORDS="${TRAIN_RECORDS:-2048}"
WIDTH="${WIDTH:-2512}"
SEED="${SEED:-1}"
SCORER="${SCORER:-$HOME/src/NEAT-AI-scorer/target/release/rust_scorer}"
OUT="${OUT:-$ROOT/.backprop/production}"
NEAT_AI="${NEAT_AI:-$ROOT/../NEAT-AI}"

if [[ ! -f "$CREATURE" ]]; then
  echo "FAIL: creature not found: $CREATURE" >&2
  exit 2
fi
if [[ ! -f "$SRC_BIN" ]]; then
  echo "FAIL: source bin not found: $SRC_BIN" >&2
  exit 2
fi

mkdir -p "$OUT"
SLICE="$OUT/slice"
TRAIN_SLICE="$OUT/train-slice"
"$ROOT/scripts/extract-bin-slice.sh" --src "$SRC_BIN" --dest "$SLICE" --records "$SLICE_RECORDS" --width "$WIDTH"
"$ROOT/scripts/extract-bin-slice.sh" --src "$SRC_BIN" --dest "$TRAIN_SLICE" --records "$TRAIN_RECORDS" --width "$WIDTH"

echo "Building neat_ai_backpropagation (release)..."
(
  cd "$ROOT"
  cargo build -p neat_ai_backpropagation --release
)
BIN="$ROOT/target/release/neat_ai_backpropagation"

echo "Rust compare on $SLICE_RECORDS production records..."
"$BIN" compare "$CREATURE" "$SLICE" --max-records "$SLICE_RECORDS" --seed "$SEED" --out "$OUT/rust-compare.json"

if [[ -d "$NEAT_AI" ]]; then
  echo "TypeScript compare on the same slice..."
  (
    cd "$NEAT_AI"
    deno run -A --config deno.json \
      "$ROOT/scripts/ts-compare.ts" \
      "$CREATURE" "$SLICE" \
      --max-records "$SLICE_RECORDS" \
      --out "$OUT/ts-compare.json"
  )
  echo "Diffing dumps..."
  "$BIN" diff "$OUT/rust-compare.json" "$OUT/ts-compare.json"
else
  echo "WARN: sibling NEAT-AI not found at $NEAT_AI — skipping TS dual-run"
fi

echo "Train on $TRAIN_RECORDS production records..."
SCORER_ARGS=()
if [[ -x "$SCORER" ]]; then
  SCORER_ARGS=(--scorer "$SCORER")
else
  echo "WARN: rust_scorer not executable at $SCORER — train without scorer"
fi

"$BIN" train "$CREATURE" "$TRAIN_SLICE" \
  --epochs 1 \
  --max-records "$TRAIN_RECORDS" \
  --seed "$SEED" \
  --learning-rate 0.01 \
  --step-scale 0.01 \
  --output-dir "$OUT/train" \
  "${SCORER_ARGS[@]}"

echo "Production run artefacts under $OUT"
