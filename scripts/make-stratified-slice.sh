#!/usr/bin/env bash
# Copy the first N records of every production `.bin` into DEST as numbered
# files so TrainingDataIterator's numeric sort visits every letter/year.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: make-stratified-slice.sh --src-dir DIR --dest DIR --records N --width W [--offset K]

  --src-dir DIR   Directory of production .bin files
  --dest DIR      Destination directory (created). Writes 0.bin, 1.bin, ...
  --records N     Records to copy from each source file
  --width W       Floats per record (creature.input + creature.output)
  --offset K      Skip the first K records of each source (default 0)
EOF
}

SRC_DIR=""
DEST=""
RECORDS=""
WIDTH=""
OFFSET=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --src-dir) SRC_DIR="${2:?}"; shift 2 ;;
    --dest) DEST="${2:?}"; shift 2 ;;
    --records) RECORDS="${2:?}"; shift 2 ;;
    --width) WIDTH="${2:?}"; shift 2 ;;
    --offset) OFFSET="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$SRC_DIR" || -z "$DEST" || -z "$RECORDS" || -z "$WIDTH" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -d "$SRC_DIR" ]]; then
  echo "FAIL: source dir not found: $SRC_DIR" >&2
  exit 2
fi

bytes_per_record=$((WIDTH * 4))
skip_bytes=$((OFFSET * bytes_per_record))
copy_bytes=$((RECORDS * bytes_per_record))
mkdir -p "$DEST"

idx=0
copied=0
while IFS= read -r src; do
  size=$(wc -c <"$src" | tr -d ' ')
  available=$(( (size - skip_bytes) / bytes_per_record ))
  if [[ "$available" -lt "$RECORDS" ]]; then
    echo "WARN: skip $(basename "$src") (only $available records after offset $OFFSET)" >&2
    continue
  fi
  dest="$DEST/${idx}.bin"
  dd if="$src" of="$dest" bs="$bytes_per_record" skip="$OFFSET" count="$RECORDS" status=none
  actual=$(wc -c <"$dest" | tr -d ' ')
  if [[ "$actual" -ne "$copy_bytes" ]]; then
    echo "FAIL: expected $copy_bytes bytes from $src, wrote $actual" >&2
    exit 1
  fi
  idx=$((idx + 1))
  copied=$((copied + RECORDS))
done < <(find "$SRC_DIR" -maxdepth 1 -type f -name '*.bin' | LC_ALL=C sort)

if [[ "$idx" -eq 0 ]]; then
  echo "FAIL: no usable .bin files in $SRC_DIR" >&2
  exit 1
fi

echo "OK   wrote $idx files under $DEST ($copied records, offset=$OFFSET)"
