#!/usr/bin/env bash
# Copy the first N records of a production `.bin` into DEST/0.bin so Rust and
# TypeScript read identical bytes (neat-core numeric sort vs GRQ A-YYYY names).
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: extract-bin-slice.sh --src FILE --dest DIR --records N --width W

  --src FILE    Source .bin (e.g. GRQ A-2007.bin)
  --dest DIR    Destination directory (created). Writes 0.bin
  --records N   Number of records to copy
  --width W     Floats per record (creature.input + creature.output)
EOF
}

SRC=""
DEST=""
RECORDS=""
WIDTH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) SRC="${2:?}"; shift 2 ;;
    --dest) DEST="${2:?}"; shift 2 ;;
    --records) RECORDS="${2:?}"; shift 2 ;;
    --width) WIDTH="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$SRC" || -z "$DEST" || -z "$RECORDS" || -z "$WIDTH" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -f "$SRC" ]]; then
  echo "FAIL: source not found: $SRC" >&2
  exit 2
fi

bytes_per_record=$((WIDTH * 4))
bytes=$((RECORDS * bytes_per_record))
mkdir -p "$DEST"
# dd count is in blocks; use bs=bytes_per_record count=RECORDS
dd if="$SRC" of="$DEST/0.bin" bs="$bytes_per_record" count="$RECORDS" status=none
actual=$(wc -c <"$DEST/0.bin" | tr -d ' ')
if [[ "$actual" -ne "$bytes" ]]; then
  echo "FAIL: expected $bytes bytes, wrote $actual (file shorter than $RECORDS records?)" >&2
  exit 1
fi
echo "OK   wrote $DEST/0.bin ($RECORDS records, $bytes bytes)"
