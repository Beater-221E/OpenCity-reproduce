#!/usr/bin/env bash
# Unzip HuggingFace downloads and generate CAD subsets.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"
cd "$DATA"

echo "=== Unzipping datasets ==="
for z in *.zip; do
  [ -f "$z" ] || continue
  case "$z" in
    ca_his_raw_2020.h5.zip|ca_meta.zip|ca_rn_adj.npy.zip) continue ;;
  esac
  name="${z%.zip}"
  if [ -d "$name" ] && [ "$(ls -A "$name" 2>/dev/null)" ]; then
    echo "skip $name (exists)"
    continue
  fi
  echo "unzip $z"
  unzip -o -q "$z"
done

for z in ca_his_raw_2020.h5.zip ca_meta.zip ca_rn_adj.npy.zip; do
  [ -f "$z" ] || continue
  case "$z" in
    ca_meta.zip) out="ca_meta.csv" ;;
    *) out="${z%.zip}" ;;
  esac
  [ -f "$out" ] && continue
  echo "unzip $z -> $out"
  unzip -o -q "$z"
done

echo "=== Generating CAD subsets ==="
if [ "${OPENCITY_SKIP_CAD:-0}" = "1" ]; then
  echo "skip CAD generation (OPENCITY_SKIP_CAD=1)"
  echo "=== Data ready (without CAD subsets) ==="
  ls -d "$DATA"/*/ 2>/dev/null | wc -l
  exit 0
fi
python3 generate_ca_data.py

echo "=== Data ready ==="
ls -d "$DATA"/*/ 2>/dev/null | wc -l
