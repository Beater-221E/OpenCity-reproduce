#!/usr/bin/env bash
# Unzip HuggingFace downloads and generate CAD subsets.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"
cd "$DATA"

echo "=== Unzipping datasets ==="
for z in *.zip; do
  [ -f "$z" ] || continue
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
  out="${z%.zip}"
  [ -f "$out" ] && continue
  echo "unzip $z -> $out"
  unzip -o -q "$z"
done

echo "=== Generating CAD subsets ==="
python3 generate_ca_data.py

echo "=== Data ready ==="
ls -d "$DATA"/*/ 2>/dev/null | wc -l
