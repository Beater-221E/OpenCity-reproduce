#!/usr/bin/env bash
# Run zero-shot eval, baselines, and summarize results (resumable).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Phase 1: zero-shot (resumable) ==="
python3 repro/run_zero_shot_fast.py

echo "=== Phase 2: baselines ==="
python3 repro/run_baselines_fast.py

echo "=== Phase 3: summarize ==="
python3 repro/summarize_results.py

echo "=== ALL DONE ==="
echo "See repro/results/*.csv and comparison_summary.txt"
