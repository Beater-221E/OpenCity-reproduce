#!/usr/bin/env bash
set -euo pipefail
cd /home/OpenCity
source repro/common/python_env.sh
MAIN_LOG="repro/logs/phase6_single.log"
mkdir -p repro/logs
log() { echo "[$(date -Iseconds)] $*" | tee -a "$MAIN_LOG"; }
export PYTHONUNBUFFERED=1
log "single-GPU queue start"
"$PYTHON" repro/check_env.py 2>&1 | tee -a "$MAIN_LOG"
bash repro/phase6/download_missing_hf.sh 2>&1 | tee -a "$MAIN_LOG"
"$PYTHON" repro/phase6/run_phase6.py --exp all 2>&1 | tee -a repro/logs/phase6_worker.log
"$PYTHON" repro/phase6/run_ablation.py --gpu-id 0 2>&1 | tee -a repro/logs/ablation_worker_gpu0.log
"$PYTHON" repro/phase6/run_ablation.py --gpu-id 1 2>&1 | tee -a repro/logs/ablation_worker_gpu1.log
"$PYTHON" repro/phase6/merge_results.py 2>&1 | tee -a "$MAIN_LOG"
"$PYTHON" repro/phase5/generate_report.py 2>&1 | tee -a "$MAIN_LOG"
log "single-GPU queue done"
