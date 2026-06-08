#!/usr/bin/env bash
# Dual-GPU queue with full logging.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source repro/common/python_env.sh

MAIN_LOG="repro/logs/phase6_dual.log"
mkdir -p repro/logs repro/results/phase6/logs

log() { echo "[$(date -Iseconds)] $*" | tee -a "$MAIN_LOG"; }

log "========== phase6 dual-GPU queue START =========="
log "python=$PYTHON main_log=$MAIN_LOG"

"$PYTHON" repro/check_env.py 2>&1 | tee -a "$MAIN_LOG"

log "---------- data download ----------"
bash repro/phase6/download_missing_hf.sh 2>&1 | tee -a repro/logs/download.log | tee -a "$MAIN_LOG"

log "---------- GPU0 worker ----------"
(
  export PYTHONUNBUFFERED=1
  "$PYTHON" repro/phase6/run_phase6.py --exp all --gpu-id 0
) >> repro/logs/phase6_worker_gpu0.log 2>&1 &
PID0=$!
log "GPU0 pid=$PID0 log=repro/logs/phase6_worker_gpu0.log"

log "---------- GPU1 worker ----------"
(
  export PYTHONUNBUFFERED=1
  "$PYTHON" repro/phase6/run_phase6.py --exp all --gpu-id 1
) >> repro/logs/phase6_worker_gpu1.log 2>&1 &
PID1=$!
log "GPU1 pid=$PID1 log=repro/logs/phase6_worker_gpu1.log"

wait "$PID0"; EC0=$?; log "GPU0 phase6 exit=$EC0"
wait "$PID1"; EC1=$?; log "GPU1 phase6 exit=$EC1"

log "---------- ablation workers ----------"
(
  export PYTHONUNBUFFERED=1
  "$PYTHON" repro/phase6/run_ablation.py --gpu-id 0
) >> repro/logs/ablation_worker_gpu0.log 2>&1 &
(
  export PYTHONUNBUFFERED=1
  "$PYTHON" repro/phase6/run_ablation.py --gpu-id 1
) >> repro/logs/ablation_worker_gpu1.log 2>&1 &
wait
log "ablation workers done"

log "---------- merge & report ----------"
"$PYTHON" repro/phase6/merge_results.py 2>&1 | tee -a "$MAIN_LOG"
"$PYTHON" repro/phase5/generate_report.py 2>&1 | tee -a "$MAIN_LOG"
"$PYTHON" repro/show_progress.py 2>&1 | tee -a "$MAIN_LOG"

log "========== phase6 dual-GPU queue DONE =========="
