#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source repro/common/python_env.sh

MAIN_LOG="repro/logs/phase7_dual.log"
mkdir -p repro/logs repro/results/phase7/logs

log() { echo "[$(date -Iseconds)] $*" | tee -a "$MAIN_LOG"; }

log "========== phase7 LoRA queue START =========="
python repro/phase7/download_assets.py 2>&1 | tee -a repro/logs/phase7_download.log | tee -a "$MAIN_LOG"

PYTHON="${PYTHON:-python}"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
log "detected GPUs: $NGPU"

if [ "$NGPU" -lt 2 ]; then
  log "single-GPU mode: run all 18 LoRA jobs on GPU 0"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" repro/phase7/run_lora_adapt.py \
    >> repro/logs/phase7_worker_gpu0.log 2>&1
  EC0=$?
  log "LoRA worker exit=$EC0"
else
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" repro/phase7/run_lora_adapt.py --gpu-id 0 \
    >> repro/logs/phase7_worker_gpu0.log 2>&1 &
  PID0=$!
  log "GPU0 pid=$PID0"
  CUDA_VISIBLE_DEVICES=1 "$PYTHON" repro/phase7/run_lora_adapt.py --gpu-id 1 \
    >> repro/logs/phase7_worker_gpu1.log 2>&1 &
  PID1=$!
  log "GPU1 pid=$PID1"
  wait "$PID0"; EC0=$?; log "GPU0 exit=$EC0"
  wait "$PID1"; EC1=$?; log "GPU1 exit=$EC1"
fi

"$PYTHON" repro/phase7/merge_results.py 2>&1 | tee -a "$MAIN_LOG"
"$PYTHON" repro/phase7/compare_lora_fast.py 2>&1 | tee -a "$MAIN_LOG"

log "========== phase7 LoRA queue DONE =========="
