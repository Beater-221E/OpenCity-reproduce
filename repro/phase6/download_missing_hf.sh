#!/usr/bin/env bash
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source repro/common/python_env.sh
LOG="repro/logs/download.log"
mkdir -p repro/results/phase6 repro/logs

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

log "download_missing_hf start"
"$PYTHON" repro/phase6/download_hf.py 2>&1 | tee -a "$LOG"
"$PYTHON" repro/phase6/build_manifest.py 2>&1 | tee -a "$LOG"
log "download_missing_hf done"
