#!/usr/bin/env bash
# Use existing instance Python/CUDA (do not install README torch 1.9).
if [[ -f /venv/main/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
fi
export PYTHON="${PYTHON:-python}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
