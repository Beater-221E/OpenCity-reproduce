#!/usr/bin/env bash
set -euo pipefail
cd /home/OpenCity
source repro/common/python_env.sh
"$PYTHON" repro/check_env.py
"$PYTHON" -c "
import sys; sys.path.insert(0,'repro/common')
from patch_pretrain_conf import set_dataset_use, restore
set_dataset_use(['PEMS07M'], gpu_id=0)
"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" repro/phase6/run_phase6.py --exp zeroshot --gpu-id 0 2>&1 | head -5 || true
# Only PEMS07M on gpu0 table1 list - run single test via Run.py
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -c "
import sys; sys.path.insert(0,'repro/common')
from patch_pretrain_conf import set_dataset_use
set_dataset_use(['PEMS07M'], 0)
"
cd model && CUDA_VISIBLE_DEVICES=0 "$PYTHON" Run.py -mode test -model OpenCity \
  -load_pretrain_path OpenCity-plus.pth --batch_size 2 \
  --embed_dim 512 --skip_dim 512 --enc_depth 6 2>&1 | tail -3
