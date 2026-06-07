#!/usr/bin/env bash
# T4: OpenCity-Plus pretrain (paper-aligned)
# Prerequisite: full dataset_use in ../conf/general_conf/pretrain.conf
cd "$(dirname "$0")/../model" || exit 1

python Run.py \
  -mode pretrain \
  -model OpenCity \
  -save_pretrain_path OpenCity-plus2.0.pth \
  -batch_size 4 \
  --embed_dim 512 \
  --skip_dim 512 \
  --enc_depth 6
