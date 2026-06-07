#!/usr/bin/env bash
# T3/T5: Zero-shot test — set exactly ONE dataset in pretrain.conf dataset_use
# Usage: ./test_zeroshot.sh [checkpoint_name]
# Default checkpoint: OpenCity-plus.pth (official) or OpenCity-plus2.0.pth (self-trained)
cd "$(dirname "$0")/../model" || exit 1

CKPT="${1:-OpenCity-plus.pth}"

python Run.py \
  -mode test \
  -model OpenCity \
  -load_pretrain_path "$CKPT" \
  -batch_size 2 \
  --embed_dim 512 \
  --skip_dim 512 \
  --enc_depth 6
