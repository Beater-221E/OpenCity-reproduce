# LoRA vs Fast Adaptation

Fast Adapt£º½öÎ¢µ÷ `predictor.linear`£¨3 epoch£©¡£LoRA£º±àÂëÆ÷ attn+FFN£¬rank 8/16/24¡£

| Variant | Dataset | ZS MAE | Fast MAE | LoRA r8 | LoRA r16 | LoRA r24 | Best LoRA |
|---------|---------|--------|----------|---------|----------|----------|----------|
| base | CD_DIDI | 5.7 | 2.59 | 2.60 | 2.58 | 2.57 | r24 (2.57) |
| base | SZ_DIDI | 4.47 | 2.30 | 2.30 | 2.28 | 2.27 | r24 (2.27) |
| mini | CD_DIDI | 5.31 | 2.57 | 2.63 | 2.59 | 2.58 | r24 (2.58) |
| mini | SZ_DIDI | 4.05 | 2.30 | 2.34 | 2.31 | 2.30 | r24 (2.30) |
| plus | CD_DIDI | 6.34 | 2.65 | 2.59 | 2.60 | 2.59 | r8 (2.59) |
| plus | SZ_DIDI | 4.66 | 2.37 | 2.30 | 2.29 | 2.29 | r16 (2.29) |
