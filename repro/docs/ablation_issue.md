# 消融实验说明（推理级）

论文 Figure 3 的消融变体通常使用**单独训练的消融模型权重**。本仓库仅提供完整 OpenCity-plus 官方权重。

## 方案

在 `repro/ablation/inject.py` 中于推理时关闭对应模块分支，加载同一 `OpenCity-plus.pth` 做 zero-shot 评测。

## 局限

- 权重仍来自完整模型，**绝对 MAE 不可与论文数值直接对比**
- `minus_DTP` 在完整权重下可能 MAE 爆炸，仅作示意

## 变体

| 变体 | 实现 |
|------|------|
| full | 无修改 |
| minus_DTP | 跳过 dynamic `t_attn` |
| minus_PTTM | 跳过 periodic `tc` |
| minus_SDM | GCN 恒等 bypass |
| minus_STC | 时空 context embedding 置零 |
