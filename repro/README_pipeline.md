# OpenCity 复现产物（精简版）

## 数据与权重（不在仓库内）

重跑 Fast Adapt / LoRA 时执行：

```bash
python repro/phase7/download_assets.py
```

会下载 `data/CD_DIDI`、`data/SZ_DIDI` 与 `model_weights/OpenCity/*.pth`。

## 结果（无需原始 npz）

| 文件 | 内容 |
|------|------|
| `results/tables/paper_table1_opencity.md` | Table1 零样本 |
| `results/tables/paper_table3_opencity.md` | Fast Adapt |
| `results/tables/lora_vs_fast_adapt.md` | LoRA vs Fast Adapt |
| `results/phase6/scaling_matrix.csv` | Scaling 18 格 |
| `results/phase6/fast_adapt_full.csv` | Fast Adapt 6 格 |
| `results/phase7/lora_adapt_full.csv` | LoRA 18 格 |
| `results/figures/scaling_heatmap.png` | 尺度热力图 |

详见 `repro/data_manifest.json`。

## 重新跑 LoRA

```bash
source /venv/main/bin/activate
python repro/phase7/download_assets.py   # 若缺数据/权重
bash repro/phase7/run_queue_dual.sh
```

## 说明

已删除：`phase6/logs`、`phase7/logs`（完整指标在 CSV）、GPU 分片 CSV、重复 `OpenCity.pth`。
