# OpenCity

Reproduction and extension of [OpenCity: Open Spatio-Temporal Foundation Models for Traffic Prediction](https://arxiv.org/abs/2408.10269).

OpenCity pre-trains on large-scale heterogeneous traffic data and supports zero-shot prediction on unseen cities. This repo adds a reproducible experiment pipeline (T0--T8) and a LoRA fine-tuning extension.

**Paper:** [arXiv:2408.10269](https://arxiv.org/abs/2408.10269) · **Upstream:** [HKUDS/OpenCity](https://github.com/HKUDS/OpenCity)

---

## Requirements

- Python 3.9+
- PyTorch with CUDA (match your GPU driver)
- Linux or WSL recommended for shell scripts

```bash
pip install torch torchvision torchaudio   # pick the CUDA build for your system
pip install -r requirements.txt
```

For newer GPUs (e.g. Blackwell sm_120), a PyTorch nightly build may be required. See `repro/env_info.txt` for a tested environment.

---

## Setup

### 1. Datasets

Download from [HuggingFace OpenCity-dataset](https://huggingface.co/datasets/hkuds/OpenCity-dataset) into `data/`:

```
data/{NAME}/{NAME}.npz
```

Unzip archives and generate CAD subsets:

```bash
bash repro/setup_data.sh
```

### 2. Checkpoints

Place official weights in `model_weights/OpenCity/`:

| File | Variant |
|------|---------|
| `OpenCity-plus.pth` | Plus (~26M) |
| `OpenCity-base.pth` | Base (~5M) |
| `OpenCity-mini.pth` | Mini (~2M) |

Download: [HuggingFace OpenCity-Plus](https://huggingface.co/hkuds/OpenCity-Plus) (Base/Mini included).

---

## Run

All `python Run.py` commands **must run from `model/`** (paths are relative to that directory).

### Smoke test (T3)

Set one dataset in `conf/general_conf/pretrain.conf`, e.g. `dataset_use = ['PEMS07M']`, then:

```bash
bash repro/test_zeroshot.sh
# Expected PEMS07M MAE ~ 4.50
```

### Full reproduction pipeline

```bash
bash repro/run_all.sh
```

Runs resumable zero-shot eval, baseline training, and writes CSV results to `repro/results/`.

### Manual commands

```bash
cd model

# Zero-shot test (T5)
python Run.py -mode test -model OpenCity \
  -load_pretrain_path OpenCity-plus.pth \
  -batch_size 2 --embed_dim 512 --skip_dim 512 --enc_depth 6

# Plus pretrain (T4) — enable all 20 datasets in pretrain.conf first
python Run.py -mode pretrain -model OpenCity \
  -save_pretrain_path OpenCity-plus2.0.pth \
  -batch_size 4 --embed_dim 512 --skip_dim 512 --enc_depth 6

# Baseline full-shot (T6)
python Run.py -mode ori -model STGCN -batch_size 64 --real_value False

# Linear fine-tuning (T8) — freeze backbone, train predictor.linear only
python Run.py -mode eval -model OpenCity \
  -load_pretrain_path OpenCity-plus.pth \
  -batch_size 2 -epochs 3 \
  --embed_dim 512 --skip_dim 512 --enc_depth 6
```

---

## Experiment Map

| Task | Mode | Description |
|------|------|-------------|
| T0 | — | Install dependencies |
| T1 | — | Download datasets |
| T2 | — | Generate CAD subsets |
| T3 | `test` | Smoke test with official weights |
| T4 | `pretrain` | OpenCity-Plus mixed pretraining |
| T5 | `test` | Zero-shot eval (Table 1, 6 datasets) |
| T6 | `ori` | Baseline full-shot (Table 1) |
| T7 | `ori` | Supervised OpenCity (Table 2) |
| T8 | `eval` | Linear-only fine-tuning (Table 3) |
| LoRA | `lora` | Parameter-efficient fine-tuning extension |

**Plus model defaults:** `embed_dim=512`, `enc_depth=6`, `his=pred=288`, `batch_size=4`.

---

## Project Layout

```
OpenCity/
├── README.md              # this file
├── requirements.txt
├── conf/                  # model and training configs
├── data/                  # datasets (not in git)
├── lib/                   # data loading, metrics, params
├── model/
│   ├── Run.py             # main entry point
│   └── OpenCity/          # model implementation
├── model_weights/OpenCity/  # checkpoints (not in git)
└── repro/                 # scripts, results, logs
    ├── run_zero_shot_fast.py
    ├── run_baselines_fast.py
    ├── summarize_results.py
    ├── run_all.sh
    └── results/*.csv
```

Config priority: CLI flags override `conf/OpenCity/OpenCity.conf`; `-` flags override `conf/general_conf/pretrain.conf`. Non-pretrain modes also read `global_baselines.conf` for val/test split and epochs.

---

## Repro Scripts

| Script | Purpose |
|--------|---------|
| `repro/setup_data.sh` | Unzip data and build CAD subsets |
| `repro/test_zeroshot.sh` | Single-dataset zero-shot test |
| `repro/pretrain_plus.sh` | T4 Plus pretraining |
| `repro/run_zero_shot_fast.py` | Resumable multi-dataset zero-shot |
| `repro/run_baselines_fast.py` | Resumable baseline training |
| `repro/summarize_results.py` | Print summary from CSV results |
| `repro/run_all.sh` | Run full eval pipeline |

Results: `repro/results/zero_shot_results.csv`, `baseline_results.csv`.
