#!/usr/bin/env python3
"""Resumable full-shot baseline training on 5-minute-interval datasets."""
import csv
import subprocess
import sys

from config_utils import parse_metrics, set_dataset
from paths import CONF, MODEL_DIR, REPO_ROOT, RESULTS_DIR

LOG_DIR = REPO_ROOT / "repro" / "logs" / "baselines"
RESULT = RESULTS_DIR / "baseline_results.csv"

MODELS = ["STGCN", "GWN", "AGCRN", "PDFormer", "MTGNN"]
DATASETS = ["PEMS07M", "METR_LA", "PEMS_BAY"]


def load_done():
    done = set()
    if RESULT.exists():
        with open(RESULT, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["model"], row["dataset"]))
    return done


def append_result(model, dataset, metrics):
    write_header = not RESULT.exists() or RESULT.stat().st_size == 0
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model", "dataset", "mae", "rmse", "mape"])
        writer.writerow([model, dataset, *metrics])


def gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            print(
                f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}",
                flush=True,
            )
        else:
            print("WARNING: CUDA not available, using CPU", flush=True)
    except Exception as exc:
        print(f"torch check failed: {exc}", flush=True)


def run_one(model, dataset):
    log = LOG_DIR / f"{model}_{dataset}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    set_dataset(CONF, dataset)
    cmd = [
        sys.executable, "Run.py",
        "-mode", "ori", "-model", model,
        "--batch_size", "64", "--real_value", "False",
    ]
    print(f"[RUN] baseline {model} {dataset} -> {log}", flush=True)
    with open(log, "w", encoding="utf-8") as lf:
        proc = subprocess.run(
            cmd, cwd=MODEL_DIR, stdout=lf, stderr=subprocess.STDOUT, text=True
        )
    if proc.returncode != 0:
        print(f"[FAIL] {model} {dataset}", flush=True)
        return False
    metrics = parse_metrics(log)
    if not metrics:
        print(f"[WARN] no metrics {model} {dataset}", flush=True)
        return False
    append_result(model, dataset, metrics)
    print(f"[OK] {model} {dataset} MAE={metrics[0]}", flush=True)
    return True


def main():
    gpu_info()
    done = load_done()
    tasks = [(m, d) for d in DATASETS for m in MODELS if (m, d) not in done]
    print(f"Pending baselines: {len(tasks)} (done={len(done)})", flush=True)
    for model, dataset in tasks:
        run_one(model, dataset)
    print(f"Baselines complete: {RESULT}", flush=True)


if __name__ == "__main__":
    main()
