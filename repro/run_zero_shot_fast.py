#!/usr/bin/env python3
"""Resumable zero-shot evaluation across OpenCity checkpoints."""
import csv
import subprocess
import sys
from pathlib import Path

from config_utils import parse_metrics, set_dataset
from paths import CONF, DATA_DIR, MODEL_DIR, REPO_ROOT, RESULTS_DIR

LOG_DIR = REPO_ROOT / "repro" / "logs" / "zero_shot"
RESULT = RESULTS_DIR / "zero_shot_results.csv"

ALL_DATASETS = [
    "PEMS04", "PEMS08", "PEMS07M", "METR_LA", "PEMS_BAY",
    "TrafficHZ", "TrafficZZ", "TrafficCD", "TrafficJN",
    "NYC_TAXI", "CD_DIDI", "SZ_DIDI",
    "CAD4-1", "CAD4-2", "CAD4-3", "CAD4-4",
    "CAD7-1", "CAD7-2", "CAD7-3", "CAD8-1", "CAD8-2", "CAD12-1", "CAD12-2",
]
SCALING_DATASETS = ["PEMS07M", "METR_LA", "PEMS04", "NYC_TAXI"]

VARIANTS = {
    "plus": ("OpenCity-plus.pth", 512, 512, 6, 2),
    "base": ("OpenCity-base.pth", 256, 256, 3, 8),
    "mini": ("OpenCity-mini.pth", 128, 128, 3, 16),
}


def load_done():
    done = set()
    if RESULT.exists():
        with open(RESULT, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["variant"], row["dataset"]))
    return done


def append_result(variant, dataset, metrics):
    write_header = not RESULT.exists() or RESULT.stat().st_size == 0
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["variant", "dataset", "mae", "rmse", "mape"])
        writer.writerow([variant, dataset, *metrics])


def dataset_exists(dataset: str) -> bool:
    dataset_dir = DATA_DIR / dataset
    return dataset_dir.is_dir() or (dataset_dir / f"{dataset}.npz").exists()


def run_one(variant, dataset, ckpt, embed, skip, enc, batch):
    if dataset.startswith("CAD") and variant == "plus":
        batch = 2
    log = LOG_DIR / f"{variant}_{dataset}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    set_dataset(CONF, dataset)
    cmd = [
        sys.executable, "Run.py",
        "-mode", "test", "-model", "OpenCity",
        "-load_pretrain_path", ckpt,
        "--batch_size", str(batch),
        "--embed_dim", str(embed),
        "--skip_dim", str(skip),
        "--enc_depth", str(enc),
    ]
    print(f"[RUN] {variant} {dataset} batch={batch}", flush=True)
    proc = subprocess.run(cmd, cwd=MODEL_DIR, capture_output=True, text=True)
    log.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        print(f"[FAIL] {variant} {dataset} -> {log}", flush=True)
        return False
    metrics = parse_metrics(log)
    if not metrics:
        print(f"[WARN] no metrics {variant} {dataset}", flush=True)
        return False
    append_result(variant, dataset, metrics)
    print(f"[OK] {variant} {dataset} MAE={metrics[0]}", flush=True)
    return True


def main():
    done = load_done()
    plus_done = sum(1 for variant, _ in done if variant == "plus")
    tasks = []
    for dataset in ALL_DATASETS:
        if not dataset_exists(dataset):
            continue
        if ("plus", dataset) not in done and plus_done < 15:
            tasks.append(("plus", dataset, *VARIANTS["plus"]))
    for dataset in SCALING_DATASETS:
        if not dataset_exists(dataset):
            continue
        for variant in ("base", "mini"):
            if (variant, dataset) not in done:
                tasks.append((variant, dataset, *VARIANTS[variant]))

    print(f"Pending tasks: {len(tasks)} (done={len(done)})", flush=True)
    for variant, dataset, ckpt, embed, skip, enc, batch in tasks:
        run_one(variant, dataset, ckpt, embed, skip, enc, batch)
    print(f"Zero-shot complete: {RESULT}", flush=True)


if __name__ == "__main__":
    main()
