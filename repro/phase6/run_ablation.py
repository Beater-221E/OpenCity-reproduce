#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "model"
CONF = ROOT / "repro" / "phase6" / "configs.yaml"
LOG_DIR = ROOT / "repro" / "results" / "phase6" / "logs"

sys.path.insert(0, str(ROOT / "repro" / "common"))
from parse_log import parse_actual_dataset, parse_test_metrics
from patch_pretrain_conf import restore, set_dataset_use
from run_logger import RunLogger, log_subprocess_run


def load_cfg():
    with open(CONF) as f:
        return yaml.safe_load(f)


def csv_path(gpu_id: int | None) -> Path:
    suf = f".gpu{gpu_id}" if gpu_id is not None else ""
    return ROOT / "repro" / "results" / "phase6" / f"ablation{suf}.csv"


def run_one(ds: str, ablation: str, gpu_id: int, log: RunLogger) -> dict:
    set_dataset_use([ds], gpu_id=gpu_id)
    detail = LOG_DIR / f"abl_{ablation}_{ds}_g{gpu_id}.log"
    env = os.environ.copy()
    env["OPENCITY_ABLATION"] = ablation
    env["OPENCITY_DATASET_USE"] = ds
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    wrapper = ROOT / "repro" / "ablation" / "run_test_wrapped.py"
    cmd = [
        sys.executable, str(wrapper),
        "-mode", "test", "-model", "OpenCity",
        "-load_pretrain_path", "OpenCity-plus.pth",
        "--batch_size", "2", "--embed_dim", "512", "--skip_dim", "512", "--enc_depth", "6",
    ]
    job = f"ablation_{ablation}_{ds}"
    rc, _ = log_subprocess_run(log, cmd, MODEL_DIR, env, detail, job)
    actual = parse_actual_dataset(detail)
    m = parse_test_metrics(detail) if actual == ds else None
    if actual != ds:
        log.error(f"dataset mismatch expected={ds} actual={actual} log={detail}")
    return {
        "dataset": ds, "ablation": ablation,
        "mae": m["mae"] if m else "", "rmse": m["rmse"] if m else "", "mape": m["mape"] if m else "",
        "status": "ok" if rc == 0 and m else "failed", "delta_mae_pct": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-id", type=int, required=True, choices=[0, 1])
    args = ap.parse_args()
    log = RunLogger("ablation_worker", gpu_id=args.gpu_id)
    cfg = load_cfg()
    key = f"gpu{args.gpu_id}_datasets"
    datasets = cfg["ablation"].get(key, cfg["ablation"]["datasets"])
    out = csv_path(args.gpu_id)
    fields = ["dataset", "ablation", "mae", "rmse", "mape", "status", "delta_mae_pct"]
    done = set()
    if out.exists():
        with open(out) as f:
            for row in csv.DictReader(f):
                done.add((row["dataset"], row["ablation"]))
    rows = []
    full_mae = {}
    log.section(f"ablation worker gpu={args.gpu_id}")
    try:
        for ds in datasets:
            if not (ROOT / "data" / ds / f"{ds}.npz").exists():
                log.warning(f"skip missing data: {ds}")
                continue
            for abl in cfg["ablation"]["variants"]:
                if (ds, abl) in done:
                    log.info(f"[skip] {abl} @ {ds}")
                    continue
                row = run_one(ds, abl, args.gpu_id, log)
                if abl == "full" and row["mae"]:
                    full_mae[ds] = float(row["mae"])
                if full_mae.get(ds) and row["mae"]:
                    try:
                        row["delta_mae_pct"] = round(
                            (float(row["mae"]) - full_mae[ds]) / full_mae[ds] * 100, 2
                        )
                    except ValueError:
                        pass
                rows.append(row)
                log.info(f"result {row}")
    finally:
        restore()
    write_header = not out.exists() or out.stat().st_size == 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)
    log.event("ablation_worker_done", rows=len(rows))


if __name__ == "__main__":
    main()
