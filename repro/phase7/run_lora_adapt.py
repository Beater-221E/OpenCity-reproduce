#!/usr/bin/env python3
"""Phase7: LoRA fine-tuning vs Fast Adaptation baseline."""
import argparse
import csv
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "model"
CONF_PATH = Path(__file__).resolve().parent / "configs.yaml"
LOG_DIR = ROOT / "repro" / "results" / "phase7" / "logs"
RESULT_DIR = ROOT / "repro" / "results" / "phase7"

sys.path.insert(0, str(ROOT / "repro" / "common"))
from parse_log import parse_actual_dataset, parse_test_metrics
from run_logger import RunLogger, log_subprocess_run

_RUN_LOGGER: RunLogger | None = None


def logger() -> RunLogger:
    global _RUN_LOGGER
    if _RUN_LOGGER is None:
        _RUN_LOGGER = RunLogger("phase7_worker", gpu_id=None)
    return _RUN_LOGGER


def set_logger(gpu_id: int | None):
    global _RUN_LOGGER
    _RUN_LOGGER = RunLogger("phase7_worker", gpu_id=gpu_id)


def load_cfg():
    with open(CONF_PATH) as f:
        return yaml.safe_load(f)


def dataset_exists(ds: str) -> bool:
    return (ROOT / "data" / ds / f"{ds}.npz").exists()


def csv_suffix(gpu_id: int | None) -> str:
    return f".gpu{gpu_id}" if gpu_id is not None else ""


def shard_jobs(jobs: list, gpu_id: int | None) -> list:
    if gpu_id is None:
        return jobs
    return [j for i, j in enumerate(jobs) if i % 2 == gpu_id]


def get_gpu_mb(cuda_visible: str | None) -> int:
    env = os.environ.copy()
    if cuda_visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, env=env,
        )
        if r.returncode == 0 and r.stdout.strip():
            vals = [int(float(x)) for x in r.stdout.strip().split("\n") if x.strip()]
            return max(vals) if vals else 0
    except Exception:
        pass
    return 0


def effective_batch(ds: str, variant_cfg: dict) -> int:
    bs = variant_cfg["batch_size"]
    if ds.startswith("CAD") or ds in ("NYC_TAXI", "CHI_TAXI") or ds.startswith("Traffic") or ds.startswith("NYC_BIKE"):
        return min(2, bs)
    return bs


def load_zeroshot_mae(cfg: dict, variant: str, dataset: str) -> float | None:
    fa_path = ROOT / cfg.get("fast_adapt_csv", "repro/results/phase6/fast_adapt_full.csv")
    if not fa_path.exists():
        return None
    with open(fa_path) as f:
        for row in csv.DictReader(f):
            if row.get("variant") == variant and row.get("dataset") == dataset and row.get("status") == "ok":
                try:
                    return float(row["zeroshot_mae"])
                except (TypeError, ValueError):
                    pass
    return None


def parse_trainable_params(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    import re
    m = re.search(r"trainable_params=(\d+)", log_path.read_text(errors="replace"))
    return int(m.group(1)) if m else None


def variant_from_name(cfg: dict, name: str) -> dict:
    v = dict(cfg["variants"][name])
    v["name"] = name
    return v


def load_done(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path) as f:
            for row in csv.DictReader(f):
                done.add((row["variant"], row["dataset"], str(row["lora_rank"])))
    return done


def append_row(path: Path, fields: list[str], row: dict):
    write_header = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)


def run_lora_job(
    ds: str,
    variant_cfg: dict,
    rank: int,
    epochs: int,
    gpu_id: int | None,
) -> tuple[int, Path, float, int]:
    tag = f"lora_{variant_cfg['name']}_r{rank}_{ds}"
    log = LOG_DIR / f"{tag}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    bs = effective_batch(ds, variant_cfg)
    cmd = [
        sys.executable,
        "Run.py",
        "-mode", "lora_eval",
        "-model", "OpenCity",
        "-load_pretrain_path", variant_cfg["ckpt"],
        "--batch_size", str(bs),
        "--embed_dim", str(variant_cfg["embed_dim"]),
        "--skip_dim", str(variant_cfg["skip_dim"]),
        "--enc_depth", str(variant_cfg["enc_depth"]),
        "--lora_rank", str(rank),
        "-epochs", str(epochs),
        "--epochs", str(epochs),
    ]
    env = os.environ.copy()
    env["OPENCITY_DATASET_USE"] = ds
    env["OPENCITY_TQDM"] = "1"
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    job_name = f"{tag}_gpu{gpu_id}"
    logger().info(f"LoRA start: {job_name} dataset={ds} rank={rank}")
    gpu_before = get_gpu_mb(str(gpu_id) if gpu_id is not None else None)
    rc, wall = log_subprocess_run(logger(), cmd, MODEL_DIR, env, log, job_name)
    gpu_after = get_gpu_mb(str(gpu_id) if gpu_id is not None else None)
    return rc, log, wall, max(gpu_before, gpu_after)


def build_jobs(cfg: dict) -> list[tuple[str, str, int]]:
    la = cfg["lora_adapt"]
    jobs = []
    for v in la["variants"]:
        for r in la["lora_ranks"]:
            for ds in la["datasets"]:
                jobs.append((v, ds, int(r)))
    return jobs


def run_all(cfg: dict, gpu_id: int | None):
    suf = csv_suffix(gpu_id)
    out = RESULT_DIR / f"lora_adapt_full{suf}.csv"
    fields = [
        "variant", "dataset", "lora_rank", "zeroshot_mae", "mae", "rmse", "mape",
        "improve_pct", "trainable_params", "lora_wall_s", "gpu_mb", "status",
    ]
    done = load_done(out)
    la = cfg["lora_adapt"]
    epochs = la["finetune_epochs"]
    jobs = shard_jobs(build_jobs(cfg), gpu_id)

    for vname, ds, rank in jobs:
        key = (vname, ds, str(rank))
        if key in done:
            logger().info(f"[skip] lora {vname} r{rank} {ds}")
            continue
        if not dataset_exists(ds):
            append_row(out, fields, {
                "variant": vname, "dataset": ds, "lora_rank": rank,
                "zeroshot_mae": "", "mae": "", "rmse": "", "mape": "",
                "improve_pct": "", "trainable_params": "",
                "lora_wall_s": "", "gpu_mb": "", "status": "no_data",
            })
            continue

        zs = load_zeroshot_mae(cfg, vname, ds)
        vc = variant_from_name(cfg, vname)
        rc, log, wall, gpu = run_lora_job(ds, vc, rank, epochs, gpu_id)
        actual = parse_actual_dataset(log)
        m = parse_test_metrics(log) if actual == ds else None
        if actual != ds:
            logger().error(f"dataset mismatch expected={ds} actual={actual} log={log}")
        tp = parse_trainable_params(log)

        if rc == 0 and m:
            zs_f = float(zs) if zs is not None else None
            ft_mae = float(m["mae"])
            imp = round((zs_f - ft_mae) / zs_f * 100, 2) if zs_f else ""
            append_row(out, fields, {
                "variant": vname, "dataset": ds, "lora_rank": rank,
                "zeroshot_mae": zs if zs is not None else "",
                **m, "improve_pct": imp,
                "trainable_params": tp if tp is not None else "",
                "lora_wall_s": round(wall, 1), "gpu_mb": gpu, "status": "ok",
            })
            logger().info(f"[ok] lora gpu{gpu_id} {vname} r{rank} {ds} MAE={m['mae']}")
        else:
            append_row(out, fields, {
                "variant": vname, "dataset": ds, "lora_rank": rank,
                "zeroshot_mae": zs if zs is not None else "",
                "mae": "", "rmse": "", "mape": "", "improve_pct": "",
                "trainable_params": tp if tp is not None else "",
                "lora_wall_s": round(wall, 1), "gpu_mb": gpu, "status": "failed",
            })
            logger().error(f"[fail] lora gpu{gpu_id} {vname} r{rank} {ds} rc={rc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-id", type=int, default=None, help="0/1 for dual-GPU shard; omit for all jobs")
    args = ap.parse_args()
    set_logger(args.gpu_id)
    cfg = load_cfg()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    logger().section(f"Phase7 LoRA start gpu_id={args.gpu_id}")
    logger().event("worker_start")
    run_all(cfg, args.gpu_id)
    logger().event("worker_done")
    logger().info(f"Phase7 LoRA finished gpu_id={args.gpu_id}")


if __name__ == "__main__":
    main()
