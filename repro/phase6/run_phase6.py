#!/usr/bin/env python3
"""Phase6 runner with dual-GPU sharding via --gpu-id."""
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "model"
CONF_PATH = ROOT / "repro" / "phase6" / "configs.yaml"
LOG_DIR = ROOT / "repro" / "results" / "phase6" / "logs"
RESULT_DIR = ROOT / "repro" / "results" / "phase6"

sys.path.insert(0, str(ROOT / "repro" / "common"))
from parse_log import parse_actual_dataset, parse_test_metrics, parse_total_params
from patch_pretrain_conf import restore, set_dataset_use
from run_logger import RunLogger, log_subprocess_run

_RUN_LOGGER: RunLogger | None = None


def logger() -> RunLogger:
    global _RUN_LOGGER
    if _RUN_LOGGER is None:
        _RUN_LOGGER = RunLogger("phase6_worker", gpu_id=None)
    return _RUN_LOGGER


def set_logger(gpu_id: int | None):
    global _RUN_LOGGER
    _RUN_LOGGER = RunLogger("phase6_worker", gpu_id=gpu_id)


def load_cfg():
    with open(CONF_PATH) as f:
        return yaml.safe_load(f)


def dataset_exists(ds: str) -> bool:
    return (ROOT / "data" / ds / f"{ds}.npz").exists()


def scan_all_datasets() -> list[str]:
    data = ROOT / "data"
    out = []
    for d in sorted(data.iterdir()):
        if d.is_dir() and (d / f"{d.name}.npz").exists():
            out.append(d.name)
    return out


def shard_list(items: list, gpu_id: int | None, n_gpus: int = 2) -> list:
    if gpu_id is None:
        return items
    return [x for i, x in enumerate(items) if i % n_gpus == gpu_id]


def shard_by_name(ds: str, gpu_id: int | None) -> bool:
    if gpu_id is None:
        return True
    return hash(ds) % 2 == gpu_id


def csv_suffix(gpu_id: int | None) -> str:
    return f".gpu{gpu_id}" if gpu_id is not None else ""


def get_gpu_mb(cuda_visible: str | None) -> int:
    env = os.environ.copy()
    if cuda_visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            env=env,
        )
        if r.returncode == 0 and r.stdout.strip():
            vals = [int(float(x)) for x in r.stdout.strip().split("\n") if x.strip()]
            return max(vals) if vals else 0
    except Exception:
        pass
    return 0


def effective_batch(ds: str, variant_cfg: dict, cfg: dict) -> int:
    if ds in cfg.get("cad_datasets", []) or ds.startswith("CAD"):
        return min(2, variant_cfg["batch_size"])
    if ds in ("NYC_TAXI", "CHI_TAXI") or ds.startswith("Traffic") or ds.startswith("NYC_BIKE"):
        return min(2, variant_cfg["batch_size"])
    return variant_cfg["batch_size"]


def run_runpy(
    mode: str,
    ds: str,
    variant_cfg: dict,
    gpu_id: int | None,
    extra_args: list | None = None,
    log_tag: str = "",
) -> tuple[int, Path, float, int]:
    set_dataset_use([ds], gpu_id=gpu_id)
    tag = log_tag or f"{mode}_{ds}_{variant_cfg.get('name', 'v')}"
    log = LOG_DIR / f"{tag}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg()
    bs = effective_batch(ds, variant_cfg, cfg)
    cmd = [
        sys.executable,
        "Run.py",
        "-mode",
        mode,
        "-model",
        "OpenCity",
        "-load_pretrain_path",
        variant_cfg["ckpt"],
        "--batch_size",
        str(bs),
        "--embed_dim",
        str(variant_cfg["embed_dim"]),
        "--skip_dim",
        str(variant_cfg["skip_dim"]),
        "--enc_depth",
        str(variant_cfg["enc_depth"]),
    ]
    if extra_args:
        cmd.extend(extra_args)
    env = os.environ.copy()
    env["OPENCITY_DATASET_USE"] = ds
    env["OPENCITY_TQDM"] = "1"
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    job_name = f"{tag}_gpu{gpu_id}"
    logger().info(f"Run.py start: {job_name} dataset={ds} mode={mode}")
    gpu_before = get_gpu_mb(str(gpu_id) if gpu_id is not None else None)
    rc, wall = log_subprocess_run(logger(), cmd, MODEL_DIR, env, log, job_name)
    gpu_after = get_gpu_mb(str(gpu_id) if gpu_id is not None else None)
    return rc, log, wall, max(gpu_before, gpu_after)


def variant_from_name(cfg: dict, name: str) -> dict:
    v = dict(cfg["variants"][name])
    v["name"] = name
    return v


def load_done(path: Path, keys: list[str]) -> set:
    done = set()
    if path.exists():
        with open(path) as f:
            for row in csv.DictReader(f):
                done.add(tuple(row[k] for k in keys))
    return done


def append_row(path: Path, fields: list[str], row: dict):
    write_header = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)


def filter_datasets(names: list[str], gpu_id: int | None, explicit: list | None) -> list[str]:
    if explicit is not None:
        return [d for d in names if d in explicit]
    return [d for d in names if shard_by_name(d, gpu_id)]


def metrics_if_dataset_matches(log_path: Path, expected: str) -> dict | None:
    actual = parse_actual_dataset(log_path)
    if actual != expected:
        logger().error(f"dataset mismatch expected={expected} actual={actual} log={log_path}")
        return None
    return parse_test_metrics(log_path)


def run_zero_shot_batch(cfg: dict, datasets: list[str], out_csv: Path, group: str, gpu_id: int | None):
    fields = ["group", "variant", "dataset", "mae", "rmse", "mape", "wall_s", "gpu_mb", "status"]
    done = load_done(out_csv, ["group", "variant", "dataset"])
    vc = variant_from_name(cfg, "plus")
    for ds in datasets:
        key = (group, "plus", ds)
        if key in done:
            logger().info(f"[skip] {group} {ds}")
            continue
        if not dataset_exists(ds):
            append_row(out_csv, fields, {
                "group": group, "variant": "plus", "dataset": ds,
                "mae": "", "rmse": "", "mape": "", "wall_s": "", "gpu_mb": "", "status": "no_data",
            })
            continue
        rc, log, wall, gpu = run_runpy("test", ds, vc, gpu_id, log_tag=f"zs_{group}_{ds}_plus")
        m = metrics_if_dataset_matches(log, ds)
        if rc == 0 and m:
            append_row(out_csv, fields, {
                "group": group, "variant": "plus", "dataset": ds, **m,
                "wall_s": wall, "gpu_mb": gpu, "status": "ok",
            })
            logger().info(f"[ok] gpu{gpu_id} {group} {ds} MAE={m['mae']} wall={wall}s")
        else:
            append_row(out_csv, fields, {
                "group": group, "variant": "plus", "dataset": ds,
                "mae": "", "rmse": "", "mape": "",
                "wall_s": wall, "gpu_mb": gpu, "status": "failed",
            })
            logger().error(f"[fail] gpu{gpu_id} {group} {ds} rc={rc} log={log}")


def run_zero_shot_all(cfg: dict, gpu_id: int | None):
    suf = csv_suffix(gpu_id)
    if gpu_id is None:
        t1_list = cfg["table1_datasets"]
        t2_list = cfg["table2_supervised_datasets"]
    else:
        t1_list = cfg.get(f"table1_gpu{gpu_id}", shard_list(cfg["table1_datasets"], gpu_id))
        t2_list = cfg.get(f"table2_gpu{gpu_id}", shard_list(cfg["table2_supervised_datasets"], gpu_id))
    run_zero_shot_batch(cfg, t1_list, RESULT_DIR / f"zero_shot_table1{suf}.csv", "table1", gpu_id)
    all_ds = filter_datasets(scan_all_datasets(), gpu_id, None)
    run_zero_shot_batch(cfg, all_ds, RESULT_DIR / f"zero_shot_all{suf}.csv", "extended", gpu_id)
    run_zero_shot_batch(cfg, t2_list, RESULT_DIR / f"zero_shot_table2_compare{suf}.csv", "table2_zs", gpu_id)


def run_fast_adapt_full(cfg: dict, gpu_id: int | None):
    suf = csv_suffix(gpu_id)
    out = RESULT_DIR / f"fast_adapt_full{suf}.csv"
    fields = [
        "variant", "dataset", "zeroshot_mae", "mae", "rmse", "mape",
        "improve_pct", "finetune_wall_s", "gpu_mb", "status",
    ]
    done = load_done(out, ["variant", "dataset"])
    epochs = cfg["fast_adapt"]["finetune_epochs"]
    ds_list = cfg["fast_adapt"].get(f"gpu{gpu_id}_datasets", cfg["fast_adapt"]["datasets"]) if gpu_id is not None else cfg["fast_adapt"]["datasets"]
    for vname in cfg["fast_adapt"]["variants"]:
        vc = variant_from_name(cfg, vname)
        for ds in ds_list:
            if (vname, ds) in done:
                logger().info(f"[skip] ft {vname} {ds}")
                continue
            if not dataset_exists(ds):
                continue
            rc, log_z, _, _ = run_runpy("test", ds, vc, gpu_id, log_tag=f"ft_zs_{vname}_{ds}")
            mz = metrics_if_dataset_matches(log_z, ds)
            zs = float(mz["mae"]) if mz else None
            extra = ["-epochs", str(epochs), "--epochs", str(epochs)]
            rc2, log_f, wall, gpu = run_runpy("eval", ds, vc, gpu_id, extra, log_tag=f"ft_eval_{vname}_{ds}")
            mf = metrics_if_dataset_matches(log_f, ds)
            if rc == 0 and rc2 == 0 and mz and mf:
                imp = round((zs - float(mf["mae"])) / zs * 100, 2) if zs else 0
                append_row(out, fields, {
                    "variant": vname, "dataset": ds, "zeroshot_mae": zs, **mf,
                    "improve_pct": imp, "finetune_wall_s": wall, "gpu_mb": gpu, "status": "ok",
                })
                logger().info(f"[ok] ft gpu{gpu_id} {vname} {ds} zs={zs} ft={mf['mae']}")
            else:
                append_row(out, fields, {
                    "variant": vname, "dataset": ds, "zeroshot_mae": zs or "",
                    "mae": "", "rmse": "", "mape": "", "improve_pct": "",
                    "finetune_wall_s": wall, "gpu_mb": gpu, "status": "failed",
                })


def run_scaling_matrix(cfg: dict, gpu_id: int | None):
    suf = csv_suffix(gpu_id)
    out = RESULT_DIR / f"scaling_matrix{suf}.csv"
    fields = ["variant", "dataset", "params_M", "mae", "rmse", "mape", "infer_wall_s", "gpu_mb", "status"]
    done = load_done(out, ["variant", "dataset"])
    jobs = [(v, ds) for ds in cfg["scaling"]["datasets"] for v in cfg["scaling"]["variants"]]
    jobs = shard_list(jobs, gpu_id)
    for vname, ds in jobs:
        if (vname, ds) in done:
            logger().info(f"[skip] scale {vname} {ds}")
            continue
        if not dataset_exists(ds):
            continue
        vc = variant_from_name(cfg, vname)
        rc, log, wall, gpu = run_runpy("test", ds, vc, gpu_id, log_tag=f"scale_{vname}_{ds}")
        m = metrics_if_dataset_matches(log, ds)
        params = parse_total_params(log)
        pm = round(params / 1e6, 2) if params else ""
        if rc == 0 and m:
            append_row(out, fields, {
                "variant": vname, "dataset": ds, "params_M": pm, **m,
                "infer_wall_s": wall, "gpu_mb": gpu, "status": "ok",
            })
            logger().info(f"[ok] scale gpu{gpu_id} {vname} {ds} MAE={m['mae']}")
        else:
            append_row(out, fields, {
                "variant": vname, "dataset": ds, "params_M": pm,
                "mae": "", "rmse": "", "mape": "",
                "infer_wall_s": wall, "gpu_mb": gpu, "status": "failed",
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["zeroshot", "fast_adapt", "scaling", "all"], default="all")
    ap.add_argument("--gpu-id", type=int, default=None, choices=[0, 1])
    args = ap.parse_args()
    set_logger(args.gpu_id)
    log = logger()
    cfg = load_cfg()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    log.section(f"Phase6 start exp={args.exp} gpu_id={args.gpu_id}")
    log.event("worker_start", exp=args.exp)
    try:
        if args.exp in ("zeroshot", "all"):
            log.section("zero-shot")
            run_zero_shot_all(cfg, args.gpu_id)
        if args.exp in ("fast_adapt", "all"):
            log.section("fast_adapt")
            run_fast_adapt_full(cfg, args.gpu_id)
        if args.exp in ("scaling", "all"):
            log.section("scaling")
            run_scaling_matrix(cfg, args.gpu_id)
    finally:
        restore()
    log.event("worker_done", exp=args.exp)
    log.info(f"Phase6 finished gpu_id={args.gpu_id}")


if __name__ == "__main__":
    main()
