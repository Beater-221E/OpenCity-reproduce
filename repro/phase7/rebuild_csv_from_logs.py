#!/usr/bin/env python3
"""Rebuild lora_adapt_full.csv from completed job logs."""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "repro/results/phase7/logs"
OUT = ROOT / "repro/results/phase7/lora_adapt_full.csv"
FAST = ROOT / "repro/results/phase6/fast_adapt_full.csv"

import sys
sys.path.insert(0, str(ROOT / "repro/common"))
from parse_log import parse_test_metrics

RE_JOB = re.compile(r"lora_(mini|base|plus)_r(\d+)_(CD_DIDI|SZ_DIDI)\.log$")
RE_TRAINABLE = re.compile(r"trainable_params=(\d+)")


def zeroshot_mae(variant: str, dataset: str) -> str:
    if not FAST.exists():
        return ""
    with open(FAST) as f:
        for row in csv.DictReader(f):
            if row.get("variant") == variant and row.get("dataset") == dataset:
                return row.get("zeroshot_mae", "")
    return ""


def main():
    rows = []
    for log in sorted(LOG_DIR.glob("lora_*.log")):
        m = RE_JOB.search(log.name)
        if not m:
            continue
        variant, rank, dataset = m.group(1), int(m.group(2)), m.group(3)
        metrics = parse_test_metrics(log)
        if not metrics:
            continue
        text = log.read_text(errors="replace")
        tp = RE_TRAINABLE.search(text)
        zs = zeroshot_mae(variant, dataset)
        zs_f = float(zs) if zs else None
        mae = float(metrics["mae"])
        imp = round((zs_f - mae) / zs_f * 100, 2) if zs_f else ""
        rows.append({
            "variant": variant,
            "dataset": dataset,
            "lora_rank": rank,
            "zeroshot_mae": zs,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mape": metrics["mape"],
            "improve_pct": imp,
            "trainable_params": tp.group(1) if tp else "",
            "lora_wall_s": "",
            "gpu_mb": "",
            "status": "ok",
        })

    fields = [
        "variant", "dataset", "lora_rank", "zeroshot_mae", "mae", "rmse", "mape",
        "improve_pct", "trainable_params", "lora_wall_s", "gpu_mb", "status",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["variant"], r["dataset"], r["lora_rank"])))
    print(f"Rebuilt {OUT} with {len(rows)} rows")


if __name__ == "__main__":
    main()
