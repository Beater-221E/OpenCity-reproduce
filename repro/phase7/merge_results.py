#!/usr/bin/env python3
"""Merge phase7 per-GPU CSV shards."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "repro" / "results" / "phase7"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    shards = [
        RESULT / "lora_adapt_full.csv",
        RESULT / "lora_adapt_full.gpu0.csv",
        RESULT / "lora_adapt_full.gpu1.csv",
    ]
    rows = []
    seen = set()
    fields = None
    for p in shards:
        if not p.exists():
            continue
        for row in read_csv(p):
            key = (row.get("variant"), row.get("dataset"), row.get("lora_rank"))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            if fields is None and row:
                fields = list(row.keys())
    if not fields:
        fields = [
            "variant", "dataset", "lora_rank", "zeroshot_mae", "mae", "rmse", "mape",
            "improve_pct", "trainable_params", "lora_wall_s", "gpu_mb", "status",
        ]
    out = RESULT / "lora_adapt_full.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r.get("variant", ""), r.get("dataset", ""), int(r.get("lora_rank", 0) or 0))))
    print(f"Wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
