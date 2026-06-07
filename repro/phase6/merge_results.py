#!/usr/bin/env python3
"""Merge per-GPU CSV shards into final phase6 results."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "repro" / "results" / "phase6"

MERGE_MAP = {
    "zero_shot_table1": ["group", "variant", "dataset"],
    "zero_shot_all": ["group", "variant", "dataset"],
    "zero_shot_table2_compare": ["group", "variant", "dataset"],
    "scaling_matrix": ["variant", "dataset"],
    "fast_adapt_full": ["variant", "dataset"],
    "ablation": ["dataset", "ablation"],
}


def merge_one(stem: str, key_fields: list[str]):
    shards = sorted(RESULT.glob(f"{stem}.gpu*.csv"))
    if not shards and (RESULT / f"{stem}.csv").exists():
        return
    rows = []
    seen = set()
    for sh in shards:
        with open(sh) as f:
            for row in csv.DictReader(f):
                k = tuple(row.get(x, "") for x in key_fields)
                if k in seen:
                    continue
                seen.add(k)
                rows.append(row)
    if not rows and not shards:
        return
    out = RESULT / f"{stem}.csv"
    fields = list(rows[0].keys()) if rows else []
    if not fields and shards:
        with open(shards[0]) as f:
            fields = f.readline().strip().split(",")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in sorted(rows, key=lambda r: tuple(r.get(k, "") for k in key_fields)):
            w.writerow(row)
    print(f"Merged {len(rows)} rows -> {out}")


def main():
    import sys
    sys.path.insert(0, str(ROOT / "repro" / "common"))
    from run_logger import RunLogger

    log = RunLogger("merge")
    log.section("merge_results start")
    RESULT.mkdir(parents=True, exist_ok=True)
    for stem, keys in MERGE_MAP.items():
        merge_one(stem, keys)
    log.event("merge_done")


if __name__ == "__main__":
    main()
