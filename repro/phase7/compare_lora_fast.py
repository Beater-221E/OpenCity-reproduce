#!/usr/bin/env python3
"""Compare LoRA fine-tuning vs Fast Adaptation (head-only)."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LORA_CSV = ROOT / "repro/results/phase7/lora_adapt_full.csv"
FAST_CSV = ROOT / "repro/results/phase6/fast_adapt_full.csv"
OUT_MD = ROOT / "repro/results/tables/lora_vs_fast_adapt.md"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    fast = {(r["variant"], r["dataset"]): r for r in read_csv(FAST_CSV) if r.get("status") == "ok"}
    lora_rows = [r for r in read_csv(LORA_CSV) if r.get("status") == "ok"]

    by_key: dict[tuple, dict] = {}
    for r in lora_rows:
        key = (r["variant"], r["dataset"])
        by_key.setdefault(key, {})[str(r["lora_rank"])] = r

    lines = [
        "# LoRA vs Fast Adaptation\n\n",
        "Fast Adapt：仅微调 `predictor.linear`（3 epoch）。LoRA：编码器 attn+FFN，rank 8/16/24。\n\n",
        "| Variant | Dataset | ZS MAE | Fast MAE | LoRA r8 | LoRA r16 | LoRA r24 | Best LoRA |\n",
        "|---------|---------|--------|----------|---------|----------|----------|----------|\n",
    ]

    keys = sorted(set(fast.keys()) | set(by_key.keys()))
    for key in keys:
        v, ds = key
        f = fast.get(key, {})
        zs = f.get("zeroshot_mae", "-")
        fast_mae = f.get("mae", "-")
        loras = by_key.get(key, {})
        cells = []
        best = ("", 1e9)
        for rank in ("8", "16", "24"):
            row = loras.get(rank)
            if row and row.get("mae"):
                m = float(row["mae"])
                cells.append(f"{m:.2f}")
                if m < best[1]:
                    best = (rank, m)
            else:
                cells.append("-")
        best_s = f"r{best[0]} ({best[1]:.2f})" if best[0] else "-"
        lines.append(
            f"| {v} | {ds} | {zs} | {fast_mae} | {cells[0]} | {cells[1]} | {cells[2]} | {best_s} |\n"
        )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("".join(lines))
    print(f"Wrote {OUT_MD} ({len(lora_rows)} LoRA rows)")


if __name__ == "__main__":
    main()
