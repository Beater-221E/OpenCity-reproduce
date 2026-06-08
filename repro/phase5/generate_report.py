#!/usr/bin/env python3
"""Generate comparison tables from phase6 results."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P6 = ROOT / "repro" / "results" / "phase6"
P6CFG = ROOT / "repro" / "phase6" / "configs.yaml"
TABLES = ROOT / "repro" / "results" / "tables"
FIGURES = ROOT / "repro" / "figures"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def pct_err(ours, paper):
    try:
        o, p = float(ours), float(paper)
        return round(abs(o - p) / p * 100, 1)
    except (TypeError, ValueError, ZeroDivisionError):
        return "-"


def load_yaml_cfg():
    import yaml
    with open(P6CFG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_table1(cfg):
    t1 = read_csv(P6 / "zero_shot_table1.csv")
    paper = cfg.get("paper_table1", {})
    lines = ["# Table 1 — OpenCity Zero-shot（复现）\n\n"]
    lines.append("| Dataset | MAE | RMSE | MAPE | Paper MAE | Err% | Status |\n")
    lines.append("|---------|-----|------|------|-----------|------|--------|\n")
    for ds in cfg.get("table1_datasets", []):
        row = next((r for r in t1 if r.get("dataset") == ds), None)
        pref = paper.get(ds, {})
        pm = pref.get("mae", "-")
        if row and row.get("status") == "ok":
            err = pct_err(row["mae"], pm) if pm != "-" else "-"
            lines.append(f"| {ds} | {row['mae']} | {row['rmse']} | {row['mape']}% | {pm} | {err} | ok |\n")
        elif row:
            status = row.get("status") or "pending"
            lines.append(f"| {ds} | - | - | - | {pm} | - | {status} |\n")
        else:
            lines.append(f"| {ds} | - | - | - | {pm} | - | pending |\n")
    (TABLES / "paper_table1_opencity.md").write_text("".join(lines))


def write_table3(cfg):
    fa = read_csv(P6 / "fast_adapt_full.csv")
    paper = cfg.get("paper_table3_plus", {})
    lines = ["# Table 3 — OpenCity Fast Adaptation（复现）\n\n"]
    lines.append("| Variant | Dataset | ZS MAE | FT MAE | Improve% | Paper ZS | Paper FT |\n")
    lines.append("|---------|---------|--------|--------|----------|----------|----------|\n")
    for ds in cfg.get("fast_adapt", {}).get("datasets", []):
        pref = paper.get(ds, {})
        for vname in cfg.get("fast_adapt", {}).get("variants", []):
            row = next((r for r in fa if r.get("dataset") == ds and r.get("variant") == vname and r.get("status") == "ok"), None)
            pzs, pft = pref.get("zeroshot", "-"), pref.get("finetune", "-")
            if vname != "plus":
                pzs, pft = "-", "-"
            if row:
                lines.append(f"| {vname} | {ds} | {row['zeroshot_mae']} | {row['mae']} | {row['improve_pct']}% | {pzs} | {pft} |\n")
            else:
                lines.append(f"| {vname} | {ds} | - | - | - | {pzs} | {pft} |\n")
    (TABLES / "paper_table3_opencity.md").write_text("".join(lines))


def write_table2_compare(cfg):
    t2 = read_csv(P6 / "zero_shot_table2_compare.csv")
    paper = cfg.get("paper_table2_opencity_plus", {})
    lines = [
        "# Zero-shot vs 论文 Table 2 OpenCity 监督行\n\n",
        "| Dataset | ZS MAE (ours) | Paper supervised MAE | Δ% |\n",
        "|---------|---------------|----------------------|-----|\n",
    ]
    for ds, pref in paper.items():
        row = next((r for r in t2 if r.get("dataset") == ds and r.get("status") == "ok"), None)
        pm = pref.get("mae", "-")
        if row and pm != "-":
            try:
                delta = round((float(row["mae"]) - float(pm)) / float(pm) * 100, 1)
            except ValueError:
                delta = "-"
            lines.append(f"| {ds} | {row['mae']} | {pm} | {delta} |\n")
        else:
            lines.append(f"| {ds} | - | {pm} | pending |\n")
    (P6 / "zero_shot_vs_table2_supervised.md").write_text("".join(lines))


def write_coverage(cfg):
    def count_ok(path, n_expect, key_fn):
        rows = read_csv(path)
        ok = sum(1 for r in rows if r.get("status") == "ok")
        return ok, n_expect

    t1_n = len(cfg["table1_datasets"])
    t1_ok, _ = count_ok(P6 / "zero_shot_table1.csv", t1_n, None)
    fa_n = len(cfg["fast_adapt"]["variants"]) * len(cfg["fast_adapt"]["datasets"])
    fa_ok, _ = count_ok(P6 / "fast_adapt_full.csv", fa_n, None)
    sc_n = len(cfg["scaling"]["variants"]) * len(cfg["scaling"]["datasets"])
    sc_ok, _ = count_ok(P6 / "scaling_matrix.csv", sc_n, None)
    ab_n = len(cfg["ablation"]["variants"]) * len(cfg["ablation"]["datasets"])
    ab_ok, _ = count_ok(P6 / "ablation.csv", ab_n, None)
    all_ok = sum(1 for r in read_csv(P6 / "zero_shot_all.csv") if r.get("status") == "ok")

    text = f"""# 论文实验覆盖矩阵

| 产出 | 完成 | 说明 |
|------|------|------|
| Table 1 ZS | {t1_ok}/{t1_n} | baseline N/A |
| Table 3 ZS+FT | {fa_ok}/{fa_n} | Cost N/A |
| Scaling | {sc_ok}/{sc_n} | 参数维 |
| 消融 | {ab_ok}/{ab_n} | 推理级 |
| 扩展 ZS | {all_ok} 集 | |
| Table 2 监督 | 不做 | 见 zero_shot_vs_table2 |
"""
    (TABLES / "coverage_matrix.md").write_text(text)


def plot_scaling(cfg):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    sc = read_csv(P6 / "scaling_matrix.csv")
    ok = [r for r in sc if r.get("status") == "ok"]
    if not ok:
        return
    variants = cfg["scaling"]["variants"]
    datasets = sorted({r["dataset"] for r in ok})
    mat = []
    for ds in datasets:
        row = [float(next((r for r in ok if r["dataset"] == ds and r["variant"] == v), {"mae": "nan"})["mae"]) for v in variants]
        mat.append(row)
    mat = np.array(mat, dtype=float)
    plt.figure(figsize=(8, max(4, len(datasets) * 0.4)))
    plt.imshow(mat, aspect="auto", cmap="YlOrRd")
    plt.colorbar(label="MAE")
    plt.xticks(range(len(variants)), variants)
    plt.yticks(range(len(datasets)), datasets)
    plt.title("Scaling matrix")
    plt.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES / "scaling_heatmap.png", dpi=120)
    plt.close()


def main():
    import sys
    sys.path.insert(0, str(ROOT / "repro" / "common"))
    from run_logger import RunLogger

    log = RunLogger("report")
    log.section("generate_report start")
    TABLES.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml_cfg()
    write_table1(cfg)
    write_table3(cfg)
    write_table2_compare(cfg)
    write_coverage(cfg)
    plot_scaling(cfg)
    log.event("report_done", tables=str(TABLES))
    log.info(f"Reports in {TABLES}")


if __name__ == "__main__":
    main()
