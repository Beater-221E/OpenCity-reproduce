#!/usr/bin/env python3
"""Show reproduction progress from events.jsonl and result CSVs."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # /home/OpenCity
P6 = ROOT / "repro" / "results" / "phase6"
LOGS = ROOT / "repro" / "logs"
EVENTS = LOGS / "events.jsonl"


def count_csv(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    rows = list(csv.DictReader(open(path)))
    ok = sum(1 for r in rows if r.get("status") == "ok")
    return ok, len(rows)


def tail_events(n: int = 15):
    if not EVENTS.exists():
        print("(no events.jsonl yet)")
        return
    lines = EVENTS.read_text(encoding="utf-8").strip().split("\n")
    for line in lines[-n:]:
        try:
            e = json.loads(line)
            print(f"{e.get('ts')} gpu{e.get('gpu_id')} {e.get('event')} {e.get('job', '')} {e.get('status', '')}")
        except json.JSONDecodeError:
            print(line[:120])


def main():
    print("=== OpenCity repro progress ===\n")
    for label, pattern in [
        ("table1", "zero_shot_table1*.csv"),
        ("all_zs", "zero_shot_all*.csv"),
        ("table2", "zero_shot_table2*.csv"),
        ("scaling", "scaling_matrix*.csv"),
        ("fast_adapt", "fast_adapt_full*.csv"),
        ("ablation", "ablation*.csv"),
    ]:
        files = sorted(P6.glob(pattern))
        if not files:
            print(f"{label}: (no csv)")
            continue
        for f in files:
            ok, total = count_csv(f)
            print(f"{label} [{f.name}]: {ok}/{total} ok")

    print("\n--- recent events ---")
    tail_events(20)

    print("\n--- worker logs (tail) ---")
    for wl in sorted(LOGS.glob("phase6_worker*.log")) + sorted(LOGS.glob("worker*.log")):
        if wl.exists() and wl.stat().st_size:
            last = wl.read_text(encoding="utf-8", errors="replace").strip().split("\n")[-1]
            print(f"{wl.name}: {last[:100]}")


if __name__ == "__main__":
    main()
