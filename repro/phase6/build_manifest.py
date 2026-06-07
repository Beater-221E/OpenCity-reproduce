#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TABLE1 = ["CAD3", "CAD5", "PEMS07M", "TrafficSH", "CHI_TAXI", "NYC_BIKE-3"]
OUT = ROOT / "repro" / "results" / "phase6" / "data_manifest.json"


def main():
    datasets = {}
    for d in sorted(DATA.iterdir()):
        if d.is_dir():
            npz = d / f"{d.name}.npz"
            datasets[d.name] = {"exists": npz.exists(), "path": str(npz)}
    ready = all(datasets.get(ds, {}).get("exists") for ds in TABLE1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "table1_required": TABLE1,
        "datasets": datasets,
        "table1_ready": ready,
    }, indent=2))
    print(f"Wrote {OUT} table1_ready={ready}")


if __name__ == "__main__":
    main()
