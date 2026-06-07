#!/usr/bin/env python3
"""Download Table1 missing datasets from HuggingFace (zip at repo root)."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MISSING = ["TrafficSH", "CHI_TAXI", "NYC_BIKE-3"]
REPO = "hkuds/OpenCity-dataset"


def extract_zip(zip_path: Path, ds: str):
    dest = DATA / ds
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA)
    npz = dest / f"{ds}.npz"
    if not npz.exists():
        for p in DATA.rglob(f"{ds}.npz"):
            if p.parent.name == ds or ds in str(p):
                p.rename(npz) if p != npz else None
                break
    if zip_path.exists():
        zip_path.unlink(missing_ok=True)


def main():
    from huggingface_hub import hf_hub_download

    for ds in MISSING:
        dest = DATA / ds
        npz = dest / f"{ds}.npz"
        if npz.exists():
            print(f"OK exists: {npz}")
            continue
        print(f"Downloading {ds}.zip ...")
        try:
            path = hf_hub_download(
                repo_id=REPO,
                filename=f"{ds}.zip",
                repo_type="dataset",
                local_dir=str(DATA),
            )
            extract_zip(Path(path), ds)
            if npz.exists():
                print(f"OK: {npz}")
            else:
                print(f"WARN: extracted but missing {npz}")
        except Exception as e:
            print(f"FAIL {ds}: {e}")


if __name__ == "__main__":
    main()
