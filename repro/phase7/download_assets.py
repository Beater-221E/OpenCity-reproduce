#!/usr/bin/env python3
"""Download LoRA experiment assets: CD_DIDI, SZ_DIDI + OpenCity weights."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
WEIGHTS = ROOT / "model_weights" / "OpenCity"
DATASETS = ["CD_DIDI", "SZ_DIDI"]
DATA_REPO = "hkuds/OpenCity-dataset"
WEIGHT_SOURCES = [
    ("hkuds/OpenCity-Mini", "OpenCity-mini.pth"),
    ("hkuds/OpenCity-Base", "OpenCity-base.pth"),
    ("hkuds/OpenCity-Plus", "OpenCity-plus.pth"),
]


def extract_dataset_zip(zip_path: Path, ds: str):
    dest = DATA / ds
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA)
    npz = dest / f"{ds}.npz"
    if not npz.exists():
        for p in DATA.rglob(f"{ds}.npz"):
            if p.name == f"{ds}.npz":
                if p != npz:
                    p.rename(npz)
                break
    if zip_path.exists():
        zip_path.unlink(missing_ok=True)


def download_datasets():
    from huggingface_hub import hf_hub_download

    for ds in DATASETS:
        npz = DATA / ds / f"{ds}.npz"
        if npz.exists():
            print(f"OK dataset: {npz}")
            continue
        print(f"Downloading {ds}.zip ...")
        path = hf_hub_download(
            repo_id=DATA_REPO,
            filename=f"{ds}.zip",
            repo_type="dataset",
            local_dir=str(DATA),
        )
        extract_dataset_zip(Path(path), ds)
        print(f"OK dataset: {npz}" if npz.exists() else f"FAIL dataset: {ds}")


def download_weights():
    from huggingface_hub import hf_hub_download

    WEIGHTS.mkdir(parents=True, exist_ok=True)
    for repo_id, name in WEIGHT_SOURCES:
        dest = WEIGHTS / name
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"OK weight: {dest}")
            continue
        print(f"Downloading {name} from {repo_id} ...")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=name,
            repo_type="model",
            local_dir=str(WEIGHTS),
        )
        p = Path(path)
        if p != dest and p.exists():
            p.rename(dest)
        print(f"OK weight: {dest}" if dest.exists() else f"FAIL weight: {name}")


def main():
    download_datasets()
    download_weights()


if __name__ == "__main__":
    main()
