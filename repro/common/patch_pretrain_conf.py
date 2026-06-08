#!/usr/bin/env python3
"""Patch dataset_use in config files.

For GPU-specific workers, this intentionally avoids copying back to the shared
pretrain.conf. Run.py should receive OPENCITY_DATASET_USE in its environment.
"""
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = ROOT / "conf" / "general_conf"
BASE_CONF = CONF_DIR / "pretrain.conf"
ACTIVE_CONF = CONF_DIR / "pretrain.conf"
_BACKUP = CONF_DIR / "pretrain.conf.repro.bak"
_GPU_COPIES: dict[int, Path] = {}
_ACTIVE_GPU: int | None = None


def _gpu_conf(gpu_id: int | None) -> Path:
    if gpu_id is None:
        return BASE_CONF
    path = CONF_DIR / f"pretrain.gpu{gpu_id}.conf"
    if not path.exists() and BASE_CONF.exists():
        shutil.copy2(BASE_CONF, path)
    return path


def set_dataset_use(datasets: list[str], gpu_id: int | None = None):
    global _ACTIVE_GPU
    gid = 0 if gpu_id is None else int(gpu_id)
    _ACTIVE_GPU = gid
    src = _gpu_conf(gid)
    if not src.exists():
        raise FileNotFoundError(src)
    if gid == 0 and not _BACKUP.exists() and BASE_CONF.exists():
        shutil.copy2(BASE_CONF, _BACKUP)
    text = src.read_text()
    repl = "dataset_use = " + str(datasets)
    if re.search(r"^dataset_use\s*=", text, flags=re.M):
        text = re.sub(r"^dataset_use\s*=\s*\[.*?\]", repl, text, count=1, flags=re.M)
    else:
        text = repl + "\n" + text
    src.write_text(text)
    if gpu_id is None:
        shutil.copy2(src, ACTIVE_CONF)


def restore():
    global _ACTIVE_GPU
    if _BACKUP.exists():
        shutil.copy2(_BACKUP, BASE_CONF)
        shutil.copy2(_BACKUP, ACTIVE_CONF)
        _BACKUP.unlink(missing_ok=True)
    _ACTIVE_GPU = None
