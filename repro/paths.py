"""Shared paths for reproduction scripts. Repo root is parent of repro/."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "model"
CONF = REPO_ROOT / "conf" / "general_conf" / "pretrain.conf"
REPRO_DIR = REPO_ROOT / "repro"
LOG_DIR = REPRO_DIR / "logs"
RESULTS_DIR = REPRO_DIR / "results"
DATA_DIR = REPO_ROOT / "data"
WEIGHTS_DIR = REPO_ROOT / "model_weights" / "OpenCity"
