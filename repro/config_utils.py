"""Helpers for editing pretrain.conf and parsing training logs."""
import re
from pathlib import Path

METRIC_PATTERN = re.compile(
    r"Average Horizon, MAE: ([\d.]+), RMSE: ([\d.]+), MAPE: ([\d.]+)%"
)


def set_dataset(conf_path: Path, dataset: str) -> None:
    text = conf_path.read_text(encoding="utf-8")
    text = re.sub(
        r"dataset_use\s*=\s*\[[^\]]*\]",
        f"dataset_use = ['{dataset}']",
        text,
    )
    conf_path.write_text(text, encoding="utf-8")


def parse_metrics(log_path: Path):
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = METRIC_PATTERN.search(text)
    return match.groups() if match else None
