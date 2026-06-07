#!/usr/bin/env python3
"""Parse OpenCity Run.py log files for losses and test metrics."""
import re
from pathlib import Path

RE_STEP = re.compile(r"step:\s+(\d+)\s+train loss is:\s+([\d.]+)", re.I)
RE_VAL = re.compile(r"Val Epoch (\d+):\s+average Loss:\s+([\d.]+)", re.I)
RE_METRICS = re.compile(
    r"Average Horizon, MAE:\s*([\d.]+),\s*RMSE:\s*([\d.]+),\s*MAPE:\s*([\d.]+)%"
)
RE_PARAMS = re.compile(r"Total params num:\s*(\d+)")


def parse_train_losses(log_path: Path) -> list[tuple[int, float]]:
    if not log_path.exists():
        return []
    text = log_path.read_text(errors="replace")
    return [(int(m.group(1)), float(m.group(2))) for m in RE_STEP.finditer(text)]


def parse_val_epochs(log_path: Path) -> list[tuple[int, float]]:
    if not log_path.exists():
        return []
    text = log_path.read_text(errors="replace")
    return [(int(m.group(1)), float(m.group(2))) for m in RE_VAL.finditer(text)]


def parse_test_metrics(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="replace")
    matches = list(RE_METRICS.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return {"mae": m.group(1), "rmse": m.group(2), "mape": m.group(3)}


def parse_total_params(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="replace")
    matches = list(RE_PARAMS.finditer(text))
    if not matches:
        return None
    return int(matches[-1].group(1))
