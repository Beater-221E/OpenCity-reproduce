#!/usr/bin/env python3
"""Verify existing CUDA/PyTorch; exit 1 if torch unavailable."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "repro" / "env_info.txt"


def main():
    lines = []
    lines.append(f"python: {sys.executable}")
    lines.append(f"version: {sys.version.split()[0]}")
    try:
        import torch

        lines.append(f"torch: {torch.__version__}")
        lines.append(f"cuda_available: {torch.cuda.is_available()}")
        lines.append(f"cuda_device_count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            lines.append(f"gpu_{i}: {torch.cuda.get_device_name(i)}")
    except ImportError as e:
        lines.append(f"torch: MISSING ({e})")
        lines.append("Use existing PyTorch in /venv/main; do not pip install README torch 1.9.")
        OUT.write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        sys.exit(1)

    r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if r.returncode == 0:
        lines.append("nvidia-smi:")
        lines.extend("  " + ln for ln in r.stdout.strip().split("\n"))
    lines.append(f"opencity_root: {ROOT}")
    lines.append(f"weights: {ROOT / 'model_weights' / 'OpenCity'}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()
