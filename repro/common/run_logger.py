#!/usr/bin/env python3
"""Unified run logging: worker log + events.jsonl + per-job Run.py logs."""
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = ROOT / "repro" / "logs"
EVENTS_FILE = LOG_ROOT / "events.jsonl"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RunLogger:
    """Per-worker logger writing to repro/logs/ and events.jsonl."""

    def __init__(self, name: str, gpu_id: int | None = None):
        self.name = name
        self.gpu_id = gpu_id
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        suffix = f"_gpu{gpu_id}" if gpu_id is not None else ""
        self.worker_log = LOG_ROOT / f"{name}{suffix}.log"
        self._logger = logging.getLogger(f"repro.{name}{suffix}")
        self._logger.setLevel(logging.INFO)
        self._logger.handlers.clear()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(self.worker_log, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        self._logger.addHandler(fh)
        self._logger.addHandler(sh)

    def info(self, msg: str):
        self._logger.info(msg)

    def warning(self, msg: str):
        self._logger.warning(msg)

    def error(self, msg: str):
        self._logger.error(msg)

    def event(self, event_type: str, **fields):
        """Append one JSON line to events.jsonl for progress tracking."""
        rec = {
            "ts": _ts(),
            "worker": self.name,
            "gpu_id": self.gpu_id,
            "event": event_type,
            **fields,
        }
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        with open(EVENTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.info(f"[{event_type}] " + " ".join(f"{k}={v}" for k, v in fields.items()))

    def job_start(self, job: str, **kw):
        self.event("job_start", job=job, **kw)

    def job_end(self, job: str, status: str, wall_s: float | None = None, **kw):
        self.event("job_end", job=job, status=status, wall_s=wall_s, **kw)

    def section(self, title: str):
        line = "=" * 60
        self.info(line)
        self.info(title)
        self.info(line)


def log_subprocess_run(
    logger: RunLogger,
    cmd: list,
    cwd: Path,
    env: dict,
    detail_log: Path,
    job_name: str,
) -> tuple[int, float]:
    """Run subprocess; write full stdout to detail_log and summary to worker log."""
    import subprocess

    logger.job_start(job_name, cmd=" ".join(cmd), detail_log=str(detail_log))
    detail_log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(detail_log, "w", encoding="utf-8") as lf:
        lf.write(f"# started {_ts()}\n# cmd: {' '.join(cmd)}\n# cwd: {cwd}\n\n")
        lf.flush()
        proc = subprocess.run(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT, text=True, env=env)
    wall = round(time.time() - t0, 1)
    status = "ok" if proc.returncode == 0 else "failed"
    with open(detail_log, "a", encoding="utf-8") as lf:
        lf.write(f"\n# finished {_ts()} exit={proc.returncode} wall_s={wall}\n")
    logger.job_end(job_name, status=status, wall_s=wall, exit_code=proc.returncode)
    return proc.returncode, wall
