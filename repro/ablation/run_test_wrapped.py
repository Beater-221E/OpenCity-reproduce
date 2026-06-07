#!/usr/bin/env python3
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.path.join(ROOT, "model")
sys.path.insert(0, ROOT)
sys.path.insert(0, MODEL)

from repro.ablation.inject import apply

apply(os.environ.get("OPENCITY_ABLATION", "full"))

run_py = os.path.join(MODEL, "Run.py")
with open(run_py) as f:
    code = compile(f.read(), run_py, "exec")
os.chdir(MODEL)
sys.argv = [run_py] + sys.argv[1:]
exec(code, {"__name__": "__main__", "__file__": run_py})
