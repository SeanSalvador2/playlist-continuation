"""Run the whole ablation suite end-to-end and report total runtime.

Each experiment is a standalone, seeded, config-driven script; this just runs
them in sequence so ``results/ablations/`` is regenerated from scratch.

Run:  python experiments/run_all.py
      python experiments/run_all.py --quick   # smaller data-scale grid
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import os

HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    ("Exp1  hyperparameters", "exp_hyperparams.py", []),
    ("Exp1e hybrid + feature ablation", "exp_hybrid.py", []),
    ("Exp2  taste engine", "exp_taste.py", []),
    ("Exp3  ensemble (LOMO + routing)", "exp_ensemble.py", []),
    ("Exp4  robustness", "exp_robustness.py", []),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    t0 = time.time()
    for label, script, extra in SCRIPTS:
        if args.quick and script == "exp_robustness.py":
            extra = extra + ["--quick"]
        print(f"\n{'='*70}\n{label}  ({script})\n{'='*70}")
        t = time.time()
        r = subprocess.run([sys.executable, os.path.join(HERE, script)] + extra,
                           env=env)
        print(f"[{script}] finished in {time.time()-t:.1f}s (exit {r.returncode})")
        if r.returncode != 0:
            print("ABORTING: a sub-experiment failed.")
            sys.exit(r.returncode)
    print(f"\nAll experiments done in {(time.time()-t0)/60:.1f} min. "
          f"See results/ablations/ and ABLATIONS.md.")


if __name__ == "__main__":
    main()
