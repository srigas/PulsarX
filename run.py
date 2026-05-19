#!/usr/bin/env python3
"""
run.py — Multi-experiment orchestrator.

Reads parameter lists from config.py, builds every combination via Cartesian
product, and calls main.py once per combination as a subprocess. Experiments
run sequentially.

Usage
-----
    python run.py
"""

import itertools
import json
import subprocess
import sys
import time

import config

combinations = list(itertools.product(
    config.SEED,
    config.R,
    config.MULTIPLIER,
    config.DOUBLE_PRECISION,
    config.BETA_SEP,
    config.TOL,
    config.CYCLE_CONFIG,
    config.SYM_SEP,
    config.ALPHA,
    config.BETA,
))

n_total = len(combinations)
print("=" * 70)
print("  Pulsar Magnetosphere PINN — Sweep Orchestrator")
print("=" * 70)
print(f"  Total combinations : {n_total}")
print("=" * 70 + "\n")

results = []

for idx, combo in enumerate(combinations, start=1):
    seed, R, multiplier, double_precision, beta_sep, (tol_sep, tol_dp), cycle_config, sym_sep, alpha, beta = combo

    cmd = [
        sys.executable, "main.py",
        "--seed",             str(seed),
        "--R",                str(R),
        "--multiplier",       str(multiplier),
        "--double-precision", str(double_precision).lower(),
        "--beta-sep",         str(beta_sep),
        "--tol-sep",          str(tol_sep),
        "--tol-dp",           str(tol_dp),
        "--cycle-config",     json.dumps({str(k): v for k, v in cycle_config.items()}),
        "--sym-sep",          str(sym_sep).lower(),
        "--alpha",            str(alpha),
        "--beta",             str(beta),
    ]

    print("─" * 70)
    print(f"  Run [{idx}/{n_total}]")
    print(f"    seed={seed}  R={R}  multiplier={multiplier}  dp={double_precision}")
    print(f"    beta_sep={beta_sep}  tol_sep={tol_sep}  tol_dp={tol_dp}")
    print(f"    cycle_config={cycle_config}")
    print(f"    sym_sep={sym_sep}  alpha={alpha}  beta={beta}")
    print("─" * 70)

    t0      = time.time()
    proc    = subprocess.run(cmd)
    elapsed = time.time() - t0

    status = "OK" if proc.returncode == 0 else f"FAILED (exit code {proc.returncode})"
    results.append((idx, proc.returncode, elapsed))
    print(f"\n  [{idx}/{n_total}] {status}  —  {elapsed / 60:.1f} min\n")

ok     = [r for r in results if r[1] == 0]
failed = [r for r in results if r[1] != 0]
total_t = sum(r[2] for r in results)

print("=" * 70)
print("  Sweep summary")
print("=" * 70)
print(f"  Completed  : {len(ok)} / {n_total}")
if failed:
    print(f"  Failed     : runs {', '.join(str(r[0]) for r in failed)}")
print(f"  Total time : {total_t / 60:.1f} min")
print("=" * 70)

if failed:
    sys.exit(1)
