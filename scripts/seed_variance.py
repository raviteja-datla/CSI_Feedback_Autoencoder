#!/usr/bin/env python
"""Aggregate the official (seed=0) CR-sweep result with additional-seed reruns
(scripts/train.py --seed 1/2 --out-dir outputs/seeds/seed{1,2}) into mean/std per CR,
so training-stochasticity noise can be checked against the network-vs-PCA margins
reported elsewhere -- especially the tight ones, where seed noise could plausibly
flip which one "wins".

Example:
    uv run python scripts/seed_variance.py --scenario indoor \
        --cr 0.25 0.0625 0.03125 0.015625 \
        --official-dir outputs --seed-dirs outputs/seeds/seed1 outputs/seeds/seed2 \
        --out-dir outputs
"""

import argparse
import json
import os

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--cr", type=float, nargs="+", required=True)
    parser.add_argument("--official-dir", default="outputs")
    parser.add_argument("--seed-dirs", nargs="+", required=True)
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    metrics_dir = os.path.join(args.out_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    run_dirs = [args.official_dir] + args.seed_dirs

    for cr in args.cr:
        values = []
        for run_dir in run_dirs:
            path = os.path.join(run_dir, "metrics", f"{args.scenario}_cr{cr}.json")
            with open(path) as f:
                d = json.load(f)
            values.append(d["test_nmse_db"])

        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))  # sample std, ddof=1 since these are 3 independent runs
        print(f"[{args.scenario}] cr={cr}: seeds={[round(v, 3) for v in values]} mean={mean:.3f} std={std:.3f}")

        out_path = os.path.join(metrics_dir, f"{args.scenario}_cr{cr}_seeds.json")
        with open(out_path, "w") as f:
            json.dump({"scenario": args.scenario, "cr": cr, "values": values, "mean": mean, "std": std}, f, indent=2)
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
