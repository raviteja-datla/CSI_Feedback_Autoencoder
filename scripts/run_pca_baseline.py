#!/usr/bin/env python
"""Run the PCA baseline at matched compression ratios, save one metrics JSON per scenario.

Example:
    uv run python scripts/run_pca_baseline.py --scenario indoor \
        --cr 0.25 0.0625 0.03125 0.015625 --data-dir outputs/data --out-dir outputs
"""

import argparse
import json
import os

from csinet.dataset import load_split
from csinet.pca_baseline import run_pca_baseline

AMBIENT_DIM = 2 * 32 * 32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--cr", type=float, nargs="+", required=True)
    parser.add_argument("--data-dir", default="outputs/data")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    metrics_dir = os.path.join(args.out_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    train_path = os.path.join(args.data_dir, f"{args.scenario}_train.npz")
    test_path = os.path.join(args.data_dir, f"{args.scenario}_test.npz")
    train_h = load_split(train_path)["h"]
    test_h = load_split(test_path)["h"]

    results = {}
    for cr in args.cr:
        m = round(cr * AMBIENT_DIM)
        nmse = run_pca_baseline(train_h, test_h, m)
        print(f"[{args.scenario}] cr={cr} m={m} pca_nmse_db={nmse:.3f}")
        results[str(cr)] = {"m": m, "nmse_db": nmse}

    out_path = os.path.join(metrics_dir, f"{args.scenario}_pca.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
