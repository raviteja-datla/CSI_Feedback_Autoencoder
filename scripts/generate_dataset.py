#!/usr/bin/env python
"""Generate train/val/test splits for a scenario and save them under --out-dir.

Example:
    uv run python scripts/generate_dataset.py --scenario indoor \
        --n-train 10000 --n-val 2000 --n-test 2000 --seed 0 --out-dir outputs/data
"""

import argparse
import os

import numpy as np

from csinet.channel_model import SCENARIOS
from csinet.dataset import build_split, save_split
from csinet.transform import energy_capture_ratio
from csinet.channel_model import generate_channel_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=list(SCENARIOS), required=True)
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-val", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-ant", type=int, default=32)
    parser.add_argument("--n-subcarriers", type=int, default=1024)
    parser.add_argument("--trunc", type=int, default=32)
    parser.add_argument("--out-dir", default="outputs/data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    scenario = SCENARIOS[args.scenario]

    # Diagnostic: energy capture ratio at the target truncation, on a held-out sample.
    diag_rng = np.random.default_rng(args.seed + 9999)
    diag_H = generate_channel_batch(
        scenario, 1000, n_ant=args.n_ant, n_subcarriers=args.n_subcarriers, rng=diag_rng
    )
    ratio = energy_capture_ratio(diag_H, n_ant=args.n_ant, trunc=args.trunc)
    print(f"[{args.scenario}] energy capture ratio at trunc={args.trunc}: {ratio:.4f}")

    splits = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    for i, (split_name, n) in enumerate(splits.items()):
        data = build_split(
            scenario,
            n,
            seed=args.seed + i,
            n_ant=args.n_ant,
            n_subcarriers=args.n_subcarriers,
            trunc=args.trunc,
        )
        h = data["h"]
        n_nan = int(np.sum(~np.isfinite(h)))
        print(
            f"[{args.scenario}/{split_name}] n={n} h.shape={h.shape} "
            f"mean|h|={np.mean(np.abs(h)):.4f} std|h|={np.std(np.abs(h)):.4f} n_nonfinite={n_nan}"
        )
        out_path = os.path.join(args.out_dir, f"{args.scenario}_{split_name}.npz")
        save_split(data, out_path)
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
