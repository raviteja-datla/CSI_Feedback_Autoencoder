#!/usr/bin/env python
"""Generate train/val/test splits from a real DeepMIMO scenario.

Requires the optional deepmimo dependency group: `uv sync --group deepmimo`.

Uses a blocked spatial holdout split by default (see
csinet.deepmimo_adapter.blocked_spatial_split_indices): train/val/test are scattered across many
small, buffered blocks covering the whole scenario, rather than one contiguous region or a plain
shuffle. Two simpler alternatives are kept for comparison, both NOT recommended for reported
results:
  --split random  : a plain shuffle. DeepMIMO users sit on a dense spatial grid (e.g. 1cm spacing
                    for indoor i2_28b), so a held-out point routinely lands right next to several
                    training points -- both PCA and the network partly interpolate between
                    near-duplicate samples instead of generalizing (produced an implausible
                    -120dB PCA result for indoor_real).
  --split spatial : one contiguous held-out region. Fixes the leakage above, but on a
                    geographically large/diverse scenario the held-out region can be a
                    statistically different *environment* (different buildings, different
                    distance/angle-to-BS distribution) -- collapsed both PCA and the network to
                    ~0dB on outdoor_real, a domain-shift artifact, not a real result.

Example:
    uv run python scripts/generate_dataset_deepmimo.py --scenario outdoor_real \
        --n-train 10000 --n-val 2000 --n-test 2000 --seed 0 --out-dir outputs/data
"""

import argparse
import os

import numpy as np

from csinet.dataset import save_split
from csinet.deepmimo_adapter import (
    REAL_SCENARIOS,
    blocked_spatial_split_indices,
    build_real_split,
    load_deepmimo_channels,
    random_split_indices,
    spatial_split_indices,
)
from csinet.transform import energy_capture_ratio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=list(REAL_SCENARIOS), required=True)
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-val", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-ant", type=int, default=32)
    parser.add_argument("--n-subcarriers", type=int, default=1024)
    parser.add_argument("--trunc", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=500, help="users processed per batch (memory bound)")
    parser.add_argument("--split", choices=["blocked", "spatial", "random"], default="blocked")
    parser.add_argument("--out-dir", default="outputs/data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    scenario = REAL_SCENARIOS[args.scenario]

    print(f"[{args.scenario}] downloading/loading DeepMIMO scenario '{scenario.name}' ...")
    import deepmimo as dm

    dm.download(scenario.name)
    dataset = dm.load(scenario.name)
    active_idxs = dataset.get_active_idxs()

    # Diagnostic: energy capture ratio on a small held-out sample (bounded memory -- mirrors
    # scripts/generate_dataset.py's synthetic-data diagnostic, which also uses a separate small
    # held-out batch rather than the full dataset).
    diag_rng = np.random.default_rng(args.seed + 9999)
    n_diag = min(1000, len(active_idxs))
    diag_idxs = diag_rng.choice(active_idxs, size=n_diag, replace=False)
    diag_h = load_deepmimo_channels(scenario, n_ant=args.n_ant, n_subcarriers=args.n_subcarriers, max_users=n_diag)
    ratio = energy_capture_ratio(diag_h, n_ant=args.n_ant, trunc=args.trunc)
    print(f"[{args.scenario}] energy capture ratio at trunc={args.trunc}: {ratio:.4f}")

    # Small buffer per split so normalize_unit_energy's (rare) zero-power drop doesn't leave us
    # short of the requested count.
    def buffered(n: int) -> int:
        return n + max(20, int(0.02 * n))

    if args.split == "blocked":
        positions = np.asarray(dataset.rx_pos)[active_idxs]
        band_idxs = blocked_spatial_split_indices(
            positions,
            n_train=buffered(args.n_train),
            n_val=buffered(args.n_val),
            n_test=buffered(args.n_test),
            seed=args.seed,
        )
        split_global_idxs = {name: active_idxs[pos] for name, pos in band_idxs.items()}
    elif args.split == "spatial":
        positions = np.asarray(dataset.rx_pos)[active_idxs]
        band_idxs = spatial_split_indices(
            positions,
            n_train=buffered(args.n_train),
            n_val=buffered(args.n_val),
            n_test=buffered(args.n_test),
            seed=args.seed,
        )
        split_global_idxs = {name: active_idxs[pos] for name, pos in band_idxs.items()}
    else:
        n_avail = len(active_idxs)
        band_idxs = random_split_indices(
            n_avail, buffered(args.n_train), buffered(args.n_val), buffered(args.n_test), seed=args.seed
        )
        split_global_idxs = {name: active_idxs[pos] for name, pos in band_idxs.items()}

    targets = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    for split_name, global_idxs in split_global_idxs.items():
        n_target = targets[split_name]
        if len(global_idxs) < n_target:
            print(
                f"[{args.scenario}/{split_name}] WARNING: only {len(global_idxs)} users available "
                f"in this region (requested {n_target}) -- using all of them"
            )

        data = build_real_split(
            scenario,
            global_idxs,
            n_ant=args.n_ant,
            n_subcarriers=args.n_subcarriers,
            trunc=args.trunc,
            chunk_size=args.chunk_size,
        )
        h, scale = data["h"][:n_target], data["scale"][:n_target]
        n_nan = int(np.sum(~np.isfinite(h)))
        print(
            f"[{args.scenario}/{split_name}] n={len(h)} h.shape={h.shape} "
            f"mean|h|={np.mean(np.abs(h)):.4f} std|h|={np.std(np.abs(h)):.4f} n_nonfinite={n_nan}"
        )
        out_path = os.path.join(args.out_dir, f"{args.scenario}_{split_name}.npz")
        save_split({"h": h, "scale": scale}, out_path)
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
