#!/usr/bin/env python
"""Train a CsiNet model per compression ratio, save checkpoints + metrics.

Example:
    uv run python scripts/train.py --scenario indoor \
        --cr 0.25 0.0625 0.03125 0.015625 --epochs 200 --batch-size 200 --lr 1e-2 \
        --data-dir outputs/data --out-dir outputs
"""

import argparse
import json
import os

import torch

from csinet.dataset import CsiDataset
from csinet.models.csinet import CsiNet
from csinet.train import evaluate, train_one_cr

AMBIENT_DIM = 2 * 32 * 32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--cr", type=float, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoder-width", type=int, nargs=2, default=[8, 16], metavar=("W1", "W2"))
    parser.add_argument("--data-dir", default="outputs/data")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()
    encoder_width = tuple(args.encoder_width)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    metrics_dir = os.path.join(args.out_dir, "metrics")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    train_path = os.path.join(args.data_dir, f"{args.scenario}_train.npz")
    val_path = os.path.join(args.data_dir, f"{args.scenario}_val.npz")
    test_path = os.path.join(args.data_dir, f"{args.scenario}_test.npz")

    train_ds = CsiDataset(train_path)
    val_ds = CsiDataset(val_path)
    test_ds = CsiDataset(test_path)

    for cr in args.cr:
        m = round(cr * AMBIENT_DIM)
        print(f"\n=== scenario={args.scenario} cr={cr} m={m} ===")
        ckpt_path = os.path.join(ckpt_dir, f"{args.scenario}_cr{cr}.pt")

        result = train_one_cr(
            train_ds,
            val_ds,
            m=m,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            seed=args.seed,
            ckpt_path=ckpt_path,
            encoder_width=encoder_width,
        )
        print(
            f"best_epoch={result['best_epoch']} best_val_nmse_db={result['best_val_nmse_db']:.3f}"
        )

        ckpt = torch.load(ckpt_path, map_location=device)
        model = CsiNet(ckpt["m"], n_refine=ckpt["n_refine"], encoder_width=tuple(ckpt.get("encoder_width", (8, 16)))).to(device)
        model.load_state_dict(ckpt["state_dict"])
        test_nmse_db = evaluate(model, test_ds, device, batch_size=args.batch_size)
        print(f"test_nmse_db={test_nmse_db:.3f}")

        metrics_path = os.path.join(metrics_dir, f"{args.scenario}_cr{cr}.json")
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "scenario": args.scenario,
                    "cr": cr,
                    "m": m,
                    "history": result["history"],
                    "best_epoch": result["best_epoch"],
                    "best_val_nmse_db": result["best_val_nmse_db"],
                    "test_nmse_db": test_nmse_db,
                },
                f,
                indent=2,
            )
        print(f"  saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
