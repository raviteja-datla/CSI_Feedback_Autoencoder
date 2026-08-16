#!/usr/bin/env python
"""Quantization-aware fine-tuning sweep: continue training existing checkpoints with
quantization noise injected, to recover accuracy lost to post-hoc quantization
(scripts/quantize_eval.py). Starts from the already-trained full-precision checkpoint.

Example:
    uv run python scripts/qat_finetune.py --scenario indoor \
        --cr 0.25 0.0625 0.03125 0.015625 --bits 3 4 --epochs 50 --lr 1e-3 \
        --data-dir outputs/data --checkpoints-dir outputs/checkpoints --out-dir outputs
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from csinet.dataset import CsiDataset
from csinet.models.csinet import CsiNet
from csinet.qat import finetune_qat, quantized_nmse_db

AMBIENT_DIM = 2 * 32 * 32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--cr", type=float, nargs="+", required=True)
    parser.add_argument("--bits", type=int, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default="outputs/data")
    parser.add_argument("--checkpoints-dir", default="outputs/checkpoints")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = os.path.join(args.out_dir, "checkpoints_qat")
    metrics_dir = os.path.join(args.out_dir, "metrics")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    train_ds = CsiDataset(os.path.join(args.data_dir, f"{args.scenario}_train.npz"))
    val_ds = CsiDataset(os.path.join(args.data_dir, f"{args.scenario}_val.npz"))
    test_ds = CsiDataset(os.path.join(args.data_dir, f"{args.scenario}_test.npz"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    for cr in args.cr:
        m = round(cr * AMBIENT_DIM)
        base_ckpt_path = os.path.join(args.checkpoints_dir, f"{args.scenario}_cr{cr}.pt")

        for n_bits in args.bits:
            print(f"\n=== scenario={args.scenario} cr={cr} m={m} bits={n_bits} (QAT) ===")
            out_ckpt_path = os.path.join(ckpt_dir, f"{args.scenario}_cr{cr}_b{n_bits}.pt")

            result = finetune_qat(
                base_ckpt_path,
                train_ds,
                val_ds,
                n_bits=n_bits,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                seed=args.seed,
                out_ckpt_path=out_ckpt_path,
            )
            print(f"best_val_nmse_db={result['best_val_nmse_db']:.3f}")

            ckpt = torch.load(out_ckpt_path, map_location=device)
            model = CsiNet(ckpt["m"], n_refine=ckpt["n_refine"], encoder_width=tuple(ckpt.get("encoder_width", (8, 16)))).to(device)
            model.load_state_dict(ckpt["state_dict"])
            test_nmse_db = quantized_nmse_db(model, test_loader, ckpt["clip_scale"], n_bits, device)
            print(f"test_nmse_db={test_nmse_db:.3f}")

            metrics_path = os.path.join(metrics_dir, f"{args.scenario}_cr{cr}_b{n_bits}_qat.json")
            with open(metrics_path, "w") as f:
                json.dump(
                    {
                        "scenario": args.scenario,
                        "cr": cr,
                        "m": m,
                        "n_bits": n_bits,
                        "clip_scale": ckpt["clip_scale"],
                        "history": result["history"],
                        "best_val_nmse_db": result["best_val_nmse_db"],
                        "test_nmse_db": test_nmse_db,
                    },
                    f,
                    indent=2,
                )
            print(f"  saved -> {metrics_path}")


if __name__ == "__main__":
    main()
