#!/usr/bin/env python
"""Post-hoc codeword quantization sweep: how much NMSE is lost feeding back B bits/dim
instead of a full-precision codeword? Uses already-trained checkpoints -- no retraining.

Example:
    uv run python scripts/quantize_eval.py --scenario indoor \
        --cr 0.25 0.0625 0.03125 0.015625 --bits 1 2 3 4 5 6 8 \
        --data-dir outputs/data --checkpoints-dir outputs/checkpoints --out-dir outputs
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from csinet.dataset import CsiDataset
from csinet.metrics import nmse_db
from csinet.models.csinet import CsiNet
from csinet.quantize import codeword_clip_scale, uniform_quantize
from csinet.transform import from_network_output

AMBIENT_DIM = 2 * 32 * 32


def _encode_dataset(model: CsiNet, dataset: CsiDataset, device: str, batch_size: int = 200):
    """Returns (codewords, h_true) as numpy arrays over the full dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    codewords, h_true_all = [], []
    model.eval()
    with torch.no_grad():
        for x, scale in loader:
            x = x.to(device)
            codeword = model.encoder(x)
            codewords.append(codeword.cpu().numpy())
            h_true_all.append(from_network_output(x.cpu().numpy(), scale.numpy()))
    return np.concatenate(codewords, axis=0), np.concatenate(h_true_all, axis=0)


def _decode_codewords(model: CsiNet, codewords: np.ndarray, scale: np.ndarray, device: str, batch_size: int = 200):
    """Decode codewords (N,M) back to complex channel matrices (N,32,32) via the decoder."""
    h_pred_all = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(codewords), batch_size):
            stop = min(start + batch_size, len(codewords))
            cw = torch.from_numpy(codewords[start:stop]).to(device)
            x_hat = model.decoder(cw)
            h_pred_all.append(from_network_output(x_hat.cpu().numpy(), scale[start:stop]))
    return np.concatenate(h_pred_all, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--cr", type=float, nargs="+", required=True)
    parser.add_argument("--bits", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8])
    parser.add_argument("--data-dir", default="outputs/data")
    parser.add_argument("--checkpoints-dir", default="outputs/checkpoints")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics_dir = os.path.join(args.out_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    train_ds = CsiDataset(os.path.join(args.data_dir, f"{args.scenario}_train.npz"))
    test_ds = CsiDataset(os.path.join(args.data_dir, f"{args.scenario}_test.npz"))
    test_scale = test_ds.scale

    for cr in args.cr:
        m = round(cr * AMBIENT_DIM)
        ckpt_path = os.path.join(args.checkpoints_dir, f"{args.scenario}_cr{cr}.pt")
        ckpt = torch.load(ckpt_path, map_location=device)
        model = CsiNet(ckpt["m"], n_refine=ckpt["n_refine"], encoder_width=tuple(ckpt.get("encoder_width", (8, 16)))).to(device)
        model.load_state_dict(ckpt["state_dict"])

        train_codewords, _ = _encode_dataset(model, train_ds, device)
        clip_scale = codeword_clip_scale(train_codewords)

        test_codewords, h_true = _encode_dataset(model, test_ds, device)

        unquantized_h_pred = _decode_codewords(model, test_codewords, test_scale, device)
        unquantized_nmse_db = nmse_db(h_true, unquantized_h_pred)
        print(f"[{args.scenario}] cr={cr} m={m} clip_scale={clip_scale:.4f} unquantized_nmse_db={unquantized_nmse_db:.3f}")

        bit_results = {}
        for n_bits in args.bits:
            q_codewords = uniform_quantize(test_codewords, clip_scale, n_bits)
            h_pred = _decode_codewords(model, q_codewords, test_scale, device)
            nmse = nmse_db(h_true, h_pred)
            bit_results[n_bits] = nmse
            feedback_bits = n_bits * m
            print(f"  bits/dim={n_bits} (feedback={feedback_bits} bits/sample) nmse_db={nmse:.3f}")

        out_path = os.path.join(metrics_dir, f"{args.scenario}_cr{cr}_quant.json")
        with open(out_path, "w") as f:
            json.dump(
                {
                    "scenario": args.scenario,
                    "cr": cr,
                    "m": m,
                    "clip_scale": clip_scale,
                    "unquantized_nmse_db": unquantized_nmse_db,
                    "bits": bit_results,
                },
                f,
                indent=2,
            )
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
