#!/usr/bin/env python
"""Produce the headline plots/table from saved metrics + checkpoints.

Example:
    uv run python scripts/make_plots.py --scenario indoor \
        --data-dir outputs/data --metrics-dir outputs/metrics --out-dir outputs
"""

import argparse
import glob
import json
import os
import re

import torch

from csinet.dataset import CsiDataset
from csinet.models.csinet import CsiNet
from csinet.transform import from_network_output
from csinet.viz import plot_heatmap_comparison, plot_nmse_vs_cr, save_comparison_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--data-dir", default="outputs/data")
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--checkpoints-dir", default="outputs/checkpoints")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    figures_dir = os.path.join(args.out_dir, "figures")
    tables_dir = os.path.join(args.out_dir, "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # --- Load network per-CR metrics ---
    # Match exactly "{scenario}_cr<number>.json" -- NOT "..._quant.json" or
    # "..._b<n>_qat.json", which live alongside these in the same directory and
    # share the same {"cr": ..., "test_nmse_db": ...} schema (easy to silently
    # glob-match and overwrite the real per-CR result with a quantization sweep's).
    cr_pattern = re.compile(rf"^{re.escape(args.scenario)}_cr[0-9.]+\.json$")
    network_results = {}
    cr_metric_paths = sorted(
        p
        for p in glob.glob(os.path.join(args.metrics_dir, f"{args.scenario}_cr*.json"))
        if cr_pattern.match(os.path.basename(p))
    )
    for path in cr_metric_paths:
        with open(path) as f:
            d = json.load(f)
        network_results[d["cr"]] = d["test_nmse_db"]

    # --- Load PCA metrics ---
    pca_path = os.path.join(args.metrics_dir, f"{args.scenario}_pca.json")
    with open(pca_path) as f:
        pca_raw = json.load(f)
    pca_results = {float(cr): v["nmse_db"] for cr, v in pca_raw.items()}

    # --- Load seed-variance metrics (optional; scripts/seed_variance.py) ---
    network_std = {}
    for path in glob.glob(os.path.join(args.metrics_dir, f"{args.scenario}_cr*_seeds.json")):
        with open(path) as f:
            d = json.load(f)
        network_std[d["cr"]] = d["std"]

    # --- NMSE vs CR plot ---
    plot_nmse_vs_cr(
        {"network": network_results, "pca": pca_results},
        scenario=args.scenario,
        out_path=os.path.join(figures_dir, f"nmse_vs_cr_{args.scenario}.png"),
        network_std=network_std or None,
    )

    # --- Comparison table ---
    rows = []
    for cr in sorted(network_results):
        m = round(cr * 2048)
        nn_nmse = network_results[cr]
        pca_nmse = pca_results.get(cr)
        gain = None if pca_nmse is None else pca_nmse - nn_nmse  # positive => network better
        rows.append(
            {
                "scenario": args.scenario,
                "cr": cr,
                "m": m,
                "nn_nmse_db": round(nn_nmse, 3),
                "nn_std_db": round(network_std[cr], 3) if cr in network_std else None,
                "pca_nmse_db": None if pca_nmse is None else round(pca_nmse, 3),
                "gain_db": None if gain is None else round(gain, 3),
            }
        )
    save_comparison_table(rows, os.path.join(tables_dir, f"comparison_{args.scenario}.csv"))

    # --- Heatmap comparison at the best (largest) CR ---
    best_cr = max(network_results)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(args.checkpoints_dir, f"{args.scenario}_cr{best_cr}.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = CsiNet(ckpt["m"], n_refine=ckpt["n_refine"], encoder_width=tuple(ckpt.get("encoder_width", (8, 16)))).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    test_ds = CsiDataset(os.path.join(args.data_dir, f"{args.scenario}_test.npz"))
    x, scale = test_ds[0]
    with torch.no_grad():
        x_hat, _ = model(x.unsqueeze(0).to(device))
    h_true = from_network_output(x.numpy(), scale.numpy())
    h_pred = from_network_output(x_hat.squeeze(0).cpu().numpy(), scale.numpy())

    plot_heatmap_comparison(
        h_true,
        h_pred,
        scenario=args.scenario,
        cr_label=f"CR={best_cr}",
        out_path=os.path.join(figures_dir, f"heatmap_comparison_{args.scenario}.png"),
    )

    print(f"[{args.scenario}] wrote figures to {figures_dir} and table to {tables_dir}")


if __name__ == "__main__":
    main()
