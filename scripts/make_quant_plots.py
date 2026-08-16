#!/usr/bin/env python
"""Plot NMSE vs. codeword bit-width (post-hoc + QAT overlay) and the combined
bits-vs-NMSE Pareto curve, from scripts/quantize_eval.py and scripts/qat_finetune.py's
saved metrics.

Example:
    uv run python scripts/make_quant_plots.py --scenario indoor \
        --metrics-dir outputs/metrics --out-dir outputs
"""

import argparse
import glob
import json
import os

from csinet.viz import plot_nmse_vs_bits, plot_pareto_bits_vs_nmse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    figures_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    results, unquantized = {}, {}
    for path in sorted(glob.glob(os.path.join(args.metrics_dir, f"{args.scenario}_cr*_quant.json"))):
        with open(path) as f:
            d = json.load(f)
        cr = d["cr"]
        results[cr] = {int(b): v for b, v in d["bits"].items()}
        unquantized[cr] = d["unquantized_nmse_db"]

    qat_results: dict[float, dict[int, float]] = {}
    for path in sorted(glob.glob(os.path.join(args.metrics_dir, f"{args.scenario}_cr*_b*_qat.json"))):
        with open(path) as f:
            d = json.load(f)
        cr, n_bits = d["cr"], d["n_bits"]
        qat_results.setdefault(cr, {})[n_bits] = d["test_nmse_db"]

    bits_out_path = os.path.join(figures_dir, f"nmse_vs_bits_{args.scenario}.png")
    plot_nmse_vs_bits(
        results, unquantized, scenario=args.scenario, out_path=bits_out_path, qat_results=qat_results or None
    )
    print(f"[{args.scenario}] wrote {bits_out_path}")

    pareto_out_path = os.path.join(figures_dir, f"pareto_bits_vs_nmse_{args.scenario}.png")
    plot_pareto_bits_vs_nmse(
        results, scenario=args.scenario, out_path=pareto_out_path, qat_results=qat_results or None
    )
    print(f"[{args.scenario}] wrote {pareto_out_path}")


if __name__ == "__main__":
    main()
