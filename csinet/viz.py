"""Static PNG plots: NMSE vs compression ratio, true-vs-reconstructed heatmaps, comparison table."""

import csv

import matplotlib.pyplot as plt
import numpy as np

NETWORK_COLOR = "#2a78d6"
PCA_COLOR = "#eb6834"
# Categorical palette slots 1-4 (fixed order, validated colorblind-safe adjacent pairs).
CR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def plot_nmse_vs_cr(
    results: dict[str, dict[float, float]],
    scenario: str,
    out_path: str,
    network_std: dict[float, float] | None = None,
) -> None:
    """results = {'network': {cr: nmse_db, ...}, 'pca': {cr: nmse_db, ...}}.

    network_std (optional) = {cr: std_across_seeds, ...} -- drawn as error bars (+-1
    std) on the network series, when a multi-seed run is available (see
    scripts/seed_variance.py). PCA has no seed variance (deterministic given a fixed
    random_state) so it never gets error bars.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)

    for key, color, marker, label in [
        ("network", NETWORK_COLOR, "o", "CsiNet"),
        ("pca", PCA_COLOR, "s", "PCA baseline"),
    ]:
        crs = sorted(results[key])
        vals = [results[key][cr] for cr in crs]
        if key == "network" and network_std:
            yerr = [network_std.get(cr, 0.0) for cr in crs]
            ax.errorbar(
                crs, vals, yerr=yerr, color=color, marker=marker, linewidth=2, markersize=7,
                label=label, capsize=4, elinewidth=1,
            )
        else:
            ax.plot(crs, vals, color=color, marker=marker, linewidth=2, markersize=7, label=label)
        for cr, v in zip(crs, vals):
            ax.annotate(f"{v:.1f}", (cr, v), textcoords="offset points", xytext=(0, 8), fontsize=8, color=color)

    ax.set_xscale("log")
    ax.set_xlabel("Compression ratio (M / 2048)")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title(f"NMSE vs. compression ratio -- {scenario}")
    ax.grid(True, which="both", linestyle="-", linewidth=0.5, color="#dddddd")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_heatmap_comparison(
    h_true: np.ndarray, h_pred: np.ndarray, scenario: str, cr_label: str, out_path: str
) -> None:
    """h_true, h_pred: (32,32) complex, single sample. Plots |h| magnitude side by side."""
    mag_true = np.abs(h_true)
    mag_pred = np.abs(h_pred)
    vmin, vmax = 0.0, mag_true.max()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=150)
    im0 = axes[0].imshow(mag_true, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    axes[0].set_title("True |H|")
    axes[1].imshow(mag_pred, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    axes[1].set_title("Reconstructed |H|")
    for ax in axes:
        ax.set_xlabel("antenna index")
        ax.set_ylabel("delay tap")
    fig.colorbar(im0, ax=axes, shrink=0.85, label="magnitude")
    fig.suptitle(f"{scenario} -- {cr_label}")
    fig.savefig(out_path)
    plt.close(fig)


def plot_nmse_vs_bits(
    results: dict[float, dict[int, float]],
    unquantized: dict[float, float],
    scenario: str,
    out_path: str,
    qat_results: dict[float, dict[int, float]] | None = None,
) -> None:
    """results = {cr: {n_bits: nmse_db, ...}, ...} (post-hoc quantization);
    unquantized = {cr: full-precision nmse_db, ...};
    qat_results (optional) = {cr: {n_bits: nmse_db, ...}, ...} (quantization-aware fine-tuned).

    One line per CR (fixed categorical color order) for post-hoc quantization, plus a
    dashed horizontal reference line per CR at its unquantized NMSE, plus -- where
    available -- star markers for the QAT-recovered NMSE at the same (cr, bits), so the
    recovery is a direct visual comparison against the post-hoc circle at that point.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)

    for i, cr in enumerate(sorted(results, reverse=True)):
        color = CR_COLORS[i % len(CR_COLORS)]
        bits = sorted(results[cr])
        vals = [results[cr][b] for b in bits]
        ax.plot(bits, vals, color=color, marker="o", linewidth=2, markersize=6, label=f"CR={cr}")
        ax.axhline(unquantized[cr], color=color, linestyle="--", linewidth=1, alpha=0.6)

        if qat_results and cr in qat_results:
            qat_bits = sorted(qat_results[cr])
            qat_vals = [qat_results[cr][b] for b in qat_bits]
            ax.scatter(qat_bits, qat_vals, color=color, marker="*", s=160, edgecolors="black", linewidths=0.5, zorder=5)

    if qat_results:
        ax.scatter([], [], color="black", marker="*", s=160, edgecolors="black", label="QAT (same bits)")

    ax.set_xlabel("Bits per codeword element")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title(f"Quantized codeword NMSE vs. bit-width -- {scenario}")
    ax.grid(True, linestyle="-", linewidth=0.5, color="#dddddd")
    ax.legend(title="(dashed = unquantized)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_pareto_bits_vs_nmse(
    quant_results: dict[float, dict[int, float]],
    scenario: str,
    out_path: str,
    qat_results: dict[float, dict[int, float]] | None = None,
) -> None:
    """The actual deployable question: for a fixed total feedback budget (M x bits/element,
    summed over CR and bit-width jointly, not either alone), what's the best achievable
    NMSE? quant_results/qat_results = {cr: {n_bits: nmse_db, ...}, ...}.

    Plots every (cr, bits) combination as its own point (one line per CR, colored by the
    fixed categorical order; QAT points as stars where available), plus a dotted Pareto
    frontier -- the lower envelope over all points sorted by total bits -- highlighting
    which combinations are actually worth choosing versus dominated by a cheaper option.
    """
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

    all_points: list[tuple[float, float]] = []
    for i, cr in enumerate(sorted(quant_results, reverse=True)):
        color = CR_COLORS[i % len(CR_COLORS)]
        m = round(cr * 2048)
        bits = sorted(quant_results[cr])
        total_bits = [m * b for b in bits]
        vals = [quant_results[cr][b] for b in bits]
        ax.plot(total_bits, vals, color=color, marker="o", linewidth=1.5, markersize=5, alpha=0.85, label=f"CR={cr}")
        all_points.extend(zip(total_bits, vals))

        if qat_results and cr in qat_results:
            qat_bits = sorted(qat_results[cr])
            qat_total_bits = [m * b for b in qat_bits]
            qat_vals = [qat_results[cr][b] for b in qat_bits]
            ax.scatter(
                qat_total_bits, qat_vals, color=color, marker="*", s=140, edgecolors="black", linewidths=0.5, zorder=5
            )
            all_points.extend(zip(qat_total_bits, qat_vals))

    all_points.sort(key=lambda p: p[0])
    frontier_x, frontier_y = [], []
    running_min = float("inf")
    for x, y in all_points:
        if y < running_min:
            running_min = y
            frontier_x.append(x)
            frontier_y.append(y)
    ax.plot(frontier_x, frontier_y, color="#4a3aa7", linewidth=2, linestyle=":", label="Pareto frontier", zorder=4)

    if qat_results:
        ax.scatter([], [], color="black", marker="*", s=140, edgecolors="black", label="QAT point")

    ax.set_xscale("log")
    ax.set_xlabel("Total feedback bits per sample (M × bits/element)")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title(f"Accuracy vs. total feedback size -- {scenario}")
    ax.grid(True, which="both", linestyle="-", linewidth=0.5, color="#dddddd")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_comparison_table(rows: list[dict], out_path_csv: str) -> None:
    """rows: [{'scenario':..., 'cr':..., 'm':..., 'nn_nmse_db':..., 'pca_nmse_db':..., 'gain_db':...}, ...]"""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
