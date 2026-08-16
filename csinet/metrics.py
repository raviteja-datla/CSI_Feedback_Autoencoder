"""Canonical NMSE(dB) metric -- reused by training validation, PCA baseline, and reporting."""

import numpy as np


def nmse_db(h_true: np.ndarray, h_pred: np.ndarray, eps: float = 1e-12) -> float:
    """Per-sample-normalized NMSE, in dB.

    h_true, h_pred: complex ndarray (N, ...) -- one or more trailing dims per sample.
    NMSE_dB = 10*log10( mean_i[ ||h_true_i - h_pred_i||_F^2 / max(||h_true_i||_F^2, eps) ] )
    """
    axes = tuple(range(1, h_true.ndim))
    err_energy = np.sum(np.abs(h_true - h_pred) ** 2, axis=axes)
    true_energy = np.sum(np.abs(h_true) ** 2, axis=axes)
    ratio = err_energy / np.maximum(true_energy, eps)
    return float(10.0 * np.log10(np.mean(ratio)))
