import numpy as np

from csinet.metrics import nmse_db


def test_perfect_reconstruction_is_very_negative():
    rng = np.random.default_rng(0)
    h = (rng.standard_normal((16, 32, 32)) + 1j * rng.standard_normal((16, 32, 32))).astype(np.complex64)
    assert nmse_db(h, h) < -100


def test_zero_baseline_is_exactly_zero_db():
    rng = np.random.default_rng(1)
    h = (rng.standard_normal((16, 32, 32)) + 1j * rng.standard_normal((16, 32, 32))).astype(np.complex64)
    h_pred = np.zeros_like(h)
    assert abs(nmse_db(h, h_pred) - 0.0) < 1e-6


def test_more_error_is_worse():
    rng = np.random.default_rng(2)
    h = (rng.standard_normal((16, 32, 32)) + 1j * rng.standard_normal((16, 32, 32))).astype(np.complex64)
    small_err = h + 0.01 * (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape))
    large_err = h + 0.5 * (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape))
    assert nmse_db(h, small_err) < nmse_db(h, large_err)


def test_works_on_flattened_shape():
    rng = np.random.default_rng(3)
    h = (rng.standard_normal((16, 32, 32)) + 1j * rng.standard_normal((16, 32, 32))).astype(np.complex64)
    h_pred = h + 0.1 * (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape))
    h_flat, h_pred_flat = h.reshape(16, -1), h_pred.reshape(16, -1)
    assert abs(nmse_db(h, h_pred) - nmse_db(h_flat, h_pred_flat)) < 1e-4
