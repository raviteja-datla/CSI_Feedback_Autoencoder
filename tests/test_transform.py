import numpy as np

from csinet.transform import (
    energy_capture_ratio,
    from_network_output,
    per_sample_scale,
    to_angular_delay,
    to_network_input,
)


def test_to_angular_delay_is_energy_preserving_and_capture_ratio_monotonic():
    rng = np.random.default_rng(0)
    h_freq = (rng.standard_normal((10, 64, 8)) + 1j * rng.standard_normal((10, 64, 8))).astype(np.complex64)

    h_ad_full = to_angular_delay(h_freq, n_ant=8, trunc=64)
    energy_before = np.sum(np.abs(h_freq) ** 2, axis=(-2, -1))
    energy_after = np.sum(np.abs(h_ad_full) ** 2, axis=(-2, -1))
    np.testing.assert_allclose(energy_before, energy_after, rtol=1e-3)

    r_small = energy_capture_ratio(h_freq, n_ant=8, trunc=8)
    r_large = energy_capture_ratio(h_freq, n_ant=8, trunc=32)
    r_full = energy_capture_ratio(h_freq, n_ant=8, trunc=64)
    assert r_small <= r_large <= r_full
    assert np.isclose(r_full, 1.0, atol=1e-3)


def test_network_input_roundtrip_and_typical_scale():
    rng = np.random.default_rng(1)
    h = (rng.standard_normal((20, 32, 32)) + 1j * rng.standard_normal((20, 32, 32))).astype(np.complex64)
    scale = per_sample_scale(h)

    x = to_network_input(h, scale)
    assert x.shape == (20, 2, 32, 32)
    # RMS-based scale (unbounded output) -- typical entries near unit magnitude,
    # not collapsed into a narrow band the way a peak-based scale would.
    assert 0.5 < x.std() < 2.0

    h_rec = from_network_output(x, scale)
    np.testing.assert_allclose(h_rec, h, rtol=1e-4, atol=1e-4)
