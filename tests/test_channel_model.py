import numpy as np

from csinet.channel_model import INDOOR, OUTDOOR, generate_channel_batch
from csinet.transform import energy_capture_ratio


def test_generate_channel_batch_shape_and_power():
    rng = np.random.default_rng(0)
    h = generate_channel_batch(INDOOR, n_samples=200, n_ant=32, n_subcarriers=1024, rng=rng, chunk_size=64)
    assert h.shape == (200, 1024, 32)
    assert h.dtype == np.complex64
    assert np.all(np.isfinite(h))

    power = np.mean(np.abs(h) ** 2, axis=(1, 2))
    np.testing.assert_allclose(power, np.ones(200), rtol=1e-4)


def test_indoor_spreads_more_energy_beyond_truncation_than_outdoor():
    rng = np.random.default_rng(1)
    h_indoor = generate_channel_batch(INDOOR, n_samples=300, rng=rng)
    h_outdoor = generate_channel_batch(OUTDOOR, n_samples=300, rng=rng)

    ratio_indoor = energy_capture_ratio(h_indoor, n_ant=32, trunc=32)
    ratio_outdoor = energy_capture_ratio(h_outdoor, n_ant=32, trunc=32)

    assert 0.0 < ratio_indoor <= 1.0
    assert 0.0 < ratio_outdoor <= 1.0
    assert ratio_indoor < ratio_outdoor
