import numpy as np

from csinet.channel_model import INDOOR, generate_channel_batch
from csinet.pca_baseline import flatten_complex, run_pca_baseline, unflatten_complex
from csinet.transform import to_angular_delay


def _make_h(n, seed):
    rng = np.random.default_rng(seed)
    h_freq = generate_channel_batch(INDOOR, n_samples=n, rng=rng)
    return to_angular_delay(h_freq, n_ant=32, trunc=32).astype(np.complex64)


def test_flatten_unflatten_roundtrip():
    h = _make_h(20, 0)
    v = flatten_complex(h)
    assert v.shape == (20, 2048)
    h_rec = unflatten_complex(v)
    np.testing.assert_allclose(h_rec, h, rtol=1e-5, atol=1e-5)


def test_pca_full_components_near_lossless():
    # PCA needs n_samples >= n_components for a full-rank fit; use a small ambient
    # dim's worth of components (2048 = 2*32*32) but keep the train set >= that size.
    train_h = _make_h(2100, 1)
    test_h = _make_h(50, 2)
    nmse = run_pca_baseline(train_h, test_h, m=2048)
    assert nmse < -40


def test_pca_more_components_is_better():
    train_h = _make_h(600, 3)
    test_h = _make_h(50, 4)
    nmse_small = run_pca_baseline(train_h, test_h, m=8)
    nmse_large = run_pca_baseline(train_h, test_h, m=300)
    assert nmse_large < nmse_small
