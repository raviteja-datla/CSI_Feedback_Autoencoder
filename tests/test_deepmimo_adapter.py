import numpy as np
import pytest

from csinet.deepmimo_adapter import (
    blocked_spatial_split_indices,
    normalize_unit_energy,
    random_split_indices,
    reshape_and_squeeze,
    spatial_split_indices,
)


def test_reshape_and_squeeze_matches_documented_deepmimo_shape():
    # Mock DeepMIMO compute_channels() output: (n_users, n_rx=1, n_tx=n_ant, n_subcarriers)
    rng = np.random.default_rng(0)
    raw = (rng.standard_normal((50, 1, 32, 1024)) + 1j * rng.standard_normal((50, 1, 32, 1024))).astype(
        np.complex64
    )
    h = reshape_and_squeeze(raw)
    assert h.shape == (50, 1024, 32)
    assert h.dtype == np.complex64


def test_normalize_unit_energy_drops_zero_power_and_normalizes():
    rng = np.random.default_rng(1)
    h = (rng.standard_normal((10, 64, 8)) + 1j * rng.standard_normal((10, 64, 8))).astype(np.complex64)
    h[3] = 0.0  # simulate a disconnected/zero-power user DeepMIMO's active-user filter missed
    out = normalize_unit_energy(h)
    assert out.shape[0] == 9
    power = np.mean(np.abs(out) ** 2, axis=(1, 2))
    np.testing.assert_allclose(power, np.ones(9), rtol=1e-5)


def test_random_split_indices_disjoint_and_deterministic():
    idxs_a = random_split_indices(100, 60, 20, 20, seed=0)
    idxs_b = random_split_indices(100, 60, 20, 20, seed=0)
    assert set(idxs_a["train"]) | set(idxs_a["val"]) | set(idxs_a["test"]) == set(range(100))
    assert len(set(idxs_a["train"]) & set(idxs_a["val"])) == 0
    assert len(set(idxs_a["train"]) & set(idxs_a["test"])) == 0
    assert len(set(idxs_a["val"]) & set(idxs_a["test"])) == 0
    for k in idxs_a:
        np.testing.assert_array_equal(idxs_a[k], idxs_b[k])  # same seed -> same split


def test_random_split_indices_raises_when_oversubscribed():
    with pytest.raises(ValueError):
        random_split_indices(100, 60, 60, 60, seed=0)


def test_spatial_split_indices_disjoint_and_gapped():
    # 1D positions along a 100-unit corridor (x-axis), evenly spaced -- mimics a dense DeepMIMO grid.
    x = np.linspace(0, 100, 1000)
    positions = np.stack([x, np.zeros_like(x)], axis=1)
    result = spatial_split_indices(positions, n_train=400, n_val=100, n_test=100, seed=0, gap_fraction=0.05)

    train, val, test = result["train"], result["val"], result["test"]
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0

    # train sits entirely at smaller x than val, which sits entirely at smaller x than test (with gaps).
    assert x[train].max() < x[val].min()
    assert x[val].max() < x[test].min()


def test_spatial_split_indices_deterministic_and_respects_axis():
    x = np.linspace(0, 100, 500)
    y = np.linspace(0, 10, 500)  # narrower extent -- axis=None should pick x, not y
    positions = np.stack([x, y], axis=1)
    a = spatial_split_indices(positions, 200, 50, 50, seed=1)
    b = spatial_split_indices(positions, 200, 50, 50, seed=1)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    # split along the wide (x) axis, so test-set y values shouldn't be systematically separated
    assert x[a["train"]].max() < x[a["test"]].min()


def test_spatial_split_indices_returns_fewer_when_region_too_small():
    x = np.linspace(0, 100, 200)
    positions = np.stack([x, np.zeros_like(x)], axis=1)
    result = spatial_split_indices(positions, n_train=1000, n_val=10, n_test=10, seed=0)
    assert len(result["train"]) < 1000  # region can't supply 1000 of only 200 total points


def _dense_grid(n_per_axis=60, extent=100.0):
    x, y = np.meshgrid(np.linspace(0, extent, n_per_axis), np.linspace(0, extent, n_per_axis))
    return np.stack([x.ravel(), y.ravel()], axis=1)


def test_blocked_spatial_split_indices_disjoint_and_deterministic():
    positions = _dense_grid()
    a = blocked_spatial_split_indices(positions, n_train=1500, n_val=300, n_test=300, seed=0)
    b = blocked_spatial_split_indices(positions, n_train=1500, n_val=300, n_test=300, seed=0)
    assert len(set(a["train"]) & set(a["val"])) == 0
    assert len(set(a["train"]) & set(a["test"])) == 0
    assert len(set(a["val"]) & set(a["test"])) == 0
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_blocked_spatial_split_indices_no_near_neighbor_leakage():
    positions = _dense_grid(n_per_axis=60, extent=100.0)
    result = blocked_spatial_split_indices(positions, n_train=1500, n_val=300, n_test=300, seed=0, n_blocks_per_axis=10)
    train_xy = positions[result["train"]]
    held_out_xy = positions[np.concatenate([result["val"], result["test"]])]

    # block width along each axis at n_blocks_per_axis=10 over a 100-unit extent is 10 units --
    # buffer removal should guarantee every train point is at least one block width from any
    # held-out point (checked via a coarse block-size lower bound, not an exact distance).
    block_width = 100.0 / 10
    # nearest train->held_out distance, computed in chunks to stay memory-light
    min_dist = np.inf
    for i in range(0, len(train_xy), 200):
        chunk = train_xy[i : i + 200]
        d = np.sqrt(((chunk[:, None, :] - held_out_xy[None, :, :]) ** 2).sum(axis=2))
        min_dist = min(min_dist, d.min())
    assert min_dist >= block_width * 0.99  # allow tiny floating-point slack


def test_blocked_spatial_split_indices_scatters_holdout_across_whole_area():
    # Unlike spatial_split_indices (one contiguous region), held-out blocks should appear at
    # both low and high coordinate values, not concentrated at one end.
    positions = _dense_grid()
    result = blocked_spatial_split_indices(positions, n_train=1500, n_val=300, n_test=300, seed=0)
    held_out_x = positions[np.concatenate([result["val"], result["test"]])][:, 0]
    assert held_out_x.min() < 30.0
    assert held_out_x.max() > 70.0
