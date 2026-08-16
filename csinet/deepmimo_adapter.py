"""Adapter: DeepMIMO ray-traced scenarios -> this project's H_freq convention.

Produces complex64 (N, n_subcarriers, n_ant) arrays, power-normalized per-sample to unit
average energy exactly like csinet.channel_model.generate_channel_batch, so downstream code
(to_angular_delay, per_sample_scale, save_split, CsiDataset, train.py, and every scripts/*.py
keyed off --scenario) needs zero changes to consume real data alongside synthetic data.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeepMIMOScenarioConfig:
    name: str  # DeepMIMO scenario handle, e.g. "asu_campus_3p5"
    label: str  # output filename prefix, e.g. "outdoor_real"
    bandwidth_hz: float = 10e6


REAL_SCENARIOS = {
    "outdoor_real": DeepMIMOScenarioConfig(name="asu_campus_3p5", label="outdoor_real"),
    "indoor_real": DeepMIMOScenarioConfig(name="i2_28b", label="indoor_real"),
}


def _channel_parameters(scenario: DeepMIMOScenarioConfig, n_ant: int, n_subcarriers: int):
    import deepmimo as dm

    ch_params = dm.ChannelParameters()
    ch_params.bs_antenna.shape = [n_ant, 1]
    ch_params.ue_antenna.shape = [1, 1]
    ch_params.ofdm.subcarriers = n_subcarriers
    # DeepMIMO only computes the subcarriers listed here (default: just [0]) -- must be set
    # explicitly to the full range or compute_channels silently returns a single-subcarrier
    # channel regardless of `ofdm.subcarriers` (verified empirically: raw.shape's last dim
    # stayed 1 until this was set).
    ch_params.ofdm.selected_subcarriers = np.arange(n_subcarriers)
    ch_params.ofdm.bandwidth = scenario.bandwidth_hz
    ch_params.freq_domain = True
    return ch_params


def load_deepmimo_channels(
    scenario: DeepMIMOScenarioConfig,
    n_ant: int = 32,
    n_subcarriers: int = 1024,
    max_users: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Download (if needed), load, filter to active users, compute channels -- for SMALL,
    bounded-size requests only (e.g. an energy-capture-ratio diagnostic on ~1000 users). A full
    1024-subcarrier, 32-antenna complex64 channel is ~256KB per sample; a full 10k+ split
    computed in one call here would need several GB of RAM for the raw array alone, before
    counting reshape/normalize's temporary copies. For an actual train/val/test split, use
    `build_real_split` instead, which processes users in memory-bounded chunks.

    max_users (optional): if given, a seeded random subset of that many active users is taken
    BEFORE calling compute_channels, rather than computing channels for every active user and
    discarding most of them. A random (not positional-prefix) subset avoids spatial bias, since
    DeepMIMO returns active indices in grid/raster order.

    Returns complex64 (min(n_active_users, max_users), n_subcarriers, n_ant) -- squeezed
    rx-antenna axis, raw (not yet normalized) DeepMIMO channel values. Imported lazily so the
    rest of the package never requires the (large, network-fetching) deepmimo dependency to be
    installed.
    """
    import deepmimo as dm

    dm.download(scenario.name)
    dataset = dm.load(scenario.name)
    active_idxs = dataset.get_active_idxs()
    if max_users is not None and max_users < len(active_idxs):
        rng = np.random.default_rng(seed)
        active_idxs = rng.choice(active_idxs, size=max_users, replace=False)
    dataset = dataset.subset(active_idxs)

    ch_params = _channel_parameters(scenario, n_ant, n_subcarriers)
    raw = dataset.compute_channels(ch_params)  # (n_users, 1, n_ant, n_subcarriers) complex64
    return reshape_and_squeeze(raw)


def build_real_split(
    scenario: DeepMIMOScenarioConfig,
    active_idxs: np.ndarray,
    n_ant: int = 32,
    n_subcarriers: int = 1024,
    trunc: int = 32,
    chunk_size: int = 500,
) -> dict:
    """Memory-bounded equivalent of csinet.dataset.build_split, sourcing from a DeepMIMO
    scenario instead of the synthetic generator. Processes `active_idxs` in chunks of
    `chunk_size`: computes each chunk's full (n_subcarriers-wide) channel, normalizes it, and
    immediately truncates to the angular-delay domain (csinet.transform.to_angular_delay) before
    moving to the next chunk -- so peak memory is bounded by chunk_size, not by the total
    requested sample count (a full-width chunk of 500 users is ~130MB; the full un-chunked
    equivalent for a 14k-sample split would be several GB, comfortably enough to exhaust RAM on
    a memory-constrained machine).

    Returns {'h': complex64 (n_valid, trunc, n_ant), 'scale': float32 (n_valid,)} -- same schema
    as csinet.dataset.build_split. n_valid <= len(active_idxs) since normalize_unit_energy drops
    any exactly-zero-power sample per chunk.
    """
    import deepmimo as dm

    from csinet.transform import per_sample_scale, to_angular_delay

    dm.download(scenario.name)
    dataset = dm.load(scenario.name)
    ch_params = _channel_parameters(scenario, n_ant, n_subcarriers)

    chunks = []
    for start in range(0, len(active_idxs), chunk_size):
        chunk_idxs = active_idxs[start : start + chunk_size]
        raw = dataset.subset(chunk_idxs).compute_channels(ch_params)
        h = normalize_unit_energy(reshape_and_squeeze(raw))
        chunks.append(to_angular_delay(h, n_ant=n_ant, trunc=trunc).astype(np.complex64))

    h_trunc = np.concatenate(chunks, axis=0)
    return {"h": h_trunc, "scale": per_sample_scale(h_trunc)}


def reshape_and_squeeze(raw: np.ndarray) -> np.ndarray:
    """(n_users, n_rx=1, n_tx=n_ant, n_subcarriers) -> (n_users, n_subcarriers, n_ant) complex64.

    Pure numpy -- independently unit-testable against a mock array of the documented DeepMIMO
    output shape, no real download needed.
    """
    assert raw.shape[1] == 1, f"expected single UE rx antenna, got shape {raw.shape}"
    h = raw[:, 0, :, :]  # (n_users, n_ant, n_subcarriers)
    h = np.transpose(h, (0, 2, 1))  # (n_users, n_subcarriers, n_ant)
    return h.astype(np.complex64)


def normalize_unit_energy(h: np.ndarray) -> np.ndarray:
    """Per-sample unit-average-energy normalization, identical convention to
    generate_channel_batch: h / sqrt(mean(|h|**2)) over the full (subcarriers, ant) matrix.

    Real DeepMIMO channel power varies enormously across users (path loss depends on distance
    to the BS), unlike the synthetic generator which normalizes at construction time -- without
    this step, PCA/NMSE comparisons would be dominated by which users happen to be close to the
    BS rather than by genuine angular-delay compressibility structure. Also drops any residual
    exactly-zero/non-finite-power sample as a second guard beyond DeepMIMO's own active-user
    filtering (a disconnected user has exactly zero energy -- no paths summed at all -- not
    merely a small one; real path-loss-attenuated power is legitimately tiny, e.g. ~1e-17 for a
    distant but perfectly valid user, so an absolute threshold like `power > 1e-12` would
    incorrectly discard most real users instead of only genuinely-blocked ones).
    """
    power = np.mean(np.abs(h) ** 2, axis=(1, 2))
    valid = np.isfinite(power) & (power > 0)
    h = h[valid]
    power = power[valid]
    return (h / np.sqrt(power)[:, None, None]).astype(np.complex64)


def random_split_indices(n_total: int, n_train: int, n_val: int, n_test: int, seed: int) -> dict:
    """Deterministic seeded shuffle-then-slice split over user indices.

    NOT recommended for DeepMIMO data -- kept only for comparison/reproducing the leakage issue
    documented in the README. DeepMIMO users sit on a dense spatial grid (e.g. 1cm spacing for
    the i2_28b indoor scenario), so a random permutation routinely places a "held-out" test point
    immediately next to several training points; both PCA and the network then partly interpolate
    between near-duplicate samples instead of generalizing, which inflated indoor_real's initial
    PCA result to an implausible -120dB. Use `blocked_spatial_split_indices` for actual reported
    results.
    """
    if n_train + n_val + n_test > n_total:
        raise ValueError(f"requested split {n_train}+{n_val}+{n_test} exceeds {n_total} valid users")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    return {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val : n_train + n_val + n_test],
    }


def spatial_split_indices(
    positions: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    axis: int | None = None,
    gap_fraction: float = 0.02,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> dict:
    """Single-region spatial holdout: train/val/test occupy physically disjoint, non-adjacent bands
    along one coordinate axis, separated by buffer zones -- so no held-out point sits next to a
    training point, unlike `random_split_indices` (see its docstring for why that matters here).

    NOT recommended for a geographically large/diverse scenario -- kept for comparison. One big
    contiguous test region (e.g. "the east half of a campus") can be a statistically different
    *environment* from the train region (different buildings, different distance/angle-to-BS
    distribution), not just a different sample of the same environment -- this produced a
    collapse to ~0dB NMSE for both PCA and the network on outdoor_real (asu_campus_3p5), a
    domain-shift artifact rather than a genuine compressibility measurement. Use
    `blocked_spatial_split_indices` for actual reported results.

    positions: (n_total, 2 or 3) array of (x, y[, z]) coordinates, in the same order/index space
    as whatever index array these results will be used to select from (typically DeepMIMO's
    active-user index array).

    axis (optional): which coordinate to split along; defaults to whichever of x/y has the larger
    extent (the "long" direction of the scene), so a narrow room or corridor splits along its
    length rather than its width.
    gap_fraction: width of each of the two buffer zones (excluded from every split), as a
    fraction of the axis's total extent.
    train_frac/val_frac: target share of the axis's extent allotted to train/val (test gets the
    remainder), by coordinate quantile -- not by point count, since point density can vary.

    Returns positional indices (into `positions`, 0..n_total-1), each a seeded random sample of
    up to the requested count drawn from within its band. If a band contains fewer than
    requested, every point in that band is returned (fewer than requested) rather than padding
    across a gap -- the caller should check the actual returned counts.
    """
    positions = np.asarray(positions)
    if axis is None:
        extent = positions[:, :2].max(axis=0) - positions[:, :2].min(axis=0)
        axis = int(np.argmax(extent))
    coord = positions[:, axis]

    t1 = np.quantile(coord, train_frac)
    t2 = np.quantile(coord, train_frac + val_frac)
    gap = gap_fraction * (coord.max() - coord.min())

    train_mask = coord <= t1 - gap / 2
    val_mask = (coord > t1 + gap / 2) & (coord <= t2 - gap / 2)
    test_mask = coord > t2 + gap / 2

    rng = np.random.default_rng(seed)

    def sample(mask: np.ndarray, n: int) -> np.ndarray:
        idx = np.flatnonzero(mask)
        if len(idx) <= n:
            return idx
        return rng.choice(idx, size=n, replace=False)

    return {
        "train": sample(train_mask, n_train),
        "val": sample(val_mask, n_val),
        "test": sample(test_mask, n_test),
    }


def blocked_spatial_split_indices(
    positions: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    n_blocks_per_axis: int = 12,
    holdout_fraction: float = 0.2,
) -> dict:
    """Recommended split for DeepMIMO data: many small blocks scattered across the whole area,
    rather than one contiguous region (`spatial_split_indices`) or a plain shuffle
    (`random_split_indices`).

    Tiles the 2D (x, y) extent into an `n_blocks_per_axis` x `n_blocks_per_axis` grid. A random
    `holdout_fraction` of blocks are held out (split between val/test); every block ADJACENT
    (sharing an edge or corner) to a held-out block is excluded entirely as a buffer, so a
    held-out point is never next to a training point. Because held-out blocks are scattered
    throughout the grid rather than concentrated in one region, train and held-out data cover the
    same overall geographic diversity (same buildings, same distance/angle-to-BS distribution) --
    fixing the domain-shift collapse `spatial_split_indices` produced on a geographically large
    scenario (outdoor_real / asu_campus_3p5 fell to ~0dB NMSE for both PCA and the network under
    a single big regional split; see README).

    positions: (n_total, 2 or 3) array of (x, y[, z]) coordinates, in the same order/index space
    as whatever index array these results will be used to select from.
    n_blocks_per_axis: grid resolution; higher = smaller, more numerous blocks (finer-grained
    scattering, but each block holds fewer points).
    holdout_fraction: target fraction of blocks assigned to val+test before buffer removal.

    Returns positional indices (into `positions`, 0..n_total-1), each a seeded random sample of
    up to the requested count. If a pool has fewer than requested (e.g. train pool shrunk by
    buffer removal), every available point is returned -- the caller should check actual counts.
    """
    positions = np.asarray(positions)
    xy = positions[:, :2]
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    block_size = np.maximum(maxs - mins, 1e-9) / n_blocks_per_axis

    block_idx = np.clip(np.floor((xy - mins) / block_size).astype(int), 0, n_blocks_per_axis - 1)
    block_keys = block_idx[:, 0] * n_blocks_per_axis + block_idx[:, 1]
    unique_blocks = np.unique(block_keys)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_blocks)
    n_holdout_blocks = max(1, int(round(holdout_fraction * len(unique_blocks))))
    holdout_blocks = set(shuffled[:n_holdout_blocks].tolist())

    def block_neighbors(key: int):
        bx, by = key // n_blocks_per_axis, key % n_blocks_per_axis
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = bx + dx, by + dy
                if 0 <= nx < n_blocks_per_axis and 0 <= ny < n_blocks_per_axis:
                    yield nx * n_blocks_per_axis + ny

    buffer_blocks = {nb for k in holdout_blocks for nb in block_neighbors(k) if nb not in holdout_blocks}
    train_blocks = {k for k in unique_blocks.tolist() if k not in holdout_blocks and k not in buffer_blocks}

    holdout_arr = rng.permutation(np.array(sorted(holdout_blocks)))
    val_share = n_val / max(n_val + n_test, 1)
    n_val_blocks = max(1, int(round(val_share * len(holdout_arr)))) if len(holdout_arr) > 1 else len(holdout_arr)
    val_blocks = set(holdout_arr[:n_val_blocks].tolist())
    test_blocks = set(holdout_arr[n_val_blocks:].tolist())

    def points_in(block_set: set) -> np.ndarray:
        return np.flatnonzero(np.isin(block_keys, list(block_set)))

    def sample(idx_pool: np.ndarray, n: int) -> np.ndarray:
        if len(idx_pool) <= n:
            return idx_pool
        return rng.choice(idx_pool, size=n, replace=False)

    return {
        "train": sample(points_in(train_blocks), n_train),
        "val": sample(points_in(val_blocks), n_val),
        "test": sample(points_in(test_blocks), n_test),
    }
