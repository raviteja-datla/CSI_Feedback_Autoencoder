"""Synthetic clustered (Saleh-Valenzuela-style) multipath MIMO-OFDM channel generator.

A real wireless channel is the superposition of a handful of physical propagation
paths, not i.i.d. noise. This module draws random clusters (physical scattering
neighborhoods) each containing several rays (sub-paths), and superposes them into
a frequency-domain MIMO channel matrix. This clustered structure is what makes the
channel compressible in the angular-delay domain -- plain Rayleigh fading would not
have this property and would defeat the entire premise of CSI feedback compression.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    n_clusters: int
    n_rays: int
    ray_aod_std_deg: float  # intra-cluster angular (Laplacian) spread
    aod_range_deg: tuple[float, float]
    cluster_delay_scale_bins: float  # mean cluster delay (Exp), in DFT-bin units
    ray_delay_scale_bins: float  # mean intra-cluster ray delay offset (Exp), in DFT-bin units
    cluster_power_decay: float  # PDP decay constant (bins) applied to cluster delay
    ray_power_decay: float  # PDP decay constant (bins) applied to ray delay offset


# Indoor: many, angularly/delay-wise spread clusters, with power decaying slowly
# relative to delay (long tail) -> harder to truncate losslessly at 32 taps.
INDOOR = ScenarioConfig(
    name="indoor",
    n_clusters=8,
    n_rays=10,
    ray_aod_std_deg=10.0,
    aod_range_deg=(-60.0, 60.0),
    cluster_delay_scale_bins=14.0,
    ray_delay_scale_bins=5.0,
    cluster_power_decay=60.0,
    ray_power_decay=20.0,
)

# Outdoor: few, angularly/delay-wise concentrated clusters, with power decaying
# quickly relative to delay -> nearly lossless truncation at 32 taps.
OUTDOOR = ScenarioConfig(
    name="outdoor",
    n_clusters=3,
    n_rays=6,
    ray_aod_std_deg=3.0,
    aod_range_deg=(-60.0, 60.0),
    cluster_delay_scale_bins=1.5,
    ray_delay_scale_bins=0.4,
    cluster_power_decay=1.2,
    ray_power_decay=0.6,
)

SCENARIOS = {"indoor": INDOOR, "outdoor": OUTDOOR}


def steering_vector(theta_deg: np.ndarray, n_ant: int) -> np.ndarray:
    """ULA steering vectors, half-wavelength spacing.

    theta_deg: (...,) array of angles-of-departure in degrees.
    Returns: (..., n_ant) complex64.
    """
    theta_rad = np.deg2rad(theta_deg)
    n = np.arange(n_ant)
    phase = np.pi * n * np.sin(theta_rad)[..., None]
    return np.exp(1j * phase).astype(np.complex64)


def generate_channel_batch(
    scenario: ScenarioConfig,
    n_samples: int,
    n_ant: int = 32,
    n_subcarriers: int = 1024,
    rng: np.random.Generator | None = None,
    chunk_size: int = 500,
) -> np.ndarray:
    """Generate a batch of frequency-domain MIMO-OFDM channels.

    Returns: H_freq, complex64 ndarray of shape (n_samples, n_subcarriers, n_ant),
    each sample power-normalized to unit average energy over the full matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    out = np.empty((n_samples, n_subcarriers, n_ant), dtype=np.complex64)
    k_idx = np.arange(n_subcarriers)

    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        b = stop - start

        cluster_aod = rng.uniform(*scenario.aod_range_deg, size=(b, scenario.n_clusters))
        cluster_delay = rng.exponential(scenario.cluster_delay_scale_bins, size=(b, scenario.n_clusters))

        ray_aod_offset = rng.laplace(0.0, scenario.ray_aod_std_deg, size=(b, scenario.n_clusters, scenario.n_rays))
        ray_delay_offset = rng.exponential(
            scenario.ray_delay_scale_bins, size=(b, scenario.n_clusters, scenario.n_rays)
        )

        ray_aod = cluster_aod[..., None] + ray_aod_offset
        ray_delay = cluster_delay[..., None] + ray_delay_offset
        ray_power = np.exp(-cluster_delay[..., None] / scenario.cluster_power_decay) * np.exp(
            -ray_delay_offset / scenario.ray_power_decay
        )
        ray_phase = rng.uniform(0.0, 2.0 * np.pi, size=(b, scenario.n_clusters, scenario.n_rays))
        ray_gain = np.sqrt(ray_power) * np.exp(1j * ray_phase)

        n_paths = scenario.n_clusters * scenario.n_rays
        ray_gain = ray_gain.reshape(b, n_paths).astype(np.complex64)
        ray_aod = ray_aod.reshape(b, n_paths)
        ray_delay = ray_delay.reshape(b, n_paths)

        steering = steering_vector(ray_aod, n_ant)  # (b, n_paths, n_ant)
        gain_steer = ray_gain[..., None] * steering  # (b, n_paths, n_ant)

        delay_phase = np.exp(
            -1j * 2.0 * np.pi * k_idx[None, None, :] * ray_delay[..., None] / n_subcarriers
        ).astype(np.complex64)  # (b, n_paths, n_subcarriers)

        h = np.einsum("npk,npa->nka", delay_phase, gain_steer, optimize=True).astype(np.complex64)

        power = np.mean(np.abs(h) ** 2, axis=(1, 2), keepdims=True)
        h = h / np.sqrt(power)

        out[start:stop] = h

    return out
