"""2D DFT to the angular-delay domain, truncation, and RMS-scaling for the network.

Convention (must stay consistent across the whole pipeline): the truncated
angular-delay matrix H_c is ground truth for everything downstream -- the network
target, the NMSE reference, and the PCA baseline input.
"""

import numpy as np
from scipy.linalg import dft


def unitary_dft_matrix(n: int) -> np.ndarray:
    """Energy-preserving (unitary) n x n DFT matrix."""
    return dft(n, scale="sqrtn").astype(np.complex64)


def _partial_unitary_idft_rows(n: int, n_rows: int) -> np.ndarray:
    """First n_rows rows of the unitary n x n INVERSE DFT matrix, without building the full matrix.

    A path with delay tau (in subcarrier-index units) was constructed in the
    frequency domain as exp(-j*2*pi*k*tau/n) (see channel_model.py) -- i.e. H_freq
    is already the forward-DFT of a delay-domain impulse train. Recovering that
    impulse (so energy concentrates near the true small delay, which is the whole
    point of truncating to few delay taps) requires the INVERSE transform
    (positive exponent), not another forward DFT.
    """
    d = np.arange(n_rows)[:, None]
    k = np.arange(n)[None, :]
    return (np.exp(2j * np.pi * d * k / n) / np.sqrt(n)).astype(np.complex64)


def to_angular_delay(H_freq: np.ndarray, n_ant: int, trunc: int = 32) -> np.ndarray:
    """Transform frequency-domain channel(s) to the truncated angular-delay domain.

    H_freq: (..., Nc, Nt) complex.
    Returns: (..., trunc, Nt) complex -- first `trunc` delay taps only.

    Only the `trunc` needed delay-domain rows are computed (not the full Nc x Nc
    IDFT matrix) -- this is the dominant cost when Nc >> trunc (e.g. 1024 vs 32).
    The antenna axis uses the same inverse-transform convention (Fa.conj().T is
    the unitary IDFT matrix, consistent with the steering-vector sign convention
    in channel_model.py).
    """
    n_subcarriers = H_freq.shape[-2]
    Fd_trunc = _partial_unitary_idft_rows(n_subcarriers, trunc)
    Fa = unitary_dft_matrix(n_ant)
    return np.einsum("dk,...ka,ab->...db", Fd_trunc, H_freq, Fa.conj().T, optimize=True)


def energy_capture_ratio(H_freq: np.ndarray, n_ant: int, trunc: int = 32) -> float:
    """Mean fraction of total Frobenius energy retained after angular-delay truncation.

    Diagnostic only -- used to validate that the indoor scenario spreads more energy
    beyond the truncation window than the outdoor scenario (indoor ratio < outdoor ratio).
    """
    H_ad_full = to_angular_delay(H_freq, n_ant=n_ant, trunc=H_freq.shape[-2])
    total_energy = np.sum(np.abs(H_ad_full) ** 2, axis=(-2, -1))
    trunc_energy = np.sum(np.abs(H_ad_full[..., :trunc, :]) ** 2, axis=(-2, -1))
    return float(np.mean(trunc_energy / total_energy))


def per_sample_scale(h_complex: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """(..., 32, 32) complex -> (...,) float32, s_i = max(eps, RMS(|h_i|)).

    RMS (not peak) magnitude: our clustered channel is sparse/heavy-tailed (a
    handful of dominant paths carry most of the energy -- peak/RMS ratio ~6-7x
    empirically), so normalizing by the peak squashes the typical entry down to a
    tiny sliver near the center of the input range, starving per-element MSE of
    gradient signal (verified: even an uncompressed CR=1 network then failed to
    beat the trivial zero-reconstruction baseline). RMS normalization keeps
    typical entries near unit scale; the rare large entries are simply large
    values for the (unbounded, no-Sigmoid) decoder to regress, not a source of
    saturation/clipping loss.
    """
    rms = np.sqrt(np.mean(np.abs(h_complex) ** 2, axis=(-2, -1)))
    return np.maximum(rms, eps).astype(np.float32)


def to_network_input(h_complex: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """(..., 32, 32) complex, (...,) float -> (..., 2, 32, 32) float32, real/imag of h/scale.

    Unbounded (no [0,1] shift) -- pairs with the decoder's linear (no Sigmoid)
    output layer. Stacked [real, imag] on a new axis before the trailing two
    spatial dims.
    """
    normalized = h_complex / scale[..., None, None]
    return np.stack([normalized.real, normalized.imag], axis=-3).astype(np.float32)


def from_network_output(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Inverse of to_network_input: (..., 2, 32, 32) float, (...,) scale -> (..., 32, 32) complex."""
    normalized = x[..., 0, :, :] + 1j * x[..., 1, :, :]
    return normalized * scale[..., None, None]
