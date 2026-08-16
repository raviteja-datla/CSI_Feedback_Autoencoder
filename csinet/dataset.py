"""Dataset construction (raw channel -> truncated angular-delay split) and torch Dataset."""

import numpy as np
import torch
from torch.utils.data import Dataset

from csinet.channel_model import ScenarioConfig, generate_channel_batch
from csinet.transform import per_sample_scale, to_angular_delay, to_network_input


def build_split(
    scenario: ScenarioConfig,
    n_samples: int,
    seed: int,
    n_ant: int = 32,
    n_subcarriers: int = 1024,
    trunc: int = 32,
    chunk_size: int = 500,
) -> dict:
    """Generate a split: raw channels -> truncated angular-delay domain + per-sample scale.

    Returns {'h': complex64 (n_samples, trunc, n_ant), 'scale': float32 (n_samples,)}.
    """
    rng = np.random.default_rng(seed)
    H_freq = generate_channel_batch(
        scenario, n_samples, n_ant=n_ant, n_subcarriers=n_subcarriers, rng=rng, chunk_size=chunk_size
    )
    h = to_angular_delay(H_freq, n_ant=n_ant, trunc=trunc).astype(np.complex64)
    scale = per_sample_scale(h)
    return {"h": h, "scale": scale}


def save_split(data: dict, path: str) -> None:
    np.savez(path, h=data["h"], scale=data["scale"])


def load_split(path: str) -> dict:
    d = np.load(path)
    return {"h": d["h"], "scale": d["scale"]}


class CsiDataset(Dataset):
    """Wraps a saved split; each item is the network's RMS-scaled (2,32,32) input/target."""

    def __init__(self, npz_path: str):
        d = load_split(npz_path)
        self.h = d["h"]  # (N, 32, 32) complex64
        self.scale = d["scale"]  # (N,) float32

    def __len__(self) -> int:
        return len(self.h)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = to_network_input(self.h[idx], self.scale[idx])  # (2,32,32) float32, RMS-normalized
        return torch.from_numpy(x), torch.tensor(self.scale[idx], dtype=torch.float32)
