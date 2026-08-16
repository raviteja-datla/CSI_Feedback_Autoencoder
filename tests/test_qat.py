import os

import torch

from csinet.channel_model import INDOOR
from csinet.dataset import CsiDataset, build_split, save_split
from csinet.models.csinet import CsiNet
from csinet.qat import finetune_qat


def _make_dataset(tmp_path, name, n_samples, seed):
    data = build_split(INDOOR, n_samples=n_samples, seed=seed)
    path = os.path.join(tmp_path, f"{name}.npz")
    save_split(data, path)
    return CsiDataset(path)


def test_finetune_qat_runs_and_saves_checkpoint(tmp_path):
    train_ds = _make_dataset(tmp_path, "train", n_samples=64, seed=0)
    val_ds = _make_dataset(tmp_path, "val", n_samples=32, seed=1)

    m = 32
    model = CsiNet(m)
    base_ckpt_path = os.path.join(tmp_path, "base.pt")
    torch.save({"m": m, "n_refine": 2, "state_dict": model.state_dict()}, base_ckpt_path)

    out_ckpt_path = os.path.join(tmp_path, "qat.pt")
    result = finetune_qat(
        base_ckpt_path,
        train_ds,
        val_ds,
        n_bits=3,
        epochs=2,
        batch_size=16,
        device="cpu",
        out_ckpt_path=out_ckpt_path,
    )

    assert len(result["history"]["train_loss"]) == 2
    assert len(result["history"]["val_nmse_db"]) == 2
    assert result["best_val_nmse_db"] == min(result["history"]["val_nmse_db"])
    assert os.path.exists(out_ckpt_path)

    ckpt = torch.load(out_ckpt_path, map_location="cpu")
    assert ckpt["m"] == m
    assert ckpt["n_bits"] == 3
    assert ckpt["clip_scale"] > 0
