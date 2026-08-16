"""Quantization-aware fine-tuning: continue training an already-trained checkpoint with
the codeword quantized (straight-through estimator) in the forward pass, so the decoder
learns to compensate for quantization noise instead of only encountering it at inference
time. Post-hoc quantization (quantize.py + scripts/quantize_eval.py) quantizes a model
that was never trained to expect it; this trains the model to expect it.
"""

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from csinet.dataset import CsiDataset
from csinet.metrics import nmse_db
from csinet.models.csinet import CsiNet
from csinet.quantize import codeword_clip_scale, quantize_ste
from csinet.transform import from_network_output


def _encode_all(model: CsiNet, dataset: CsiDataset, device: str, batch_size: int = 200) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    codewords = []
    model.eval()
    with torch.no_grad():
        for x, _scale in loader:
            codewords.append(model.encoder(x.to(device)).cpu().numpy())
    return np.concatenate(codewords, axis=0)


def quantized_nmse_db(model: CsiNet, loader: DataLoader, clip_scale: float, n_bits: int, device: str) -> float:
    """NMSE(dB) with the codeword quantized in the forward pass -- the deployed-system metric."""
    model.eval()
    h_true_all, h_pred_all = [], []
    with torch.no_grad():
        for x, scale in loader:
            x = x.to(device)
            codeword = model.encoder(x)
            q = quantize_ste(codeword, clip_scale, n_bits)
            x_hat = model.decoder(q)
            h_true_all.append(from_network_output(x.cpu().numpy(), scale.numpy()))
            h_pred_all.append(from_network_output(x_hat.cpu().numpy(), scale.numpy()))
    h_true = np.concatenate(h_true_all, axis=0)
    h_pred = np.concatenate(h_pred_all, axis=0)
    return nmse_db(h_true, h_pred)


def finetune_qat(
    ckpt_path: str,
    train_ds: CsiDataset,
    val_ds: CsiDataset,
    n_bits: int,
    epochs: int = 50,
    batch_size: int = 200,
    lr: float = 1e-3,
    device: str | None = None,
    seed: int = 0,
    out_ckpt_path: str | None = None,
) -> dict:
    """Fine-tune a trained checkpoint with quantization noise injected at n_bits.

    clip_scale is fixed once, from the starting (pre-fine-tune) model's codeword
    statistics on the training set -- matching how a real system fixes its quantizer
    range in advance, and keeping the target stable across fine-tuning epochs.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    ckpt = torch.load(ckpt_path, map_location=device)
    encoder_width = tuple(ckpt.get("encoder_width", (8, 16)))
    model = CsiNet(ckpt["m"], n_refine=ckpt["n_refine"], encoder_width=encoder_width).to(device)
    model.load_state_dict(ckpt["state_dict"])

    train_codewords = _encode_all(model, train_ds, device, batch_size)
    clip_scale = codeword_clip_scale(train_codewords)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_nmse_db": []}
    best_val_nmse_db = float("inf")
    best_state = None

    for _epoch in range(epochs):
        model.train()
        loss_sum, n_batches = 0.0, 0
        for x, _scale in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            codeword = model.encoder(x)
            q = quantize_ste(codeword, clip_scale, n_bits)
            x_hat = model.decoder(q)
            loss = loss_fn(x_hat, x)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n_batches += 1

        val_nmse = quantized_nmse_db(model, val_loader, clip_scale, n_bits, device)
        history["train_loss"].append(loss_sum / n_batches)
        history["val_nmse_db"].append(val_nmse)

        if val_nmse < best_val_nmse_db:
            best_val_nmse_db = val_nmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if out_ckpt_path is not None and best_state is not None:
        torch.save(
            {
                "m": ckpt["m"],
                "n_refine": ckpt["n_refine"],
                "encoder_width": encoder_width,
                "state_dict": best_state,
                "clip_scale": clip_scale,
                "n_bits": n_bits,
            },
            out_ckpt_path,
        )

    return {"history": history, "best_val_nmse_db": best_val_nmse_db, "clip_scale": clip_scale}
