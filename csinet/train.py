"""Training loop for a single compression ratio (importable core, no CLI parsing)."""

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from csinet.dataset import CsiDataset
from csinet.metrics import nmse_db
from csinet.models.csinet import CsiNet
from csinet.transform import from_network_output


def _val_nmse_db(model: CsiNet, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    mse_sum, n_batches = 0.0, 0
    h_true_all, h_pred_all = [], []
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        for x, scale in loader:
            x = x.to(device)
            x_hat, _ = model(x)
            mse_sum += loss_fn(x_hat, x).item()
            n_batches += 1

            x_np = x.cpu().numpy()
            x_hat_np = x_hat.cpu().numpy()
            scale_np = scale.numpy()
            h_true_all.append(from_network_output(x_np, scale_np))
            h_pred_all.append(from_network_output(x_hat_np, scale_np))

    h_true = np.concatenate(h_true_all, axis=0)
    h_pred = np.concatenate(h_pred_all, axis=0)
    return mse_sum / n_batches, nmse_db(h_true, h_pred)


def train_one_cr(
    train_ds: CsiDataset,
    val_ds: CsiDataset,
    m: int,
    epochs: int = 200,
    batch_size: int = 200,
    lr: float = 1e-2,
    device: str | None = None,
    seed: int = 0,
    ckpt_path: str | None = None,
    n_refine: int = 2,
    encoder_width: tuple[int, int] = (8, 16),
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = CsiNet(m, n_refine=n_refine, encoder_width=encoder_width).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_mse": [], "val_nmse_db": []}
    best_val_nmse_db = float("inf")
    best_epoch = -1
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss_sum, n_batches = 0.0, 0
        for x, _scale in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, _ = model(x)
            loss = loss_fn(x_hat, x)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1

        val_mse, val_nmse = _val_nmse_db(model, val_loader, device)
        scheduler.step(val_mse)

        history["train_loss"].append(train_loss_sum / n_batches)
        history["val_mse"].append(val_mse)
        history["val_nmse_db"].append(val_nmse)

        if val_nmse < best_val_nmse_db:
            best_val_nmse_db = val_nmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if ckpt_path is not None and best_state is not None:
        torch.save(
            {"m": m, "n_refine": n_refine, "encoder_width": encoder_width, "state_dict": best_state}, ckpt_path
        )

    return {
        "history": history,
        "best_val_nmse_db": best_val_nmse_db,
        "best_epoch": best_epoch,
    }


def evaluate(model: CsiNet, dataset: CsiDataset, device: str, batch_size: int = 200) -> float:
    """Full-dataset NMSE(dB) via the same inversion + metrics path as validation."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    _, test_nmse = _val_nmse_db(model, loader, device)
    return test_nmse
