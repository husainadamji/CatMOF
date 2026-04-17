"""Training and validation loop with early stopping."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_and_validate(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 4000,
    patience: int = 100,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[List[float], List[float]]:
    train_losses: List[float] = []
    val_losses: List[float] = []
    best_loss = float("inf")
    patience_counter = 0
    ckpt = checkpoint_path if checkpoint_path is not None else Path("best_model.pth")

    for epoch in range(epochs):
        model.train()
        epoch_train_loss: List[float] = []
        for batch in train_dataloader:
            xb, yb = batch
            xb = {k: v.to(device) for k, v in xb.items()}
            yb = yb.to(device)

            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            epoch_train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_losses.append(float(np.mean(epoch_train_loss)))

        model.eval()
        epoch_val_loss: List[float] = []
        with torch.no_grad():
            for batch in val_dataloader:
                xb, yb = batch
                xb = {k: v.to(device) for k, v in xb.items()}
                yb = yb.to(device)
                pred = model(xb)
                epoch_val_loss.append(torch.nn.functional.mse_loss(pred, yb).item())

        val_mean = float(np.mean(epoch_val_loss))
        val_losses.append(val_mean)

        print(f"epoch {epoch + 1}, train loss: {train_losses[-1]:.6f}, val loss: {val_mean:.6f}")
        # Early stopping logic
        if val_mean < best_loss:
            best_loss = val_mean
            patience_counter = 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"early stopping triggered after {epoch + 1} epochs.")
                break

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    return train_losses, val_losses
