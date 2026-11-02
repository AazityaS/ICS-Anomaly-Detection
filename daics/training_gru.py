"""Independent training path for the proposed GRU predictor."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from daics.config import DATA, SPLIT, WINDOW
from daics.gru_config import GRU_CONFIG, GRUTrainingConfig
from daics.models.gru import GRUPredictor, parameter_count


def set_gru_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sectionwise_mse_sum(predictions, targets) -> torch.Tensor:
    """Use the WDNN-compatible loss: sum of per-section mean MSE losses."""
    if len(predictions) != len(targets):
        raise ValueError("Prediction and target section counts differ")
    if not predictions:
        raise ValueError("At least one section is required")
    return sum(nn.functional.mse_loss(pred, target) for pred, target in zip(predictions, targets))


def _epoch_loss(model, loader, device, optimizer=None, gradient_clip_norm=1.0):
    training = optimizer is not None
    model.train(training)
    losses = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            x = batch[0].to(device, non_blocking=(device.type == "cuda"))
            targets = [batch[g + 1].to(device, non_blocking=(device.type == "cuda"))
                       for g in range(len(DATA.mgse_per_section))]
            if training:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(x)
            loss = sectionwise_mse_sum(predictions, targets)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            losses.append(float(loss.detach().item()))
    if not losses:
        raise ValueError("DataLoader produced no batches")
    return float(np.mean(losses))


def train_gru(
    model: GRUPredictor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    config: GRUTrainingConfig = GRU_CONFIG,
    device: Optional[torch.device] = None,
    checkpoint_path: Path = Path("checkpoints/gru/best_checkpoint.pt"),
    log_path: Path = Path("logs/gru/training.csv"),
    max_epochs: Optional[int] = None,
    patience: Optional[int] = None,
) -> dict:
    """Train with early stopping; only the best validation checkpoint is saved."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    num_epochs = config.max_epochs if max_epochs is None else max_epochs
    stop_patience = config.patience if patience is None else patience
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = Path(checkpoint_path)
    log_path = Path(log_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # This function starts a new run and writes a fresh epoch log.
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "best_val_loss"])
        writer.writeheader()
        for epoch in range(num_epochs):
            train_loss = _epoch_loss(model, train_loader, device, optimizer,
                                     config.gradient_clip_norm)
            val_loss = _epoch_loss(model, val_loader, device)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                torch.save({
                    "model_name": "GRUPredictor",
                    "epoch": best_epoch,
                    "best_val_loss": best_val,
                    "train_loss": train_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                    "config": config.to_dict(),
                    "split": {
                        "train_start": str(SPLIT.train_start),
                        "train_end_exclusive": str(SPLIT.train_end),
                        "validation_normal_only": True,
                    },
                    "loss_definition": "sum of three section-wise mean MSE losses",
                }, checkpoint_path)
            else:
                epochs_without_improvement += 1

            writer.writerow({"epoch": epoch + 1, "train_loss": train_loss,
                             "val_loss": val_loss, "best_val_loss": best_val})
            f.flush()
            if epochs_without_improvement >= stop_patience:
                break

    # Restore best validation weights in the returned model.
    best = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    history.update({"best_epoch": best_epoch, "best_val_loss": best_val,
                    "epochs_completed": len(history["train_loss"]),
                    "checkpoint": str(checkpoint_path),
                    "parameter_count": parameter_count(model),
                    "device": str(device)})
    return history


def make_gru_loaders(train_frame, val_frame, normalizer, batch_size=GRU_CONFIG.batch_size):
    """Build the existing source-local on-demand window datasets and loaders."""
    from daics.training import make_lazy_window_dataset

    if not train_frame["Normal/Attack"].eq("Normal").all():
        raise ValueError("GRU training data must contain only normal observations")
    if not val_frame["Normal/Attack"].eq("Normal").all():
        raise ValueError("GRU validation data must contain only normal observations")
    if not train_frame["Timestamp"].is_monotonic_increasing:
        raise ValueError("GRU train sequence must be chronological")
    if not val_frame["Timestamp"].is_monotonic_increasing:
        raise ValueError("GRU validation sequence must be chronological")
    train_ds = make_lazy_window_dataset(train_frame, normalizer)
    val_ds = make_lazy_window_dataset(val_frame, normalizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = {"batch_size": batch_size, "num_workers": 0,
              "pin_memory": device.type == "cuda"}
    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
    )


def main():
    """Train the separate GRU model with the existing approved split/scaler."""
    import joblib

    from daics.data.loader import load_raw_csv
    from daics.data.split import chronological_split
    from daics.models.gru import GRUPredictor

    root = Path(__file__).resolve().parents[1]
    normal = load_raw_csv(root / "data/processed/normal_clean.csv")
    attack = load_raw_csv(root / "data/processed/attack_clean.csv")
    split = chronological_split(normal, attack, build_test_combined=False)
    if not split.train["Timestamp"].max() < split.val["Timestamp"].min():
        raise ValueError("Train and validation split boundaries overlap")
    normalizer = joblib.load(root / "models/norm_params.joblib")
    train_loader, val_loader = make_gru_loaders(split.train, split.val, normalizer)
    set_gru_seed(GRU_CONFIG.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUPredictor()
    history = train_gru(
        model, train_loader, val_loader, config=GRU_CONFIG, device=device,
        checkpoint_path=root / "checkpoints/gru/best_checkpoint.pt",
        log_path=root / "logs/gru/training.csv",
    )
    (root / "logs/gru/training_summary.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps(history, indent=2), flush=True)


if __name__ == "__main__":
    main()
