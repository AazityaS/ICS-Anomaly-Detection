"""GRU-specific benign validation error and TTNN calibration.

The existing TTNN architecture is reused without modification. The original
paper does not fully specify its supervised target/loss; this module follows
the project's documented next-MSE regression approximation and records that
assumption in the resulting checkpoint. GRU thresholds/checkpoints are saved
separately from WDNN artifacts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from daics.config import DATA, TTNN as TTNN_CONFIG, WINDOW
from daics.detection.threshold import (
    Thresholds,
    apply_median_filter,
    compute_base_threshold,
)
from daics.models.ttnn import TTNNEnsemble


def compute_gru_validation_mse(model, val_loader: DataLoader, device: torch.device):
    """Return chronological per-section MSE vectors, computed on benign val only."""
    if not isinstance(val_loader.sampler, SequentialSampler):
        raise ValueError("GRU validation MSE requires shuffle=False")
    model.eval()
    values = [[] for _ in DATA.mgse_per_section]
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            predictions = model(x)
            for section, prediction in enumerate(predictions):
                target = batch[section + 1].to(device)
                values[section].append(
                    torch.mean((prediction - target) ** 2, dim=(1, 2)).cpu().numpy()
                )
    if any(not section for section in values):
        raise ValueError("Validation loader produced no GRU prediction errors")
    return [np.concatenate(section).astype(np.float64, copy=False) for section in values]


def _next_mse_pairs(mse: np.ndarray, win: int):
    if len(mse) <= win:
        raise ValueError(f"Need more than {win} validation errors for TTNN calibration")
    windows = np.lib.stride_tricks.sliding_window_view(mse, win)
    return windows[:-1].copy(), mse[win:].copy()


def calibrate_gru_ttnn(
    model,
    val_loader: DataLoader,
    *,
    device: torch.device,
    checkpoint_dir: Path = Path("checkpoints/gru"),
    learning_rate: float = TTNN_CONFIG.learning_rate,
    batch_size: int = TTNN_CONFIG.batch_size,
    epochs: int = TTNN_CONFIG.tuning_epochs,
):
    """Fit separate TTNN weights and validation-derived thresholds for GRU.

    TTNN fitting uses only benign validation prediction errors. Next-MSE is an
    explicit approximation to an unspecified paper target, not paper-exact.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    mse_by_section = compute_gru_validation_mse(model, val_loader, device)
    train_pairs = [_next_mse_pairs(errors, WINDOW.Win) for errors in mse_by_section]
    ttnn = TTNNEnsemble().to(device)
    optimizer = optim.SGD(ttnn.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    history = []

    for epoch in range(epochs):
        ttnn.train()
        losses = []
        for section, (windows, targets) in enumerate(train_pairs):
            dataset = TensorDataset(
                torch.from_numpy(windows.astype(np.float32)).unsqueeze(1),
                torch.from_numpy(targets.astype(np.float32)).unsqueeze(1),
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            for inputs, target in loader:
                inputs, target = inputs.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = ttnn.forward_section(section, inputs)
                loss = criterion(prediction, target)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().item()))
        history.append(float(np.mean(losses)) if losses else float("nan"))

    t_base = compute_base_threshold(mse_by_section)
    ttnn.eval()
    section_thresholds = []
    validation_ttnn_outputs = []
    with torch.no_grad():
        for section, (windows, _) in enumerate(train_pairs):
            filtered = np.stack([
                apply_median_filter(row, TTNN_CONFIG.median_kernel_size) for row in windows
            ])
            outputs = []
            for start in range(0, len(filtered), batch_size):
                inputs = torch.from_numpy(filtered[start:start + batch_size].astype(np.float32))
                inputs = inputs.unsqueeze(1).to(device)
                outputs.append(ttnn.forward_section(section, inputs).reshape(-1).cpu().numpy())
            section_predictions = np.concatenate(outputs)
            validation_ttnn_outputs.append(section_predictions)
            section_thresholds.append(float(t_base + np.max(section_predictions)))

    thresholds = Thresholds(t_base=float(t_base), t_sections=section_thresholds)
    torch.save({
        "model_name": "GRU-specific TTNNEnsemble",
        "model_state_dict": ttnn.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epochs,
        "history": history,
        "target_assumption": "next_mse_regression_approximation",
        "training_source": "benign_validation_prediction_errors_only",
    }, checkpoint_dir / "ttnn_checkpoint.pt")
    import joblib
    joblib.dump(thresholds, checkpoint_dir / "thresholds.joblib")
    return ttnn, thresholds, {
        "train_loss": history,
        "validation_mse_per_section": mse_by_section,
        "validation_ttnn_outputs": validation_ttnn_outputs,
        "target_assumption": "next_mse_regression_approximation",
    }
