"""
DAICS Training Pipeline
End-to-end training and evaluation of WDNN + TTNN.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, Dataset
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import joblib

from daics.config import WDNN as WDNNConfig, TTNN, WINDOW, DATA, SPLIT
from daics.data.loader import load_normal, load_attack, load_raw_csv, FEATURE_COLUMNS, SENSOR_COLUMNS, ACTUATOR_COLUMNS
from daics.data.split import chronological_split, get_feature_label_arrays
from daics.data.preprocessing import (
    prepare_training_data, create_windows_for_section, combine_test_windows,
    WindowDataset, fit_normalizer, normalize_features
)
from daics.data.actuator_db import build_actuator_database, save_actuator_database, load_actuator_database
from daics.data.sensor_groups import SENSOR_GROUPS
from daics.models.wdnn import WDNN
from daics.models.ttnn import TTNNEnsemble
from daics.detection.threshold import (
    compute_thresholds_from_validation,
    detect_anomalies,
    Thresholds,
    get_actuator_anomaly_scores,
    apply_median_filter,
    historical_thresholds_for_sequence
)
from daics.evaluation.metrics import evaluate, print_evaluation


def prepare_section_data(windowed_data, section_idx: int):
    """Extract data for a specific section"""
    X, y_section, timestamps, labels = create_windows_for_section(windowed_data, section_idx)
    return X, y_section, timestamps, labels


class JointSectionDataset(Dataset):
    """Expose on-demand windows plus target-time actuator state and label."""
    def __init__(self, window_dataset: WindowDataset):
        self.window_dataset = window_dataset
        self.sections = window_dataset.mgse_per_section

    def __len__(self):
        return len(self.window_dataset)

    def __getitem__(self, idx):
        sample = self.window_dataset[idx]
        y = sample['y']
        targets = []
        start = 0
        for count in self.sections:
            targets.append(y[start:start + count])
            start += count
        target_idx = idx + self.window_dataset.Win + self.window_dataset.H
        actuator_state = self.window_dataset.X[target_idx, -DATA.n_actuators:]
        return (sample['X'], *targets,
                torch.as_tensor(actuator_state.copy(), dtype=torch.float32),
                torch.as_tensor(sample['label'], dtype=torch.long))


def make_lazy_window_dataset(frame, normalizer) -> JointSectionDataset:
    """Create source-local windows on demand without storing overlapping windows."""
    X = normalize_features(frame, normalizer)
    y_raw = frame[SENSOR_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    # Transform once per split instead of invoking sklearn for every window.
    y_sensors = normalizer.sensor_scaler.transform(y_raw).astype(np.float32, copy=False)
    labels = (frame['Normal/Attack'] == 'Attack').to_numpy(dtype=np.int32)
    base = WindowDataset(
        X, y_sensors, labels, frame['Timestamp'], normalizer=None
    )
    return JointSectionDataset(base)


def create_dataloaders(
    X: np.ndarray,
    y_sections: List[np.ndarray],
    batch_size: int = 32,
    shuffle: bool = True
) -> List[DataLoader]:
    """Create dataloaders for each section"""
    dataloaders = []
    for y_section in y_sections:
        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(y_section)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        dataloaders.append(loader)
    return dataloaders


def create_joint_dataloader(
    X: np.ndarray,
    y_sections: List[np.ndarray],
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    """Create a single dataloader with all section targets for TTNN training"""
    tensors = [torch.FloatTensor(X)] + [torch.FloatTensor(y) for y in y_sections]
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_wdnn(
    wdnn: WDNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 0.01,
    device: torch.device = torch.device('cpu'),
    checkpoint_path: Path = None,
    log_path: Path = None,
) -> Dict:
    """Train WDNN from joint, on-demand window batches."""
    wdnn = wdnn.to(device)
    optimizer = optim.SGD(wdnn.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': []}
    start_epoch = 0
    if checkpoint_path is not None and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        wdnn.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        history = checkpoint.get('history', history)
        start_epoch = int(checkpoint['epoch'])
        print(f"Resuming WDNN from checkpoint epoch {start_epoch}.", flush=True)

    for epoch in range(start_epoch, epochs):
        # Training
        wdnn.train()
        train_losses = []

        for batch in train_loader:
            X_batch = batch[0].to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = wdnn(X_batch)  # List of (batch, mgse, Wout)

            loss = 0
            for g in range(WDNNConfig.G):
                target = batch[g + 1].to(device, non_blocking=True)
                loss += criterion(outputs[g], target)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        wdnn.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                X_batch = batch[0].to(device, non_blocking=True)
                outputs = wdnn(X_batch)

                loss = 0
                for g in range(WDNNConfig.G):
                    target = batch[g + 1].to(device, non_blocking=True)
                    loss += criterion(outputs[g], target)

                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)

        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': wdnn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train,
                'val_loss': avg_val,
                'history': history,
                'config': {'epochs': epochs, 'learning_rate': lr, 'batch_size': WDNNConfig.batch_size},
            }, checkpoint_path)

        if log_path is not None:
            import csv
            log_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not log_path.exists()
            with log_path.open('a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss'])
                if write_header:
                    writer.writeheader()
                writer.writerow({'epoch': epoch + 1, 'train_loss': avg_train, 'val_loss': avg_val})

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={avg_train:.6f}, val_loss={avg_val:.6f}")

    return history


def compute_mse_sequences(
    wdnn: WDNN,
    data_loader,
    device: torch.device,
    Win: int = WINDOW.Win,
    max_windows: int = None
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Compute MSE sequences for TTNN training.
    For each section, collects MSE per window, then creates sliding windows of length Win.

    Args:
        data_loader: Non-shuffled DataLoader with (X, y_section0, y_section1, ..., y_sectionG, labels)
    Returns: Tuple of (sequences, targets)
        sequences: List of G tensors, each (n_sequences, 1, Win) - MSE sequences for TTNN input
        targets: List of G tensors, each (n_sequences, 1) - next MSE value (target for TTNN)
    """
    if not isinstance(data_loader.sampler, SequentialSampler):
        raise ValueError("MSE time series require a DataLoader with shuffle=False")

    wdnn.eval()
    all_mse_per_section = [[] for _ in range(WDNNConfig.G)]

    with torch.no_grad():
        for batch in data_loader:
            X_batch = batch[0].to(device)

            wdnn_outputs = wdnn(X_batch)

            for g, pred in enumerate(wdnn_outputs):
                target = batch[g + 1].to(device)  # Targets are at positions 1..G
                mse = torch.mean((pred - target) ** 2, dim=(1, 2)).cpu().numpy()
                all_mse_per_section[g].extend(mse)

    # Convert to tensors and create sliding windows of length Win
    sequences_list = []
    targets_list = []
    for g in range(WDNNConfig.G):
        mse_array = np.array(all_mse_per_section[g])
        if max_windows is not None and len(mse_array) > max_windows:
            mse_array = mse_array[-max_windows:]

        # Create sliding windows of length Win
        # Sequence i: MSE[t-Win+1 ... t], Target i: MSE[t+1]
        n_sequences = len(mse_array) - Win
        if n_sequences <= 0:
            sequences_list.append(torch.zeros((0, 1, Win), device=device))
            targets_list.append(torch.zeros((0, 1), device=device))
            continue

        sequences = []
        targets = []
        for i in range(n_sequences):
            seq = mse_array[i:i+Win]
            target = mse_array[i+Win]  # Next MSE value
            sequences.append(seq)
            targets.append(target)

        seq_tensor = torch.FloatTensor(np.array(sequences)).unsqueeze(1).to(device)  # (n_seq, 1, Win)
        tgt_tensor = torch.FloatTensor(np.array(targets)).unsqueeze(1).to(device)  # (n_seq, 1)
        sequences_list.append(seq_tensor)
        targets_list.append(tgt_tensor)

    return sequences_list, targets_list


def train_ttnn(
    ttnn_ensemble: TTNNEnsemble,
    wdnn: WDNN,
    val_loader,    # Benign validation data is TTNN's training set per paper
    epochs: int = TTNN.tuning_epochs,
    lr: float = 0.01,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Tune TTNN from benign validation prediction errors. The paper specifies
    this training set, length-Win inputs, and SGD, but not the target/loss.
    Next-MSE regression is an explicit approximation, not paper-exact.
    """
    ttnn_ensemble = ttnn_ensemble.to(device)
    wdnn = wdnn.to(device)
    wdnn.eval()  # Freeze WDNN

    optimizer = optim.SGD(ttnn_ensemble.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': [], 'target_assumption': 'next_mse_forecasting'}

    # Validation-only, chronological MSE series; never use WDNN training errors.
    print("  Computing benign validation MSE sequences for TTNN tuning (non-shuffled)...")
    tuning_sequences, tuning_targets = compute_mse_sequences(wdnn, val_loader, device)

    for epoch in range(epochs):
        ttnn_ensemble.train()
        train_losses = []

        # One paper-specified SGD tuning epoch over validation-derived errors.
        for g in range(WDNNConfig.G):
            mse_seq = tuning_sequences[g]
            mse_tgt = tuning_targets[g]
            if len(mse_seq) == 0:
                continue

            # Batch the sequences
            n_seq = mse_seq.shape[0]
            for i in range(0, n_seq, TTNN.batch_size):
                batch_seq = mse_seq[i:i+TTNN.batch_size]
                batch_tgt = mse_tgt[i:i+TTNN.batch_size]

                optimizer.zero_grad()

                # TTNN forward
                out = ttnn_ensemble.forward_section(g, batch_seq)  # (batch, 1)

                # Assumption: next-MSE target (the paper omits target/loss).
                loss = criterion(out, batch_tgt)

                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

        avg_train = np.mean(train_losses) if train_losses else 0
        history['train_loss'].append(avg_train)
        print(f"  Epoch {epoch+1}/{epochs}: validation-set tuning_loss={avg_train:.6f}")

    return history


def evaluate_model(
    wdnn: WDNN,
    ttnn_ensemble: TTNNEnsemble,
    thresholds: Thresholds,
    actuator_db,
    test_data_loader,
    device: torch.device
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Evaluate full model on test set.

    Pipeline: WDNN -> MSE per section -> MSE sequence (length Win) -> TTNN -> Dynamic Threshold -> Detection

    Algorithm 1 per batch:
    1. For each batch of s samples, get MSE sequences of length Win
    2. Apply median filter to MSE sequences
    3. TTNN predicts on filtered sequences
    4. T_g = T_base + max(TTNN_predictions)
    5. Compare MSE to T_g
    6. Apply Wanom consecutive detection

    Returns:
        final_anomalies: (n_windows,) boolean - final anomaly decisions
        section_anomalies: List of G arrays (n_windows,) - per-section anomalies
    """
    wdnn.eval()
    ttnn_ensemble.eval()
    if not isinstance(test_data_loader.sampler, SequentialSampler):
        raise ValueError("Test MSE sequences require a DataLoader with shuffle=False")

    all_mse_per_section = [[] for _ in range(WDNNConfig.G)]
    all_labels = []
    all_actuator_states = []

    with torch.no_grad():
        for batch in test_data_loader:
            X_batch = batch[0].to(device)

            # Extract section targets from batch
            y_sections = [batch[g+1].to(device) for g in range(WDNNConfig.G)]

            # WDNN forward
            wdnn_outputs = wdnn(X_batch)

            # Compute MSE per section
            for g, (pred, target) in enumerate(zip(wdnn_outputs, y_sections)):
                mse = torch.mean((pred - target) ** 2, dim=(1, 2)).cpu().numpy()
                all_mse_per_section[g].extend(mse)

            # Labels
            all_labels.extend(batch[-1].cpu().numpy())

            # Actuator tuple at the timestamp being evaluated. JointSectionDataset
            # supplies target-time states; legacy batches fall back to the input
            # window end for compatibility with old smoke fixtures.
            if len(batch) >= WDNNConfig.G + 3:
                actuator_states = batch[-2].cpu().numpy().astype(int)
            else:
                actuator_states = X_batch[:, -DATA.n_actuators:, -1].cpu().numpy().astype(int)
            all_actuator_states.append(actuator_states)

    # Convert to arrays
    mse_per_section = [np.array(m) for m in all_mse_per_section]
    labels = np.array(all_labels)
    actuator_states = np.vstack(all_actuator_states)

    # Query actuator database for anomaly scores
    actuator_anomaly_scores = get_actuator_anomaly_scores(actuator_db, actuator_states)

    # Compute thresholds from the same independently sourced sequence as the
    # current test loader. The helper uses Algorithm 1's t0=tbar-H-Win.
    t_base = thresholds.t_base
    batch_size = WDNNConfig.batch_size  # s=32

    ttnn_predictions_per_section = []
    dynamic_thresholds_per_section = []
    for g in range(WDNNConfig.G):
        thresholds_g, predictions_g = historical_thresholds_for_sequence(
            mse_per_section[g], t_base, ttnn_ensemble, g, device,
            batch_size=batch_size,
        )
        dynamic_thresholds_per_section.append(thresholds_g)
        ttnn_predictions_per_section.append(predictions_g)

    # Detect anomalies using dynamic thresholds
    final_anomalies, section_anomalies = detect_anomalies(
        mse_per_section,
        t_base,
        ttnn_predictions_per_section,
        actuator_anomaly_scores,
        wanom=WINDOW.Wanom,
        batch_size=batch_size,
        dynamic_thresholds_per_section=dynamic_thresholds_per_section,
    )

    return final_anomalies, section_anomalies


def main():
    """Train and validate WDNN using on-demand windows; do not test here."""
    import gc
    import psutil

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=== DAICS WDNN Baseline Training ===", flush=True)
    print(f"Device: {device}", flush=True)

    processed_dir = Path("data/processed")
    if (processed_dir / "normal_clean.csv").is_file() and (processed_dir / "attack_clean.csv").is_file():
        normal = load_raw_csv(processed_dir / "normal_clean.csv")
        attack = load_raw_csv(processed_dir / "attack_clean.csv")
    else:
        normal = load_normal(Path("data/SWaT"))
        attack = load_attack(Path("data/SWaT"))

    split = chronological_split(normal, attack, build_test_combined=False)
    if not (split.train['Timestamp'].max() < split.val['Timestamp'].min()
            <= split.val['Timestamp'].max() < split.test_normal['Timestamp'].min()):
        raise ValueError("Normal train/validation/test boundaries are not strictly ordered")
    if not (split.train['Normal/Attack'].eq('Normal').all()
            and split.val['Normal/Attack'].eq('Normal').all()):
        raise ValueError("Attack-labelled records found in WDNN train/validation split")
    if split.test_attack is split.test_normal:
        raise ValueError("Normal and attack test sources must remain distinct")

    print(f"Rows: train={len(split.train):,}, val={len(split.val):,}, "
          f"test_normal={len(split.test_normal):,}, test_attack={len(split.test_attack):,}", flush=True)
    print(f"Boundaries: train_end={split.train['Timestamp'].max()}, "
          f"val={split.val['Timestamp'].min()}..{split.val['Timestamp'].max()}, "
          f"test_normal_start={split.test_normal['Timestamp'].min()}", flush=True)
    print("Validation/attack calendar overlap is retained by the approved Option A split; "
          "attack rows are excluded from validation.", flush=True)

    norm_params = fit_normalizer(split.train)
    train_dataset = make_lazy_window_dataset(split.train, norm_params)
    val_dataset = make_lazy_window_dataset(split.val, norm_params)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    actuator_db = build_actuator_database(split.train)
    save_actuator_database(actuator_db, processed_dir / "actuator_db.joblib")
    total_windows = sum(max(0, len(df) - WINDOW.Win - WINDOW.H - WINDOW.Wout + 1)
                        for df in (split.train, split.val, split.test_normal, split.test_attack))
    del normal, attack, split
    gc.collect()

    train_loader = DataLoader(
        train_dataset, batch_size=WDNNConfig.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=WDNNConfig.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == 'cuda')
    )
    expanded_gb = total_windows * (DATA.n_features * WINDOW.Win + DATA.n_sensors * WINDOW.Wout) * 4 / 1e9
    print(f"Windowing is on-demand; no overlapping windows are pre-materialized "
          f"(equivalent expanded arrays would be about {expanded_gb:.2f} GB).", flush=True)
    print(f"Available host RAM: {psutil.virtual_memory().available / 1e9:.2f} GB; "
          f"CUDA free: {torch.cuda.mem_get_info(device)[0] / 1e9:.2f} GB" if device.type == 'cuda'
          else f"Available host RAM: {psutil.virtual_memory().available / 1e9:.2f} GB", flush=True)

    joblib.dump(norm_params, Path("models/norm_params.joblib"))

    wdnn = WDNN()
    print(f"Training configuration: epochs={WDNNConfig.tuning_epochs}, batch_size={WDNNConfig.batch_size}, "
          f"SGD lr={WDNNConfig.learning_rate}, Win={WINDOW.Win}, H={WINDOW.H}, Wout={WINDOW.Wout}; "
          f"validation is normal-only and chronological.", flush=True)
    history = train_wdnn(
        wdnn, train_loader, val_loader,
        epochs=WDNNConfig.tuning_epochs,
        lr=WDNNConfig.learning_rate,
        device=device,
        checkpoint_path=Path("models/wdnn_checkpoint.pt"),
        log_path=Path("logs/wdnn_training.csv"),
    )
    Path("models").mkdir(parents=True, exist_ok=True)
    torch.save(wdnn.state_dict(), "models/wdnn.pth")
    joblib.dump(history, Path("logs/wdnn_training_history.joblib"))
    print(f"WDNN training complete: final train_loss={history['train_loss'][-1]:.6f}, "
          f"val_loss={history['val_loss'][-1]:.6f}", flush=True)
    print("Saved models/wdnn.pth, models/wdnn_checkpoint.pt, "
          "logs/wdnn_training.csv, and logs/wdnn_training_history.joblib.", flush=True)
    print("No TTNN training or test evaluation was started.", flush=True)


if __name__ == "__main__":
    main()
