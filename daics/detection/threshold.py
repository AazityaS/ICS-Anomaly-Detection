"""
DAICS Detection - Threshold Computation and Anomaly Detection
Implements adaptive thresholding with actuator normal-state database.
Paper: "Dynamic threshold: T_g = T_base + max(TTNN_predictions)"
"""

import numpy as np
import torch
from torch.utils.data import SequentialSampler
from typing import List, Tuple, Optional
from dataclasses import dataclass
from daics.config import DATA, WINDOW, WDNN as WDNNConfig, TTNN as TTNNConfig
from daics.models.wdnn import WDNN
from daics.models.ttnn import TTNNEnsemble
from daics.data.actuator_db import ActuatorDatabase, query_actuator_database_vectorized


@dataclass
class ThresholdConfig:
    """Threshold configuration"""
    wanom: int = WINDOW.Wanom  # 30 consecutive samples
    median_kernel_size: int = 59  # Median filter size
    percentile: float = None  # Not used - paper uses mean + std


@dataclass
class Thresholds:
    """Computed thresholds for each section"""
    t_base: float
    t_sections: List[float]  # T_g = T_base + max(TTNN_predictions_g)


def compute_base_threshold(
    val_mse_per_section: List[np.ndarray]
) -> float:
    """
    Compute base threshold T_base = mean(val_MSE) + std(val_MSE)
    Across all sections combined.
    """
    all_mse = np.concatenate(val_mse_per_section)
    t_base = all_mse.mean() + all_mse.std()
    return float(t_base)


def compute_section_thresholds(
    t_base: float,
    ttnn_predictions: List[np.ndarray]
) -> List[float]:
    """
    Compute per-section thresholds: T_g = T_base + max(TTNN_predictions_g)
    """
    t_sections = []
    for pred in ttnn_predictions:
        t_g = t_base + pred.max()
        t_sections.append(float(t_g))
    return t_sections


def apply_median_filter(
    scores: np.ndarray,
    kernel_size: int = 59
) -> np.ndarray:
    """
    Apply median filter to anomaly scores.
    Paper: "median filter of size 59"
    """
    from scipy.ndimage import median_filter
    return median_filter(scores, size=kernel_size)


def historical_thresholds_for_sequence(
    mse: np.ndarray,
    t_base: float,
    ttnn: TTNNEnsemble,
    section_idx: int,
    device: torch.device,
    batch_size: int = WDNNConfig.batch_size,
    win: int = WINDOW.Win,
    horizon: int = WINDOW.H,
    median_kernel_size: int = TTNNConfig.median_kernel_size,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate causal, batchwise Algorithm 1 thresholds for one MSE sequence.

    For current interval [i, i+s), Eg consists of s length-win rows starting
    at i-horizon-win through i-horizon-win+s-1. Thus its latest MSE index is
    strictly before i. The first horizon+win samples are warm-up and use T_base.

    The paper does not specify the median-filter axis/edge rule. We apply the
    size-59 median independently along each row's temporal axis; scipy's
    default reflect boundary mode is retained and documented as an assumption.
    """
    mse = np.asarray(mse, dtype=np.float64).reshape(-1)
    thresholds = np.full(len(mse), float(t_base), dtype=np.float64)
    batch_predictions = np.full(len(mse), np.nan, dtype=np.float64)
    ttnn.eval()

    with torch.no_grad():
        for current_start in range(0, len(mse), batch_size):
            current_end = min(current_start + batch_size, len(mse))
            count = current_end - current_start
            if current_start < horizon + win:
                continue

            t0 = current_start - horizon - win
            eg = np.stack([mse[t0 + j:t0 + j + win] for j in range(count)])
            # Algorithm 1: median(Eg); explicit per-row temporal-axis
            # interpretation because the paper leaves the matrix axis unstated.
            filtered = np.stack([apply_median_filter(row, median_kernel_size) for row in eg])
            x = torch.as_tensor(filtered, dtype=torch.float32, device=device).unsqueeze(1)
            predictions = ttnn.forward_section(section_idx, x).reshape(-1).cpu().numpy()
            dynamic = float(t_base + np.max(predictions))
            thresholds[current_start:current_end] = dynamic
            batch_predictions[current_start:current_end] = predictions.max()

    return thresholds, batch_predictions


def apply_wanom_consecutive(
    binary_scores: np.ndarray,
    wanom: int = 30
) -> np.ndarray:
    """
    Apply consecutive anomaly detection (Wanom).
    Anomaly is confirmed only if Wanom consecutive samples are above threshold.
    """
    n = len(binary_scores)
    confirmed = np.zeros(n, dtype=bool)

    consecutive = 0
    for i in range(n):
        if binary_scores[i]:
            consecutive += 1
        else:
            consecutive = 0

        if consecutive >= wanom:
            confirmed[i] = True
            # Also mark the previous wanom-1 samples as confirmed
            start = max(0, i - wanom + 1)
            confirmed[start:i+1] = True

    return confirmed


def compute_section_mse(
    predictions: List[np.ndarray],
    targets: List[np.ndarray]
) -> List[np.ndarray]:
    """
    Compute MSE per section.
    predictions: List of G arrays, each (n_samples, mgse_g, Wout)
    targets: List of G arrays, each (n_samples, mgse_g, Wout)
    Returns: List of G arrays, each (n_samples,) - MSE per sample
    """
    mse_per_section = []
    for pred, target in zip(predictions, targets):
        # MSE across sensors and time steps: (n_samples,)
        mse = np.mean((pred - target) ** 2, axis=(1, 2))
        mse_per_section.append(mse)
    return mse_per_section


def detect_anomalies(
    mse_per_section: List[np.ndarray],
    t_base: float,
    ttnn_predictions_per_section: List[np.ndarray],
    actuator_anomaly_scores: np.ndarray,
    wanom: int = WINDOW.Wanom,
    batch_size: int = WDNNConfig.batch_size,
    dynamic_thresholds_per_section: Optional[List[np.ndarray]] = None
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Full anomaly detection pipeline using dynamic thresholds.

    Paper (Section VI-D): Threshold T_g is constant for a batch of s samples.
    T_g = T_base + max(TTNN_predictions for the batch)

    Args:
        mse_per_section: List of G arrays (n_samples,) - MSE per section
        t_base: Base threshold constant
        ttnn_predictions_per_section: List of G arrays (n_sequences,) - TTNN outputs
            Note: n_sequences = n_samples - Win + 1 (approximately)
        actuator_anomaly_scores: (n_samples,) - 1 if anomaly, 0 if normal
        wanom: Consecutive samples required
        batch_size: s (typically 32)

    Returns:
        final_anomalies: (n_samples,) boolean
        section_anomalies: List of G arrays (n_samples,)
    """
    n_samples = len(mse_per_section[0])
    section_anomalies = []

    for g in range(len(mse_per_section)):
        mse_array = mse_per_section[g]
        ttnn_preds = (ttnn_predictions_per_section[g]
                      if ttnn_predictions_per_section is not None else np.array([]))

        # Binary results for this section
        section_binary = np.zeros(n_samples, dtype=bool)

        # We need to match ttnn_preds to the corresponding batches of MSEs.
        # TTNN prediction at index i corresponds to MSE window starting at i.
        # Algorithm 1: Threshold for batch [t, t+s) is based on TTNN outputs for that batch.

        if dynamic_thresholds_per_section is not None:
            thresholds = np.asarray(dynamic_thresholds_per_section[g])
            if len(thresholds) != n_samples:
                raise ValueError("Dynamic threshold sequence must align with MSE samples")
            section_binary[:] = mse_array > thresholds
        else:
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)

                # Legacy mode: caller supplies predictions already aligned to
                # each current batch. Production evaluation passes explicit
                # causal threshold arrays via dynamic_thresholds_per_section.
                batch_ttnn = ttnn_preds[i:end_idx]

                if len(batch_ttnn) > 0:
                    t_g = t_base + np.max(batch_ttnn)
                else:
                    t_g = t_base

                # Apply this threshold to the batch of MSEs
                section_binary[i:end_idx] = mse_array[i:end_idx] > t_g

        # Apply Wanom consecutive detection (Equation 3)
        section_confirmed = apply_wanom_consecutive(section_binary, wanom)
        section_anomalies.append(section_confirmed)

    # Combine sections (OR)
    combined = np.zeros(n_samples, dtype=bool)
    for sec_anom in section_anomalies:
        combined = combined | sec_anom

    # Combine with Actuator anomalies
    final_anomalies = combined | actuator_anomaly_scores.astype(bool)

    return final_anomalies, section_anomalies


def compute_thresholds_from_validation(
    wdnn: WDNN,
    ttnn_ensemble: TTNNEnsemble,
    val_data_loader,
    actuator_db: ActuatorDatabase,
    device: torch.device
) -> Thresholds:
    """
    Compute thresholds using validation data.
    Algorithm 1:
    1. Get WDNN predictions and compute MSE per section
    2. Compute T_base = mean + std of all validation MSE
    3. Apply Median Filter (size 59) to MSE sequences
    4. Create windows of filtered MSEs (length Win)
    5. Pass to TTNN to get Estimated prediction errors
    6. T_g = T_base + max(TTNN_outputs)
    """
    wdnn.eval()
    ttnn_ensemble.eval()
    if not isinstance(val_data_loader.sampler, SequentialSampler):
        raise ValueError("Validation MSE thresholds require shuffle=False chronological order")

    all_mse_per_section = [[] for _ in range(WDNNConfig.G)]
    all_actuator_states = []

    with torch.no_grad():
        for batch in val_data_loader:
            # Loader format: (X, y_section0, y_section1, ..., y_sectionG, labels)
            X = batch[0].to(device)
            y_sections = [batch[g+1].to(device) for g in range(WDNNConfig.G)]

            # WDNN forward
            wdnn_outputs = wdnn(X)

            # Compute MSE per section
            for g, (pred, target) in enumerate(zip(wdnn_outputs, y_sections)):
                mse = torch.mean((pred - target) ** 2, dim=(1, 2)).cpu().numpy()
                all_mse_per_section[g].extend(mse)

            # Actuator states (last time step)
            actuator_states = X[:, -DATA.n_actuators:, -1].cpu().numpy().astype(int)
            all_actuator_states.append(actuator_states)

    # 1. Compute T_base
    mse_per_section = [np.array(m) for m in all_mse_per_section]
    t_base = compute_base_threshold(mse_per_section)

    # Validation-only summary thresholds use the same causal batch alignment.
    # During test, evaluate_model builds new thresholds from each independent
    # test source's past MSE and never uses these validation summaries.
    t_sections = []
    for g, mse_array in enumerate(mse_per_section):
        dynamic, _ = historical_thresholds_for_sequence(
            mse_array, t_base, ttnn_ensemble, g, device,
            batch_size=WDNNConfig.batch_size,
            median_kernel_size=TTNNConfig.median_kernel_size,
        )
        eligible = dynamic[WINDOW.H + WINDOW.Win:]
        t_sections.append(float(eligible[-1]) if len(eligible) else float(t_base))

    return Thresholds(t_base=t_base, t_sections=t_sections)


def get_actuator_anomaly_scores(
    db: ActuatorDatabase,
    actuator_states: np.ndarray
) -> np.ndarray:
    """Get actuator anomaly scores (0=normal, 1=anomalous)"""
    in_db = query_actuator_database_vectorized(db, actuator_states)
    return (~in_db).astype(np.float32)


if __name__ == "__main__":
    # Quick test
    print("Testing threshold computation...")

    # Mock validation MSE
    val_mse = [
        np.random.exponential(0.01, 1000) for _ in range(3)
    ]
    t_base = compute_base_threshold(val_mse)
    print(f"T_base: {t_base:.6f}")

    # Mock TTNN predictions
    ttnn_preds = [np.random.uniform(0, 0.1, 1000) for _ in range(3)]
    t_sections = compute_section_thresholds(t_base, ttnn_preds)
    print(f"T_sections: {[f'{t:.6f}' for t in t_sections]}")

    # Test median filter
    scores = np.random.rand(100) > 0.7
    filtered = apply_median_filter(scores.astype(float), 5)
    print(f"Median filter: {scores.sum()} -> {filtered.sum()} anomalies")

    # Test Wanom
    binary = np.array([0,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0])
    confirmed = apply_wanom_consecutive(binary, 5)
    print(f"Wanom=5: {binary.sum()} -> {confirmed.sum()} confirmed")

    print("Threshold module tests passed!")
