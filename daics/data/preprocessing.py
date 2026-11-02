"""
DAICS Data Preprocessing - Normalization and Time-Series Windowing
Implements MinMax normalization (fit on train, apply to all) and sliding window generation.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
from sklearn.preprocessing import MinMaxScaler
import joblib
from daics.config import WINDOW, DATA, SPLIT
from daics.data.loader import FEATURE_COLUMNS, SENSOR_COLUMNS, ACTUATOR_COLUMNS
from daics.data.split import SplitResult, get_feature_label_arrays


@dataclass
class NormalizationParams:
    """Stores fitted normalization parameters for sensors"""
    sensor_scaler: MinMaxScaler
    sensor_min: np.ndarray
    sensor_max: np.ndarray
    feature_columns: List[str]
    sensor_columns: List[str]
    actuator_columns: List[str]


def fit_normalizer(train_df: pd.DataFrame) -> NormalizationParams:
    """
    Fit MinMax scaler on training data sensors only.
    Actuators are NOT normalized (discrete states).
    """
    sensor_data = train_df[SENSOR_COLUMNS].values.astype(np.float32)

    scaler = MinMaxScaler(feature_range=(DATA.sensor_min, DATA.sensor_max))
    scaler.fit(sensor_data)

    return NormalizationParams(
        sensor_scaler=scaler,
        sensor_min=scaler.data_min_.astype(np.float32),
        sensor_max=scaler.data_max_.astype(np.float32),
        feature_columns=FEATURE_COLUMNS,
        sensor_columns=SENSOR_COLUMNS,
        actuator_columns=ACTUATOR_COLUMNS
    )


def normalize_features(
    df: pd.DataFrame,
    params: NormalizationParams
) -> np.ndarray:
    """
    Normalize sensor columns using fitted scaler.
    Actuator columns passed through unchanged.
    Returns (n_samples, n_features) array.
    """
    X = df[params.feature_columns].values.astype(np.float32)
    X_normalized = X.copy()

    # Normalize only sensor columns (first n_sensors columns in FEATURE_COLUMNS)
    sensor_indices = [params.feature_columns.index(c) for c in params.sensor_columns]
    X_normalized[:, sensor_indices] = params.sensor_scaler.transform(X[:, sensor_indices])

    return X_normalized


def normalize_sensors(
    sensor_data: np.ndarray,
    params: NormalizationParams
) -> np.ndarray:
    """
    Normalize sensor data using fitted scaler.
    Input: (n_samples, n_sensors) or (n_samples, n_sensors, ...)
    """
    # Handle different shapes
    original_shape = sensor_data.shape
    if sensor_data.ndim == 2:
        # (n_samples, n_sensors)
        return params.sensor_scaler.transform(sensor_data.astype(np.float32))
    elif sensor_data.ndim == 3:
        # (n_samples, n_sensors, Wout) - reshape, transform, reshape back
        n_samples, n_sensors, wout = sensor_data.shape
        flat = sensor_data.reshape(n_samples * n_sensors, wout)
        # This won't work directly - need to reshape to 2D for scaler
        # Actually scaler expects (n_samples, n_features)
        flat = sensor_data.reshape(-1, n_sensors)
        normalized = params.sensor_scaler.transform(flat.astype(np.float32))
        return normalized.reshape(original_shape)
    else:
        raise ValueError(f"Unsupported shape: {original_shape}")


def denormalize_sensors(
    sensor_data: np.ndarray,
    params: NormalizationParams
) -> np.ndarray:
    """Denormalize sensor predictions back to original scale"""
    return params.sensor_scaler.inverse_transform(sensor_data)


def save_normalizer(params: NormalizationParams, filepath: Path) -> None:
    """Save fitted normalizer to disk"""
    joblib.dump(params, filepath)


def load_normalizer(filepath: Path) -> NormalizationParams:
    """Load fitted normalizer from disk"""
    return joblib.load(filepath)


@dataclass
class WindowedData:
    """Container for windowed time-series data"""
    X: np.ndarray           # (n_windows, n_features, Win) - sensors + actuators
    y: np.ndarray           # (n_windows, n_sensors, Wout) - sensor targets
    timestamps: np.ndarray  # (n_windows,) - timestamp of first output sample
    labels: np.ndarray      # (n_windows,) - 0=Normal, 1=Attack (for output window)
    section_masks: List[np.ndarray]  # List of G boolean masks for sensor sections
    source: str = "unknown"  # Source identifier for test split separation


def create_windows(
    X: np.ndarray,
    y_sensors: np.ndarray,
    labels: np.ndarray,
    timestamps: pd.Series,
    Win: int = WINDOW.Win,
    H: int = WINDOW.H,
    Wout: int = WINDOW.Wout,
    mgse_per_section: Tuple[int, ...] = DATA.mgse_per_section,
    normalizer: 'NormalizationParams' = None,
    max_windows: int = None,
    source: str = "unknown"
) -> WindowedData:
    """
    Create sliding windows for WDNN training/inference.

    Input: X (n_samples, n_features), y_sensors (n_samples, n_sensors)
    Output: windows where each window predicts Wout future sensor values
    after horizon H from the end of input window Win.

    Timeline:
    [t-Win ... t-1] -> input window (Win samples)
    [t ... t+H-1]   -> horizon (H samples, ignored)
    [t+H ... t+H+Wout-1] -> output window (Wout samples, prediction targets)

    For sample i (0-indexed in original data):
    - Input: X[i : i+Win]
    - Target: y_sensors[i+Win+H : i+Win+H+Wout]

    Valid range: i from 0 to n_samples - Win - H - Wout

    Memory-efficient: If max_windows is specified, only create up to that many windows
    (useful for testing). Otherwise creates all windows but uses memory-efficient
    batch processing for normalization.
    """
    n_samples, n_features = X.shape
    n_sensors = y_sensors.shape[1]

    # Number of valid windows
    n_windows = n_samples - Win - H - Wout + 1
    if n_windows <= 0:
        raise ValueError(f"Not enough samples for windowing: {n_samples} < {Win + H + Wout}")

    if max_windows is not None:
        n_windows = min(n_windows, max_windows)

    # Create section masks for output sections
    section_masks = []
    sensor_idx = 0
    for mgse in mgse_per_section:
        mask = np.zeros(n_sensors, dtype=bool)
        mask[sensor_idx:sensor_idx + mgse] = True
        section_masks.append(mask)
        sensor_idx += mgse

    # Pre-allocate arrays (still needed for return, but we fill incrementally)
    X_windows = np.zeros((n_windows, n_features, Win), dtype=np.float32)
    y_windows = np.zeros((n_windows, n_sensors, Wout), dtype=np.float32)
    window_timestamps = np.zeros(n_windows, dtype='datetime64[ns]')
    window_labels = np.zeros(n_windows, dtype=np.int32)

    # Generate windows
    for i in range(n_windows):
        # Input window: Win samples of features (sensors + actuators)
        X_windows[i] = X[i:i+Win].T  # (n_features, Win)

        # Output window: Wout samples of sensors (targets)
        out_start = i + Win + H
        out_end = out_start + Wout
        y_windows[i] = y_sensors[out_start:out_end].T  # (n_sensors, Wout)

        # Timestamp of first output sample
        window_timestamps[i] = timestamps.iloc[out_start].value  # nanoseconds since epoch

        # Label: 1 if ANY sample in output window is Attack
        window_labels[i] = labels[out_start:out_end].max()

    # Normalize targets if normalizer provided (batch process for memory efficiency)
    if normalizer is not None:
        # Process in chunks to avoid large intermediate arrays
        chunk_size = 10000
        for start_idx in range(0, n_windows, chunk_size):
            end_idx = min(start_idx + chunk_size, n_windows)
            chunk = y_windows[start_idx:end_idx]
            n_chunk = chunk.shape[0]

            # Reshape: (n_chunk, n_sensors, Wout) -> (n_chunk * Wout, n_sensors)
            chunk_flat = chunk.transpose(0, 2, 1).reshape(-1, n_sensors)
            chunk_normalized = normalizer.sensor_scaler.transform(chunk_flat.astype(np.float32))
            y_windows[start_idx:end_idx] = chunk_normalized.reshape(n_chunk, Wout, n_sensors).transpose(0, 2, 1)

    return WindowedData(
        X=X_windows,
        y=y_windows,
        timestamps=window_timestamps,
        labels=window_labels,
        section_masks=section_masks,
        source=source
    )


def create_windows_for_section(
    windowed: WindowedData,
    section_idx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract data for a specific output section.
    Returns (X, y_section, timestamps, labels) for that section.
    """
    mask = windowed.section_masks[section_idx]
    y_section = windowed.y[:, mask, :]  # (n_windows, mgse, Wout)
    return windowed.X, y_section, windowed.timestamps, windowed.labels


def prepare_training_data(split: SplitResult, max_windows_per_split: int = None) -> Tuple[WindowedData, WindowedData, List[WindowedData], NormalizationParams]:
    """
    Full preprocessing pipeline for train/val/test.
    Returns (train_windows, val_windows, test_windows_list, norm_params)

    Test windows are generated independently from normal and attack source sequences
    to avoid creating windows that cross source file boundaries.

    Args:
        split: SplitResult from chronological_split
        max_windows_per_split: If set, limits windows per split (for testing)
    """
    # Fit normalizer on training data only
    norm_params = fit_normalizer(split.train)

    # Normalize all splits
    X_train = normalize_features(split.train, norm_params)
    X_val = normalize_features(split.val, norm_params)
    X_test_normal = normalize_features(split.test_normal, norm_params)
    X_test_attack = normalize_features(split.test_attack, norm_params)

    # Get sensor targets and labels
    _, y_sensors_train, labels_train = get_feature_label_arrays(split.train)
    _, y_sensors_val, labels_val = get_feature_label_arrays(split.val)
    _, y_sensors_test_normal, labels_test_normal = get_feature_label_arrays(split.test_normal)
    _, y_sensors_test_attack, labels_test_attack = get_feature_label_arrays(split.test_attack)

    # Create windows for train/val
    train_windows = create_windows(
        X_train, y_sensors_train, labels_train, split.train["Timestamp"],
        normalizer=norm_params,
        max_windows=max_windows_per_split,
        source="train"
    )
    val_windows = create_windows(
        X_val, y_sensors_val, labels_val, split.val["Timestamp"],
        normalizer=norm_params,
        max_windows=max_windows_per_split,
        source="val"
    )

    # Create test windows independently from normal and attack sources
    test_normal_windows = create_windows(
        X_test_normal, y_sensors_test_normal, labels_test_normal, split.test_normal["Timestamp"],
        normalizer=norm_params,
        max_windows=max_windows_per_split,
        source="test_normal"
    )
    test_attack_windows = create_windows(
        X_test_attack, y_sensors_test_attack, labels_test_attack, split.test_attack["Timestamp"],
        normalizer=norm_params,
        max_windows=max_windows_per_split,
        source="test_attack"
    )

    test_windows_list = [test_normal_windows, test_attack_windows]

    return train_windows, val_windows, test_windows_list, norm_params


def combine_test_windows(test_windows_list: List[WindowedData]) -> WindowedData:
    """
    Combine independent test window sources into a single WindowedData.
    Used for evaluation where we want one continuous stream.
    Windows remain separated by source - no cross-source windows exist.
    """
    assert len(test_windows_list) == 2, "Expected [test_normal, test_attack]"

    # Sort by timestamp and concatenate
    all_X = np.concatenate([w.X for w in test_windows_list], axis=0)
    all_y = np.concatenate([w.y for w in test_windows_list], axis=0)
    all_ts = np.concatenate([w.timestamps for w in test_windows_list], axis=0)
    all_labels = np.concatenate([w.labels for w in test_windows_list], axis=0)
    section_masks = test_windows_list[0].section_masks

    # Sort by timestamp
    sort_idx = np.argsort(all_ts)
    all_X = all_X[sort_idx]
    all_y = all_y[sort_idx]
    all_ts = all_ts[sort_idx]
    all_labels = all_labels[sort_idx]

    return WindowedData(
        X=all_X,
        y=all_y,
        timestamps=all_ts,
        labels=all_labels,
        section_masks=section_masks,
        source="test_combined"
    )


class WindowDataset(torch.utils.data.Dataset):
    """
    Memory-efficient dataset that generates windows on-the-fly.
    Avoids storing all overlapping windows in RAM.

    This is the recommended approach for training on large datasets.
    """
    def __init__(
        self,
        X: np.ndarray,
        y_sensors: np.ndarray,
        labels: np.ndarray,
        timestamps: pd.Series,
        Win: int = WINDOW.Win,
        H: int = WINDOW.H,
        Wout: int = WINDOW.Wout,
        mgse_per_section: Tuple[int, ...] = DATA.mgse_per_section,
        normalizer: 'NormalizationParams' = None,
        max_windows: int = None
    ):
        self.X = X
        self.y_sensors = y_sensors
        self.labels = labels
        self.timestamps = timestamps
        self.Win = Win
        self.H = H
        self.Wout = Wout
        self.mgse_per_section = mgse_per_section
        self.normalizer = normalizer
        self.n_sensors = y_sensors.shape[1]
        self.n_features = X.shape[1]

        # Number of valid windows
        n_windows = len(X) - Win - H - Wout + 1
        if n_windows <= 0:
            raise ValueError(f"Not enough samples for windowing: {len(X)} < {Win + H + Wout}")

        if max_windows is not None:
            n_windows = min(n_windows, max_windows)

        self.n_windows = n_windows

        # Pre-compute section masks
        self.section_masks = []
        sensor_idx = 0
        for mgse in mgse_per_section:
            mask = np.zeros(self.n_sensors, dtype=bool)
            mask[sensor_idx:sensor_idx + mgse] = True
            self.section_masks.append(mask)
            sensor_idx += mgse

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        if idx >= self.n_windows:
            raise IndexError(f"Index {idx} out of range ({self.n_windows})")

        # Input window
        X_window = self.X[idx:idx+self.Win].T  # (n_features, Win)

        # Output window
        out_start = idx + self.Win + self.H
        out_end = out_start + self.Wout
        y_window = self.y_sensors[out_start:out_end].T  # (n_sensors, Wout)

        # Timestamp
        timestamp = self.timestamps.iloc[out_start].value

        # Label
        label = self.labels[out_start:out_end].max()

        # Normalize target if normalizer provided
        if self.normalizer is not None:
            # y_window: (n_sensors, Wout) -> (Wout, n_sensors) -> (Wout * n_sensors,)
            # Actually scaler expects (n_samples, n_features)
            y_flat = y_window.T  # (Wout, n_sensors)
            y_normalized = self.normalizer.sensor_scaler.transform(y_flat.astype(np.float32))
            y_window = y_normalized.T  # (n_sensors, Wout)

        return {
            'X': torch.FloatTensor(X_window),
            'y': torch.FloatTensor(y_window),
            'timestamp': timestamp,
            'label': label,
            'section_masks': self.section_masks
        }


def create_window_dataloader(
    dataset: WindowDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> torch.utils.data.DataLoader:
    """Create DataLoader from WindowDataset"""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def verify_windowing(train_windows: WindowedData, val_windows: WindowedData, test_windows: WindowedData) -> Dict[str, bool]:
    """Verify windowed data integrity"""
    checks = {}

    # Check shapes
    n_features = DATA.n_features
    n_sensors = DATA.n_sensors

    checks["train_X_shape"] = train_windows.X.shape[1:] == (n_features, WINDOW.Win)
    checks["train_y_shape"] = train_windows.y.shape[1:] == (n_sensors, WINDOW.Wout)
    checks["val_X_shape"] = val_windows.X.shape[1:] == (n_features, WINDOW.Win)
    checks["val_y_shape"] = val_windows.y.shape[1:] == (n_sensors, WINDOW.Wout)
    checks["test_X_shape"] = test_windows.X.shape[1:] == (n_features, WINDOW.Win)
    checks["test_y_shape"] = test_windows.y.shape[1:] == (n_sensors, WINDOW.Wout)

    # Check no NaN
    checks["train_no_nan"] = not np.isnan(train_windows.X).any() and not np.isnan(train_windows.y).any()
    checks["val_no_nan"] = not np.isnan(val_windows.X).any() and not np.isnan(val_windows.y).any()
    checks["test_no_nan"] = not np.isnan(test_windows.X).any() and not np.isnan(test_windows.y).any()

    # Check sensor values in [0, 1] (normalized) - with tolerance for floating point
    eps = 1e-6
    sensor_indices = [FEATURE_COLUMNS.index(c) for c in SENSOR_COLUMNS]
    checks["train_sensors_in_range"] = (train_windows.X[:, sensor_indices, :] >= -eps).all() and \
                                        (train_windows.X[:, sensor_indices, :] <= 1 + eps).all()

    # Check section masks cover all sensors
    total_mgse = sum(DATA.mgse_per_section)
    checks["section_masks_complete"] = total_mgse == n_sensors

    # Check labels are binary
    checks["train_labels_binary"] = set(np.unique(train_windows.labels)).issubset({0, 1})
    checks["val_labels_binary"] = set(np.unique(val_windows.labels)).issubset({0, 1})
    checks["test_labels_binary"] = set(np.unique(test_windows.labels)).issubset({0, 1})

    return checks


if __name__ == "__main__":
    from daics.data.loader import load_normal, load_attack
    from daics.data.split import chronological_split

    data_dir = Path("C:/Users/Aaditya/Desktop/Programming/IMDAI/data/SWaT")
    normal = load_normal(data_dir)
    attack = load_attack(data_dir)

    split = chronological_split(normal, attack)
    train_windows, val_windows, test_windows_list, norm_params = prepare_training_data(split)

    # Combine test windows for display
    test_windows = combine_test_windows(test_windows_list)

    print("\n=== Windowed Data Shapes ===")
    print(f"Train: X={train_windows.X.shape}, y={train_windows.y.shape}, labels={train_windows.labels.shape}")
    print(f"Val:   X={val_windows.X.shape}, y={val_windows.y.shape}, labels={val_windows.labels.shape}")
    print(f"Test Normal:  X={test_windows_list[0].X.shape}, labels={test_windows_list[0].labels.shape}")
    print(f"Test Attack:  X={test_windows_list[1].X.shape}, labels={test_windows_list[1].labels.shape}")
    print(f"Test Combined: X={test_windows.X.shape}, y={test_windows.y.shape}, labels={test_windows.labels.shape}")

    print(f"\nSection masks:")
    for i, mask in enumerate(train_windows.section_masks):
        print(f"  Section {i}: {mask.sum()} sensors")

    print(f"\nLabel distribution:")
    print(f"  Train: {np.bincount(train_windows.labels)}")
    print(f"  Val:   {np.bincount(val_windows.labels)}")
    print(f"  Test Normal: {np.bincount(test_windows_list[0].labels)}")
    print(f"  Test Attack: {np.bincount(test_windows_list[1].labels)}")
    print(f"  Test Combined: {np.bincount(test_windows.labels)}")

    checks = verify_windowing(train_windows, val_windows, test_windows)
    print("\n=== Windowing Verification ===")
    for check, passed in checks.items():
        print(f"  {check}: {'PASS' if passed else 'FAIL'}")

    # Test normalizer save/load
    save_normalizer(norm_params, Path("C:/Users/Aaditya/Desktop/Programming/IMDAI/data/processed/normalizer.joblib"))
    print("\nNormalizer saved successfully")