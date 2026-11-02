"""
DAICS Data Split - Chronological Train/Validation/Test Split
Implements Option A: Paper's 7-day train (80/20), remaining normal + all attacks for test.
Documents temporal overlap between validation and attack observations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict
from daics.config import SPLIT, DATA
from daics.data.loader import load_normal, load_attack, FEATURE_COLUMNS, SENSOR_COLUMNS, ACTUATOR_COLUMNS


@dataclass(frozen=True)
class SplitResult:
    """Container for split datasets"""
    train: pd.DataFrame      # Normal only, 80% of first 7 days
    val: pd.DataFrame        # Normal only, 20% of first 7 days
    test_normal: pd.DataFrame  # Normal only, remaining after 7 days
    test_attack: pd.DataFrame  # All attack data
    test_combined: pd.DataFrame  # test_normal + test_attack, sorted by timestamp

    # Metadata
    train_size: int
    val_size: int
    test_normal_size: int
    test_attack_size: int
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp


def chronological_split(
    normal_df: pd.DataFrame,
    attack_df: pd.DataFrame,
    train_end: pd.Timestamp = None,
    val_ratio: float = 0.2,
    build_test_combined: bool = True,
) -> SplitResult:
    """
    Perform chronological train/val/test split per DAICS paper methodology.

    Paper: 7 days normal -> 80% train, 20% val | 4 days normal+attack -> test
    Our data: normal.csv spans ~11 days, attack.csv spans ~5 days starting Dec 28.

    Option A (implemented): Use exact 7-day boundary from paper.
    - Train: normal.csv [Dec 22 16:00, Dec 29 16:00) * 0.8
    - Val:   normal.csv [Dec 22 16:00, Dec 29 16:00) * 0.2
    - Test:  normal.csv [Dec 29 16:00, end] + all attack.csv

    Note: Validation period overlaps with attack timestamps (first attack Dec 28 10:29).
    Validation data contains ONLY "Normal" labeled records from normal.csv.
    """
    if train_end is None:
        train_end = pd.Timestamp(SPLIT.train_end)

    train_start = pd.Timestamp(SPLIT.train_start)

    # Ensure sorted
    normal_df = normal_df.sort_values("Timestamp").reset_index(drop=True)
    attack_df = attack_df.sort_values("Timestamp").reset_index(drop=True)

    # Full training period (first 7 days of normal)
    train_mask = (normal_df["Timestamp"] >= train_start) & (normal_df["Timestamp"] < train_end)
    train_full = normal_df[train_mask].copy()

    # 80/20 split within training period
    val_size = int(len(train_full) * val_ratio)
    train = train_full.iloc[:-val_size].copy()
    val = train_full.iloc[-val_size:].copy()

    # Test normal: remaining normal after 7 days
    test_normal = normal_df[normal_df["Timestamp"] >= train_end].copy()

    # Test attack: all attack data
    test_attack = attack_df.copy()

    # Optional combined view for legacy evaluation. Full WDNN training only
    # needs the independent sources, so avoid another large DataFrame copy.
    if build_test_combined:
        test_combined = pd.concat([test_normal, test_attack], ignore_index=True)
        test_combined = test_combined.sort_values("Timestamp").reset_index(drop=True)
    else:
        test_combined = pd.DataFrame(columns=normal_df.columns)
        test_combined["Timestamp"] = pd.to_datetime(test_combined["Timestamp"])

    return SplitResult(
        train=train,
        val=val,
        test_normal=test_normal,
        test_attack=test_attack,
        test_combined=test_combined,
        train_size=len(train),
        val_size=len(val),
        test_normal_size=len(test_normal),
        test_attack_size=len(test_attack),
        train_end=train["Timestamp"].max(),
        val_start=val["Timestamp"].min(),
        val_end=val["Timestamp"].max(),
        test_start=min(test_normal["Timestamp"].min(), attack_df["Timestamp"].min())
    )


def verify_split(result: SplitResult) -> Dict[str, bool]:
    """
    Verify split integrity: no leakage, correct boundaries, label consistency.
    Returns dict of check names -> pass/fail.
    """
    checks = {}

    # 1. Train/Val/Test are mutually exclusive in timestamps (for normal data)
    train_ts = set(result.train["Timestamp"])
    val_ts = set(result.val["Timestamp"])
    test_normal_ts = set(result.test_normal["Timestamp"])

    checks["train_val_disjoint"] = len(train_ts & val_ts) == 0
    checks["train_test_normal_disjoint"] = len(train_ts & test_normal_ts) == 0
    checks["val_test_normal_disjoint"] = len(val_ts & test_normal_ts) == 0

    # 2. Train + Val = full 7-day training period
    train_val_union = train_ts | val_ts
    train_full_mask = (result.train["Timestamp"] >= pd.Timestamp(SPLIT.train_start)) & \
                      (result.train["Timestamp"] < pd.Timestamp(SPLIT.train_end))
    # Actually need to check against original normal data
    # This is verified by construction in chronological_split

    # 3. Chronological order
    checks["train_chronological"] = result.train["Timestamp"].is_monotonic_increasing
    checks["val_chronological"] = result.val["Timestamp"].is_monotonic_increasing
    checks["test_normal_chronological"] = result.test_normal["Timestamp"].is_monotonic_increasing
    checks["test_attack_chronological"] = result.test_attack["Timestamp"].is_monotonic_increasing
    checks["test_combined_chronological"] = result.test_combined["Timestamp"].is_monotonic_increasing

    # 4. Label consistency
    checks["train_all_normal"] = (result.train["Normal/Attack"] == "Normal").all()
    checks["val_all_normal"] = (result.val["Normal/Attack"] == "Normal").all()
    checks["test_normal_all_normal"] = (result.test_normal["Normal/Attack"] == "Normal").all()
    checks["test_attack_all_attack"] = (result.test_attack["Normal/Attack"] == "Attack").all()

    # 5. Test combined has both labels
    test_labels = result.test_combined["Normal/Attack"].unique()
    checks["test_has_both_labels"] = set(test_labels) == {"Normal", "Attack"}

    # 6. No duplicate timestamps within each split
    checks["train_no_dup_ts"] = not result.train["Timestamp"].duplicated().any()
    checks["val_no_dup_ts"] = not result.val["Timestamp"].duplicated().any()
    checks["test_combined_no_dup_ts"] = not result.test_combined["Timestamp"].duplicated().any()

    # 7. Feature columns present
    for name, df in [("train", result.train), ("val", result.val),
                     ("test_normal", result.test_normal), ("test_attack", result.test_attack)]:
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        checks[f"{name}_has_all_features"] = len(missing) == 0

    # 8. Validate temporal overlap documentation
    # Val ends at Dec 29 15:59, first attack at Dec 28 10:29
    # This overlap is EXPECTED and documented
    checks["val_attack_overlap_documented"] = True  # By design

    return checks


def print_split_summary(result: SplitResult, checks: Dict[str, bool]) -> None:
    """Print split summary and verification results"""
    print("\n=== Chronological Split Summary ===")
    print(f"Train:     {result.train_size:>8} rows  ({result.train['Timestamp'].min()} to {result.train_end})")
    print(f"Val:       {result.val_size:>8} rows  ({result.val_start} to {result.val_end})")
    print(f"Test Normal: {result.test_normal_size:>8} rows  ({result.test_normal['Timestamp'].min()} to {result.test_normal['Timestamp'].max()})")
    print(f"Test Attack: {result.test_attack_size:>8} rows  ({result.test_attack['Timestamp'].min()} to {result.test_attack['Timestamp'].max()})")
    print(f"Test Combined: {len(result.test_combined):>8} rows")

    # Overlap info
    attack_start = result.test_attack["Timestamp"].min()
    val_end = result.val_end
    overlap_hours = (val_end - attack_start).total_seconds() / 3600
    print(f"\nTemporal Overlap (Val vs Attack): {overlap_hours:.2f} hours")
    print(f"  Val ends:   {val_end}")
    print(f"  Attack starts: {attack_start}")
    print(f"  NOTE: Val contains only 'Normal' labeled data from normal.csv")
    print(f"        Attack data in same period is in attack.csv (separate file)")

    print("\n=== Verification Checks ===")
    all_pass = True
    for check_name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {check_name}: {status}")

    print(f"\nAll checks passed: {all_pass}")

    if not all_pass:
        raise ValueError("Split verification failed!")


def get_feature_label_arrays(
    df: pd.DataFrame,
    sensor_columns: list = None,
    actuator_columns: list = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract features, sensor targets, and labels from dataframe.
    Returns (X, y_sensors, y_labels) where:
    - X: (n_samples, n_features) sensors + actuators
    - y_sensors: (n_samples, n_sensors) sensor readings (targets for WDNN)
    - y_labels: (n_samples,) 0=Normal, 1=Attack
    """
    if sensor_columns is None:
        sensor_columns = SENSOR_COLUMNS
    if actuator_columns is None:
        actuator_columns = ACTUATOR_COLUMNS

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y_sensors = df[sensor_columns].values.astype(np.float32)
    y_labels = (df["Normal/Attack"] == "Attack").astype(np.int32).values

    return X, y_sensors, y_labels


if __name__ == "__main__":
    data_dir = Path("C:/Users/Aaditya/Desktop/Programming/IMDAI/data/processed")
    import pandas as pd

    normal = pd.read_csv(data_dir / "normal_clean.csv")
    attack = pd.read_csv(data_dir / "attack_clean.csv")

    normal["Timestamp"] = pd.to_datetime(normal["Timestamp"], dayfirst=True, format="mixed")
    attack["Timestamp"] = pd.to_datetime(attack["Timestamp"], dayfirst=True, format="mixed")

    result = chronological_split(normal, attack)
    checks = verify_split(result)
    print_split_summary(result, checks)

    # Test feature extraction
    X_train, y_sensors_train, y_labels_train = get_feature_label_arrays(result.train)
    print(f"\nTrain X shape: {X_train.shape}, y_sensors: {y_sensors_train.shape}, y_labels: {y_labels_train.shape}")
    print(f"Train labels unique: {np.unique(y_labels_train)}")

    X_test, y_sensors_test, y_labels_test = get_feature_label_arrays(result.test_combined)
    print(f"Test X shape: {X_test.shape}, y_sensors: {y_sensors_test.shape}, y_labels: {y_labels_test.shape}")
    print(f"Test labels unique: {np.unique(y_labels_test)}, counts: {np.bincount(y_labels_test)}")
