"""
DAICS Data Loader
Loads raw SWaT CSV files, handles leading spaces in column names, parses timestamps.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from daics.config import DATA, SPLIT


# Raw column names (with leading spaces as they appear in CSV)
RAW_COLUMNS = [
    " Timestamp",
    "FIT101", "LIT101", " MV101", "P101", "P102", " AIT201", "AIT202", "AIT203",
    "FIT201", " MV201", " P201", " P202", "P203", " P204", "P205", "P206",
    "DPIT301", "FIT301", "LIT301", "MV301", "MV302", " MV303", "MV304",
    "P301", "P302", "AIT401", "AIT402", "FIT401", "LIT401", "P401", "P402",
    "P403", "P404", "UV401", "AIT501", "AIT502", "AIT503", "AIT504",
    "FIT501", "FIT502", "FIT503", "FIT504", "P501", "P502",
    "PIT501", "PIT502", "PIT503", "FIT601", "P601", "P602", "P603",
    "Normal/Attack"
]

# Cleaned column names (stripped)
CLEAN_COLUMNS = [c.strip() for c in RAW_COLUMNS]

# Sensor columns (24 in our cleaned data - AIT201 excluded)
SENSOR_COLUMNS = [
    "FIT101", "LIT101", "AIT202", "AIT203", "FIT201",
    "DPIT301", "FIT301", "LIT301", "AIT401", "AIT402",
    "FIT401", "LIT401", "AIT501", "AIT502", "AIT503", "AIT504",
    "FIT501", "FIT502", "FIT503", "FIT504",
    "PIT501", "PIT502", "PIT503", "FIT601"
]

# Actuator columns (20 in our cleaned data - 6 excluded)
ACTUATOR_COLUMNS = [
    "P101", "P102", "P203", "P205", "P206",
    "MV301", "MV302", "MV304",
    "P301", "P302", "P401", "P402", "P403", "P404",
    "P501", "P502", "P601", "P602", "P603",
    "UV401"   # UV401 is discrete {1,2} -> actuator
]

# All feature columns (sensor + actuator)
FEATURE_COLUMNS = SENSOR_COLUMNS + ACTUATOR_COLUMNS

# Excluded columns (had leading spaces, removed during preprocessing)
EXCLUDED_COLUMNS = ["MV101", "AIT201", "MV201", "P201", "P202", "P204", "MV303"]


def load_raw_csv(filepath: Path) -> pd.DataFrame:
    """
    Load raw SWaT CSV with original column names (including leading spaces).
    Parses timestamps and strips column names.
    """
    df = pd.read_csv(filepath)

    # Strip column names
    df.columns = [c.strip() for c in df.columns]

    # Parse timestamp
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, format="mixed")

    # Ensure chronological order
    df = df.sort_values("Timestamp").reset_index(drop=True)

    return df


def load_normal(data_dir: Path) -> pd.DataFrame:
    """Load normal.csv"""
    return load_raw_csv(data_dir / "normal.csv")


def load_attack(data_dir: Path) -> pd.DataFrame:
    """Load attack.csv"""
    return load_raw_csv(data_dir / "attack.csv")


def load_merged(data_dir: Path) -> pd.DataFrame:
    """Load merged.csv"""
    return load_raw_csv(data_dir / "merged.csv")


def verify_columns(df: pd.DataFrame, expected_features: int = 44) -> bool:
    """
    Verify dataframe has expected columns.
    Returns True if valid.
    """
    # Check feature columns exist
    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        print(f"WARNING: Missing feature columns: {missing_features}")
        return False

    # Check timestamp and label
    if "Timestamp" not in df.columns:
        print("ERROR: Missing Timestamp column")
        return False

    if "Normal/Attack" not in df.columns:
        print("ERROR: Missing Normal/Attack column")
        return False

    # Check feature count
    feature_cols = [c for c in df.columns if c not in ["Timestamp", "Normal/Attack"]]
    if len(feature_cols) != expected_features:
        print(f"WARNING: Expected {expected_features} features, got {len(feature_cols)}")
        return False

    return True


def get_sensor_actuator_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into sensor and actuator dataframes.
    Returns (sensors_df, actuators_df) with Timestamp index.
    """
    sensors = df[["Timestamp"] + SENSOR_COLUMNS].copy()
    actuators = df[["Timestamp"] + ACTUATOR_COLUMNS].copy()

    sensors = sensors.set_index("Timestamp")
    actuators = actuators.set_index("Timestamp")

    return sensors, actuators


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Check missing values per column"""
    return df[FEATURE_COLUMNS].isnull().sum()


def print_data_summary(name: str, df: pd.DataFrame) -> None:
    """Print summary of loaded data"""
    print(f"\n=== {name} ===")
    print(f"  Rows: {len(df)}")
    print(f"  Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    print(f"  Duration: {(df['Timestamp'].max() - df['Timestamp'].min()).total_seconds() / 86400:.2f} days")
    print(f"  Sorted: {df['Timestamp'].is_monotonic_increasing}")
    print(f"  Label distribution:")
    print(df["Normal/Attack"].value_counts().to_string())
    print(f"  Missing values:")
    missing = check_missing_values(df)
    for col, count in missing.items():
        if count > 0:
            print(f"    {col}: {count}")


if __name__ == "__main__":
    data_dir = Path("C:/Users/Aaditya/Desktop/Programming/IMDAI/data/SWaT")

    # Load all three files
    normal = load_normal(data_dir)
    attack = load_attack(data_dir)
    merged = load_merged(data_dir)

    print_data_summary("NORMAL", normal)
    print_data_summary("ATTACK", attack)
    print_data_summary("MERGED", merged)

    # Verify columns
    print("\n=== Column Verification ===")
    for name, df in [("normal", normal), ("attack", attack), ("merged", merged)]:
        ok = verify_columns(df)
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    # Check sensor/actuator split
    sensors, actuators = get_sensor_actuator_data(normal)
    print(f"\nSensor shape: {sensors.shape}")
    print(f"Actuator shape: {actuators.shape}")
    print(f"Sensor columns ({len(SENSOR_COLUMNS)}): {SENSOR_COLUMNS}")
    print(f"Actuator columns ({len(ACTUATOR_COLUMNS)}): {ACTUATOR_COLUMNS}")