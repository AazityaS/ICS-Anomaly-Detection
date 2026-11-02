"""
DAICS Actuator Normal-State Database
Builds and queries database of normal actuator combinations from training data.
Paper: "The database entries are n-tuples, where n is the number of actuators...
Each entry is a combination of actuators states labelled as normal, for a total
of 146 entries available in the SWaT dataset" (our cleaned data: ~70)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Set, List, Optional, Dict
from dataclasses import dataclass
import joblib
from daics.config import DATA
from daics.data.loader import ACTUATOR_COLUMNS
from daics.data.split import SplitResult


@dataclass
class ActuatorDatabase:
    """Container for actuator normal-state database"""
    # Unique normal actuator combinations (n_combinations, n_actuators)
    combinations: np.ndarray
    # Column names for actuators
    actuator_columns: List[str]
    # Number of unique combinations
    n_combinations: int
    # Number of actuators
    n_actuators: int


def build_actuator_database(train_df: pd.DataFrame) -> ActuatorDatabase:
    """
    Build actuator normal-state database from training data.
    Returns unique actuator state combinations observed during normal training.
    """
    actuator_data = train_df[ACTUATOR_COLUMNS].values
    # Get unique combinations
    unique_combinations = np.unique(actuator_data, axis=0)

    return ActuatorDatabase(
        combinations=unique_combinations.astype(np.int32),
        actuator_columns=ACTUATOR_COLUMNS.copy(),
        n_combinations=len(unique_combinations),
        n_actuators=len(ACTUATOR_COLUMNS)
    )


def query_actuator_database(
    db: ActuatorDatabase,
    actuator_states: np.ndarray
) -> np.ndarray:
    """
    Query if actuator combinations exist in normal database.
    Input: (n_samples, n_actuators) array of actuator states
    Output: (n_samples,) boolean array, True = found in database (normal)
    """
    # Convert to tuples for set lookup (faster for repeated queries)
    db_tuples = set(map(tuple, db.combinations))

    results = np.zeros(len(actuator_states), dtype=bool)
    for i, state in enumerate(actuator_states):
        results[i] = tuple(state) in db_tuples

    return results


def query_actuator_database_vectorized(
    db: ActuatorDatabase,
    actuator_states: np.ndarray
) -> np.ndarray:
    """
    Vectorized query using numpy broadcasting.
    More efficient for large batches.
    """
    # Reshape for broadcasting: (n_queries, 1, n_actuators) vs (1, n_db, n_actuators)
    queries = actuator_states[:, np.newaxis, :]  # (n_q, 1, n_act)
    db_combos = db.combinations[np.newaxis, :, :]  # (1, n_db, n_act)

    # Check equality across all actuators
    matches = np.all(queries == db_combos, axis=2)  # (n_q, n_db)
    # Any match in database
    return np.any(matches, axis=1)


def get_actuator_anomaly_scores(
    db: ActuatorDatabase,
    actuator_states: np.ndarray
) -> np.ndarray:
    """
    Get anomaly scores for actuator combinations.
    Score = 0 if in database (normal), 1 if not in database (anomaly).
    """
    in_db = query_actuator_database_vectorized(db, actuator_states)
    return (~in_db).astype(np.float32)


def save_actuator_database(db: ActuatorDatabase, filepath: Path) -> None:
    """Save actuator database to disk"""
    joblib.dump(db, filepath)


def load_actuator_database(filepath: Path) -> ActuatorDatabase:
    """Load actuator database from disk"""
    return joblib.load(filepath)


def print_actuator_database_summary(db: ActuatorDatabase) -> None:
    """Print summary of actuator database"""
    print("\n=== Actuator Normal-State Database ===")
    print(f"Actuators ({db.n_actuators}): {db.actuator_columns}")
    print(f"Unique normal combinations: {db.n_combinations}")
    print(f"Paper reports 146 for SWaT (our cleaned data: {db.n_combinations})")

    # Show value ranges per actuator
    print("\nActuator value ranges in database:")
    for i, col in enumerate(db.actuator_columns):
        unique_vals = np.unique(db.combinations[:, i])
        print(f"  {col}: {unique_vals}")


def verify_actuator_database(db: ActuatorDatabase, train_df: pd.DataFrame) -> Dict[str, bool]:
    """Verify database correctness"""
    checks = {}

    # All training actuator combinations should be in database
    train_actuators = train_df[ACTUATOR_COLUMNS].values
    in_db = query_actuator_database_vectorized(db, train_actuators)
    checks["all_train_in_db"] = in_db.all()

    # Database size matches unique combinations in training
    expected = np.unique(train_actuators, axis=0)
    checks["size_matches"] = len(db.combinations) == len(expected)

    # Combinations match exactly
    checks["combinations_match"] = np.array_equal(
        np.sort(db.combinations, axis=0),
        np.sort(expected, axis=0)
    )

    # No duplicates in database
    checks["no_duplicates"] = len(db.combinations) == len(np.unique(db.combinations, axis=0))

    return checks


if __name__ == "__main__":
    from daics.data.loader import load_normal
    from daics.data.split import chronological_split

    data_dir = Path("C:/Users/Aaditya/Desktop/Programming/IMDAI/data/SWaT")
    normal = load_normal(data_dir)
    attack = load_normal(data_dir)  # dummy
    split = chronological_split(normal, attack)

    db = build_actuator_database(split.train)
    print_actuator_database_summary(db)

    checks = verify_actuator_database(db, split.train)
    print("\n=== Database Verification ===")
    for check, passed in checks.items():
        print(f"  {check}: {'PASS' if passed else 'FAIL'}")

    # Test query on validation set
    val_actuators = split.val[ACTUATOR_COLUMNS].values
    in_db = query_actuator_database_vectorized(db, val_actuators)
    print(f"\nValidation set in database: {in_db.sum()}/{len(in_db)} ({100*in_db.mean():.1f}%)")

    # Test query on test attack set
    test_actuators = split.test_attack[ACTUATOR_COLUMNS].values
    in_db = query_actuator_database_vectorized(db, test_actuators)
    print(f"Test attack set in database: {in_db.sum()}/{len(in_db)} ({100*in_db.mean():.1f}%)")

    # Save database
    save_actuator_database(db, Path("C:/Users/Aaditya/Desktop/Programming/IMDAI/data/processed/actuator_db.joblib"))
    print("\nActuator database saved successfully")