"""Deterministic, reusable sensor-noise realizations for robustness evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np


NOISE_CONDITIONS = {
    "clean": 0.0,
    "low": 0.01,
    "medium": 0.05,
    "high": 0.10,
}
BASE_SEED = 20261031


def load_or_create_noise(
    path: Path,
    shape: tuple[int, int],
    training_std: np.ndarray,
    multiplier: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a saved exact realization or create and persist it.

    Noise is stored in normalized sensor units and is reusable by later model
    comparisons. ``training_std`` is one population standard deviation per
    sensor, calculated from normalized training-only observations.
    """
    path = Path(path)
    sigma = np.asarray(training_std, dtype=np.float32) * np.float32(multiplier)
    if sigma.shape != (shape[1],):
        raise ValueError("training_std must have one value per sensor")
    if path.exists():
        with np.load(path) as saved:
            noise = saved["noise"]
            saved_sigma = saved["sigma"]
            saved_seed = int(saved["seed"])
            saved_multiplier = float(saved["multiplier"])
        if noise.shape != shape or saved_seed != seed or saved_multiplier != multiplier:
            raise ValueError(f"Saved noise realization metadata mismatch: {path}")
        if not np.array_equal(saved_sigma, sigma):
            raise ValueError(f"Training noise scale changed since realization was saved: {path}")
        return noise.astype(np.float32, copy=False), sigma

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=shape).astype(np.float32)
    np.savez_compressed(path, noise=noise, sigma=sigma,
                        seed=np.int64(seed), multiplier=np.float64(multiplier))
    return noise, sigma


def apply_normalized_sensor_noise(window_dataset, noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Add one source-length noise array to sensor inputs and targets in place.

    The arrays are normalized time-series stored by ``WindowDataset`` before
    any sliding windows are materialized. Actuators, labels and timestamps are
    not modified. Values are deliberately not clipped to [0, 1]. Returns
    actuator and label snapshots for verification.
    """
    noise = np.asarray(noise, dtype=np.float32)
    base = window_dataset
    if noise.shape != (len(base.X), base.n_sensors):
        raise ValueError(f"Noise shape {noise.shape} does not match source {(len(base.X), base.n_sensors)}")
    n_sensors = base.n_sensors
    actuator_snapshot = base.X[:, n_sensors:].copy()
    label_snapshot = base.labels.copy()
    base.X[:, :n_sensors] += noise
    base.y_sensors += noise
    if not np.array_equal(base.X[:, n_sensors:], actuator_snapshot):
        raise RuntimeError("Actuator values changed during sensor noise injection")
    if not np.array_equal(base.labels, label_snapshot):
        raise RuntimeError("Labels changed during sensor noise injection")
    return actuator_snapshot, label_snapshot
