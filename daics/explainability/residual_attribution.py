"""Sensor-level post-hoc residual attribution for frozen DAICS detections.

This module does not alter WDNN predictions, thresholds, or anomaly decisions.
It describes residual magnitude, not causal responsibility.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def attribute_section_residuals(
    actual: np.ndarray,
    predicted: np.ndarray,
    sensor_names: Sequence[str],
    *,
    section: str,
    prediction_timestamps: Sequence[str],
    detection_timestamp: str,
    anomaly_label: str = "Anomaly",
    top_k: int = 5,
) -> list[dict]:
    """Return top-k sensor residual attributions for one detected section event.

    ``actual`` and ``predicted`` must be ``(sensors, Wout)`` arrays. Per-sensor
    error is the mean squared residual across Wout. Contributions are
    normalized over *all* sensors in the section before selecting top-k.
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.shape != predicted.shape or actual.ndim != 2:
        raise ValueError("actual and predicted must have the same (sensors, Wout) shape")
    if actual.shape[0] != len(sensor_names):
        raise ValueError("sensor_names must match the first tensor dimension")
    if actual.shape[1] != len(prediction_timestamps):
        raise ValueError("prediction_timestamps must contain one timestamp per Wout step")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    squared = np.square(actual - predicted)
    sensor_errors = squared.mean(axis=1)
    total_error = float(sensor_errors.sum())
    contributions = (sensor_errors / total_error * 100.0
                     if total_error > 0.0 else np.zeros_like(sensor_errors))
    order = np.argsort(-sensor_errors, kind="stable")[:min(top_k, len(sensor_names))]

    records = []
    for rank, sensor_idx in enumerate(order, start=1):
        records.append({
            "timestamp": str(detection_timestamp),
            "prediction_timestamp": str(prediction_timestamps[0]),
            "prediction_timestamps": [str(t) for t in prediction_timestamps],
            "value_scale": "training_fitted_minmax_normalized",
            "anomaly_label": anomaly_label,
            "section": section,
            "sensor_name": str(sensor_names[sensor_idx]),
            "actual_readings": actual[sensor_idx].tolist(),
            "predicted_readings": predicted[sensor_idx].tolist(),
            "squared_prediction_errors": squared[sensor_idx].tolist(),
            "sensor_error_mse": float(sensor_errors[sensor_idx]),
            "contribution_percent": float(contributions[sensor_idx]),
            "contribution_rank": rank,
            "section_total_sensor_error": total_error,
        })
    return records


def find_anomaly_events(
    section_flags: Sequence[bool], timestamps: Sequence[str], wanom: int
) -> list[dict]:
    """Find contiguous detected section episodes and their Wanom confirmation.

    The saved detector backdates a confirmed run across its first Wanom
    samples. The episode onset is therefore the first true output row; the
    online confirmation time is onset + Wanom - 1 rows.
    """
    flags = np.asarray(section_flags, dtype=bool)
    if len(flags) != len(timestamps):
        raise ValueError("section_flags and timestamps must have equal length")
    events = []
    i = 0
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue
        start = i
        while i < len(flags) and flags[i]:
            i += 1
        end = i - 1
        confirmation_idx = min(start + wanom - 1, end)
        events.append({
            "onset_index": start,
            "end_index": end,
            "confirmation_index": confirmation_idx,
            "prediction_timestamp": str(timestamps[start]),
            "detection_timestamp": str(timestamps[confirmation_idx]),
        })
    return events


def summarize_attributions(records: Iterable[dict]) -> dict:
    """Summarize sensor frequency among top-five and rank-one explanations."""
    rows = list(records)
    top_five = Counter(row["sensor_name"] for row in rows)
    rank_one = Counter(row["sensor_name"] for row in rows
                       if row["contribution_rank"] == 1)
    return {
        "explanation_records": len(rows),
        "top_five_frequency": dict(top_five.most_common()),
        "rank_one_frequency": dict(rank_one.most_common()),
    }


def write_attribution_results(records: Sequence[dict], output_dir: Path) -> dict:
    """Write CSV records, structured JSON, and a compact sensor summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_attributions(records)

    csv_path = output_dir / "sensor_attributions.csv"
    json_path = output_dir / "anomaly_explanations.json"
    summary_path = output_dir / "sensor_summary.json"
    columns = [
        "source", "timestamp", "prediction_timestamp", "prediction_timestamps", "value_scale",
        "anomaly_label", "section", "sensor_name", "actual_readings",
        "predicted_readings", "squared_prediction_errors", "sensor_error_mse",
        "contribution_percent", "contribution_rank", "section_total_sensor_error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({
                key: json.dumps(record[key]) if isinstance(record.get(key), list)
                else record.get(key, "")
                for key in columns
            })
    json_path.write_text(json.dumps({"explanations": records}, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "summary": summary_path}
