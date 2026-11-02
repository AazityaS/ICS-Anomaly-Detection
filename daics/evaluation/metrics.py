"""
DAICS Evaluation - Point-based Metrics
Paper: Precision, Recall, F1, FPR, Confusion Matrix, Detection Delay, Attacks Detected
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    # Point-based metrics
    precision: float
    recall: float
    f1: float
    fpr: float  # False Positive Rate

    # Confusion matrix
    tp: int
    fp: int
    tn: int
    fn: int

    # Detection delay
    detection_delays: List[float]  # Per attack (in seconds/samples)
    avg_detection_delay: float

    # Attacks detected
    n_attacks_total: int
    n_attacks_detected: int
    attacks_detected_detail: List[Dict]


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[int, int, int, int]:
    """
    Compute confusion matrix elements.
    y_true: (n_samples,) 0=Normal, 1=Attack
    y_pred: (n_samples,) 0=Normal, 1=Attack
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return int(tp), int(fp), int(tn), int(fn)


def compute_point_metrics(
    tp: int, fp: int, tn: int, fn: int
) -> Tuple[float, float, float, float]:
    """Compute precision, recall, F1, FPR"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return precision, recall, f1, fpr


def find_attacks(y_true: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find contiguous attack segments in ground truth.
    Returns list of (start_idx, end_idx) for each attack.
    """
    attacks = []
    in_attack = False
    start = 0

    for i, label in enumerate(y_true):
        if label == 1 and not in_attack:
            in_attack = True
            start = i
        elif label == 0 and in_attack:
            in_attack = False
            attacks.append((start, i - 1))

    if in_attack:
        attacks.append((start, len(y_true) - 1))

    return attacks


def compute_detection_delay(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_rate: float = 1.0  # samples per second
) -> Tuple[List[float], float, int, int, List[Dict]]:
    """
    Compute detection delay for each attack.

    Detection delay = time from attack start to first detected anomaly.
    If attack not detected, delay = attack duration (or NaN).

    Returns:
        delays: List of detection delays per attack (in seconds)
        avg_delay: Average detection delay (only for detected attacks)
        n_detected: Number of attacks detected
        n_total: Total number of attacks
        detail: Per-attack details
    """
    attacks = find_attacks(y_true)
    n_total = len(attacks)

    delays = []
    detail = []
    n_detected = 0

    for start, end in attacks:
        attack_duration = end - start + 1

        # Find first detection within attack window
        attack_pred = y_pred[start:end+1]
        detected_indices = np.where(attack_pred == 1)[0]

        if len(detected_indices) > 0:
            first_detection = detected_indices[0]
            delay_samples = first_detection
            delay_seconds = delay_samples / sample_rate
            detected = True
            n_detected += 1
        else:
            delay_samples = attack_duration  # Not detected
            delay_seconds = delay_samples / sample_rate
            detected = False

        delays.append(delay_seconds)
        detail.append({
            'attack_start': start,
            'attack_end': end,
            'duration_samples': attack_duration,
            'delay_samples': delay_samples,
            'delay_seconds': delay_seconds,
            'detected': detected
        })

    avg_delay = np.mean([d for d, det in zip(delays, [d['detected'] for d in detail]) if det]) if n_detected > 0 else float('nan')

    return delays, avg_delay, n_detected, n_total, detail


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_rate: float = 1.0
) -> EvaluationResult:
    """
    Full evaluation pipeline.

    Args:
        y_true: (n_samples,) ground truth labels (0=Normal, 1=Attack)
        y_pred: (n_samples,) predicted labels (0=Normal, 1=Attack)
        sample_rate: Samples per second (default 1 for SWaT)

    Returns:
        EvaluationResult with all metrics
    """
    # Confusion matrix
    tp, fp, tn, fn = compute_confusion_matrix(y_true, y_pred)

    # Point-based metrics
    precision, recall, f1, fpr = compute_point_metrics(tp, fp, tn, fn)

    # Detection delay
    delays, avg_delay, n_detected, n_total, detail = compute_detection_delay(
        y_true, y_pred, sample_rate
    )

    return EvaluationResult(
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=fpr,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        detection_delays=delays,
        avg_detection_delay=avg_delay,
        n_attacks_total=n_total,
        n_attacks_detected=n_detected,
        attacks_detected_detail=detail
    )


def print_evaluation(result: EvaluationResult) -> None:
    """Pretty print evaluation results"""
    print("\n=== DAICS Evaluation Results ===")
    print(f"Point-based Metrics:")
    print(f"  Precision: {result.precision:.4f}")
    print(f"  Recall:    {result.recall:.4f}")
    print(f"  F1 Score:  {result.f1:.4f}")
    print(f"  FPR:       {result.fpr:.6f}")

    print(f"\nConfusion Matrix:")
    print(f"  TP: {result.tp:6d}  FP: {result.fp:6d}")
    print(f"  FN: {result.fn:6d}  TN: {result.tn:6d}")

    print(f"\nAttack Detection:")
    print(f"  Total attacks: {result.n_attacks_total}")
    print(f"  Detected:      {result.n_attacks_detected}")
    if result.n_attacks_detected > 0:
        print(f"  Avg delay:     {result.avg_detection_delay:.2f} seconds")

    print(f"\nPer-Attack Detail:")
    for i, d in enumerate(result.attacks_detected_detail):
        status = "DETECTED" if d['detected'] else "MISSED"
        print(f"  Attack {i+1}: [{d['attack_start']}-{d['attack_end']}] "
              f"dur={d['duration_samples']}s delay={d['delay_seconds']:.1f}s {status}")


def evaluate_per_section(
    y_true: np.ndarray,
    section_anomalies: List[np.ndarray],
    sample_rate: float = 1.0
) -> Dict[int, EvaluationResult]:
    """Evaluate each section independently"""
    results = {}
    for g, y_pred in enumerate(section_anomalies):
        results[g] = evaluate(y_true, y_pred, sample_rate)
        print(f"\n--- Section {g} ---")
        print_evaluation(results[g])
    return results


if __name__ == "__main__":
    # Quick test
    print("Testing evaluation metrics...")

    # Simple test case
    y_true = np.array([0,0,0,1,1,1,0,0,1,1,0,0,0])
    y_pred = np.array([0,0,1,1,1,0,0,0,1,1,1,0,0])

    result = evaluate(y_true, y_pred)
    print_evaluation(result)

    # Test perfect detection
    y_true2 = np.array([0]*100 + [1]*50 + [0]*100)
    y_pred2 = np.array([0]*100 + [1]*50 + [0]*100)
    result2 = evaluate(y_true2, y_pred2)
    assert result2.precision == 1.0 and result2.recall == 1.0
    assert result2.n_attacks_detected == 1
    assert result2.avg_detection_delay == 0.0
    print("\nPerfect detection test: PASSED")

    # Test no detection
    y_pred3 = np.zeros_like(y_true2)
    result3 = evaluate(y_true2, y_pred3)
    assert result3.recall == 0.0
    assert result3.n_attacks_detected == 0
    print("No detection test: PASSED")

    print("\nAll evaluation tests passed!")