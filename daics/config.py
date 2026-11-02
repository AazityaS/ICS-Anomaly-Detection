"""
DAICS Configuration - Exact hyperparameters from Table IV (SWaT column)
Paper: "DAICS: A Deep Learning Solution for Anomaly Detection in Industrial Control Systems"
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class WindowConfig:
    """Time-series windowing parameters (Table IV)"""
    Win: int = 60          # Input time window (seconds/samples)
    H: int = 50            # Horizon (separation between input and output)
    Wout: int = 4          # Output time window (seconds/samples)
    Wanom: int = 30        # Anomaly waiting time (consecutive samples)


@dataclass(frozen=True)
class WDNNConfig:
    """WDNN architecture hyperparameters (Table IV - SWaT column)"""
    # Training
    learning_rate: float = 0.01
    batch_size: int = 32
    tuning_epochs: int = 100       # EPtuning for few-time-steps

    # Layer sizes (exact from Table IV)
    DL1_neurons: int = 4           # Wout (wide branch output)
    DL2_neurons: int = 180         # 3 * Win
    DL3_neurons: int = 4           # Wout
    DL4_neurons: int = 80

    # Per output section (formulas from Table IV, mgse varies per section)
    DL5_factor: float = 2.25       # 2.25 * mgse
    DL6_factor: float = 1.5        # 1.5 * mgse
    DL7_factor: float = 1.0        # mgse

    # Convolutional layers
    CL1_kernels: int = 64
    CL1_kernel_size: int = 2
    MP1_pool_size: int = 2
    CL2_kernels: int = 128
    CL2_kernel_size: int = 2
    MP2_pool_size: int = 2

    # Activation
    leaky_relu_alpha: float = 0.01

    # Number of output sections (Figure 2 shows 3)
    G: int = 3


@dataclass(frozen=True)
class TTNNConfig:
    """TTNN architecture hyperparameters (Table IV - SWaT column)"""
    learning_rate: float = 0.01
    batch_size: int = 32
    tuning_epochs: int = 1          # 1 epoch per online update

    # Median filter
    median_kernel_size: int = 59

    # Convolutional layers (exact from Table IV)
    CL1_kernels: int = 2
    CL1_kernel_size: int = 2
    MP1_pool_size: int = 2
    CL2_kernels: int = 4           # Wout (4 for SWaT)
    CL2_kernel_size: int = 2
    MP2_pool_size: int = 4         # Wout

    # Output
    DL_neurons: int = 1             # Single output neuron

    # Activation
    leaky_relu_alpha: float = 0.01


@dataclass(frozen=True)
class DataConfig:
    """Dataset configuration for our SWaT distribution"""
    # Our cleaned dataset has 24 sensors + 20 actuators = 44 features
    # (Paper uses 25 sensors + 26 actuators = 51; we exclude AIT201 and 6 actuators due to missing data)
    n_sensors: int = 24
    n_actuators: int = 20
    n_features: int = 44            # n_sensors + n_actuators

    # Sensor groups (stage-based approximation, see sensor_groups.py)
    # Section 1: Stage 1-2 sensors (5)
    # Section 2: Stage 3-4 sensors (7, excl UV401 which is actuator)
    # Section 3: Stage 5-6 sensors (12)
    mgse_per_section: Tuple[int, int, int] = (5, 7, 12)

    # Normalization
    sensor_min: float = 0.0
    sensor_max: float = 1.0


@dataclass(frozen=True)
class SplitConfig:
    """Chronological train/val/test split (Option A)"""
    # Training period: first 7 days of normal data
    train_start: str = "2015-12-22 16:00:00"
    train_end: str = "2015-12-29 16:00:00"        # Exactly 7 days
    val_ratio: float = 0.2                         # 20% of training for validation

    # Test period: remaining normal + all attacks
    test_start: str = "2015-12-29 16:00:00"

    # Note: Validation period (Dec 28 04:52 - Dec 29 15:59) overlaps with
    # attack observations (first attack Dec 28 10:29). Validation data contains
    # only "Normal" labeled records from normal.csv. This is documented as
    # a reproduction limitation in REPRODUCTION_NOTES.md.


# Global instances
WINDOW = WindowConfig()
WDNN = WDNNConfig()
TTNN = TTNNConfig()
DATA = DataConfig()
SPLIT = SplitConfig()


def get_dl5_neurons(mgse: int) -> int:
    """DL5 neurons = 2.25 * mgse (rounded to nearest int)"""
    return int(round(WDNN.DL5_factor * mgse))


def get_dl6_neurons(mgse: int) -> int:
    """DL6 neurons = 1.5 * mgse (rounded to nearest int)"""
    return int(round(WDNN.DL6_factor * mgse))


def get_dl7_neurons(mgse: int) -> int:
    """DL7 neurons = mgse"""
    return mgse


def verify_mgse_sum() -> bool:
    """Verify mgse_per_section sums to n_sensors"""
    return sum(DATA.mgse_per_section) == DATA.n_sensors


if __name__ == "__main__":
    print("=== DAICS Configuration ===")
    print(f"Window: Win={WINDOW.Win}, H={WINDOW.H}, Wout={WINDOW.Wout}, Wanom={WINDOW.Wanom}")
    print(f"WDNN: G={WDNN.G}, lr={WDNN.learning_rate}, batch={WDNN.batch_size}")
    print(f"  DL1={WDNN.DL1_neurons}, DL2={WDNN.DL2_neurons}, DL3={WDNN.DL3_neurons}, DL4={WDNN.DL4_neurons}")
    print(f"  CL1: {WDNN.CL1_kernels}x{WDNN.CL1_kernel_size}, MP1: {WDNN.MP1_pool_size}")
    print(f"  CL2: {WDNN.CL2_kernels}x{WDNN.CL2_kernel_size}, MP2: {WDNN.MP2_pool_size}")
    print(f"TTNN: lr={TTNN.learning_rate}, batch={TTNN.batch_size}")
    print(f"  CL1: {TTNN.CL1_kernels}x{TTNN.CL1_kernel_size}, MP1: {TTNN.MP1_pool_size}")
    print(f"  CL2: {TTNN.CL2_kernels}x{TTNN.CL2_kernel_size}, MP2: {TTNN.MP2_pool_size}")
    print(f"  Median kernel: {TTNN.median_kernel_size}")
    print(f"DATA: {DATA.n_sensors} sensors, {DATA.n_actuators} actuators = {DATA.n_features} features")
    print(f"  mgse per section: {DATA.mgse_per_section} (sum={sum(DATA.mgse_per_section)})")
    print(f"  DL5 neurons per section: {[get_dl5_neurons(m) for m in DATA.mgse_per_section]}")
    print(f"  DL6 neurons per section: {[get_dl6_neurons(m) for m in DATA.mgse_per_section]}")
    print(f"  DL7 neurons per section: {[get_dl7_neurons(m) for m in DATA.mgse_per_section]}")
    print(f"SPLIT: train={SPLIT.train_start} to {SPLIT.train_end}, val_ratio={SPLIT.val_ratio}")
    print(f"  mgse sum verified: {verify_mgse_sum()}")