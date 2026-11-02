"""
DAICS Sensor Groups - PLC-based Output Section Mapping
Maps 24 sensors to 3 output sections based on SWaT testbed stages.
Paper: "output section is split into multiple branches to serve large-scale ICSs
where the sensors are controlled by several PLCs"

Reference: Goh et al. 2016 "A dataset to support research in the design of secure
water treatment systems" - defines 6 stages with sensor/actuator mapping.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
from daics.config import DATA, WDNN
from daics.data.loader import SENSOR_COLUMNS


# SWaT Stage Definitions (from Goh et al. 2016 / iTrust documentation)
# Stage 1: Raw Water Supply
# Stage 2: Pre-treatment (Coagulation/Flocculation)
# Stage 3: Ultra-filtration
# Stage 4: Dechlorination / UV
# Stage 5: Reverse Osmosis
# Stage 6: Backwash / Distribution

STAGE_SENSORS = {
    1: ["FIT101", "LIT101"],                    # Flow, Level - raw water
    2: ["AIT201", "AIT202", "AIT203", "FIT201"], # Analyzers, Flow - pre-treatment
    3: ["DPIT301", "FIT301", "LIT301"],         # Pressure, Flow, Level - UF
    4: ["AIT401", "AIT402", "FIT401", "LIT401"], # Analyzers, Flow, Level - dechlorination/UV
    5: ["AIT501", "AIT502", "AIT503", "AIT504",
        "FIT501", "FIT502", "FIT503", "FIT504"], # Analyzers, Flows - RO
    6: ["PIT501", "PIT502", "PIT503", "FIT601"]  # Pressures, Flow - backwash/distribution
}

# Our available sensors (24, AIT201 excluded)
AVAILABLE_SENSORS = [s for s in SENSOR_COLUMNS if s != "AIT201"]

# Verify all available sensors are mapped
MAPPED_SENSORS = []
for stage, sensors in STAGE_SENSORS.items():
    MAPPED_SENSORS.extend(sensors)

UNMAPPED = [s for s in AVAILABLE_SENSORS if s not in MAPPED_SENSORS]
if UNMAPPED:
    print(f"WARNING: Unmapped sensors: {UNMAPPED}")

# Paper's Figure 2 shows G=3 output sections
# Map 6 stages -> 3 sections (2 stages per section, matching "pair of PLCs" in paper)
SECTION_STAGES = {
    0: [1, 2],   # Section 1: Stages 1-2 (Raw water + Pre-treatment)
    1: [3, 4],   # Section 2: Stages 3-4 (UF + Dechlorination/UV)
    2: [5, 6],   # Section 3: Stages 5-6 (RO + Backwash)
}


@dataclass(frozen=True)
class SensorGroup:
    """Sensor group for one WDNN output section"""
    section_idx: int
    stages: List[int]
    sensor_names: List[str]
    mgse: int
    sensor_indices: List[int]  # Indices into full sensor array (0-23)


def build_sensor_groups() -> List[SensorGroup]:
    """
    Build sensor groups for G=3 output sections.
    Returns list of SensorGroup objects.
    """
    groups = []

    for section_idx in range(WDNN.G):  # G=3
        stages = SECTION_STAGES[section_idx]
        sensors = []
        for stage in stages:
            sensors.extend(STAGE_SENSORS[stage])

        # Filter to available sensors only
        available_sensors = [s for s in sensors if s in AVAILABLE_SENSORS]

        # Get indices in full sensor array
        sensor_indices = [AVAILABLE_SENSORS.index(s) for s in available_sensors]

        groups.append(SensorGroup(
            section_idx=section_idx,
            stages=stages,
            sensor_names=available_sensors,
            mgse=len(available_sensors),
            sensor_indices=sensor_indices
        ))

    return groups


def get_section_masks(sensor_groups: List[SensorGroup], n_sensors: int = DATA.n_sensors) -> List:
    """Get boolean masks for each output section"""
    masks = []
    for group in sensor_groups:
        mask = np.zeros(n_sensors, dtype=bool)
        mask[group.sensor_indices] = True
        masks.append(mask)
    return masks


def get_mgse_per_section(sensor_groups: List[SensorGroup]) -> Tuple[int, ...]:
    """Get mgse (number of sensors) per section as tuple"""
    return tuple(g.mgse for g in sensor_groups)


def print_sensor_groups(groups: List[SensorGroup]) -> None:
    """Print sensor group mapping"""
    print("\n=== Sensor Groups (Output Sections) ===")
    print(f"Total sensors: {sum(g.mgse for g in groups)} (expected {DATA.n_sensors})")
    print(f"G = {WDNN.G} sections\n")

    for g in groups:
        print(f"Section {g.section_idx + 1} (Stages {g.stages}): {g.mgse} sensors")
        for s in g.sensor_names:
            print(f"  {s}")
        print(f"  Indices: {g.sensor_indices}")


def verify_sensor_groups(groups: List[SensorGroup]) -> Dict[str, bool]:
    """Verify sensor group correctness"""
    checks = {}

    # All sensors covered exactly once
    all_indices = []
    for g in groups:
        all_indices.extend(g.sensor_indices)
    checks["all_sensors_covered"] = len(all_indices) == DATA.n_sensors
    checks["no_duplicate_sensors"] = len(all_indices) == len(set(all_indices))
    checks["indices_in_range"] = all(0 <= i < DATA.n_sensors for i in all_indices)

    # mgse matches config
    mgse_tuple = get_mgse_per_section(groups)
    checks["mgse_matches_config"] = mgse_tuple == DATA.mgse_per_section

    # Number of sections matches G
    checks["correct_num_sections"] = len(groups) == WDNN.G

    return checks


# Build groups at module import
import numpy as np
SENSOR_GROUPS = build_sensor_groups()
SECTION_MASKS = get_section_masks(SENSOR_GROUPS)
MGSE_PER_SECTION = get_mgse_per_section(SENSOR_GROUPS)


if __name__ == "__main__":
    print_sensor_groups(SENSOR_GROUPS)

    checks = verify_sensor_groups(SENSOR_GROUPS)
    print("\n=== Sensor Group Verification ===")
    for check, passed in checks.items():
        print(f"  {check}: {'PASS' if passed else 'FAIL'}")

    print(f"\nmgse_per_section: {MGSE_PER_SECTION}")
    print(f"Sum: {sum(MGSE_PER_SECTION)}")