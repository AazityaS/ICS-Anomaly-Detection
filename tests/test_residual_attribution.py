import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from daics.data.sensor_groups import SENSOR_GROUPS
from daics.explainability.residual_attribution import (
    attribute_section_residuals,
    find_anomaly_events,
    write_attribution_results,
)


class ResidualAttributionTests(unittest.TestCase):
    def test_manual_squared_error_and_contribution(self):
        actual = np.array([[1.0, 3.0, 2.0, 4.0], [0.0, 0.0, 0.0, 0.0]])
        predicted = np.array([[0.0, 1.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0]])
        # Sensor errors: (1+4+0+4)/4=2.25; second sensor=1.0.
        rows = attribute_section_residuals(
            actual, predicted, ["S_A", "S_B"], section="section_1",
            prediction_timestamps=["t0", "t1", "t2", "t3"],
            detection_timestamp="alarm", top_k=2,
        )
        self.assertEqual(rows[0]["sensor_name"], "S_A")
        self.assertEqual(rows[0]["squared_prediction_errors"], [1.0, 4.0, 0.0, 4.0])
        self.assertAlmostEqual(rows[0]["sensor_error_mse"], 2.25)
        self.assertAlmostEqual(rows[0]["contribution_percent"], 2.25 / 3.25 * 100)
        self.assertAlmostEqual(sum(row["contribution_percent"] for row in rows), 100.0)

    def test_wout_aggregation_uses_mean_per_sensor(self):
        actual = np.zeros((2, 4))
        predicted = np.array([[2.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        rows = attribute_section_residuals(
            actual, predicted, ["one", "two"], section="section_1",
            prediction_timestamps=list("abcd"), detection_timestamp="d", top_k=2,
        )
        by_sensor = {row["sensor_name"]: row for row in rows}
        self.assertEqual(by_sensor["one"]["sensor_error_mse"], 1.0)
        self.assertEqual(by_sensor["two"]["sensor_error_mse"], 1.0)
        self.assertAlmostEqual(sum(r["contribution_percent"] for r in rows), 100.0)

    def test_group_names_and_section_sizes(self):
        self.assertEqual([len(g.sensor_names) for g in SENSOR_GROUPS], [5, 7, 12])
        self.assertEqual(SENSOR_GROUPS[0].sensor_names,
                         ["FIT101", "LIT101", "AIT202", "AIT203", "FIT201"])
        self.assertNotIn("UV401", [s for group in SENSOR_GROUPS for s in group.sensor_names])
        self.assertEqual(len(set(s for group in SENSOR_GROUPS for s in group.sensor_names)), 24)

    def test_prediction_and_detection_timestamp_alignment(self):
        ts = [f"t{i}" for i in range(40)]
        flags = [False] * 4 + [True] * 31 + [False] * 5
        event, = find_anomaly_events(flags, ts, wanom=30)
        self.assertEqual(event["prediction_timestamp"], "t4")
        self.assertEqual(event["detection_timestamp"], "t33")
        actual = np.ones((1, 4))
        rows = attribute_section_residuals(
            actual, np.zeros_like(actual), ["FIT101"], section="section_1",
            prediction_timestamps=["t4", "t5", "t6", "t7"],
            detection_timestamp=event["detection_timestamp"], top_k=1,
        )
        self.assertEqual(rows[0]["prediction_timestamp"], "t4")
        self.assertEqual(rows[0]["timestamp"], "t33")
        self.assertEqual(rows[0]["prediction_timestamps"], ["t4", "t5", "t6", "t7"])

    def test_zero_total_error_is_safe(self):
        zeros = np.zeros((2, 4))
        rows = attribute_section_residuals(
            zeros, zeros, ["S1", "S2"], section="section_1",
            prediction_timestamps=list("abcd"), detection_timestamp="alarm", top_k=2,
        )
        self.assertTrue(all(row["contribution_percent"] == 0.0 for row in rows))
        self.assertTrue(all(row["sensor_error_mse"] == 0.0 for row in rows))

    def test_non_anomaly_outputs_remain_unchanged(self):
        predictions = np.array([0, 1, 0, 1], dtype=np.int32)
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        pred_before, labels_before = predictions.copy(), labels.copy()
        actual = np.ones((1, 4))
        attribute_section_residuals(
            actual, np.zeros_like(actual), ["S1"], section="section_1",
            prediction_timestamps=list("abcd"), detection_timestamp="alarm", top_k=1,
        )
        np.testing.assert_array_equal(predictions, pred_before)
        np.testing.assert_array_equal(labels, labels_before)

    def test_outputs_are_written_as_csv_and_json(self):
        record = attribute_section_residuals(
            np.ones((1, 4)), np.zeros((1, 4)), ["S1"], section="section_1",
            prediction_timestamps=list("abcd"), detection_timestamp="alarm", top_k=1,
        )[0]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "tmp") as tmp:
            paths = write_attribution_results([record], Path(tmp))
            self.assertTrue(paths["csv"].is_file())
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["explanations"][0]["sensor_name"], "S1")


if __name__ == "__main__":
    unittest.main()
