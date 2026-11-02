import tempfile
import unittest
from pathlib import Path

import numpy as np

from daics.robustness.noise import (
    NOISE_CONDITIONS,
    apply_normalized_sensor_noise,
    load_or_create_noise,
)


class RobustnessNoiseTests(unittest.TestCase):
    def test_conditions_and_deterministic_saved_realization(self):
        self.assertEqual(list(NOISE_CONDITIONS.values()), [0.0, 0.01, 0.05, 0.1])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "tmp") as tmp:
            path = Path(tmp) / "noise.npz"
            std = np.array([0.2, 0.4], dtype=np.float32)
            n1, sigma1 = load_or_create_noise(path, (30, 2), std, 0.1, 1234)
            n2, sigma2 = load_or_create_noise(path, (30, 2), std, 0.1, 1234)
            np.testing.assert_array_equal(n1, n2)
            np.testing.assert_array_equal(sigma1, sigma2)
            np.testing.assert_allclose(sigma1, [0.02, 0.04])

    def test_only_normalized_sensors_change_without_clipping(self):
        class Dummy:
            pass
        data = Dummy()
        data.X = np.column_stack([np.full((3, 2), 0.99, dtype=np.float32),
                                  np.array([[1, 2], [2, 1], [1, 1]], dtype=np.float32)])
        data.y_sensors = np.full((3, 2), 0.99, dtype=np.float32)
        data.labels = np.array([0, 1, 0], dtype=np.int32)
        data.n_sensors = 2
        noise = np.full((3, 2), 0.05, dtype=np.float32)
        actuators_before, labels_before = apply_normalized_sensor_noise(data, noise)
        self.assertTrue(np.all(data.X[:, :2] > 1.0))
        np.testing.assert_array_equal(data.X[:, 2:], actuators_before)
        np.testing.assert_array_equal(data.labels, labels_before)
        np.testing.assert_array_equal(data.y_sensors, np.full((3, 2), 1.04, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
