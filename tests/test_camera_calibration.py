"""Точечные тесты схемы и XY-коррекции калибровки камеры."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from camera_geometry import (  # noqa: E402
    CameraCalibration,
    _corrected_k_percent,
    _distortion_factor_xy,
    _distortion_profile_is_valid,
)


class DistortionFactorTests(unittest.TestCase):
    def test_center_has_no_correction(self):
        self.assertEqual(_distortion_factor_xy(0, 0, 0.2, 0.3, -0.1, 0.4), 1.0)

    def test_independent_axes_are_multiplied(self):
        rx, ry = 0.5, 0.25
        fx = 1 + 0.2 * rx**2 + 0.3 * rx**4
        fy = 1 - 0.1 * ry**2 + 0.4 * ry**4
        self.assertAlmostEqual(
            _distortion_factor_xy(rx, ry, 0.2, 0.3, -0.1, 0.4),
            fx * fy,
        )

    def test_corrected_k_uses_both_axes_and_both_pair_ends(self):
        pair = {
            "size_pixels_start": 100.0,
            "size_pixels": 120.0,
            "k_percent": 20.0,
            "k_raw_percent": 20.0,
            "cos_tilt": 1.0,
            "rx_norm_start": 0.0,
            "ry_norm_start": 0.0,
            "rx_norm": 1.0,
            "ry_norm": 1.0,
        }
        # Конечный множитель: 1.1 * 1.2 = 1.32; delta_depth исходной пары = 1 м.
        self.assertAlmostEqual(
            _corrected_k_percent(pair, 0.1, 0.0, 0.2, 0.0),
            58.4,
            places=7,
        )

    def test_invalid_negative_axis_profile_is_rejected(self):
        self.assertTrue(_distortion_profile_is_valid(0.2, 0.3, -0.1, 0.4))
        self.assertFalse(_distortion_profile_is_valid(-1.0, -1.0, 0.0, 0.0))


class CalibrationSchemaTests(unittest.TestCase):
    def test_schema_v2_round_trip(self):
        calibration = CameraCalibration(
            distortion_x_k1=0.1,
            distortion_x_k2=0.2,
            distortion_y_k1=-0.05,
            distortion_y_k2=0.3,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "calibration.json"
            calibration.to_json(str(path))
            loaded = CameraCalibration.from_json(str(path))
        self.assertEqual(loaded.distortion_x_k1, 0.1)
        self.assertEqual(loaded.distortion_x_k2, 0.2)
        self.assertEqual(loaded.distortion_y_k1, -0.05)
        self.assertEqual(loaded.distortion_y_k2, 0.3)

    def test_legacy_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.json"
            path.write_text(
                json.dumps({"distortion_k1": -0.7, "distortion_k2": 0.5}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "schema_version=2"):
                CameraCalibration.from_json(str(path))


if __name__ == "__main__":
    unittest.main()
