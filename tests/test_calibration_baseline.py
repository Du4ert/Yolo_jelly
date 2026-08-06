"""Regression tests for the frozen camera-calibration baseline."""

import json
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from calibration_baseline import _summary, build_baseline_report  # noqa: E402
from camera_geometry import (  # noqa: E402
    CameraCalibration,
    _calculate_distance_from_k,
    _calculate_k_for_pair,
    _calculate_pixel_calibration,
    _calculate_size_mm,
    _discover_test_dirs,
    _find_size_pairs,
    calibrate_coefficients,
)


class CalculationBaselineTests(unittest.TestCase):
    def test_k_distance_and_size_chain(self):
        calibration = CameraCalibration(
            distance_coef_A=80.0,
            distance_coef_B=-0.9,
            pixel_calib_C=4.35,
            pixel_calib_D=-1.25,
        )
        k_fraction = _calculate_k_for_pair(100.0, 110.0, 1.0, 1.5)
        self.assertAlmostEqual(k_fraction, 0.2)
        distance = _calculate_distance_from_k(k_fraction * 100.0, calibration)
        self.assertAlmostEqual(distance, 80.0 * 20.0**-0.9)
        pixel_calibration = _calculate_pixel_calibration(distance, calibration)
        self.assertAlmostEqual(pixel_calibration, 4.35 * distance**-1.25)
        self.assertAlmostEqual(
            _calculate_size_mm(110.0, pixel_calibration),
            110.0 / pixel_calibration,
        )

    def test_summary_is_deterministic(self):
        self.assertEqual(
            _summary([1.0, 2.0, 3.0, float("nan")]),
            {
                "count": 3,
                "mean": 2.0,
                "median": 2.0,
                "p95": 2.9,
                "min": 1.0,
                "max": 3.0,
            },
        )

    def test_pair_depth_and_growth_thresholds_are_independent(self):
        detections = pd.DataFrame({
            "frame": [0, 1],
            "depth_m": [1.0, 1.02],
            "size_pix": [100.0, 120.0],
            "x_center": [0.5, 0.5],
            "y_center": [0.5, 0.5],
        })
        calibration = CameraCalibration(frame_width=100, frame_height=100)

        accepted, _, _ = _find_size_pairs(
            detections, 1.1, 0.01, False, None, calibration
        )
        rejected_by_depth, _, _ = _find_size_pairs(
            detections, 1.1, 0.05, False, None, calibration
        )
        rejected_by_growth, _, _ = _find_size_pairs(
            detections, 1.3, 0.01, False, None, calibration
        )

        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected_by_depth, [])
        self.assertEqual(rejected_by_growth, [])


class LocalCalibrationDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = ROOT / "test_video"
        cls.expected_path = ROOT / "calibration_baseline.json"
        if not cls.test_dir.exists():
            raise unittest.SkipTest("Local ignored calibration dataset is unavailable")
        cls.actual = build_baseline_report(
            cls.test_dir,
            ROOT / "calibration_xy.json",
        )
        cls.expected = json.loads(cls.expected_path.read_text(encoding="utf-8"))

    def test_dataset_structure_matches_baseline(self):
        self.assertEqual(self.actual["settings"], self.expected["settings"])
        self.assertEqual(self.actual["totals"], self.expected["totals"])
        self.assertEqual(self.actual["videos"], self.expected["videos"])
        self.assertEqual(self.actual["inputs"], self.expected["inputs"])

    def test_calculation_metrics_match_baseline(self):
        self.assertEqual(self.actual["calibration"], self.expected["calibration"])
        self.assertEqual(self.actual["metrics"], self.expected["metrics"])
        self.assertEqual(self.actual["tracks"], self.expected["tracks"])

    def test_refit_reproduces_calibration_coefficients(self):
        calibration = calibrate_coefficients(
            _discover_test_dirs(str(self.test_dir)),
            verbose=False,
        )
        actual = {
            "distance_coef_A": calibration.distance_coef_A,
            "distance_coef_B": calibration.distance_coef_B,
            "pixel_calib_C": calibration.pixel_calib_C,
            "pixel_calib_D": calibration.pixel_calib_D,
            "distortion_x_k1": calibration.distortion_x_k1,
            "distortion_x_k2": calibration.distortion_x_k2,
            "distortion_y_k1": calibration.distortion_y_k1,
            "distortion_y_k2": calibration.distortion_y_k2,
        }
        expected = {
            key: self.expected["calibration"][key]
            for key in actual
        }
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
