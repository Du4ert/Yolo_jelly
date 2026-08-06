"""Тесты простой угловой модели вертикального зазора."""

import sys
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from angle_vertical_calibration_experiment import (  # noqa: E402
    exact_pair_distances,
    exact_vertical_k_percent,
    run_experiment,
    select_pair_angle,
    simple_gamma,
)
from camera_geometry import CameraCalibration  # noqa: E402
from camera_geometry import _find_size_pairs  # noqa: E402


class SimpleGammaTests(unittest.TestCase):
    def setUp(self):
        self.calibration = CameraCalibration(
            frame_width=1000,
            frame_height=500,
            fov_horizontal=100.0,
            fov_vertical=50.0,
        )

    def test_object_at_foe_has_zero_angle(self):
        gamma, cosine = simple_gamma(0.6, 0.4, 600.0, 200.0, self.calibration)
        self.assertAlmostEqual(gamma, 0.0)
        self.assertAlmostEqual(cosine, 1.0)

    def test_uses_independent_horizontal_and_vertical_fov(self):
        gamma, cosine = simple_gamma(0.6, 0.6, 500.0, 250.0, self.calibration)
        self.assertAlmostEqual(gamma, np.hypot(10.0, 5.0))
        self.assertAlmostEqual(cosine, np.cos(np.radians(np.hypot(10.0, 5.0))))

    def test_missing_local_foe_returns_none(self):
        geometry = pd.DataFrame([{
            "frame_start": 100,
            "frame_end": 120,
            "foe_x": 500.0,
            "foe_y": 250.0,
            "confidence": 0.9,
            "flow_regime": "radial",
        }])
        result = select_pair_angle(
            10, 20, 0.5, 0.5, geometry, self.calibration
        )
        self.assertIsNone(result)

    def test_low_confidence_or_non_radial_foe_returns_none(self):
        for confidence, regime in ((0.4, "radial"), (0.9, "uniform")):
            with self.subTest(confidence=confidence, regime=regime):
                geometry = pd.DataFrame([{
                    "frame_start": 10,
                    "frame_end": 20,
                    "foe_x": 500.0,
                    "foe_y": 250.0,
                    "confidence": confidence,
                    "flow_regime": regime,
                }])
                result = select_pair_angle(
                    10, 20, 0.5, 0.5, geometry, self.calibration
                )
                self.assertIsNone(result)


class ExactVerticalPairTests(unittest.TestCase):
    def test_recovers_endpoint_vertical_offset(self):
        vertical_offset = 1.6
        cosine = 0.8
        endpoint_slant = vertical_offset / cosine
        delta_depth = 0.5
        start_slant = np.sqrt(
            endpoint_slant**2
            + delta_depth**2
            + 2 * endpoint_slant * delta_depth * cosine
        )
        ratio = start_slant / endpoint_slant

        k_percent = exact_vertical_k_percent(
            100.0,
            100.0 * ratio,
            delta_depth,
            cosine,
        )

        self.assertAlmostEqual(k_percent, 100.0 / vertical_offset)

        slant, recovered_vertical = exact_pair_distances(
            100.0,
            100.0 * ratio,
            delta_depth,
            cosine,
        )
        self.assertAlmostEqual(slant, endpoint_slant)
        self.assertAlmostEqual(recovered_vertical, vertical_offset)

    def test_axial_case_reduces_to_finite_growth(self):
        k_percent = exact_vertical_k_percent(100.0, 125.0, 0.5, 1.0)
        self.assertAlmostEqual(k_percent, 50.0)

    def test_rejects_missing_growth(self):
        with self.assertRaisesRegex(ValueError, "увеличиваться"):
            exact_vertical_k_percent(100.0, 100.0, 0.5, 0.8)


class ProductionAnglePairTests(unittest.TestCase):
    def setUp(self):
        self.calibration = CameraCalibration(
            frame_width=1000,
            frame_height=500,
            fov_horizontal=100.0,
            fov_vertical=50.0,
            distance_coef_A=100.0,
            distance_coef_B=-1.0,
            pixel_calib_C=1.0,
            pixel_calib_D=0.0,
            angle_distance_coef_A=100.0,
            angle_distance_coef_B=-1.0,
            angle_pixel_calib_C=1.0,
            angle_pixel_calib_D=0.0,
        )
        vertical_offset = 1.6
        cosine = 0.8
        endpoint_slant = vertical_offset / cosine
        delta_depth = 0.5
        start_slant = np.sqrt(
            endpoint_slant**2
            + delta_depth**2
            + 2 * endpoint_slant * delta_depth * cosine
        )
        ratio = start_slant / endpoint_slant
        gamma_deg = np.degrees(np.arccos(cosine))
        self.valid_df = pd.DataFrame([
            {
                "frame": 10,
                "depth_m": 1.0,
                "size_pix": 100.0,
                "x_center": 0.5 + gamma_deg / 100.0,
                "y_center": 0.5,
            },
            {
                "frame": 20,
                "depth_m": 1.5,
                "size_pix": 100.0 * ratio,
                "x_center": 0.5 + gamma_deg / 100.0,
                "y_center": 0.5,
            },
        ])

    def test_production_pair_uses_angle_vertical_offset(self):
        geometry = pd.DataFrame([{
            "frame_start": 0,
            "frame_end": 30,
            "foe_x": 500.0,
            "foe_y": 250.0,
            "confidence": 0.9,
            "flow_regime": "radial",
            "tilt_horizontal_deg": 0.0,
            "tilt_vertical_deg": 0.0,
        }])
        pairs, applied, _ = _find_size_pairs(
            self.valid_df,
            1.1,
            0.01,
            True,
            geometry,
            self.calibration,
        )
        self.assertTrue(applied)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["distance_model"], "simple_endpoint_angle")
        self.assertAlmostEqual(pairs[0]["vertical_offset_m"], 1.6)
        self.assertAlmostEqual(pairs[0]["slant_distance_m"], 2.0)
        self.assertAlmostEqual(pairs[0]["object_depth"], 3.1)

    def test_missing_local_foe_uses_legacy_pair(self):
        geometry = pd.DataFrame([{
            "frame_start": 100,
            "frame_end": 120,
            "foe_x": 500.0,
            "foe_y": 250.0,
            "confidence": 0.9,
            "flow_regime": "radial",
            "tilt_horizontal_deg": 0.0,
            "tilt_vertical_deg": 0.0,
        }])
        pairs, _, _ = _find_size_pairs(
            self.valid_df,
            1.1,
            0.01,
            True,
            geometry,
            self.calibration,
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["distance_model"], "legacy_vertical")
        self.assertIsNone(pairs[0]["slant_distance_m"])


class LocalAngleVerticalExperimentTests(unittest.TestCase):
    def test_step4_experiment_is_reproducible(self):
        geometry_path = (
            ROOT / "test_video" / "2down" / "output" / "ball_geometry_step4.csv"
        )
        expected_path = ROOT / "calibration_angle_vertical_experiment.json"
        if not geometry_path.exists():
            self.skipTest("Локальная geometry шага 4 отсутствует")
        report = run_experiment(
            ROOT / "test_video",
            geometry_name="ball_geometry_step4.csv",
            production_calibration_path=ROOT / "calibration_xy.json",
        )
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        self.assertEqual(report["coverage"], expected["coverage"])
        self.assertEqual(report["videos"], expected["videos"])
        self.assertEqual(report["full_fit"], expected["full_fit"])
        self.assertEqual(
            report["leave_one_video_out"],
            expected["leave_one_video_out"],
        )


if __name__ == "__main__":
    unittest.main()
