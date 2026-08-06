"""Синтетические тесты чистой геометрии лучей камеры."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from camera_geometry import (  # noqa: E402
    CameraCalibration,
    cos_gamma,
    foe_to_motion_ray,
    motion_ray_for_range,
    pixel_to_ray,
    select_foe_for_range,
)


class PixelRayTests(unittest.TestCase):
    def setUp(self):
        self.calibration = CameraCalibration(
            frame_width=1200,
            frame_height=600,
            fov_horizontal=120.0,
            fov_vertical=60.0,
        )

    def test_optical_center_is_forward_axis(self):
        ray = pixel_to_ray(600.0, 300.0, self.calibration)
        np.testing.assert_allclose(ray, [0.0, 0.0, 1.0], atol=1e-12)

    def test_horizontal_edge_has_half_fov_angle(self):
        ray = pixel_to_ray(1200.0, 300.0, self.calibration)
        expected = [np.sin(np.radians(60.0)), 0.0, np.cos(np.radians(60.0))]
        np.testing.assert_allclose(ray, expected, atol=1e-12)

    def test_vertical_edge_uses_vertical_fov(self):
        ray = pixel_to_ray(600.0, 600.0, self.calibration)
        expected = [0.0, np.sin(np.radians(30.0)), np.cos(np.radians(30.0))]
        np.testing.assert_allclose(ray, expected, atol=1e-12)

    def test_gamma_combines_motion_and_object_directions(self):
        # Объект находится на +20°, направление движения на -10° по горизонтали.
        object_x = 600.0 + 1200.0 * 20.0 / 120.0
        foe_x = 600.0 - 1200.0 * 10.0 / 120.0
        object_ray = pixel_to_ray(object_x, 300.0, self.calibration)
        motion_ray = foe_to_motion_ray(foe_x, 300.0, self.calibration)
        self.assertAlmostEqual(cos_gamma(object_ray, motion_ray), np.cos(np.radians(30.0)))


class FOESelectionTests(unittest.TestCase):
    def setUp(self):
        self.calibration = CameraCalibration(
            frame_width=1200,
            frame_height=600,
            fov_horizontal=120.0,
            fov_vertical=60.0,
        )
        self.geometry = pd.DataFrame({
            "frame_start": [0, 30, 60],
            "frame_end": [30, 60, 90],
            "foe_x": [500.0, 700.0, 900.0],
            "foe_y": [300.0, 320.0, 300.0],
            "confidence": [0.9, 0.8, 0.2],
        })

    def test_selection_uses_only_overlapping_confident_intervals(self):
        selection = select_foe_for_range(20, 50, self.geometry)
        self.assertIsNotNone(selection)
        self.assertEqual(selection.n_intervals, 2)
        self.assertEqual(selection.x_px, 600.0)
        self.assertEqual(selection.y_px, 310.0)
        self.assertAlmostEqual(selection.confidence, 0.85)

    def test_selection_does_not_fall_back_to_unrelated_video_interval(self):
        self.assertIsNone(select_foe_for_range(100, 120, self.geometry))
        self.assertIsNone(select_foe_for_range(70, 80, self.geometry))

    def test_missing_foe_returns_optical_axis(self):
        ray, selection = motion_ray_for_range(
            100, 120, self.geometry, self.calibration
        )
        self.assertIsNone(selection)
        np.testing.assert_allclose(ray, [0.0, 0.0, 1.0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
