"""Тесты геометрии угла γ между направлением на объект и движением камеры.

Проверяемое соотношение: видимый размер s ∝ 1/d, поэтому k = cos(γ)/d,
а вертикальный зазор до объекта равен d·cos(γ), а не d.
"""

import math
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
    DEFAULT_FOV_HORIZONTAL,
    DEFAULT_FOV_VERTICAL,
    MIN_COS_GAMMA,
    CameraCalibration,
    _find_size_pairs,
    _pair_slant_distance,
    _unit_direction,
    cos_gamma_for_point,
    get_foe_for_range,
)


class FieldOfViewTests(unittest.TestCase):
    """156° из спецификации — диагональный угол, для пересчёта углов он неприменим."""

    def test_default_fov_is_working_not_diagonal(self):
        calibration = CameraCalibration()
        self.assertEqual(calibration.fov_horizontal, 95.0)
        self.assertEqual(calibration.fov_vertical, 55.0)

    def test_pixels_per_degree_matches_both_axes(self):
        """Согласованность масштаба по осям подтверждает эквидистантную проекцию."""
        calibration = CameraCalibration()
        by_width = calibration.frame_width / DEFAULT_FOV_HORIZONTAL
        by_height = calibration.frame_height / DEFAULT_FOV_VERTICAL
        self.assertAlmostEqual(calibration.pixels_per_degree, by_width)
        self.assertLess(abs(by_width - by_height) / by_width, 0.05)

    def test_pixels_per_radian_is_degree_scale_converted(self):
        calibration = CameraCalibration()
        self.assertAlmostEqual(
            calibration.pixels_per_radian,
            calibration.pixels_per_degree * 180.0 / math.pi,
            places=6,
        )

    def test_fov_survives_json_round_trip(self):
        import tempfile

        calibration = CameraCalibration(fov_horizontal=100.0, fov_vertical=58.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "calibration.json"
            calibration.to_json(str(path))
            loaded = CameraCalibration.from_json(str(path))
        self.assertEqual(loaded.fov_horizontal, 100.0)
        self.assertEqual(loaded.fov_vertical, 58.0)


class UnitDirectionTests(unittest.TestCase):
    def test_optical_center_maps_to_axis(self):
        np.testing.assert_allclose(_unit_direction(0.0, 0.0, 2316.0), [0, 0, 1])

    def test_offset_of_one_radian_scale(self):
        """Эквидистантная проекция: смещение в px_per_rad пикселей = 1 радиан."""
        px_per_rad = 2316.0
        direction = _unit_direction(px_per_rad, 0.0, px_per_rad)
        np.testing.assert_allclose(
            direction, [math.sin(1.0), 0.0, math.cos(1.0)], atol=1e-12
        )

    def test_result_is_unit_length(self):
        direction = _unit_direction(1500.0, -700.0, 2316.0)
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0, places=12)


class CosGammaTests(unittest.TestCase):
    def setUp(self):
        self.calibration = CameraCalibration()
        self.ocx, self.ocy = self.calibration.optical_center_px

    def test_object_at_foe_has_no_correction(self):
        """Объект точно по направлению движения — камера идёт прямо на него."""
        foe = (self.ocx + 400.0, self.ocy - 250.0)
        self.assertAlmostEqual(
            cos_gamma_for_point(foe[0], foe[1], foe, self.calibration), 1.0, places=12
        )

    def test_offaxis_object_with_vertical_camera(self):
        """FOE в оптическом центре: γ сводится к внеосевому углу объекта."""
        px_per_rad = self.calibration.pixels_per_radian
        offset = px_per_rad * math.radians(30.0)
        cos_gamma = cos_gamma_for_point(
            self.ocx + offset, self.ocy, (self.ocx, self.ocy), self.calibration
        )
        self.assertAlmostEqual(cos_gamma, math.cos(math.radians(30.0)), places=9)

    def test_frame_edge_correction_is_substantial(self):
        """У края кадра поправка гораздо больше типичного наклона камеры (3-6°)."""
        cos_gamma = cos_gamma_for_point(
            self.calibration.frame_width, self.ocy,
            (self.ocx, self.ocy), self.calibration,
        )
        self.assertAlmostEqual(
            cos_gamma, math.cos(math.radians(DEFAULT_FOV_HORIZONTAL / 2)), places=9
        )
        self.assertLess(cos_gamma, 0.7)

    def test_tilt_and_offaxis_cancel_when_aligned(self):
        """Наклон камеры к объекту компенсирует его смещение в кадре."""
        offset = self.calibration.pixels_per_radian * math.radians(20.0)
        point = (self.ocx + offset, self.ocy)
        self.assertAlmostEqual(
            cos_gamma_for_point(point[0], point[1], point, self.calibration),
            1.0, places=12,
        )

    def test_result_is_floored(self):
        """Перпендикулярное направление не должно давать деления на ноль."""
        px_per_rad = self.calibration.pixels_per_radian
        cos_gamma = cos_gamma_for_point(
            self.ocx + px_per_rad * math.pi / 2, self.ocy,
            (self.ocx, self.ocy), self.calibration,
        )
        self.assertGreaterEqual(cos_gamma, MIN_COS_GAMMA)


class SlantDistanceTests(unittest.TestCase):
    def test_vertical_gap_is_converted_to_slant_range(self):
        """Ошибка исходной реализации: Δz подставлялась вместо d = Δz/cos(γ)."""
        pair = {"depth_camera": 10.0, "cos_gamma_end": 0.5}
        self.assertAlmostEqual(_pair_slant_distance(pair, 11.0), 2.0)

    def test_on_axis_object_is_unchanged(self):
        pair = {"depth_camera": 10.0, "cos_gamma_end": 1.0}
        self.assertAlmostEqual(_pair_slant_distance(pair, 11.5), 1.5)

    def test_missing_cos_gamma_falls_back_to_vertical_gap(self):
        self.assertAlmostEqual(_pair_slant_distance({"depth_camera": 4.0}, 6.0), 2.0)


class FoeLookupTests(unittest.TestCase):
    def setUp(self):
        self.calibration = CameraCalibration()
        self.optical_center = self.calibration.optical_center_px

    def _geometry(self, rows):
        return pd.DataFrame(rows)

    def test_missing_geometry_returns_optical_axis(self):
        foe, tilt, measured = get_foe_for_range(0, 100, None, self.calibration)
        self.assertEqual(foe, self.optical_center)
        self.assertEqual(tilt, 0.0)
        self.assertFalse(measured)

    def test_overlapping_confident_interval_is_used(self):
        geometry = self._geometry([
            {"frame_start": 0, "frame_end": 60, "foe_x": 2000.0,
             "foe_y": 1100.0, "confidence": 0.8},
        ])
        foe, tilt, measured = get_foe_for_range(10, 50, geometry, self.calibration)
        self.assertTrue(measured)
        self.assertEqual(foe, (2000.0, 1100.0))
        self.assertGreater(tilt, 0.0)

    def test_low_confidence_is_ignored(self):
        geometry = self._geometry([
            {"frame_start": 0, "frame_end": 60, "foe_x": 3000.0,
             "foe_y": 1800.0, "confidence": 0.1},
        ])
        foe, _, measured = get_foe_for_range(10, 50, geometry, self.calibration)
        self.assertFalse(measured)
        self.assertEqual(foe, self.optical_center)

    def test_no_overlap_does_not_fall_back_to_whole_video(self):
        """Старая реализация молча брала среднее по всему видео."""
        geometry = self._geometry([
            {"frame_start": 5000, "frame_end": 5060, "foe_x": 3000.0,
             "foe_y": 1800.0, "confidence": 0.9},
        ])
        foe, _, measured = get_foe_for_range(10, 50, geometry, self.calibration)
        self.assertFalse(measured)
        self.assertEqual(foe, self.optical_center)


class SizePairGammaTests(unittest.TestCase):
    """Сквозная проверка: k = cos(γ)/d должно восстанавливать 1/d."""

    def _track(self, x_center):
        # Объект на постоянном горизонтальном удалении; камера опускается.
        return pd.DataFrame([
            {"frame": 0, "size_pix": 100.0, "depth_m": 0.0,
             "x_center": x_center, "y_center": 0.5},
            {"frame": 30, "size_pix": 140.0, "depth_m": 0.5,
             "x_center": x_center, "y_center": 0.5},
        ])

    def test_on_axis_track_is_not_corrected(self):
        calibration = CameraCalibration()
        pairs, _, _ = _find_size_pairs(
            self._track(0.5), 1.10, True, None, calibration
        )
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs[0]["cos_gamma"], 1.0, places=9)
        self.assertAlmostEqual(pairs[0]["k_percent"], pairs[0]["k_raw_percent"], places=6)

    def test_offaxis_track_raises_k_and_shortens_vertical_gap(self):
        calibration = CameraCalibration()
        on_axis, _, _ = _find_size_pairs(
            self._track(0.5), 1.10, True, None, calibration
        )
        off_axis, _, _ = _find_size_pairs(
            self._track(0.95), 1.10, True, None, calibration
        )
        self.assertLess(off_axis[0]["cos_gamma"], 0.95)
        # k = cos(γ)/d, поэтому исправленный k строго больше сырого
        self.assertGreater(off_axis[0]["k_percent"], off_axis[0]["k_raw_percent"])
        # Вертикальный зазор до объекта — проекция дальности, а не сама дальность
        gap = off_axis[0]["object_depth"] - off_axis[0]["depth_camera"]
        self.assertLess(gap, off_axis[0]["distance"])
        self.assertAlmostEqual(
            gap, off_axis[0]["distance"] * off_axis[0]["cos_gamma_end"], places=9
        )
        # На оси проекция вырождается в тождество
        on_gap = on_axis[0]["object_depth"] - on_axis[0]["depth_camera"]
        self.assertAlmostEqual(on_gap, on_axis[0]["distance"], places=9)


if __name__ == "__main__":
    unittest.main()
