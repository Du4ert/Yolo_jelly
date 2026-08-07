"""Тесты выбора метода оценки размера и деградации уверенности по наклону.

Проверяется порядок k_method → parallax → typical и то, что наклон камеры
влияет на confidence, а не только на величину коррекции k.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from camera_geometry import (  # noqa: E402
    K_TILT_FULL_CONF_DEG,
    K_TILT_MIN_FACTOR,
    K_TILT_REJECT_DEG,
    MIN_DISTANCE_CONFIDENCE,
    CameraCalibration,
    _tilt_confidence_factor,
    estimate_size_by_k_method,
    estimate_size_by_parallax,
    process_detections_with_size,
)


N_FRAMES = 12
FRAME_WIDTH = 3840
FRAME_HEIGHT = 2160


def make_calibration() -> CameraCalibration:
    """Калибровка без дисторсии — чтобы тест проверял выбор метода, а не оптику."""
    return CameraCalibration(
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        distortion_x_k1=0.0,
        distortion_x_k2=0.0,
        distortion_y_k1=0.0,
        distortion_y_k2=0.0,
    )


def make_track(
    track_id: int = 1,
    class_name: str = "Aurelia aurita",
    growth_per_frame: float = 1.12,
    depth_step: float = 0.1,
    x_drift: float = 0.0,
    start_size_norm: float = 0.02,
) -> pd.DataFrame:
    """Трек с монотонно растущим bbox — штатный вход k-метода.

    growth_per_frame задаёт удельный прирост (k), depth_step — шаг погружения,
    x_drift — покадровое смещение центра по X (для параллакса).
    """
    frames = np.arange(N_FRAMES)
    sizes = start_size_norm * growth_per_frame ** frames
    return pd.DataFrame({
        "track_id": track_id,
        "frame": frames,
        "class_name": class_name,
        "depth_m": 10.0 + depth_step * frames,
        "width": sizes,
        "height": sizes,
        "x_center": 0.5 + x_drift * frames,
        "y_center": 0.5,
    })


def make_geometry(
    tilt_deg: float = 0.0,
    flow_regime: str = "radial",
    direction_std: float = 10.0,
    p95_speed: float = 20.0,
    median_speed: float = 10.0,
    confidence: float = 0.9,
    n_intervals: int = 4,
) -> pd.DataFrame:
    """Интервалы геометрии, покрывающие весь трек.

    Наклон раскладывается поровну между горизонтальной и вертикальной
    компонентами: полный угол = sqrt(h² + v²).
    """
    component = tilt_deg / np.sqrt(2.0)
    step = max(N_FRAMES // n_intervals, 1)
    starts = np.arange(0, N_FRAMES, step)
    return pd.DataFrame({
        "frame_start": starts,
        "frame_end": starts + step,
        "tilt_horizontal_deg": component,
        "tilt_vertical_deg": component,
        "confidence": confidence,
        "flow_median_speed": median_speed,
        "flow_p95_speed": p95_speed,
        "flow_direction_std_deg": direction_std,
        "flow_regime": flow_regime,
    })


class TiltConfidenceFactorTests(unittest.TestCase):
    def test_no_penalty_below_full_confidence_angle(self):
        self.assertEqual(_tilt_confidence_factor(0.0), 1.0)
        self.assertEqual(_tilt_confidence_factor(K_TILT_FULL_CONF_DEG), 1.0)

    def test_linear_decay_between_thresholds(self):
        midpoint = (K_TILT_FULL_CONF_DEG + K_TILT_REJECT_DEG) / 2
        expected = 1.0 - (1.0 - K_TILT_MIN_FACTOR) / 2
        self.assertAlmostEqual(_tilt_confidence_factor(midpoint), expected)
        self.assertAlmostEqual(
            _tilt_confidence_factor(K_TILT_REJECT_DEG), K_TILT_MIN_FACTOR)

    def test_rejection_above_threshold(self):
        self.assertEqual(_tilt_confidence_factor(K_TILT_REJECT_DEG + 0.1), 0.0)
        self.assertEqual(_tilt_confidence_factor(80.0), 0.0)


class KMethodTiltTests(unittest.TestCase):
    """Наклон должен влиять на confidence, а не только на коррекцию k.

    Проверки относительные: у синтетического трека есть собственный базовый
    уровень уверенности (штрафы за нестабильность размера и т.п.), не связанный
    с наклоном. Значим именно множитель, который добавляет наклон.
    """

    def setUp(self):
        self.calibration = make_calibration()
        # База: наклон 0° — tilt-фактор равен 1.0 по построению
        baseline, _ = self._run(0.0)
        self.assertIsNotNone(baseline)
        self.baseline_confidence = baseline.confidence

    def _run(self, tilt_deg):
        return estimate_size_by_k_method(
            make_track(),
            self.calibration,
            geometry_df=make_geometry(tilt_deg=tilt_deg),
            apply_tilt_correction=True,
        )

    def test_small_tilt_keeps_baseline_confidence(self):
        estimate, reasons = self._run(10.0)
        self.assertIsNotNone(estimate)
        self.assertEqual(estimate.method, "k_method")
        self.assertEqual(estimate.confidence, self.baseline_confidence)
        self.assertEqual(reasons, [])
        self.assertNotIn("k_tilt_penalized", ";".join(estimate.warnings))

    def test_moderate_tilt_reduces_confidence(self):
        estimate, _ = self._run(40.0)
        self.assertIsNotNone(estimate)
        self.assertEqual(estimate.method, "k_method")
        # 40° — середина диапазона 20..60°, фактор 0.75
        self.assertAlmostEqual(
            estimate.confidence, self.baseline_confidence * 0.75, places=2)
        self.assertTrue(
            any(w.startswith("k_tilt_penalized") for w in estimate.warnings))

    def test_confidence_decreases_monotonically_with_tilt(self):
        confidences = []
        for tilt in (0.0, 25.0, 40.0, 55.0):
            estimate, _ = self._run(tilt)
            self.assertIsNotNone(estimate, msg=f"наклон {tilt}° отклонён")
            confidences.append(estimate.confidence)
        self.assertEqual(confidences, sorted(confidences, reverse=True))
        self.assertGreater(confidences[0], confidences[-1])

    def test_high_tilt_rejects_k_method(self):
        estimate, reasons = self._run(K_TILT_REJECT_DEG + 5.0)
        self.assertIsNone(estimate)
        self.assertTrue(
            any(r.startswith("k_rejected_high_tilt") for r in reasons),
            msg=f"ожидалась причина отказа по наклону, получено: {reasons}",
        )

    def test_tilt_ignored_when_correction_disabled(self):
        estimate, _ = estimate_size_by_k_method(
            make_track(),
            self.calibration,
            geometry_df=make_geometry(tilt_deg=70.0),
            apply_tilt_correction=False,
        )
        self.assertIsNotNone(estimate)
        self.assertEqual(estimate.confidence, self.baseline_confidence)


class ParallaxSelectionTests(unittest.TestCase):
    def setUp(self):
        self.calibration = make_calibration()
        # Дрейф 0.0008 кадр⁻¹ → ~3 px/кадр при ширине 3840
        self.track = make_track(x_drift=0.0008)

    def _run(self, geometry_df, track=None):
        return estimate_size_by_parallax(
            track if track is not None else self.track,
            self.calibration,
            geometry_df,
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
        )

    def test_parallel_interval_accepted(self):
        estimate, reasons = self._run(
            make_geometry(flow_regime="parallel", direction_std=10.0,
                          confidence=0.2, p95_speed=10.0))
        self.assertIsNotNone(estimate, msg=f"отказ по причинам: {reasons}")
        self.assertEqual(estimate.method, "parallax")
        self.assertGreaterEqual(estimate.confidence, MIN_DISTANCE_CONFIDENCE)

    def test_tilted_radial_interval_accepted(self):
        # Наклон выше порога k-метода: FOE далеко за кадром, поток
        # локально параллельный, но классификатор пометил интервал radial
        estimate, reasons = self._run(
            make_geometry(tilt_deg=70.0, flow_regime="radial",
                          direction_std=10.0, p95_speed=10.0))
        self.assertIsNotNone(estimate, msg=f"отказ по причинам: {reasons}")
        self.assertEqual(estimate.method, "parallax")

    def test_incoherent_flow_rejected(self):
        estimate, reasons = self._run(
            make_geometry(tilt_deg=70.0, flow_regime="radial",
                          direction_std=60.0, p95_speed=10.0))
        self.assertIsNone(estimate)
        self.assertIn("parallax_no_usable_interval", reasons)

    def test_low_tilt_radial_interval_rejected(self):
        # Штатное вертикальное погружение — работа k-метода, не параллакса
        estimate, reasons = self._run(
            make_geometry(tilt_deg=5.0, flow_regime="radial",
                          direction_std=10.0, p95_speed=10.0))
        self.assertIsNone(estimate)
        self.assertIn("parallax_no_usable_interval", reasons)

    def test_distance_out_of_range_rejected(self):
        # Очень быстрый фон при медленном объекте → d_obj далеко за 3.0 м
        estimate, reasons = self._run(
            make_geometry(flow_regime="parallel", direction_std=10.0,
                          p95_speed=400.0))
        self.assertIsNone(estimate)
        self.assertTrue(
            any(r.startswith("parallax_rejected_distance_out_of_range")
                for r in reasons),
            msg=f"получено: {reasons}",
        )

    def test_stationary_object_rejected(self):
        estimate, reasons = self._run(
            make_geometry(flow_regime="parallel", direction_std=10.0),
            track=make_track(x_drift=0.0),
        )
        self.assertIsNone(estimate)
        self.assertIn("parallax_object_too_slow", reasons)

    def test_low_confidence_rejected(self):
        # v_obj ≈ v_ref: плохая дискриминация (×0.7) плюс размер вне
        # типичного диапазона (×0.5) → 0.175 < MIN_DISTANCE_CONFIDENCE
        track = make_track(x_drift=0.0026, start_size_norm=0.9)
        estimate, reasons = self._run(
            make_geometry(flow_regime="parallel", direction_std=10.0,
                          p95_speed=10.0),
            track=track,
        )
        self.assertIsNone(estimate)
        self.assertTrue(
            any(r.startswith("parallax_rejected_low_confidence")
                for r in reasons),
            msg=f"получено: {reasons}",
        )


class MethodSelectionPipelineTests(unittest.TestCase):
    """Сквозная проверка деградации k_method → parallax → typical."""

    def setUp(self):
        self.calibration = make_calibration()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def _run(self, track, geometry_df):
        workdir = Path(self.tmpdir.name)
        detections_csv = workdir / "detections.csv"
        geometry_csv = workdir / "geometry.csv"
        track.to_csv(detections_csv, index=False)
        geometry_df.to_csv(geometry_csv, index=False)

        _, tracks_df = process_detections_with_size(
            str(detections_csv),
            geometry_csv=str(geometry_csv),
            calibration=self.calibration,
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
            apply_tilt_correction=True,
            verbose=False,
        )
        self.assertEqual(len(tracks_df), 1)
        return tracks_df.iloc[0]

    def test_high_tilt_with_coherent_flow_falls_back_to_parallax(self):
        row = self._run(
            make_track(x_drift=0.0008),
            make_geometry(tilt_deg=70.0, flow_regime="radial",
                          direction_std=10.0, p95_speed=10.0),
        )
        self.assertEqual(row["method"], "parallax")
        self.assertIn("k_rejected_high_tilt", row["warnings"])

    def test_high_tilt_with_incoherent_flow_falls_back_to_typical(self):
        row = self._run(
            make_track(x_drift=0.0008),
            make_geometry(tilt_deg=70.0, flow_regime="radial",
                          direction_std=60.0, p95_speed=10.0),
        )
        self.assertEqual(row["method"], "typical")
        self.assertEqual(row["confidence"], 0.2)
        self.assertIn("k_rejected_high_tilt", row["warnings"])
        self.assertIn("parallax_no_usable_interval", row["warnings"])

    def test_low_tilt_keeps_k_method(self):
        row = self._run(
            make_track(x_drift=0.0008),
            make_geometry(tilt_deg=10.0),
        )
        self.assertEqual(row["method"], "k_method")
        self.assertNotIn("k_tilt_penalized", row["warnings"])
        self.assertNotIn("k_rejected_high_tilt", row["warnings"])
        self.assertGreaterEqual(row["confidence"], 0.5)

    def test_pileus_stays_fixed_regardless_of_tilt(self):
        row = self._run(
            make_track(class_name="Pleurobrachia pileus", x_drift=0.0008),
            make_geometry(tilt_deg=70.0, flow_regime="radial",
                          direction_std=10.0, p95_speed=10.0),
        )
        self.assertEqual(row["method"], "fixed")
        self.assertEqual(row["real_size_mm"], 10.0)


if __name__ == "__main__":
    unittest.main()
