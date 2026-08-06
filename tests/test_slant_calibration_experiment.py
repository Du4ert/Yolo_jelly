"""Тесты точной конечной геометрии slant-калибровки."""

import sys
import json
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slant_calibration_experiment import exact_slant_k_percent, run_experiment  # noqa: E402


class ExactSlantPairTests(unittest.TestCase):
    def test_recovers_endpoint_range_for_finite_pair(self):
        endpoint_range = 2.0
        delta_depth = 0.5
        cosine = 0.8
        start_range = np.sqrt(
            endpoint_range**2
            + delta_depth**2
            + 2 * endpoint_range * delta_depth * cosine
        )
        ratio = start_range / endpoint_range

        k_percent = exact_slant_k_percent(
            100.0,
            100.0 * ratio,
            delta_depth,
            cosine,
        )

        self.assertAlmostEqual(k_percent, 100.0 / endpoint_range)

    def test_rejects_non_growing_pair(self):
        with self.assertRaisesRegex(ValueError, "увеличиваться"):
            exact_slant_k_percent(100.0, 100.0, 0.5, 0.8)

    def test_rejects_non_descending_pair(self):
        with self.assertRaisesRegex(ValueError, "положительный спуск"):
            exact_slant_k_percent(100.0, 120.0, -0.5, 0.8)


class LocalSlantExperimentTests(unittest.TestCase):
    def test_step4_experiment_is_reproducible(self):
        geometry_path = ROOT / "test_video" / "2down" / "output" / "ball_geometry_step4.csv"
        expected_path = ROOT / "calibration_slant_experiment.json"
        if not geometry_path.exists():
            self.skipTest("Локальная geometry шага 4 отсутствует")
        report = run_experiment(
            ROOT / "test_video",
            geometry_name="ball_geometry_step4.csv",
            geometry_frame_step=4,
            baseline_report_path=ROOT / "calibration_baseline.json",
            vertical_calibration_path=ROOT / "calibration_xy.json",
            min_pair_depth_change_m=0.01,
        )
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        self.assertEqual(report["totals"], expected["totals"])
        self.assertEqual(report["videos"], expected["videos"])
        self.assertEqual(report["coefficients"], expected["coefficients"])
        self.assertEqual(report["metrics"], expected["metrics"])
        self.assertEqual(report["diagnostics"], expected["diagnostics"])
        self.assertEqual(
            report["same_pairs_vertical_baseline"],
            expected["same_pairs_vertical_baseline"],
        )
        self.assertEqual(
            report["comparison_to_vertical_baseline"],
            expected["comparison_to_vertical_baseline"],
        )


if __name__ == "__main__":
    unittest.main()
