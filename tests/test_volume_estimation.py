"""Тесты отдельного расчёта объёма для Pleurobrachia pileus."""

import math
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from camera_geometry import (  # noqa: E402
    CameraCalibration,
    calculate_surveyed_volume,
    estimate_effective_distance,
    process_volume_estimation,
)
from batch.core.worker import Worker  # noqa: E402
from batch.database.models import SubTaskType  # noqa: E402


class PleurobrachiaVolumeTests(unittest.TestCase):
    def setUp(self):
        self.detections = pd.DataFrame(
            {
                "frame": [0, 1, 2],
                "timestamp_s": [0.0, 1.0, 2.0],
                "depth_m": [0.0, 5.0, 10.0],
                "track_id": [1, 2, 3],
                "class_name": [
                    "Aurelia aurita",
                    "Pleurobrachia pileus",
                    "Pleurobrachia pileus",
                ],
            }
        )
        self.count_tracks = pd.DataFrame(
            {
                "track_id": [1, 2, 3],
                "class_name": [
                    "Aurelia aurita",
                    "Pleurobrachia pileus",
                    "Pleurobrachia pileus",
                ],
            }
        )

    def _calculate(self, pileus_distance=None):
        calibration = CameraCalibration(
            min_reliable_distance=0.2,
            max_reliable_distance=3.0,
        )
        return calculate_surveyed_volume(
            detections_df=self.detections,
            count_tracks_df=self.count_tracks,
            calibration=calibration,
            fov_horizontal_deg=90.0,
            fov_vertical_deg=90.0,
            detection_distance_m=2.0,
            pleurobrachia_detection_distance_m=pileus_distance,
            depth_range=(0.0, 10.0),
            total_duration_s=10.0,
            verbose=False,
        )

    def test_fallback_uses_min_reliable_distance_and_separate_volume(self):
        result = self._calculate()

        expected_common_volume = 4.0 * math.pi * 12.0
        expected_pileus_area = 0.04 * math.pi
        expected_pileus_volume = expected_pileus_area * 10.2

        self.assertEqual(result.pleurobrachia_effective_distance_m, 0.2)
        self.assertAlmostEqual(
            result.pleurobrachia_cross_section_area_m2,
            expected_pileus_area,
        )
        self.assertAlmostEqual(
            result.pleurobrachia_volume_m3,
            expected_pileus_volume,
        )
        self.assertAlmostEqual(
            result.density_by_class["Aurelia aurita"],
            1.0 / expected_common_volume,
        )
        self.assertAlmostEqual(
            result.density_by_class["Pleurobrachia pileus"],
            2.0 / expected_pileus_volume,
        )

    def test_manual_distance_overrides_fallback(self):
        result = self._calculate(pileus_distance=0.4)

        self.assertEqual(result.pleurobrachia_effective_distance_m, 0.4)
        self.assertAlmostEqual(
            result.pleurobrachia_cross_section_area_m2, 0.16 * math.pi
        )
        self.assertAlmostEqual(
            result.pleurobrachia_volume_m3, 0.16 * math.pi * 10.4
        )

    def test_small_positive_distance_keeps_nonzero_geometry(self):
        result = self._calculate(pileus_distance=0.01)

        self.assertGreater(result.pleurobrachia_cross_section_area_m2, 0)
        self.assertGreater(result.pleurobrachia_volume_m3, 0)

    def test_manual_distance_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "P. pileus"):
            self._calculate(pileus_distance=0.0)

    def test_pileus_tracks_do_not_affect_common_auto_distance(self):
        tracks = pd.DataFrame(
            {
                "class_name": [
                    "Aurelia aurita",
                    "Aurelia aurita",
                    "Aurelia aurita",
                    "Pleurobrachia pileus",
                    "Pleurobrachia pileus",
                    "Pleurobrachia pileus",
                ],
                "max_detection_distance_m": [0.5, 0.6, 0.7, 2.7, 2.8, 2.9],
            }
        )

        distance = estimate_effective_distance(
            detections_df=pd.DataFrame(),
            tracks_df=tracks,
            calibration=CameraCalibration(
                min_reliable_distance=0.1,
                max_reliable_distance=3.0,
            ),
        )

        self.assertAlmostEqual(distance, 0.68)

    def test_volume_csv_contains_separate_pileus_geometry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            detections_path = temp_path / "detections.csv"
            tracks_path = temp_path / "tracks.csv"
            output_path = temp_path / "volume.csv"
            self.detections.to_csv(detections_path, index=False)
            self.count_tracks.assign(real_size_cm=1.0).to_csv(
                tracks_path, index=False
            )

            process_volume_estimation(
                detections_csv=str(detections_path),
                tracks_csv=str(tracks_path),
                output_csv=str(output_path),
                fov_horizontal=90.0,
                fov_vertical=90.0,
                detection_distance=2.0,
                pleurobrachia_detection_distance=0.4,
                depth_min=0.0,
                depth_max=10.0,
                total_duration=10.0,
                verbose=False,
            )

            values = pd.read_csv(output_path).set_index("parameter")["value"]

        self.assertIn("total_volume_m3", values.index)
        self.assertEqual(values["pleurobrachia_effective_distance_m"], 0.4)
        self.assertAlmostEqual(
            values["pleurobrachia_cross_section_area_m2"], 0.16 * math.pi
        )
        self.assertAlmostEqual(
            values["density_Pleurobrachia_pileus_per_m2"],
            2.0 / (0.16 * math.pi),
        )


class AutoPostprocessParamsTests(unittest.TestCase):
    class FakeRepository:
        def __init__(self):
            self.subtasks = []

        def create_subtask(self, **kwargs):
            self.subtasks.append(kwargs)

    def test_pileus_distance_is_forwarded_to_volume_subtask(self):
        repo = self.FakeRepository()
        worker = Worker(repo)
        worker._create_auto_postprocess_subtasks(
            task_id=42,
            params_json=json.dumps(
                {
                    "geometry": False,
                    "size": False,
                    "size_video": False,
                    "volume": True,
                    "analysis": False,
                    "pleurobrachia_detection_distance": 0.7,
                }
            ),
        )

        volume = next(
            item
            for item in repo.subtasks
            if item["subtask_type"] == SubTaskType.VOLUME
        )
        params = json.loads(volume["params_json"])
        self.assertEqual(params["pleurobrachia_detection_distance"], 0.7)

    def test_missing_pileus_distance_stays_none_for_fallback(self):
        repo = self.FakeRepository()
        worker = Worker(repo)
        worker._create_auto_postprocess_subtasks(
            task_id=42,
            params_json=json.dumps(
                {
                    "geometry": False,
                    "size": False,
                    "size_video": False,
                    "volume": True,
                    "analysis": False,
                    "min_reliable_distance": 0.3,
                }
            ),
        )

        volume = next(
            item
            for item in repo.subtasks
            if item["subtask_type"] == SubTaskType.VOLUME
        )
        params = json.loads(volume["params_json"])
        self.assertIsNone(params["pleurobrachia_detection_distance"])
        self.assertEqual(params["min_reliable_distance"], 0.3)


if __name__ == "__main__":
    unittest.main()
