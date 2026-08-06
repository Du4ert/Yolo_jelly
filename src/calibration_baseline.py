"""Build a deterministic report for the current camera-calibration pipeline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from camera_geometry import (
    REFERENCE_FRAME_WIDTH,
    CameraCalibration,
    _compute_sizes_from_pairs,
    _corrected_k_percent,
    _discover_test_dirs,
    _distortion_factor_xy,
    _extract_calibration_pairs,
)


REPORT_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "count": int(len(array)),
        "mean": round(float(np.mean(array)), 6),
        "median": round(float(np.median(array)), 6),
        "p95": round(float(np.percentile(array, 95)), 6),
        "min": round(float(np.min(array)), 6),
        "max": round(float(np.max(array)), 6),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _calibration_values(calibration: CameraCalibration) -> dict:
    return {
        "distance_coef_A": calibration.distance_coef_A,
        "distance_coef_B": calibration.distance_coef_B,
        "pixel_calib_C": calibration.pixel_calib_C,
        "pixel_calib_D": calibration.pixel_calib_D,
        "distortion_x_k1": calibration.distortion_x_k1,
        "distortion_x_k2": calibration.distortion_x_k2,
        "distortion_y_k1": calibration.distortion_y_k1,
        "distortion_y_k2": calibration.distortion_y_k2,
        "min_reliable_distance": calibration.min_reliable_distance,
        "max_reliable_distance": calibration.max_reliable_distance,
    }


def build_baseline_report(
    test_dir: Path,
    calibration_path: Path,
    frame_width: int = 3840,
    frame_height: int = 2160,
    apply_tilt_correction: bool = True,
    min_track_depth_span_m: float = 0.1,
    min_pair_depth_change_m: float = 0.01,
    min_size_change_pct: float = 10.0,
) -> dict:
    """Evaluate one calibration against all discovered calibration tracks."""
    test_dir = test_dir.resolve()
    calibration_path = calibration_path.resolve()
    calibration = CameraCalibration.from_json(str(calibration_path))
    calibration.frame_width = frame_width
    calibration.frame_height = frame_height
    specs = _discover_test_dirs(str(test_dir))
    if not specs:
        raise ValueError(f"No calibration datasets found in {test_dir}")

    resolution_scale = frame_width / REFERENCE_FRAME_WIDTH
    coefficients = _calibration_values(calibration)
    A = calibration.distance_coef_A
    B = calibration.distance_coef_B
    C = calibration.pixel_calib_C
    D = calibration.pixel_calib_D
    x_k1 = calibration.distortion_x_k1
    x_k2 = calibration.distortion_x_k2
    y_k1 = calibration.distortion_y_k1
    y_k2 = calibration.distortion_y_k2

    input_paths = {calibration_path}
    videos = []
    tracks = []
    distance_signed_errors = []
    track_distance_signed_errors = []
    pipeline_signed_errors = []
    direct_signed_errors = []
    predicted_distances = []
    coverage_x = []
    coverage_y = []

    for spec in specs:
        detections_path = Path(spec["detections_csv"]).resolve()
        geometry_path = Path(spec["geometry_csv"]).resolve() if spec.get("geometry_csv") else None
        dataset_dir = detections_path.parent.parent
        ground_truth_path = dataset_dir / "size-depth.txt"
        if not ground_truth_path.exists():
            ground_truth_path = detections_path.parent / "size-depth.txt"
        input_paths.update((detections_path, ground_truth_path))
        if geometry_path:
            input_paths.add(geometry_path)

        detections_df = pd.read_csv(detections_path)
        geometry_rows = len(pd.read_csv(geometry_path)) if geometry_path else 0
        pairs_by_track = _extract_calibration_pairs(
            str(detections_path),
            str(geometry_path) if geometry_path else None,
            frame_width,
            frame_height,
            apply_tilt_correction=apply_tilt_correction,
            min_track_depth_span_m=min_track_depth_span_m,
            min_pair_depth_change_m=min_pair_depth_change_m,
            min_size_change_pct=min_size_change_pct,
        )

        contributing_tracks = 0
        pair_count = 0
        for track_id in sorted(spec["known_sizes"]):
            pairs = pairs_by_track.get(track_id, [])
            if not pairs:
                continue
            contributing_tracks += 1
            pair_count += len(pairs)
            known_size = float(spec["known_sizes"][track_id])
            known_depth = float(spec["known_depths"][track_id])

            estimated_sizes = _compute_sizes_from_pairs(
                pairs, A, B, C, D,
                x_k1, x_k2, y_k1, y_k2,
                resolution_scale,
            )
            median_size = float(np.median(estimated_sizes))
            pipeline_error = (median_size - known_size) / known_size * 100.0
            pipeline_signed_errors.append(pipeline_error)

            direct_sizes = []
            track_distances = []
            true_distances = []
            valid_predicted_distances = []
            for pair in pairs:
                true_distance = known_depth - float(pair["depth_camera"])
                corrected_k = _corrected_k_percent(pair, x_k1, x_k2, y_k1, y_k2)
                predicted_distance = A * corrected_k**B
                predicted_distances.append(predicted_distance)
                track_distances.append(predicted_distance)
                true_distances.append(true_distance)
                if true_distance > 0.05:
                    valid_predicted_distances.append(predicted_distance)
                    distance_signed_errors.append(
                        (predicted_distance - true_distance) / true_distance * 100.0
                    )
                    factor = _distortion_factor_xy(
                        pair.get("rx_norm", 0.0), pair.get("ry_norm", 0.0),
                        x_k1, x_k2, y_k1, y_k2,
                    )
                    pixels_ref = pair["size_pixels"] * factor / resolution_scale
                    pixel_calibration = C * max(true_distance, 0.1)**D
                    direct_sizes.append(pixels_ref / pixel_calibration)
                coverage_x.append(pair.get("rx_norm", 0.0))
                coverage_y.append(pair.get("ry_norm", 0.0))

            direct_error = None
            median_direct_size = None
            if direct_sizes:
                median_direct_size = float(np.median(direct_sizes))
                direct_error = (median_direct_size - known_size) / known_size * 100.0
                direct_signed_errors.append(direct_error)

            valid_true_distances = [value for value in true_distances if value > 0.05]
            track_distance_error = None
            if valid_true_distances:
                median_predicted_distance = float(np.median(valid_predicted_distances))
                median_true_distance = float(np.median(valid_true_distances))
                track_distance_error = (
                    (median_predicted_distance - median_true_distance)
                    / median_true_distance * 100.0
                )
                track_distance_signed_errors.append(track_distance_error)

            tracks.append({
                "video": spec["name"],
                "track_id": int(track_id),
                "pairs": len(pairs),
                "known_size_mm": known_size,
                "known_depth_m": known_depth,
                "pipeline_size_mm": round(median_size, 6),
                "pipeline_error_pct": round(pipeline_error, 6),
                "direct_size_mm": round(median_direct_size, 6) if median_direct_size is not None else None,
                "direct_error_pct": round(direct_error, 6) if direct_error is not None else None,
                "median_predicted_distance_m": round(float(np.median(track_distances)), 6),
                "median_true_vertical_distance_m": round(float(np.median(true_distances)), 6),
                "track_distance_error_pct": (
                    round(track_distance_error, 6)
                    if track_distance_error is not None else None
                ),
            })

        videos.append({
            "name": spec["name"],
            "detection_rows": int(len(detections_df)),
            "ground_truth_tracks": int(len(spec["known_sizes"])),
            "contributing_tracks": contributing_tracks,
            "pairs": pair_count,
            "geometry_rows": int(geometry_rows),
        })

    predicted_array = np.asarray(predicted_distances, dtype=float)
    reliable = {
        "below_min_count": int(np.sum(predicted_array < calibration.min_reliable_distance)),
        "above_max_count": int(np.sum(predicted_array > calibration.max_reliable_distance)),
        "below_min_fraction": round(float(np.mean(predicted_array < calibration.min_reliable_distance)), 6),
        "above_max_fraction": round(float(np.mean(predicted_array > calibration.max_reliable_distance)), 6),
    }

    input_files = []
    for path in sorted(input_paths, key=lambda item: str(item).lower()):
        try:
            relative_path = path.relative_to(test_dir.parent)
        except ValueError:
            relative_path = Path(path.name)
        input_files.append({"path": relative_path.as_posix(), "sha256": _sha256(path)})

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "model_contract": {
            "distance_in_current_calibration": "vertical_depth_difference",
            "object_depth_formula": "camera_depth_m + distance_m",
            "k_tilt_formula": "k_corrected = k_raw / cos(tilt)",
        },
        "settings": {
            "frame_width": frame_width,
            "frame_height": frame_height,
            "apply_tilt_correction": apply_tilt_correction,
            "min_track_depth_span_m": min_track_depth_span_m,
            "min_pair_depth_change_m": min_pair_depth_change_m,
            "min_track_points": 3,
            "min_size_change_pct": min_size_change_pct,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scipy": _package_version("scipy"),
            "opencv_python": _package_version("opencv-python"),
        },
        "inputs": input_files,
        "calibration": coefficients,
        "totals": {
            "videos": len(videos),
            "detection_rows": sum(video["detection_rows"] for video in videos),
            "ground_truth_tracks": sum(video["ground_truth_tracks"] for video in videos),
            "contributing_tracks": len(tracks),
            "pairs": sum(video["pairs"] for video in videos),
        },
        "videos": videos,
        "metrics": {
            "distance_signed_error_pct": _summary(distance_signed_errors),
            "distance_absolute_error_pct": _summary(abs(value) for value in distance_signed_errors),
            "track_distance_signed_error_pct": _summary(track_distance_signed_errors),
            "track_distance_absolute_error_pct": _summary(
                abs(value) for value in track_distance_signed_errors
            ),
            "pipeline_size_signed_error_pct": _summary(pipeline_signed_errors),
            "pipeline_size_absolute_error_pct": _summary(abs(value) for value in pipeline_signed_errors),
            "direct_size_signed_error_pct": _summary(direct_signed_errors),
            "direct_size_absolute_error_pct": _summary(abs(value) for value in direct_signed_errors),
            "predicted_distance_m": _summary(predicted_distances),
            "coverage_x": _summary(coverage_x),
            "coverage_y": _summary(coverage_y),
            "reliable_range": reliable,
        },
        "tracks": tracks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=Path("test_video"))
    parser.add_argument("--calibration", type=Path, default=Path("calibration_xy.json"))
    parser.add_argument("--output", type=Path, default=Path("calibration_baseline.json"))
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--no-tilt-correction", action="store_true")
    parser.add_argument("--min-track-depth-span", type=float, default=0.1)
    parser.add_argument("--min-pair-depth-change", type=float, default=0.01)
    parser.add_argument("--min-size-change-pct", type=float, default=10.0)
    args = parser.parse_args()

    report = build_baseline_report(
        args.test_dir,
        args.calibration,
        frame_width=args.width,
        frame_height=args.height,
        apply_tilt_correction=not args.no_tilt_correction,
        min_track_depth_span_m=args.min_track_depth_span,
        min_pair_depth_change_m=args.min_pair_depth_change,
        min_size_change_pct=args.min_size_change_pct,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    totals = report["totals"]
    metrics = report["metrics"]
    print(
        f"Baseline saved to {args.output}: {totals['videos']} videos, "
        f"{totals['contributing_tracks']} tracks, {totals['pairs']} pairs; "
        f"mean pipeline error "
        f"{metrics['pipeline_size_absolute_error_pct']['mean']:.2f}%"
    )


if __name__ == "__main__":
    main()
