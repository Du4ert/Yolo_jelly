"""Сравнение порогов глубины пары с leave-one-video-out проверкой."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from camera_geometry import (
    REFERENCE_FRAME_WIDTH,
    _compute_sizes_from_pairs,
    _corrected_k_percent,
    _discover_test_dirs,
    _extract_calibration_pairs,
    calibrate_coefficients,
)


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(len(array)),
        "mean": round(float(np.mean(array)), 6),
        "median": round(float(np.median(array)), 6),
        "p95": round(float(np.percentile(array, 95)), 6),
        "max": round(float(np.max(array)), 6),
    }


def _evaluate_specs(
    specs: list[dict],
    calibration,
    frame_width: int,
    frame_height: int,
    min_pair_depth_change_m: float,
) -> dict:
    resolution_scale = frame_width / REFERENCE_FRAME_WIDTH
    size_errors = []
    distance_errors = []
    track_distance_errors = []
    pairs_used = 0
    tracks_used = 0

    for spec in specs:
        pairs_by_track = _extract_calibration_pairs(
            spec["detections_csv"],
            spec.get("geometry_csv"),
            frame_width,
            frame_height,
            apply_tilt_correction=True,
            min_track_depth_span_m=0.1,
            min_pair_depth_change_m=min_pair_depth_change_m,
            min_size_change_pct=10.0,
        )
        for track_id in sorted(spec["known_sizes"]):
            pairs = pairs_by_track.get(track_id, [])
            if not pairs:
                continue
            tracks_used += 1
            pairs_used += len(pairs)
            known_size = float(spec["known_sizes"][track_id])
            known_depth = float(spec["known_depths"][track_id])
            estimated_sizes = _compute_sizes_from_pairs(
                pairs,
                calibration.distance_coef_A,
                calibration.distance_coef_B,
                calibration.pixel_calib_C,
                calibration.pixel_calib_D,
                calibration.distortion_x_k1,
                calibration.distortion_x_k2,
                calibration.distortion_y_k1,
                calibration.distortion_y_k2,
                resolution_scale,
            )
            median_size = float(np.median(estimated_sizes))
            size_errors.append((median_size - known_size) / known_size * 100.0)

            for pair in pairs:
                true_distance = known_depth - float(pair["depth_camera"])
                if true_distance <= 0.05:
                    continue
                k_percent = _corrected_k_percent(
                    pair,
                    calibration.distortion_x_k1,
                    calibration.distortion_x_k2,
                    calibration.distortion_y_k1,
                    calibration.distortion_y_k2,
                )
                estimated_distance = (
                    calibration.distance_coef_A
                    * k_percent**calibration.distance_coef_B
                )
                distance_errors.append(
                    (estimated_distance - true_distance) / true_distance * 100.0
                )
            valid_pairs = [
                pair for pair in pairs
                if known_depth - float(pair["depth_camera"]) > 0.05
            ]
            if valid_pairs:
                estimated_track_distances = []
                true_track_distances = []
                for pair in valid_pairs:
                    k_percent = _corrected_k_percent(
                        pair,
                        calibration.distortion_x_k1,
                        calibration.distortion_x_k2,
                        calibration.distortion_y_k1,
                        calibration.distortion_y_k2,
                    )
                    estimated_track_distances.append(
                        calibration.distance_coef_A
                        * k_percent**calibration.distance_coef_B
                    )
                    true_track_distances.append(
                        known_depth - float(pair["depth_camera"])
                    )
                median_true = float(np.median(true_track_distances))
                track_distance_errors.append(
                    (float(np.median(estimated_track_distances)) - median_true)
                    / median_true * 100.0
                )

    return {
        "tracks": tracks_used,
        "pairs": pairs_used,
        "distance_signed_error_pct": _summary(distance_errors),
        "distance_absolute_error_pct": _summary([abs(value) for value in distance_errors]),
        "track_distance_signed_error_pct": _summary(track_distance_errors),
        "track_distance_absolute_error_pct": _summary(
            [abs(value) for value in track_distance_errors]
        ),
        "size_signed_error_pct": _summary(size_errors),
        "size_absolute_error_pct": _summary([abs(value) for value in size_errors]),
    }


def run_sensitivity(
    test_dir: Path,
    thresholds: list[float],
    frame_width: int = 3840,
    frame_height: int = 2160,
) -> dict:
    specs = _discover_test_dirs(str(test_dir))
    if len(specs) < 2:
        raise ValueError("Для leave-one-video-out нужны минимум два видео")

    results = []
    for threshold in thresholds:
        if threshold <= 0:
            raise ValueError("Порог пары должен быть больше 0")
        full_calibration = calibrate_coefficients(
            specs,
            frame_width=frame_width,
            frame_height=frame_height,
            min_pair_depth_change_m=threshold,
            verbose=False,
        )
        in_sample = _evaluate_specs(
            specs, full_calibration, frame_width, frame_height, threshold
        )

        folds = []
        cv_size_errors = []
        cv_distance_errors = []
        cv_track_distance_errors = []
        cv_tracks = 0
        cv_pairs = 0
        for held_out in specs:
            training = [spec for spec in specs if spec is not held_out]
            calibration = calibrate_coefficients(
                training,
                frame_width=frame_width,
                frame_height=frame_height,
                min_pair_depth_change_m=threshold,
                verbose=False,
            )
            evaluation = _evaluate_specs(
                [held_out], calibration, frame_width, frame_height, threshold
            )
            folds.append({
                "held_out_video": held_out["name"],
                **evaluation,
            })
            cv_tracks += evaluation["tracks"]
            cv_pairs += evaluation["pairs"]
            # Для общей CV-метрики каждое видео сохраняет равный вес.
            size_mean = evaluation["size_absolute_error_pct"]["mean"]
            distance_mean = evaluation["distance_absolute_error_pct"]["mean"]
            track_distance_mean = evaluation["track_distance_absolute_error_pct"]["mean"]
            if size_mean is not None:
                cv_size_errors.append(size_mean)
            if distance_mean is not None:
                cv_distance_errors.append(distance_mean)
            if track_distance_mean is not None:
                cv_track_distance_errors.append(track_distance_mean)

        results.append({
            "min_pair_depth_change_m": threshold,
            "full_fit_coefficients": {
                "distance_coef_A": full_calibration.distance_coef_A,
                "distance_coef_B": full_calibration.distance_coef_B,
                "pixel_calib_C": full_calibration.pixel_calib_C,
                "pixel_calib_D": full_calibration.pixel_calib_D,
                "distortion_x_k1": full_calibration.distortion_x_k1,
                "distortion_x_k2": full_calibration.distortion_x_k2,
                "distortion_y_k1": full_calibration.distortion_y_k1,
                "distortion_y_k2": full_calibration.distortion_y_k2,
            },
            "in_sample": in_sample,
            "leave_one_video_out": {
                "tracks": cv_tracks,
                "pairs": cv_pairs,
                "video_mean_distance_mae_pct": round(float(np.mean(cv_distance_errors)), 6),
                "video_mean_track_distance_mae_pct": round(
                    float(np.mean(cv_track_distance_errors)), 6
                ),
                "video_mean_size_mae_pct": round(float(np.mean(cv_size_errors)), 6),
                "folds": folds,
            },
        })

    baseline = results[0]
    for result in results:
        cv = result["leave_one_video_out"]
        baseline_cv = baseline["leave_one_video_out"]
        result["comparison_to_first_threshold"] = {
            "distance_improvement_pct_points": round(
                baseline_cv["video_mean_distance_mae_pct"]
                - cv["video_mean_distance_mae_pct"],
                6,
            ),
            "track_distance_improvement_pct_points": round(
                baseline_cv["video_mean_track_distance_mae_pct"]
                - cv["video_mean_track_distance_mae_pct"],
                6,
            ),
            "size_improvement_pct_points": round(
                baseline_cv["video_mean_size_mae_pct"]
                - cv["video_mean_size_mae_pct"],
                6,
            ),
            "same_track_coverage": cv["tracks"] == baseline_cv["tracks"],
        }

    return {
        "schema_version": 1,
        "method": "leave_one_video_out",
        "frame_width": frame_width,
        "frame_height": frame_height,
        "videos": [spec["name"] for spec in specs],
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=Path("test_video"))
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    parser.add_argument("--output", type=Path, default=Path("calibration_threshold_sensitivity.json"))
    args = parser.parse_args()

    report = run_sensitivity(args.test_dir, args.thresholds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for result in report["results"]:
        cv = result["leave_one_video_out"]
        print(
            f"pair step {result['min_pair_depth_change_m']:.2f} m: "
            f"CV distance MAE={cv['video_mean_distance_mae_pct']:.2f}%, "
            f"track distance MAE={cv['video_mean_track_distance_mae_pct']:.2f}%, "
            f"size MAE={cv['video_mean_size_mae_pct']:.2f}%, "
            f"tracks={cv['tracks']}, pairs={cv['pairs']}"
        )


if __name__ == "__main__":
    main()
