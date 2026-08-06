"""LOVO-проверка простой угловой модели вертикального зазора."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from camera_geometry import (
    CameraCalibration,
    _discover_test_dirs,
    _distortion_factor_xy,
    _extract_calibration_pairs,
    calibrate_coefficients,
    exact_pair_distances as _production_exact_pair_distances,
    exact_vertical_k_percent as _production_exact_vertical_k_percent,
    select_pair_angle as _production_select_pair_angle,
    simple_gamma_from_foe,
)
from slant_calibration_experiment import _pair_delta_depth


MIN_COS_GAMMA = 0.17


def simple_gamma(
    x_normalized: float,
    y_normalized: float,
    foe_x_px: float,
    foe_y_px: float,
    calibration: CameraCalibration,
) -> tuple[float, float]:
    """Возвращает угол до направления движения без построения 3D-лучей."""
    return simple_gamma_from_foe(
        x_normalized,
        y_normalized,
        foe_x_px,
        foe_y_px,
        calibration,
    )


def select_pair_angle(
    frame_start: int,
    frame_end: int,
    x_normalized_end: float,
    y_normalized_end: float,
    geometry_df: pd.DataFrame | None,
    calibration: CameraCalibration,
    min_confidence: float = 0.5,
) -> dict | None:
    """Выбирает только локальный уверенный radial FOE для конца пары."""
    return _production_select_pair_angle(
        frame_start,
        frame_end,
        x_normalized_end,
        y_normalized_end,
        geometry_df,
        calibration,
        min_confidence=min_confidence,
    )


def exact_pair_distances(
    corrected_size_start: float,
    corrected_size_end: float,
    delta_depth_m: float,
    cos_gamma_end: float,
) -> tuple[float, float]:
    """Возвращает конечные (slant range, vertical offset) в метрах."""
    return _production_exact_pair_distances(
        corrected_size_start,
        corrected_size_end,
        delta_depth_m,
        cos_gamma_end,
    )


def exact_vertical_k_percent(
    corrected_size_start: float,
    corrected_size_end: float,
    delta_depth_m: float,
    cos_gamma_end: float,
) -> float:
    """Восстанавливает 100 / endpoint_vertical_offset из конечной пары."""
    return _production_exact_vertical_k_percent(
        corrected_size_start,
        corrected_size_end,
        delta_depth_m,
        cos_gamma_end,
    )


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "rmse": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(len(array)),
        "mean": round(float(np.mean(array)), 6),
        "median": round(float(np.median(array)), 6),
        "rmse": round(float(np.sqrt(np.mean(array**2))), 6),
        "p95": round(float(np.percentile(array, 95)), 6),
        "max": round(float(np.max(array)), 6),
    }


def _collect_records(
    specs: list[dict],
    frame_width: int,
    frame_height: int,
    min_confidence: float,
    min_pair_depth_change_m: float,
) -> tuple[list[dict], list[dict]]:
    calibration = CameraCalibration(frame_width=frame_width, frame_height=frame_height)
    records = []
    video_reports = []

    for spec in specs:
        geometry_path = spec.get("geometry_csv")
        rejected = Counter()
        candidate_pairs = 0
        video_records = 0
        if not geometry_path:
            video_reports.append({
                "name": spec["name"],
                "status": "missing_geometry",
                "pairs": 0,
                "candidate_pairs": 0,
                "rejected": {},
            })
            continue

        geometry_df = pd.read_csv(geometry_path)
        pairs_by_track = _extract_calibration_pairs(
            spec["detections_csv"],
            None,
            frame_width,
            frame_height,
            apply_tilt_correction=False,
            min_pair_depth_change_m=min_pair_depth_change_m,
        )

        for track_id in sorted(spec["known_depths"]):
            known_depth = float(spec["known_depths"][track_id])
            known_size = float(spec["known_sizes"][track_id])
            for pair in pairs_by_track.get(track_id, []):
                candidate_pairs += 1
                delta_depth = _pair_delta_depth(pair)
                if not np.isfinite(delta_depth) or delta_depth <= 0:
                    rejected["non_descending_or_invalid_depth"] += 1
                    continue
                angle = select_pair_angle(
                    int(pair["frame_start"]),
                    int(pair["frame_end"]),
                    float(pair["x_center"]),
                    float(pair["y_center"]),
                    geometry_df,
                    calibration,
                    min_confidence=min_confidence,
                )
                if angle is None:
                    rejected["no_local_confident_radial_foe"] += 1
                    continue
                true_vertical = known_depth - float(pair["depth_camera"])
                if true_vertical <= 0.05:
                    rejected["vertical_offset_too_small"] += 1
                    continue

                record = dict(pair)
                record.update({
                    "video": spec["name"],
                    "track_id": int(track_id),
                    "known_depth_m": known_depth,
                    "known_size_mm": known_size,
                    "delta_depth_m": float(delta_depth),
                    "true_vertical_offset_m": true_vertical,
                    **angle,
                })
                records.append(record)
                video_records += 1

        video_reports.append({
            "name": spec["name"],
            "status": "used" if video_records else "no_valid_pairs",
            "pairs": video_records,
            "candidate_pairs": candidate_pairs,
            "rejected": dict(sorted(rejected.items())),
        })
    return records, video_reports


def _features(records: list[dict], calibration: CameraCalibration) -> tuple[list[dict], np.ndarray, np.ndarray]:
    valid_records = []
    angle_features = []
    baseline_features = []
    for record in records:
        factor_start = _distortion_factor_xy(
            record.get("rx_norm_start", 0.0),
            record.get("ry_norm_start", 0.0),
            calibration.distortion_x_k1,
            calibration.distortion_x_k2,
            calibration.distortion_y_k1,
            calibration.distortion_y_k2,
        )
        factor_end = _distortion_factor_xy(
            record.get("rx_norm", 0.0),
            record.get("ry_norm", 0.0),
            calibration.distortion_x_k1,
            calibration.distortion_x_k2,
            calibration.distortion_y_k1,
            calibration.distortion_y_k2,
        )
        corrected_start = float(record["size_pixels_start"]) * factor_start
        corrected_end = float(record["size_pixels"]) * factor_end
        try:
            angle_feature = exact_vertical_k_percent(
                corrected_start,
                corrected_end,
                float(record["delta_depth_m"]),
                float(record["cos_gamma_end"]),
            )
        except ValueError:
            continue
        ratio = corrected_end / corrected_start
        baseline_feature = (
            (ratio - 1.0)
            / float(record["delta_depth_m"])
            / float(record["cos_camera_tilt"])
            * 100.0
        )
        if not np.isfinite(baseline_feature) or baseline_feature <= 0:
            continue
        valid_records.append(record)
        angle_features.append(angle_feature)
        baseline_features.append(baseline_feature)
    return valid_records, np.asarray(angle_features), np.asarray(baseline_features)


def _fit_power_law_balanced(
    features: np.ndarray,
    targets: np.ndarray,
    track_keys: list[tuple[str, int]],
) -> tuple[float, float]:
    if len(features) < 2 or len(features) != len(targets):
        raise ValueError("Для калибровки нужны минимум две пары")
    counts = Counter(track_keys)
    weights = np.sqrt(np.asarray([1.0 / counts[key] for key in track_keys]))
    slope, intercept = np.polyfit(np.log(features), np.log(targets), 1, w=weights)
    return float(np.exp(intercept)), float(slope)


def _fit_angle_pixel_calibration(
    records: list[dict],
    angle_features: np.ndarray,
    angle_coefficients: tuple[float, float],
    calibration: CameraCalibration,
) -> tuple[float, float]:
    distances = angle_coefficients[0] * angle_features**angle_coefficients[1]
    pixels_per_mm = []
    track_keys = []
    for record in records:
        factor_end = _distortion_factor_xy(
            record.get("rx_norm", 0.0),
            record.get("ry_norm", 0.0),
            calibration.distortion_x_k1,
            calibration.distortion_x_k2,
            calibration.distortion_y_k1,
            calibration.distortion_y_k2,
        )
        pixels_ref = (
            float(record["size_pixels"])
            * factor_end
            / calibration.resolution_scale
        )
        pixels_per_mm.append(pixels_ref / float(record["known_size_mm"]))
        track_keys.append((record["video"], record["track_id"]))
    return _fit_power_law_balanced(
        distances,
        np.asarray(pixels_per_mm),
        track_keys,
    )


def _evaluate(
    records: list[dict],
    angle_features: np.ndarray,
    baseline_features: np.ndarray,
    angle_coefficients: tuple[float, float],
    baseline_coefficients: tuple[float, float],
) -> dict:
    angle_offsets = angle_coefficients[0] * angle_features**angle_coefficients[1]
    baseline_offsets = baseline_coefficients[0] * baseline_features**baseline_coefficients[1]
    tracks = []
    angle_absolute_errors = []
    baseline_absolute_errors = []
    angle_relative_errors = []
    baseline_relative_errors = []

    grouped = defaultdict(list)
    for index, record in enumerate(records):
        grouped[(record["video"], record["track_id"])].append(index)
    for track_key in sorted(grouped):
        indices = np.asarray(grouped[track_key], dtype=int)
        known_depth = float(records[int(indices[0])]["known_depth_m"])
        camera_depths = np.asarray([
            float(records[index]["depth_camera"]) for index in indices
        ])
        angle_depth = float(np.median(camera_depths + angle_offsets[indices]))
        baseline_depth = float(np.median(camera_depths + baseline_offsets[indices]))
        angle_error = angle_depth - known_depth
        baseline_error = baseline_depth - known_depth
        true_offsets = np.asarray([
            float(records[index]["true_vertical_offset_m"]) for index in indices
        ])
        angle_relative = float(np.median(
            (angle_offsets[indices] - true_offsets) / true_offsets * 100.0
        ))
        baseline_relative = float(np.median(
            (baseline_offsets[indices] - true_offsets) / true_offsets * 100.0
        ))
        angle_absolute_errors.append(abs(angle_error))
        baseline_absolute_errors.append(abs(baseline_error))
        angle_relative_errors.append(abs(angle_relative))
        baseline_relative_errors.append(abs(baseline_relative))
        tracks.append({
            "video": track_key[0],
            "track_id": track_key[1],
            "pairs": len(indices),
            "known_object_depth_m": round(known_depth, 6),
            "angle_object_depth_m": round(angle_depth, 6),
            "baseline_object_depth_m": round(baseline_depth, 6),
            "angle_error_m": round(angle_error, 6),
            "baseline_error_m": round(baseline_error, 6),
            "improvement_m": round(abs(baseline_error) - abs(angle_error), 6),
            "median_gamma_deg": round(float(np.median([
                records[index]["gamma_end_deg"] for index in indices
            ])), 6),
        })

    improvements = [track["improvement_m"] for track in tracks]
    return {
        "tracks": len(tracks),
        "pairs": len(records),
        "angle_object_depth_absolute_error_m": _summary(angle_absolute_errors),
        "baseline_object_depth_absolute_error_m": _summary(baseline_absolute_errors),
        "angle_vertical_offset_absolute_error_pct": _summary(angle_relative_errors),
        "baseline_vertical_offset_absolute_error_pct": _summary(baseline_relative_errors),
        "tracks_improved": sum(value > 0 for value in improvements),
        "tracks_worsened": sum(value < 0 for value in improvements),
        "tracks_unchanged": sum(value == 0 for value in improvements),
        "tracks_detail": tracks,
    }


def run_experiment(
    test_dir: Path,
    frame_width: int = 3840,
    frame_height: int = 2160,
    geometry_name: str = "ball_geometry_step4.csv",
    min_confidence: float = 0.5,
    min_pair_depth_change_m: float = 0.01,
    production_calibration_path: Path | None = None,
) -> dict:
    specs = _discover_test_dirs(str(test_dir), geometry_name=geometry_name)
    if len(specs) < 2:
        raise ValueError("Для leave-one-video-out нужны минимум два видео")
    records, videos = _collect_records(
        specs,
        frame_width,
        frame_height,
        min_confidence,
        min_pair_depth_change_m,
    )
    if not records:
        raise ValueError("Нет пар с локальным уверенным FOE")

    folds = []
    pooled_angle_errors = []
    pooled_baseline_errors = []
    video_angle_mae = []
    video_baseline_mae = []
    improved_tracks = 0
    evaluated_tracks = 0
    for held_out in specs:
        training_specs = [spec for spec in specs if spec is not held_out]
        calibration = calibrate_coefficients(
            training_specs,
            frame_width=frame_width,
            frame_height=frame_height,
            min_pair_depth_change_m=min_pair_depth_change_m,
            verbose=False,
        )
        train_records = [record for record in records if record["video"] != held_out["name"]]
        test_records = [record for record in records if record["video"] == held_out["name"]]
        train_records, train_angle, train_baseline = _features(train_records, calibration)
        test_records, test_angle, test_baseline = _features(test_records, calibration)
        if not test_records:
            folds.append({
                "held_out_video": held_out["name"],
                "status": "no_evaluable_angle_pairs",
                "tracks": 0,
                "pairs": 0,
            })
            continue

        targets = np.asarray([
            record["true_vertical_offset_m"] for record in train_records
        ])
        track_keys = [
            (record["video"], record["track_id"]) for record in train_records
        ]
        angle_coefficients = _fit_power_law_balanced(train_angle, targets, track_keys)
        baseline_coefficients = _fit_power_law_balanced(train_baseline, targets, track_keys)
        evaluation = _evaluate(
            test_records,
            test_angle,
            test_baseline,
            angle_coefficients,
            baseline_coefficients,
        )
        angle_errors = [
            abs(track["angle_error_m"]) for track in evaluation["tracks_detail"]
        ]
        baseline_errors = [
            abs(track["baseline_error_m"]) for track in evaluation["tracks_detail"]
        ]
        pooled_angle_errors.extend(angle_errors)
        pooled_baseline_errors.extend(baseline_errors)
        video_angle_mae.append(float(np.mean(angle_errors)))
        video_baseline_mae.append(float(np.mean(baseline_errors)))
        improved_tracks += evaluation["tracks_improved"]
        evaluated_tracks += evaluation["tracks"]
        folds.append({
            "held_out_video": held_out["name"],
            "status": "evaluated",
            "angle_coefficients": {
                "A": round(angle_coefficients[0], 6),
                "B": round(angle_coefficients[1], 6),
            },
            "baseline_coefficients": {
                "A": round(baseline_coefficients[0], 6),
                "B": round(baseline_coefficients[1], 6),
            },
            **evaluation,
        })

    if production_calibration_path is not None:
        full_calibration = CameraCalibration.from_json(
            str(production_calibration_path)
        )
    else:
        full_calibration = calibrate_coefficients(
            specs,
            frame_width=frame_width,
            frame_height=frame_height,
            min_pair_depth_change_m=min_pair_depth_change_m,
            verbose=False,
        )
    full_records, full_angle, full_baseline = _features(records, full_calibration)
    full_targets = np.asarray([
        record["true_vertical_offset_m"] for record in full_records
    ])
    full_keys = [(record["video"], record["track_id"]) for record in full_records]
    full_angle_coefficients = _fit_power_law_balanced(full_angle, full_targets, full_keys)
    full_baseline_coefficients = _fit_power_law_balanced(full_baseline, full_targets, full_keys)
    full_angle_pixel_coefficients = _fit_angle_pixel_calibration(
        full_records,
        full_angle,
        full_angle_coefficients,
        full_calibration,
    )

    angle_macro = float(np.mean(video_angle_mae))
    baseline_macro = float(np.mean(video_baseline_mae))
    angle_pooled = float(np.mean(pooled_angle_errors))
    baseline_pooled = float(np.mean(pooled_baseline_errors))
    transition_recommended = bool(
        angle_macro < baseline_macro
        and angle_pooled < baseline_pooled
        and improved_tracks > evaluated_tracks / 2
    )
    return {
        "schema_version": 1,
        "experimental": True,
        "runtime_compatible": False,
        "model": "simple_endpoint_angle_vertical_offset",
        "assumptions": [
            "stationary_object",
            "vertical_camera_translation",
            "local_confident_radial_foe",
            "equidistant_angular_plane",
            "constant_object_shape",
        ],
        "settings": {
            "frame_width": frame_width,
            "frame_height": frame_height,
            "fov_horizontal_deg": full_calibration.fov_horizontal,
            "fov_vertical_deg": full_calibration.fov_vertical,
            "geometry_name": geometry_name,
            "min_foe_confidence": min_confidence,
            "min_cos_gamma": MIN_COS_GAMMA,
            "min_pair_depth_change_m": min_pair_depth_change_m,
        },
        "missing_foe_policy": {
            "pair": "skip",
            "track_with_valid_pairs": "median_object_depth_from_valid_pairs",
            "track_without_valid_pairs": "legacy_vertical_model_with_low_confidence",
            "whole_video_foe_fallback": False,
        },
        "coverage": {
            "videos_discovered": len(specs),
            "videos_with_pairs": sum(video["status"] == "used" for video in videos),
            "tracks": len(set(full_keys)),
            "pairs": len(full_records),
        },
        "videos": videos,
        "full_fit": {
            "distortion_source": (
                production_calibration_path.name
                if production_calibration_path is not None else "refit_all_videos"
            ),
            "angle_coefficients": {
                "A": round(full_angle_coefficients[0], 6),
                "B": round(full_angle_coefficients[1], 6),
                "pixel_C": round(full_angle_pixel_coefficients[0], 6),
                "pixel_D": round(full_angle_pixel_coefficients[1], 6),
            },
            "baseline_coefficients": {
                "A": round(full_baseline_coefficients[0], 6),
                "B": round(full_baseline_coefficients[1], 6),
            },
        },
        "leave_one_video_out": {
            "evaluable_videos": len(video_angle_mae),
            "tracks": evaluated_tracks,
            "angle_video_macro_object_depth_mae_m": round(angle_macro, 6),
            "baseline_video_macro_object_depth_mae_m": round(baseline_macro, 6),
            "angle_pooled_track_object_depth_mae_m": round(angle_pooled, 6),
            "baseline_pooled_track_object_depth_mae_m": round(baseline_pooled, 6),
            "improvement_video_macro_m": round(baseline_macro - angle_macro, 6),
            "improvement_pooled_m": round(baseline_pooled - angle_pooled, 6),
            "tracks_improved": improved_tracks,
            "tracks_worsened_or_unchanged": evaluated_tracks - improved_tracks,
            "transition_recommended": transition_recommended,
            "folds": folds,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=Path("test_video"))
    parser.add_argument("--geometry-name", default="ball_geometry_step4.csv")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-pair-depth-change", type=float, default=0.01)
    parser.add_argument(
        "--production-calibration",
        type=Path,
        default=Path("calibration_xy.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("calibration_angle_vertical_experiment.json"),
    )
    args = parser.parse_args()
    report = run_experiment(
        args.test_dir,
        geometry_name=args.geometry_name,
        min_confidence=args.min_confidence,
        min_pair_depth_change_m=args.min_pair_depth_change,
        production_calibration_path=args.production_calibration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    cv = report["leave_one_video_out"]
    print(
        f"Angle vertical CV: angle MAE={cv['angle_video_macro_object_depth_mae_m']:.3f} m, "
        f"baseline MAE={cv['baseline_video_macro_object_depth_mae_m']:.3f} m, "
        f"tracks={cv['tracks']}, recommended={cv['transition_recommended']}"
    )


if __name__ == "__main__":
    main()
