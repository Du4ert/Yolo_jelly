"""Экспериментальная калибровка наклонной дальности без изменения production."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

from camera_geometry import (
    REFERENCE_FRAME_WIDTH,
    CameraCalibration,
    _discover_test_dirs,
    _distortion_factor_xy,
    _distortion_profile_is_valid,
    _extract_calibration_pairs,
    _fit_power_law,
    cos_gamma,
    motion_ray_for_range,
    pixel_to_ray,
)


MIN_COS_GAMMA = 0.17


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_slant_k_percent(
    corrected_size_start: float,
    corrected_size_end: float,
    delta_depth_m: float,
    cos_gamma_end: float,
) -> float:
    """Восстанавливает 100 / endpoint_slant_range из конечной пары.

    Для rho=s2/s1 и вертикального перемещения Delta:
        k_slant = (sqrt(cos(gamma)^2 + rho^2 - 1) - cos(gamma)) / Delta.
    """
    values = (
        corrected_size_start, corrected_size_end,
        delta_depth_m, cos_gamma_end,
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Параметры пары должны быть конечными")
    if corrected_size_start <= 0 or corrected_size_end <= 0:
        raise ValueError("Размеры пары должны быть больше 0")
    if delta_depth_m <= 0:
        raise ValueError("Эксперимент поддерживает только положительный спуск")
    if cos_gamma_end <= 0 or cos_gamma_end > 1:
        raise ValueError("cos_gamma_end должен находиться в диапазоне (0, 1]")

    ratio = corrected_size_end / corrected_size_start
    if ratio <= 1:
        raise ValueError("Видимый размер объекта должен увеличиваться")
    radicand = cos_gamma_end**2 + ratio**2 - 1.0
    if radicand <= 0:
        raise ValueError("Геометрия пары не даёт положительную дальность")
    inverse_range = (np.sqrt(radicand) - cos_gamma_end) / delta_depth_m
    if inverse_range <= 0:
        raise ValueError("Обратная дальность должна быть больше 0")
    return float(inverse_range * 100.0)


def _pair_delta_depth(pair: dict) -> float:
    """Восстанавливает Delta depth из сохранённого сырого k пары."""
    pixels_start = float(pair["size_pixels_start"])
    pixels_end = float(pair["size_pixels"])
    k_raw = float(pair.get("k_raw_percent", pair["k_percent"])) / 100.0
    if pixels_start <= 0 or abs(k_raw) < 1e-12:
        return np.nan
    return ((pixels_end - pixels_start) / pixels_start) / k_raw


def _collect_records(
    test_dir: Path,
    frame_width: int,
    frame_height: int,
    min_confidence: float,
    geometry_name: str,
    min_pair_depth_change_m: float,
) -> tuple[list[dict], list[dict]]:
    calibration = CameraCalibration(frame_width=frame_width, frame_height=frame_height)
    optical_center_ray = pixel_to_ray(*calibration.optical_center_px, calibration)
    records = []
    videos = []

    for spec in _discover_test_dirs(str(test_dir), geometry_name=geometry_name):
        geometry_path = spec.get("geometry_csv")
        if not geometry_path:
            videos.append({"name": spec["name"], "status": "missing_geometry", "pairs": 0})
            continue
        geometry_file = Path(geometry_path)
        geometry_df = pd.read_csv(geometry_file)
        pairs_by_track = _extract_calibration_pairs(
            spec["detections_csv"],
            None,
            frame_width,
            frame_height,
            apply_tilt_correction=False,
            min_pair_depth_change_m=min_pair_depth_change_m,
        )
        video_pairs = 0
        candidate_pairs = 0
        rejected = defaultdict(int)

        for track_id in sorted(spec["known_sizes"]):
            known_size = float(spec["known_sizes"][track_id])
            known_depth = float(spec["known_depths"][track_id])
            for pair in pairs_by_track.get(track_id, []):
                candidate_pairs += 1
                delta_depth = _pair_delta_depth(pair)
                if not np.isfinite(delta_depth) or delta_depth <= 0:
                    rejected["non_descending_or_invalid_depth"] += 1
                    continue
                motion_ray, selection = motion_ray_for_range(
                    int(pair["frame_start"]),
                    int(pair["frame_end"]),
                    geometry_df,
                    calibration,
                    min_confidence=min_confidence,
                )
                if selection is None:
                    rejected["no_local_confident_foe"] += 1
                    continue

                object_ray = pixel_to_ray(
                    float(pair["x_center"]) * frame_width,
                    float(pair["y_center"]) * frame_height,
                    calibration,
                )
                cosine = cos_gamma(object_ray, motion_ray)
                vertical_offset = known_depth - float(pair["depth_camera"])
                if cosine <= MIN_COS_GAMMA:
                    rejected["cos_gamma_too_small"] += 1
                    continue
                if vertical_offset <= 0.05:
                    rejected["vertical_offset_too_small"] += 1
                    continue

                record = dict(pair)
                record.update({
                    "video": spec["name"],
                    "track_id": int(track_id),
                    "known_size_mm": known_size,
                    "known_depth_m": known_depth,
                    "delta_depth_m": float(delta_depth),
                    "cos_gamma_end": float(cosine),
                    "gamma_end_deg": float(np.degrees(np.arccos(cosine))),
                    "cos_camera_tilt": cos_gamma(optical_center_ray, motion_ray),
                    "true_slant_distance_m": float(vertical_offset / cosine),
                    "true_vertical_distance_m": float(vertical_offset),
                    "foe_confidence": selection.confidence,
                    "raw_size_ratio": float(pair["size_pixels"] / pair["size_pixels_start"]),
                })
                records.append(record)
                video_pairs += 1

        videos.append({
            "name": spec["name"],
            "status": "used" if video_pairs > 0 else "no_valid_pairs",
            "pairs": video_pairs,
            "candidate_pairs": candidate_pairs,
            "rejected": dict(sorted(rejected.items())),
            "geometry_rows": len(geometry_df),
            "geometry_sha256": _sha256(geometry_file),
        })

    return records, videos


def _refit(records: list[dict], distortion: np.ndarray, resolution_scale: float):
    x_k1, x_k2, y_k1, y_k2 = distortion
    if not _distortion_profile_is_valid(x_k1, x_k2, y_k1, y_k2):
        return None

    k_values = []
    true_distances = []
    corrected_pixels = []
    known_sizes = []
    track_keys = []
    valid_records = []
    corrected_ratios = []

    for record in records:
        factor_start = _distortion_factor_xy(
            record.get("rx_norm_start", 0.0), record.get("ry_norm_start", 0.0),
            x_k1, x_k2, y_k1, y_k2,
        )
        factor_end = _distortion_factor_xy(
            record.get("rx_norm", 0.0), record.get("ry_norm", 0.0),
            x_k1, x_k2, y_k1, y_k2,
        )
        if factor_start <= 0.2 or factor_end <= 0.2:
            return None
        try:
            corrected_start = record["size_pixels_start"] * factor_start
            corrected_end = record["size_pixels"] * factor_end
            k_percent = exact_slant_k_percent(
                corrected_start,
                corrected_end,
                record["delta_depth_m"],
                record["cos_gamma_end"],
            )
        except ValueError:
            continue
        k_values.append(k_percent)
        true_distances.append(record["true_slant_distance_m"])
        corrected_pixels.append(record["size_pixels"] * factor_end / resolution_scale)
        known_sizes.append(record["known_size_mm"])
        track_keys.append((record["video"], record["track_id"]))
        valid_records.append(record)
        corrected_ratios.append(corrected_end / corrected_start)

    if len(k_values) < 2:
        return None
    k_values = np.asarray(k_values)
    true_distances = np.asarray(true_distances)
    corrected_pixels = np.asarray(corrected_pixels)
    known_sizes = np.asarray(known_sizes)

    A, B = _fit_power_law(k_values, true_distances)
    estimated_distances = A * k_values**B
    C, D = _fit_power_law(estimated_distances, corrected_pixels / known_sizes)
    estimated_sizes = corrected_pixels / (C * estimated_distances**D)
    size_log_error = np.log(estimated_sizes / known_sizes)
    distance_log_error = np.log(estimated_distances / true_distances)

    losses = []
    for track_key in sorted(set(track_keys)):
        mask = np.asarray([key == track_key for key in track_keys])
        losses.append(
            np.mean(np.minimum(size_log_error[mask]**2, 0.5))
            + 0.3 * np.mean(np.minimum(distance_log_error[mask]**2, 0.5))
        )
    regularization = 0.01 * float(np.sum(np.asarray(distortion)**2))
    loss = float(np.mean(losses) + regularization)
    return {
        "loss": loss,
        "coefficients": (A, B, C, D, x_k1, x_k2, y_k1, y_k2),
        "k_values": k_values,
        "true_distances": true_distances,
        "estimated_distances": estimated_distances,
        "estimated_sizes": estimated_sizes,
        "known_sizes": known_sizes,
        "track_keys": track_keys,
        "records": valid_records,
        "corrected_ratios": np.asarray(corrected_ratios),
    }


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "count": int(len(values)),
        "mean": round(float(np.mean(values)), 6),
        "median": round(float(np.median(values)), 6),
        "p95": round(float(np.percentile(values, 95)), 6),
        "max": round(float(np.max(values)), 6),
    }


def _track_size_metrics(estimated_sizes, known_sizes, track_keys) -> tuple[dict, list[float]]:
    errors = []
    for track_key in sorted(set(track_keys)):
        mask = np.asarray([key == track_key for key in track_keys])
        estimated_size = float(np.median(estimated_sizes[mask]))
        known_size = float(known_sizes[mask][0])
        errors.append((estimated_size - known_size) / known_size * 100.0)
    return {
        "signed_error_pct": _summary(np.asarray(errors)),
        "absolute_error_pct": _summary(np.abs(errors)),
    }, errors


def _vertical_baseline_same_pairs(
    records: list[dict],
    calibration_path: Path,
    resolution_scale: float,
) -> dict:
    calibration = CameraCalibration.from_json(str(calibration_path))
    distances = []
    true_distances = []
    sizes = []
    known_sizes = []
    track_keys = []

    for record in records:
        factor_start = _distortion_factor_xy(
            record.get("rx_norm_start", 0.0), record.get("ry_norm_start", 0.0),
            calibration.distortion_x_k1, calibration.distortion_x_k2,
            calibration.distortion_y_k1, calibration.distortion_y_k2,
        )
        factor_end = _distortion_factor_xy(
            record.get("rx_norm", 0.0), record.get("ry_norm", 0.0),
            calibration.distortion_x_k1, calibration.distortion_x_k2,
            calibration.distortion_y_k1, calibration.distortion_y_k2,
        )
        ratio = (
            record["size_pixels"] * factor_end
            / (record["size_pixels_start"] * factor_start)
        )
        if ratio <= 1:
            continue
        cosine = max(float(record["cos_camera_tilt"]), MIN_COS_GAMMA)
        k_percent = (ratio - 1.0) / record["delta_depth_m"] / cosine * 100.0
        distance = calibration.distance_coef_A * k_percent**calibration.distance_coef_B
        pixels_ref = record["size_pixels"] * factor_end / resolution_scale
        pixel_calibration = calibration.pixel_calib_C * distance**calibration.pixel_calib_D
        distances.append(distance)
        true_distances.append(record["true_vertical_distance_m"])
        sizes.append(pixels_ref / pixel_calibration)
        known_sizes.append(record["known_size_mm"])
        track_keys.append((record["video"], record["track_id"]))

    distances = np.asarray(distances)
    true_distances = np.asarray(true_distances)
    sizes = np.asarray(sizes)
    known_sizes = np.asarray(known_sizes)
    distance_errors = (distances - true_distances) / true_distances * 100.0
    size_metrics, _ = _track_size_metrics(sizes, known_sizes, track_keys)
    track_distance_errors = []
    for track_key in sorted(set(track_keys)):
        mask = np.asarray([key == track_key for key in track_keys])
        median_true = float(np.median(true_distances[mask]))
        track_distance_errors.append(
            (float(np.median(distances[mask])) - median_true) / median_true * 100.0
        )
    return {
        "pairs": len(distances),
        "tracks": len(set(track_keys)),
        "distance_signed_error_pct": _summary(distance_errors),
        "distance_absolute_error_pct": _summary(np.abs(distance_errors)),
        "track_distance_signed_error_pct": _summary(np.asarray(track_distance_errors)),
        "track_distance_absolute_error_pct": _summary(np.abs(track_distance_errors)),
        "size_signed_error_pct": size_metrics["signed_error_pct"],
        "size_absolute_error_pct": size_metrics["absolute_error_pct"],
    }


def _diagnostics(fitted: dict, distance_errors: np.ndarray) -> dict:
    records = fitted["records"]
    absolute_errors = np.abs(distance_errors)
    geometry_distances = 100.0 / fitted["k_values"]
    geometry_errors = (
        geometry_distances - fitted["true_distances"]
    ) / fitted["true_distances"] * 100.0
    features = {
        "gamma_deg": np.asarray([record["gamma_end_deg"] for record in records]),
        "delta_depth_m": np.asarray([record["delta_depth_m"] for record in records]),
        "true_slant_distance_m": fitted["true_distances"],
        "corrected_size_ratio": fitted["corrected_ratios"],
        "foe_confidence": np.asarray([record["foe_confidence"] for record in records]),
        "bbox_radius_norm": np.asarray([
            np.hypot(record.get("rx_norm", 0.0), record.get("ry_norm", 0.0))
            for record in records
        ]),
    }
    correlations = {}
    for name, values in features.items():
        correlation, p_value = spearmanr(values, absolute_errors)
        correlations[name] = {
            "spearman_r": round(float(correlation), 6),
            "p_value": round(float(p_value), 6),
        }

    per_video = []
    for video in sorted({record["video"] for record in records}):
        mask = np.asarray([record["video"] == video for record in records])
        per_video.append({
            "video": video,
            "pairs": int(np.sum(mask)),
            "power_law_distance_mae_pct": round(float(np.mean(absolute_errors[mask])), 6),
            "exact_geometry_distance_mae_pct": round(float(np.mean(np.abs(geometry_errors[mask]))), 6),
            "median_gamma_deg": round(float(np.median(features["gamma_deg"][mask])), 6),
            "median_delta_depth_m": round(float(np.median(features["delta_depth_m"][mask])), 6),
        })

    return {
        "exact_geometry_distance_signed_error_pct": _summary(geometry_errors),
        "exact_geometry_distance_absolute_error_pct": _summary(np.abs(geometry_errors)),
        "absolute_distance_error_correlations": correlations,
        "per_video": per_video,
    }


def run_experiment(
    test_dir: Path,
    frame_width: int = 3840,
    frame_height: int = 2160,
    min_confidence: float = 0.5,
    geometry_name: str = "ball_geometry.csv",
    geometry_frame_step: int | None = None,
    baseline_report_path: Path | None = None,
    vertical_calibration_path: Path | None = None,
    min_pair_depth_change_m: float = 0.01,
) -> dict:
    if min_pair_depth_change_m <= 0:
        raise ValueError("min_pair_depth_change_m должен быть больше 0")
    records, videos = _collect_records(
        test_dir, frame_width, frame_height, min_confidence, geometry_name,
        min_pair_depth_change_m,
    )
    if not records:
        raise ValueError("Нет пар с локальным уверенным FOE")
    resolution_scale = frame_width / REFERENCE_FRAME_WIDTH

    result = minimize(
        lambda params: (
            fitted["loss"] if (fitted := _refit(records, params, resolution_scale)) else 1e6
        ),
        np.zeros(4),
        method="L-BFGS-B",
        bounds=[(-1.0, 1.0)] * 4,
        options={"maxiter": 10000, "ftol": 1e-14, "gtol": 1e-9},
    )
    fitted = _refit(records, result.x, resolution_scale)
    if fitted is None:
        raise RuntimeError("Не удалось получить slant-калибровку")

    A, B, C, D, x_k1, x_k2, y_k1, y_k2 = fitted["coefficients"]
    distance_errors = (
        fitted["estimated_distances"] - fitted["true_distances"]
    ) / fitted["true_distances"] * 100.0

    tracks = []
    track_size_errors = []
    track_distance_errors = []
    for track_key in sorted(set(fitted["track_keys"])):
        mask = np.asarray([key == track_key for key in fitted["track_keys"]])
        estimated_size = float(np.median(fitted["estimated_sizes"][mask]))
        known_size = float(fitted["known_sizes"][mask][0])
        size_error = (estimated_size - known_size) / known_size * 100.0
        track_size_errors.append(size_error)
        median_estimated_distance = float(np.median(fitted["estimated_distances"][mask]))
        median_true_distance = float(np.median(fitted["true_distances"][mask]))
        distance_error = (
            (median_estimated_distance - median_true_distance)
            / median_true_distance * 100.0
        )
        track_distance_errors.append(distance_error)
        tracks.append({
            "video": track_key[0],
            "track_id": track_key[1],
            "pairs": int(np.sum(mask)),
            "known_size_mm": known_size,
            "estimated_size_mm": round(estimated_size, 6),
            "size_error_pct": round(size_error, 6),
            "median_estimated_distance_m": round(median_estimated_distance, 6),
            "median_true_distance_m": round(median_true_distance, 6),
            "distance_error_pct": round(distance_error, 6),
            "median_gamma_deg": round(float(np.median([
                record["gamma_end_deg"]
                for record, selected in zip(fitted["records"], mask) if selected
            ])), 6),
        })

    report = {
        "schema_version": 1,
        "experimental": True,
        "distance_basis": "slant",
        "runtime_compatible": False,
        "assumptions": [
            "stationary_object",
            "vertical_camera_translation",
            "local_confident_foe",
            "equidistant_projection",
            "constant_object_shape",
        ],
        "settings": {
            "frame_width": frame_width,
            "frame_height": frame_height,
            "min_foe_confidence": min_confidence,
            "min_cos_gamma": MIN_COS_GAMMA,
            "geometry_name": geometry_name,
            "geometry_frame_step": geometry_frame_step,
            "min_pair_depth_change_m": min_pair_depth_change_m,
        },
        "optimizer": {
            "success": bool(result.success),
            "message": str(result.message),
            "loss": round(float(fitted["loss"]), 8),
        },
        "coefficients": {
            "distance_coef_A": round(float(A), 4),
            "distance_coef_B": round(float(B), 4),
            "pixel_calib_C": round(float(C), 4),
            "pixel_calib_D": round(float(D), 4),
            "distortion_x_k1": round(float(x_k1), 6),
            "distortion_x_k2": round(float(x_k2), 6),
            "distortion_y_k1": round(float(y_k1), 6),
            "distortion_y_k2": round(float(y_k2), 6),
        },
        "totals": {
            "videos_discovered": len(videos),
            "videos_used": sum(video["status"] == "used" for video in videos),
            "tracks_used": len(tracks),
            "pairs_used": len(fitted["records"]),
        },
        "videos": videos,
        "metrics": {
            "distance_signed_error_pct": _summary(distance_errors),
            "distance_absolute_error_pct": _summary(np.abs(distance_errors)),
            "track_distance_signed_error_pct": _summary(np.asarray(track_distance_errors)),
            "track_distance_absolute_error_pct": _summary(np.abs(track_distance_errors)),
            "size_signed_error_pct": _summary(np.asarray(track_size_errors)),
            "size_absolute_error_pct": _summary(np.abs(track_size_errors)),
            "gamma_deg": _summary(np.asarray([
                record["gamma_end_deg"] for record in fitted["records"]
            ])),
        },
        "diagnostics": _diagnostics(fitted, distance_errors),
        "tracks": tracks,
    }
    if vertical_calibration_path is not None:
        report["same_pairs_vertical_baseline"] = _vertical_baseline_same_pairs(
            fitted["records"], vertical_calibration_path, resolution_scale
        )
    if baseline_report_path is not None:
        baseline = json.loads(baseline_report_path.read_text(encoding="utf-8"))
        baseline_distance = baseline["metrics"]["distance_absolute_error_pct"]["mean"]
        baseline_track_distance = baseline["metrics"]["track_distance_absolute_error_pct"]["mean"]
        baseline_size = baseline["metrics"]["pipeline_size_absolute_error_pct"]["mean"]
        baseline_tracks = baseline["totals"]["contributing_tracks"]
        slant_distance = report["metrics"]["distance_absolute_error_pct"]["mean"]
        slant_track_distance = report["metrics"]["track_distance_absolute_error_pct"]["mean"]
        slant_size = report["metrics"]["size_absolute_error_pct"]["mean"]
        same_pairs = report.get("same_pairs_vertical_baseline")
        comparison_distance = (
            same_pairs["distance_absolute_error_pct"]["mean"]
            if same_pairs else baseline_distance
        )
        comparison_track_distance = (
            same_pairs["track_distance_absolute_error_pct"]["mean"]
            if same_pairs else baseline_track_distance
        )
        comparison_size = (
            same_pairs["size_absolute_error_pct"]["mean"]
            if same_pairs else baseline_size
        )
        distance_improvement = comparison_distance - slant_distance
        track_distance_improvement = comparison_track_distance - slant_track_distance
        full_coverage = report["totals"]["tracks_used"] >= baseline_tracks
        material_distance_improvement = track_distance_improvement >= 5.0
        size_not_worse = slant_size <= comparison_size
        transition_recommended = bool(
            result.success
            and full_coverage
            and material_distance_improvement
            and size_not_worse
        )
        reasons = []
        if not full_coverage:
            reasons.append("Новая модель не покрывает все baseline-треки")
        if not material_distance_improvement:
            reasons.append("Улучшение ошибки дистанции по трекам меньше 5 п.п.")
        if not size_not_worse:
            reasons.append("Средняя ошибка размера ухудшилась")
        if not result.success:
            reasons.append("Оптимизатор не подтвердил сходимость")
        report["comparison_to_vertical_baseline"] = {
            "baseline_report": baseline_report_path.name,
            "baseline_tracks": baseline_tracks,
            "baseline_distance_mae_pct": baseline_distance,
            "slant_distance_mae_pct": slant_distance,
            "baseline_track_distance_mae_pct": baseline_track_distance,
            "slant_track_distance_mae_pct": slant_track_distance,
            "distance_improvement_pct_points": round(distance_improvement, 6),
            "track_distance_improvement_pct_points": round(
                track_distance_improvement, 6
            ),
            "baseline_size_mae_pct": baseline_size,
            "slant_size_mae_pct": slant_size,
            "same_pairs_vertical_distance_mae_pct": comparison_distance,
            "same_pairs_vertical_track_distance_mae_pct": comparison_track_distance,
            "same_pairs_vertical_size_mae_pct": comparison_size,
            "full_track_coverage": full_coverage,
            "material_distance_improvement": material_distance_improvement,
            "size_not_worse": size_not_worse,
            "transition_recommended": transition_recommended,
            "reasons": reasons,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=Path("test_video"))
    parser.add_argument("--output", type=Path, default=Path("calibration_slant_experiment.json"))
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--geometry-name", default="ball_geometry.csv")
    parser.add_argument("--geometry-frame-step", type=int)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--vertical-calibration", type=Path)
    parser.add_argument("--min-pair-depth-change", type=float, default=0.01)
    args = parser.parse_args()

    report = run_experiment(
        args.test_dir,
        frame_width=args.width,
        frame_height=args.height,
        min_confidence=args.min_confidence,
        geometry_name=args.geometry_name,
        geometry_frame_step=args.geometry_frame_step,
        baseline_report_path=args.baseline_report,
        vertical_calibration_path=args.vertical_calibration,
        min_pair_depth_change_m=args.min_pair_depth_change,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Slant experiment saved to {args.output}: "
        f"{report['totals']['videos_used']} videos, "
        f"{report['totals']['tracks_used']} tracks, "
        f"{report['totals']['pairs_used']} pairs; "
        f"distance MAE={report['metrics']['distance_absolute_error_pct']['mean']:.2f}%, "
        f"size MAE={report['metrics']['size_absolute_error_pct']['mean']:.2f}%"
    )


if __name__ == "__main__":
    main()
