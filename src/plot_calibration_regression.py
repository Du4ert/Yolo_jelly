"""Построение графиков регрессий калибровки камеры."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from camera_geometry import (
    REFERENCE_FRAME_WIDTH,
    _discover_test_dirs,
    _extract_calibration_pairs,
)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _collect_points(test_dir: Path, frame_width: int, frame_height: int) -> dict[str, np.ndarray]:
    specs = _discover_test_dirs(str(test_dir))
    resolution_scale = frame_width / REFERENCE_FRAME_WIDTH
    values = {key: [] for key in ("k", "distance", "pixel_calib")}
    pipeline_values = {"rx": [], "ry": []}

    for spec in specs:
        pairs_by_track = _extract_calibration_pairs(
            spec["detections_csv"], spec.get("geometry_csv"), frame_width, frame_height,
            apply_tilt_correction=True,
        )
        for track_id, known_size in spec["known_sizes"].items():
            if track_id not in spec["known_depths"]:
                continue
            track_depth = spec["known_depths"][track_id]
            for pair in pairs_by_track.get(track_id, []):
                pixels = pair["size_pixels"] / resolution_scale
                pipeline_values["rx"].append(pair.get("rx_norm", 0.0))
                pipeline_values["ry"].append(pair.get("ry_norm", 0.0))
                distance = track_depth - pair["depth_camera"]
                if distance < 0.05:
                    continue
                values["k"].append(pair["k_percent"])
                values["distance"].append(distance)
                values["pixel_calib"].append(pixels / known_size)

    result = {key: np.asarray(value, dtype=float) for key, value in values.items()}
    result.update({key: np.asarray(value, dtype=float) for key, value in pipeline_values.items()})
    return result


def _curve_label(calibration: dict, kind: str) -> str:
    if kind == "distance":
        return f"d = {calibration['distance_coef_A']:.4f}·k$^{{{calibration['distance_coef_B']:.4f}}}$"
    if kind == "pixel":
        return f"p = {calibration['pixel_calib_C']:.4f}·d$^{{{calibration['pixel_calib_D']:.4f}}}$"
    axis = kind[-1]
    k1 = calibration[f"distortion_{axis}_k1"]
    k2 = calibration[f"distortion_{axis}_k2"]
    symbol = "rₓ" if axis == "x" else "rᵧ"
    return f"f({symbol}) = 1 {k1:+.6f}{symbol}² {k2:+.6f}{symbol}⁴"


def plot(points: dict[str, np.ndarray], primary: dict, output: Path,
         comparison: dict | None = None) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6), constrained_layout=True)
    point_style = dict(s=18, alpha=0.42, color="#3977a8", edgecolors="none", rasterized=True)
    primary_style = dict(color="#b32323", linewidth=2.5)
    comparison_style = dict(color="#e08b22", linewidth=2.2, linestyle="--")

    # 1. Оценка расстояния по коэффициенту изменения масштаба.
    ax = axes[0]
    ax.scatter(points["k"], points["distance"], **point_style, label="Наблюдения")
    k_grid = np.geomspace(max(1.0, points["k"].min() * 0.9), points["k"].max() * 1.1, 400)
    for calibration, style, prefix in (
        (primary, primary_style, "Регрессия"),
        (comparison, comparison_style, "Контрольный пересчёт"),
    ):
        if calibration is not None:
            distance = calibration["distance_coef_A"] * k_grid ** calibration["distance_coef_B"]
            ax.plot(k_grid, distance, **style, label=f"{prefix}: {_curve_label(calibration, 'distance')}")
    ax.set(xlabel="Коэффициент изменения масштаба k, %/м", ylabel="Расстояние d, м",
           title="а) Регрессия расстояния")

    # 2. Масштаб изображения в зависимости от расстояния.
    ax = axes[1]
    ax.scatter(points["distance"], points["pixel_calib"], **point_style, label="Наблюдения")
    d_grid = np.geomspace(points["distance"].min() * 0.9, points["distance"].max() * 1.1, 400)
    for calibration, style, prefix in (
        (primary, primary_style, "Регрессия"),
        (comparison, comparison_style, "Контрольный пересчёт"),
    ):
        if calibration is not None:
            pixel = calibration["pixel_calib_C"] * d_grid ** calibration["pixel_calib_D"]
            ax.plot(d_grid, pixel, **style, label=f"{prefix}: {_curve_label(calibration, 'pixel')}")
    ax.set(xlabel="Расстояние d, м", ylabel="Масштаб p, пикс./мм",
           title="б) Регрессия масштаба")

    # 3–4. Независимые компоненты совместной XY-поправки.
    for axis_index, (axis_key, symbol, title) in enumerate((
        ("x", "rₓ", "в) Горизонтальная поправка"),
        ("y", "rᵧ", "г) Вертикальная поправка"),
    ), start=2):
        ax = axes[axis_index]
        r_grid = np.linspace(0.0, 1.0, 400)
        for calibration, style, prefix in (
            (primary, primary_style, "Поправка"),
            (comparison, comparison_style, "Контрольный пересчёт"),
        ):
            if calibration is not None:
                k1 = calibration[f"distortion_{axis_key}_k1"]
                k2 = calibration[f"distortion_{axis_key}_k2"]
                factor = 1 + k1 * r_grid**2 + k2 * r_grid**4
                ax.plot(
                    r_grid, factor, **style,
                    label=f"{prefix}: {_curve_label(calibration, 'distortion_' + axis_key)}",
                )
        ax.axhline(1.0, color="#777777", linewidth=0.8, zorder=0)
        ax.set(
            xlabel=f"Нормированное удаление {symbol}",
            ylabel=f"Коэффициент f({symbol})",
            title=title,
        )
        ax.set_ylim(bottom=min(0.95, ax.get_ylim()[0]))
        coverage_axis = ax.twinx()
        coverage_axis.hist(
            points[f"r{axis_key}"], bins=np.linspace(0.0, 1.0, 21),
            color="#3977a8", alpha=0.18, edgecolor="none",
        )
        coverage_axis.set_ylabel("Число наблюдений", color="#58768c")
        coverage_axis.tick_params(axis="y", colors="#58768c")
        coverage_axis.spines["top"].set_visible(False)
        coverage_axis.spines["right"].set_color("#9bb0be")

    for ax in axes:
        ax.grid(True, alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="best")

    fig.suptitle("Регрессионные зависимости калибровки камеры", fontsize=14, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=Path("test_video"))
    parser.add_argument("--calibration", type=Path, default=Path("calibration_xy.json"))
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Путь без расширения")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    args = parser.parse_args()

    points = _collect_points(args.test_dir, args.width, args.height)
    plot(points, _load_json(args.calibration), args.output,
         _load_json(args.comparison) if args.comparison else None)
    print(
        f"Построено: {len(points['distance'])} точек прямой регрессии, "
        f"{len(points['rx'])} pipeline-пар; {args.output}.png, {args.output}.svg"
    )


if __name__ == "__main__":
    main()
