"""Расчёт параметров CTD и плотности организмов по вертикальным слоям."""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from interactive_plot import (
    calculate_density_layer_depths,
    calculate_thermocline_depth,
    find_depth_column,
    load_ctd_data,
    to_numeric_series,
)


DEFAULT_CTD_PARAMETERS = {"Tav": 6, "Sav": 7, "Chlav": 12}
LAYER_NAMES = ("UML", "STL", "OXY", "SOL")
SPECIES_COLUMNS = {
    "Aurelia aurita": "A_a",
    "Mnemiopsis leidyi": "M_l",
    "Beroe ovata": "B_o",
    "Rhizostoma pulmo": "R_p",
    "Pleurobrachia pileus": "P_p",
}


def parse_ctd_parameters(value) -> Dict[str, int]:
    """Разбирает расширяемое описание колонок вида ``Tav:6,Sav:7``."""
    if isinstance(value, dict):
        result = {}
        for name, index in value.items():
            try:
                result[str(name).strip()] = int(index)
            except (TypeError, ValueError):
                continue
        return result or DEFAULT_CTD_PARAMETERS.copy()

    result = {}
    for item in str(value or "").split(","):
        if ":" not in item:
            continue
        name, index = item.split(":", 1)
        try:
            result[name.strip()] = int(index.strip())
        except ValueError:
            continue
    return result or DEFAULT_CTD_PARAMETERS.copy()


def _depth_weighted_average(
    ctd_df: pd.DataFrame,
    depth_col: str,
    value_col: str,
    top: float,
    bottom: float,
) -> Optional[float]:
    """Усредняет параметр по толщине слоя без экстраполяции профиля."""
    profile = ctd_df[[depth_col, value_col]].copy()
    profile[depth_col] = to_numeric_series(profile[depth_col])
    profile[value_col] = to_numeric_series(profile[value_col])
    profile = profile.dropna().groupby(depth_col, as_index=False)[value_col].mean()
    profile = profile.sort_values(depth_col)
    if len(profile) < 2:
        return None

    depths = profile[depth_col].to_numpy(dtype=float)
    values = profile[value_col].to_numpy(dtype=float)
    effective_top = max(top, float(depths[0]))
    effective_bottom = min(bottom, float(depths[-1]))
    if effective_bottom <= effective_top:
        return None

    inner = (depths > effective_top) & (depths < effective_bottom)
    layer_depths = np.concatenate(
        ([effective_top], depths[inner], [effective_bottom])
    )
    layer_values = np.interp(layer_depths, depths, values)
    integral = np.sum(np.diff(layer_depths) * (layer_values[:-1] + layer_values[1:]) / 2)
    return float(integral / (effective_bottom - effective_top))


def _survey_bottom(ctd_df: pd.DataFrame, depth_col: str):
    ctd_depths = to_numeric_series(ctd_df[depth_col]).dropna()
    if ctd_depths.empty:
        return None
    return float(ctd_depths.max())


def _build_layer_bounds(
    end_depth: Optional[float],
    thermocline_depth: Optional[float],
    density_depths: Dict[float, float],
):
    bounds = {name: None for name in LAYER_NAMES}
    if end_depth is None or end_depth <= 0:
        return bounds

    sigma_145 = density_depths.get(14.5)
    sigma_155 = density_depths.get(15.5)
    sigma_162 = density_depths.get(16.2)

    if thermocline_depth is not None and thermocline_depth > 0:
        bounds["UML"] = (0.0, min(thermocline_depth, end_depth))

    stl_top = thermocline_depth if thermocline_depth is not None else 0.0
    valid_145 = sigma_145 is not None and sigma_145 > stl_top
    if stl_top < end_depth:
        stl_bottom = min(sigma_145, end_depth) if valid_145 else end_depth
        if (sigma_145 is None or valid_145) and stl_bottom > stl_top:
            bounds["STL"] = (float(stl_top), float(stl_bottom))

    valid_155 = valid_145 and sigma_155 is not None and sigma_155 > sigma_145
    if valid_145 and sigma_145 < end_depth:
        oxy_bottom = min(sigma_155, end_depth) if valid_155 else end_depth
        if (sigma_155 is None or valid_155) and oxy_bottom > sigma_145:
            bounds["OXY"] = (float(sigma_145), float(oxy_bottom))

    valid_162 = valid_155 and sigma_162 is not None and sigma_162 > sigma_155
    if valid_155 and sigma_155 < end_depth:
        sol_bottom = min(sigma_162, end_depth) if valid_162 else end_depth
        if (sigma_162 is None or valid_162) and sol_bottom > sigma_155:
            bounds["SOL"] = (float(sigma_155), float(sol_bottom))
    return bounds


def export_layer_density_csv(
    output_path: str,
    station_name: str,
    track_sizes_path: Optional[str],
    ctd_path: str,
    ctd_parameters=None,
    sigma_t_column: int = 11,
    cross_section_area_m2: Optional[float] = None,
    pleurobrachia_cross_section_area_m2: Optional[float] = None,
    thermocline_threshold: float = 0.2,
    thermocline_mode: str = "threshold",
) -> str:
    """Создаёт CSV с четырьмя стандартными слоями UML/STL/OXY/SOL."""
    parameter_columns = parse_ctd_parameters(ctd_parameters)
    output_columns = [
        "St", "Layer", "Dt", "Db", *parameter_columns.keys(),
        *SPECIES_COLUMNS.values(),
    ]
    rows = [{column: None for column in output_columns} for _ in LAYER_NAMES]
    for row, layer_name in zip(rows, LAYER_NAMES):
        row["St"] = station_name
        row["Layer"] = layer_name

    ctd_df = load_ctd_data(ctd_path)
    depth_col = find_depth_column(ctd_df)
    all_columns = list(ctd_df.columns)
    if depth_col is None:
        raise ValueError("В CTD-файле не найдена колонка глубины")
    if not 0 <= sigma_t_column < len(all_columns):
        raise ValueError(f"В CTD-файле нет колонки SigmaT с индексом {sigma_t_column}")

    valid_parameters = {
        name: all_columns[index]
        for name, index in parameter_columns.items()
        if 0 <= index < len(all_columns)
    }
    temperature_index = parameter_columns.get("Tav", 6)
    temperature_col = (
        all_columns[temperature_index]
        if 0 <= temperature_index < len(all_columns)
        else None
    )
    thermocline = None
    if temperature_col:
        thermocline = calculate_thermocline_depth(
            ctd_df,
            depth_col,
            temperature_col,
            thermocline_threshold,
            thermocline_mode,
        )
    thermocline_depth = thermocline[0] if thermocline else None
    density_depths = dict(calculate_density_layer_depths(
        ctd_df,
        depth_col,
        all_columns[sigma_t_column],
        [14.5, 15.5, 16.2],
    ))
    end_depth = _survey_bottom(ctd_df, depth_col)
    layer_bounds = _build_layer_bounds(end_depth, thermocline_depth, density_depths)

    tracks_df = (
        pd.read_csv(track_sizes_path)
        if track_sizes_path and Path(track_sizes_path).exists()
        else pd.DataFrame()
    )
    track_data_available = {
        "track_id", "object_depth_m", "class_name"
    }.issubset(tracks_df.columns)
    if "object_depth_m" in tracks_df.columns:
        tracks_df["object_depth_m"] = to_numeric_series(tracks_df["object_depth_m"])
        tracks_df = tracks_df.dropna(subset=["object_depth_m"])
    else:
        tracks_df = tracks_df.iloc[0:0]
    if "track_id" in tracks_df.columns and "class_name" in tracks_df.columns:
        tracks_df = tracks_df.drop_duplicates(subset=["class_name", "track_id"])

    existing_layers = [name for name in LAYER_NAMES if layer_bounds[name] is not None]
    for row, layer_name in zip(rows, LAYER_NAMES):
        bounds = layer_bounds[layer_name]
        if bounds is None:
            continue
        top, bottom = bounds
        row["Dt"] = top
        row["Db"] = bottom
        for output_name, source_col in valid_parameters.items():
            row[output_name] = _depth_weighted_average(
                ctd_df, depth_col, source_col, top, bottom
            )

        is_last = layer_name == existing_layers[-1]
        depths = tracks_df.get("object_depth_m", pd.Series(dtype=float))
        in_layer = (depths >= top) & ((depths <= bottom) if is_last else (depths < bottom))
        layer_tracks = tracks_df[in_layer]
        for species_name, output_name in SPECIES_COLUMNS.items():
            area = (
                pleurobrachia_cross_section_area_m2
                if species_name == "Pleurobrachia pileus"
                else cross_section_area_m2
            )
            if area is not None and area > 0 and track_data_available:
                count = int((layer_tracks["class_name"] == species_name).sum())
                row[output_name] = count / area

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=output_columns).to_csv(
        output, sep=";", index=False, encoding="utf-8-sig", float_format="%.6g"
    )
    return str(output)
