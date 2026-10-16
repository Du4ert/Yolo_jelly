"""Сбор готовых измерений организмов с параметрами CTD на их глубине."""

from pathlib import Path

import numpy as np
import pandas as pd

from interactive_plot import find_depth_column, load_ctd_data, to_numeric_series


BASE_COLUMNS = [
    "id", "station", "class_name", "depth_m", "layer", "real_size_mm", "real_size_cm",
]
DEFAULT_CTD_COLUMNS = [6, 11, 12, 16]


def parse_ctd_columns(value):
    """Разбирает индексы колонок так же, как в настройках графика."""
    if value is None:
        return DEFAULT_CTD_COLUMNS.copy()
    items = value.split(",") if isinstance(value, str) else value
    try:
        columns = list(dict.fromkeys(int(str(item).strip()) for item in items))
    except (TypeError, ValueError) as error:
        raise ValueError("Укажите неотрицательные индексы колонок CTD через запятую.") from error
    if not columns or any(index < 0 for index in columns):
        raise ValueError("Укажите неотрицательные индексы колонок CTD через запятую.")
    return columns


def _ctd_column_names(columns):
    """Защищает основные поля от совпадений с заголовками CTD."""
    used = set(BASE_COLUMNS)
    result = {}
    for column in columns:
        name = str(column)
        while name in used:
            name = "ctd_" + name
        used.add(name)
        result[column] = name
    return result


def _sample_parameter(ctd, depth_column, column, target_depths):
    """Интерполирует числа, для текстовых параметров берёт ближайшую глубину."""
    depths = to_numeric_series(ctd[depth_column])
    depths = depths.where(np.isfinite(depths))
    numbers = to_numeric_series(ctd[column])
    numbers = numbers.where(np.isfinite(numbers))
    if numbers.notna().any():
        profile = pd.DataFrame({"depth": depths, "value": numbers}).dropna()
        profile = profile.groupby("depth", as_index=False)["value"].mean().sort_values("depth")
        if profile.empty:
            return np.full(len(target_depths), np.nan)
        return np.interp(
            target_depths.to_numpy(dtype=float), profile["depth"], profile["value"],
            left=np.nan, right=np.nan,
        )

    profile = pd.DataFrame({"depth": depths, "value": ctd[column]}).dropna()
    profile = profile.sort_values("depth").drop_duplicates("depth")
    result = np.full(len(target_depths), None, dtype=object)
    if profile.empty:
        return result
    source_depths = profile["depth"].to_numpy(dtype=float)
    values = profile["value"].to_numpy()
    for index, depth in enumerate(target_depths):
        if pd.notna(depth) and source_depths[0] <= depth <= source_depths[-1]:
            result[index] = values[np.abs(source_depths - depth).argmin()]
    return result


def _assign_layers(depths, layers_path):
    """Использует готовые границы и правила включения границ из расчёта слоёв."""
    result = pd.Series("", index=depths.index, dtype=object)
    layers = load_ctd_data(layers_path)
    if not {"Layer", "Dt", "Db"}.issubset(layers.columns):
        raise ValueError("В layers.csv отсутствуют Layer, Dt или Db.")
    layers["Dt"] = to_numeric_series(layers["Dt"])
    layers["Db"] = to_numeric_series(layers["Db"])
    layers = layers[layers["Layer"].isin(["UML", "STL", "OXY", "SOL"])]
    layers = layers.dropna(subset=["Dt", "Db"]).sort_values("Dt")
    for index, (_, layer) in enumerate(layers.iterrows()):
        last = index == len(layers) - 1
        mask = (depths >= layer["Dt"]) & (
            (depths <= layer["Db"]) if last else (depths < layer["Db"])
        )
        result.loc[mask] = layer["Layer"]
    return result


def collect_tracks_ctd(source, ctd_columns=None, all_parameters=False):
    """Собирает одно видео без пересчёта геометрии, размеров или границ слоёв."""
    warnings = []
    tracks = pd.read_csv(source["track_sizes_path"], encoding="utf-8-sig")
    required = {"track_id", "class_name", "object_depth_m"}
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError("В CSV размеров отсутствуют поля: " + ", ".join(sorted(missing)))
    if not {"real_size_mm", "real_size_cm"}.intersection(tracks.columns):
        raise ValueError("В CSV размеров отсутствует посчитанный размер организма.")
    tracks = tracks.drop_duplicates(["class_name", "track_id"]).reset_index(drop=True)
    tracks = tracks[tracks["track_id"].notna()].reset_index(drop=True)
    ids = tracks["track_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    prefix = source.get("id_prefix", "")
    result = pd.DataFrame({
        "id": prefix + ids,
        "station": source["station"],
        "class_name": tracks["class_name"],
        "depth_m": to_numeric_series(tracks["object_depth_m"]),
        "layer": "",
    })
    for name, other, factor in (
        ("real_size_mm", "real_size_cm", 10), ("real_size_cm", "real_size_mm", 0.1),
    ):
        values = (
            to_numeric_series(tracks[name]) if name in tracks
            else pd.Series(np.nan, index=tracks.index)
        )
        if other in tracks:
            values = values.fillna(to_numeric_series(tracks[other]) * factor)
        result[name] = values.where(np.isfinite(values))
    result["depth_m"] = result["depth_m"].where(np.isfinite(result["depth_m"]))
    missing_depths = int(result["depth_m"].isna().sum())
    if missing_depths:
        warnings.append(f"Организмов без рассчитанной глубины: {missing_depths}.")
    missing_sizes = int(result["real_size_mm"].isna().sum())
    if missing_sizes:
        warnings.append(f"Организмов без рассчитанного размера: {missing_sizes}.")

    layers_path = source.get("layers_path")
    if layers_path and Path(layers_path).is_file():
        try:
            result["layer"] = _assign_layers(result["depth_m"], layers_path)
        except Exception as error:
            warnings.append(f"Не удалось прочитать границы слоёв: {error}")
    else:
        warnings.append("Нет layers.csv: поле layer оставлено пустым.")

    ctd_path = source.get("ctd_path")
    if not ctd_path or not Path(ctd_path).is_file():
        warnings.append("Нет доступного CTD-файла: параметры CTD оставлены пустыми.")
        return result, warnings
    try:
        ctd = load_ctd_data(ctd_path)
    except Exception as error:
        warnings.append(f"Не удалось прочитать CTD: {error}; измерения организмов сохранены.")
        return result, warnings
    depth_column = find_depth_column(ctd)
    indices = list(range(len(ctd.columns))) if all_parameters else parse_ctd_columns(ctd_columns)
    names = _ctd_column_names(ctd.columns)
    for index in indices:
        if index >= len(ctd.columns):
            warnings.append(f"В CTD нет колонки с индексом {index}.")
            continue
        column = ctd.columns[index]
        result[names[column]] = (
            _sample_parameter(ctd, depth_column, column, result["depth_m"])
            if depth_column is not None else np.nan
        )
    if depth_column is None:
        warnings.append("В CTD не найдена глубина: параметры CTD оставлены пустыми.")
        return result, warnings
    profile_depths = to_numeric_series(ctd[depth_column])
    profile_depths = profile_depths[np.isfinite(profile_depths)]
    if profile_depths.empty:
        warnings.append("В CTD нет числовых глубин.")
    else:
        outside = (
            (result["depth_m"] < profile_depths.min())
            | (result["depth_m"] > profile_depths.max())
        ).sum()
        if outside:
            warnings.append(f"Организмов за диапазоном глубин CTD: {outside}; CTD без экстраполяции.")
    return result, warnings


def export_tracks_ctd_csv(sources, output_path, ctd_columns=None, all_parameters=False):
    """Объединяет источники и возвращает количество организмов и замечания."""
    frames, warnings = [], []
    seen = set()
    for source in sources:
        key = str(Path(source["track_sizes_path"]).resolve())
        if key in seen:
            continue
        try:
            frame, source_warnings = collect_tracks_ctd(source, ctd_columns, all_parameters)
        except Exception as error:
            warnings.append(f"{source['station']}: {error}")
            continue
        seen.add(key)
        frames.append(frame)
        warnings.extend(f"{source['station']}: {message}" for message in source_warnings)
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=BASE_COLUMNS)
    table.to_csv(output_path, index=False, encoding="utf-8-sig")
    return len(table), warnings
