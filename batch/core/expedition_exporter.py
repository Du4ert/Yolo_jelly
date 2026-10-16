"""
Экспорт данных, треков и слоёв экспедиции или выбранных задач в CSV.
"""

import csv
import os
import sys
from pathlib import Path
from typing import Any, Optional

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from constants import CLASS_NAMES

from ..database.models import OutputType


def parse_optional_float(value: str) -> float | str:
    value = value.strip() if isinstance(value, str) else value
    if value in (None, ""):
        return ""
    return float(value)


VOLUME_FIELD_MAPPING: dict[str, dict[str, Any]] = {
    "total_volume_m3": {"key": "total_volume_m3", "default": 0.0, "parser": float},
    "effective_distance_m": {"key": "effective_distance_m", "default": 0.0, "parser": float},
    "cylinder_height_m": {"key": "cylinder_height_m", "default": 0.0, "parser": float},
    "depth_min_m": {"key": "depth_min_m", "default": 0.0, "parser": float},
    "depth_max_m": {"key": "depth_max_m", "default": 0.0, "parser": float},
    "depth_traversed_m": {"key": "depth_traversed_m", "default": 0.0, "parser": float},
    "cross_section_area_m2": {"key": "cross_section_area_m2", "default": 0.0, "parser": float},
    "pleurobrachia_volume_m3": {"key": "pleurobrachia_volume_m3", "default": 0.0, "parser": float},
    "pleurobrachia_effective_distance_m": {"key": "pleurobrachia_effective_distance_m", "default": 0.0, "parser": float},
    "pleurobrachia_cylinder_height_m": {"key": "pleurobrachia_cylinder_height_m", "default": 0.0, "parser": float},
    "pleurobrachia_cross_section_area_m2": {"key": "pleurobrachia_cross_section_area_m2", "default": 0.0, "parser": float},
    "fov_horizontal_deg": {"key": "fov_horizontal_deg", "default": 0.0, "parser": float},
    "fov_vertical_deg": {"key": "fov_vertical_deg", "default": 0.0, "parser": float},
    "duration_s": {"key": "duration_s", "default": 0.0, "parser": float},
    "descent_rate_m_s": {"key": "descent_rate_m_s", "default": 0.0, "parser": float},
}

for name in CLASS_NAMES.values():
    key = name.replace(" ", "_")
    VOLUME_FIELD_MAPPING[f"count_{key}"] = {
        "key": f"count_{key}", "default": 0, "parser": lambda x: int(float(x)),
    }
    VOLUME_FIELD_MAPPING[f"median_size_{key}_cm"] = {
        "key": f"median_size_{key}_cm", "default": "", "parser": parse_optional_float,
    }
    VOLUME_FIELD_MAPPING[f"std_size_{key}_cm"] = {
        "key": f"std_size_{key}_cm", "default": "", "parser": parse_optional_float,
    }
    VOLUME_FIELD_MAPPING[f"density_{key}_per_m2"] = {
        "key": f"density_{key}_per_m2", "default": 0.0, "parser": float,
    }
    VOLUME_FIELD_MAPPING[f"density_{key}_per_m3"] = {
        "key": f"density_{key}_per_m3", "default": 0.0, "parser": float,
    }

CSV_COLUMNS = ["Id"] + list(VOLUME_FIELD_MAPPING.keys())
LAYER_CSV_COLUMNS = [
    "St", "Layer", "Dt", "Db", "Tav", "Sav", "Chlav",
    "A_a", "M_l", "B_o", "R_p", "P_p",
]


def parse_volume_csv(filepath: str) -> dict[str, str]:
    data: dict[str, str] = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                param = row.get("parameter", "").strip()
                value = row.get("value", "").strip()
                if param:
                    data[param] = value
    except (FileNotFoundError, PermissionError, OSError):
        pass
    return data


def _get_export_tasks(repo, catalog_id: Optional[int], task_ids: Optional[list[int]]):
    """Выбирает завершённые задачи экспедиции или явно заданного выделения."""
    if task_ids is not None:
        return repo.get_completed_tasks_by_ids(task_ids)
    return repo.get_tasks_by_catalog(catalog_id)


def collect_expedition_data(
    repo, catalog_id: Optional[int], task_ids: Optional[list[int]] = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tasks = _get_export_tasks(repo, catalog_id, task_ids)
    for task in tasks:
        dive = task.video_file.dive if task.video_file else None
        dive_name = dive.name if dive else "???"
        row: dict[str, Any] = {"Id": dive_name}

        vol_output = repo.get_task_output_by_type(task.id, OutputType.VOLUME_CSV)
        vol_data: dict[str, str] = {}
        if vol_output and vol_output.filepath:
            vol_data = parse_volume_csv(vol_output.filepath)

        for col_name, spec in VOLUME_FIELD_MAPPING.items():
            raw = vol_data.get(spec["key"])
            if raw is not None:
                try:
                    row[col_name] = spec["parser"](raw)
                except (ValueError, TypeError):
                    row[col_name] = spec["default"]
            else:
                row[col_name] = spec["default"]

        rows.append(row)
    return rows


def export_expedition_csv(
    repo, catalog_id: Optional[int], output_path: str,
    task_ids: Optional[list[int]] = None,
) -> tuple[int, int]:
    rows = collect_expedition_data(repo, catalog_id, task_ids)
    rows_with_data = sum(
        1 for r in rows
        if r.get("total_volume_m3", 0) != 0
        or r.get("depth_max_m", 0) != 0
    )

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), rows_with_data


def _format_track_id(track_id: Any) -> str:
    """Возвращает track_id без лишней .0 для целочисленных значений."""
    if track_id is None:
        return ""

    value = str(track_id).strip()
    if not value:
        return ""

    try:
        number = float(value)
    except ValueError:
        return value

    if number.is_integer():
        return str(int(number))
    return value


def collect_expedition_tracks(
    repo, catalog_id: Optional[int], task_ids: Optional[list[int]] = None,
) -> tuple[list[dict[str, Any]], list[str], int]:
    """Собирает строки из _track_sizes.csv экспедиции или выбранных задач."""
    expedition_names: dict[Optional[int], str] = {}

    rows: list[dict[str, Any]] = []
    columns: list[str] = ["u_track_id"]
    files_with_data = 0

    tasks = _get_export_tasks(repo, catalog_id, task_ids)
    for task in tasks:
        dive = task.video_file.dive if task.video_file else None
        dive_name = dive.name if dive else "???"

        source_catalog_id = dive.catalog_id if task_ids is not None and dive else catalog_id
        if source_catalog_id not in expedition_names:
            catalog = repo.get_catalog(source_catalog_id) if source_catalog_id is not None else None
            expedition_names[source_catalog_id] = catalog.name if catalog else "???"
        expedition_name = expedition_names[source_catalog_id]

        track_sizes_output = repo.get_task_output_by_type(task.id, OutputType.TRACK_SIZES_CSV)
        track_sizes_path = track_sizes_output.filepath if track_sizes_output else None
        if (not track_sizes_path or not os.path.exists(track_sizes_path)) and task.video_file and dive:
            base_name = Path(task.video_file.filename).stem
            candidate = Path(dive.folder_path) / "output" / f"{base_name}_track_sizes.csv"
            if candidate.exists():
                track_sizes_path = str(candidate)

        if not track_sizes_path or not os.path.exists(track_sizes_path):
            continue

        file_rows = []
        with open(track_sizes_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue

            for field in reader.fieldnames:
                if field != "u_track_id" and field not in columns:
                    columns.append(field)

            for source_row in reader:
                original_track_id = _format_track_id(source_row.get("track_id"))
                row = dict(source_row)
                row["u_track_id"] = f"{expedition_name}_{dive_name}_{original_track_id}"
                file_rows.append(row)

        if file_rows:
            files_with_data += 1
            rows.extend(file_rows)

    return rows, columns, files_with_data


def export_expedition_tracks_csv(
    repo, catalog_id: Optional[int], output_path: str,
    task_ids: Optional[list[int]] = None,
) -> tuple[int, int, int]:
    rows, columns, files_with_data = collect_expedition_tracks(repo, catalog_id, task_ids)

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return len(_get_export_tasks(repo, catalog_id, task_ids)), files_with_data, len(rows)


def collect_expedition_layers(
    repo,
    catalog_id: Optional[int],
    task_ids: Optional[list[int]] = None,
) -> tuple[list[dict[str, Any]], list[str], int, int]:
    """Собирает строки layers.csv, включая каждый файл погружения один раз."""
    rows: list[dict[str, Any]] = []
    columns = list(LAYER_CSV_COLUMNS)
    files_with_data = 0
    dive_ids: set[int] = set()

    tasks = _get_export_tasks(repo, catalog_id, task_ids)
    total_dive_ids = {
        task.video_file.dive.id
        for task in tasks
        if task.video_file and task.video_file.dive
    }

    for task in tasks:
        dive = task.video_file.dive if task.video_file else None
        if not dive or dive.id in dive_ids:
            continue

        layer_output = repo.get_task_output_by_type(
            task.id, OutputType.LAYER_DENSITY_CSV
        )
        layers_path = layer_output.filepath if layer_output else None
        if not layers_path or not os.path.exists(layers_path):
            candidate = Path(dive.folder_path) / "output" / "analysis" / "layers.csv"
            if candidate.exists():
                layers_path = str(candidate)

        if not layers_path or not os.path.exists(layers_path):
            continue

        file_rows = []
        with open(layers_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=";")
            if not reader.fieldnames:
                continue

            for field in reader.fieldnames:
                if field not in columns:
                    columns.append(field)
            file_rows.extend(dict(source_row) for source_row in reader)

        dive_ids.add(dive.id)
        if file_rows:
            files_with_data += 1
            rows.extend(file_rows)

    return rows, columns, files_with_data, len(total_dive_ids)


def export_expedition_layers_csv(
    repo,
    catalog_id: Optional[int],
    output_path: str,
    task_ids: Optional[list[int]] = None,
) -> tuple[int, int, int]:
    """Объединяет layers.csv всех погружений экспедиции в один CSV."""
    rows, columns, files_with_data, total_dives = collect_expedition_layers(
        repo, catalog_id, task_ids
    )

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=columns, delimiter=";", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)

    return total_dives, files_with_data, len(rows)
