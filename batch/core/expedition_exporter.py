"""
Экспорт данных экспедиции — собирает данные из volume.csv всех задач в сводную таблицу CSV.
"""

import csv
import sys
from pathlib import Path
from typing import Any, Optional

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from constants import CLASS_NAMES

from ..database.models import OutputType


VOLUME_FIELD_MAPPING: dict[str, dict[str, Any]] = {
    "depth_max_m": {"key": "depth_max_m", "default": 0.0, "parser": float},
    "total_volume_m3": {"key": "total_volume_m3", "default": 0.0, "parser": float},
}

for name in CLASS_NAMES.values():
    key = name.replace(" ", "_")
    VOLUME_FIELD_MAPPING[f"count_{key}"] = {
        "key": f"count_{key}", "default": 0, "parser": lambda x: int(float(x)),
    }
    VOLUME_FIELD_MAPPING[f"density_{key}_per_m3"] = {
        "key": f"density_{key}_per_m3", "default": 0.0, "parser": float,
    }

CSV_COLUMNS = ["Id"] + list(VOLUME_FIELD_MAPPING.keys())


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


def collect_expedition_data(repo, catalog_id: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tasks = repo.get_tasks_by_catalog(catalog_id)
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


def export_expedition_csv(repo, catalog_id: int, output_path: str) -> tuple[int, int]:
    rows = collect_expedition_data(repo, catalog_id)
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
