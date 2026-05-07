"""Общие defaults калибровки для batch UI и воркера."""

import sys
import json
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def get_calibration_defaults(calibration_json: Optional[str] = None) -> dict:
    """Возвращает min/max reliable из calibration JSON или дефолтов камеры."""
    from camera_geometry import CameraCalibration

    calibration_keys = set()
    if calibration_json and Path(calibration_json).exists():
        calibration = CameraCalibration.from_json(calibration_json)
        try:
            with open(calibration_json, "r", encoding="utf-8") as f:
                calibration_keys = set(json.load(f).keys())
        except Exception:
            calibration_keys = set()
    else:
        calibration = CameraCalibration()

    try:
        from .config import get_config
        ui = get_config().ui
        if ("min_reliable_distance" not in calibration_keys
                and ui.min_reliable_distance is not None):
            calibration.min_reliable_distance = ui.min_reliable_distance
        if ("max_reliable_distance" not in calibration_keys
                and ui.max_reliable_distance is not None):
            calibration.max_reliable_distance = ui.max_reliable_distance
    except Exception:
        pass

    effective_distance_auto = True
    effective_distance = None
    try:
        from .config import get_config
        ui = get_config().ui
        if ui.effective_distance_auto is not None:
            effective_distance_auto = ui.effective_distance_auto
        effective_distance = ui.effective_distance
    except Exception:
        pass

    return {
        "calibration_json": calibration_json if calibration_json and Path(calibration_json).exists() else None,
        "min_reliable_distance": calibration.min_reliable_distance,
        "max_reliable_distance": calibration.max_reliable_distance,
        "effective_distance_auto": effective_distance_auto,
        "effective_distance": effective_distance,
    }
