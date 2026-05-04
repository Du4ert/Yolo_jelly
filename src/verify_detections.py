"""
Модуль ручной проверки результатов детекции.

Позволяет загружать треки, детекции, сохранять состояние проверки,
применять изменения (удаление трека, смена класса) к CSV-файлам.
"""

import os
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

from constants import CLASS_NAMES


@dataclass
class TrackInfo:
    """Информация о треке из _tracks.csv + состояние верификации."""

    track_id: int
    class_id: int
    class_name: str
    first_frame: int
    last_frame: int
    frame_span: int
    duration_s: float
    first_timestamp_s: float
    last_timestamp_s: float
    first_depth_m: Optional[float]
    last_depth_m: Optional[float]
    detections_count: int
    avg_confidence: float
    depth_change_m: Optional[float] = None

    deleted: bool = False
    new_class_id: Optional[int] = None

    @property
    def effective_class_id(self) -> int:
        if self.new_class_id is not None:
            return self.new_class_id
        return self.class_id

    @property
    def effective_class_name(self) -> str:
        cid = self.effective_class_id
        return CLASS_NAMES.get(cid, f"unknown_{cid}")

    @property
    def is_modified(self) -> bool:
        return self.deleted or self.new_class_id is not None

    @property
    def status_text(self) -> str:
        if self.deleted:
            return "Удалён"
        if self.new_class_id is not None:
            return f"→ {CLASS_NAMES.get(self.new_class_id, str(self.new_class_id))}"
        return "OK"


@dataclass
class VerificationResult:
    """Результат применения изменений."""

    total_tracks: int
    deleted_count: int
    changed_class_count: int
    verified_csv_path: str


def load_tracks(tracks_csv_path: str) -> List[TrackInfo]:
    """Загружает треки из _tracks.csv."""
    if not os.path.exists(tracks_csv_path):
        return []

    df = pd.read_csv(tracks_csv_path)
    tracks: List[TrackInfo] = []
    for _, row in df.iterrows():
        t = TrackInfo(
            track_id=int(row["track_id"]),
            class_id=int(row["class_id"]),
            class_name=str(row["class_name"]),
            first_frame=int(row["first_frame"]),
            last_frame=int(row["last_frame"]),
            frame_span=int(row["frame_span"]),
            duration_s=float(row["duration_s"]),
            first_timestamp_s=float(row["first_timestamp_s"]),
            last_timestamp_s=float(row["last_timestamp_s"]),
            first_depth_m=_safe_float(row, "first_depth_m"),
            last_depth_m=_safe_float(row, "last_depth_m"),
            detections_count=int(row["detections_count"]),
            avg_confidence=float(row["avg_confidence"]),
            depth_change_m=_safe_float(row, "depth_change_m"),
        )
        tracks.append(t)
    return tracks


def load_verified(verified_csv_path: str) -> Dict[int, dict]:
    """Загружает сохранённое состояние проверки из _verified.csv.

    Returns:
        Словарь {track_id: {'deleted': bool, 'new_class_id': int | None}}
    """
    if not os.path.exists(verified_csv_path):
        return {}

    df = pd.read_csv(verified_csv_path)
    state: Dict[int, dict] = {}
    for _, row in df.iterrows():
        tid = int(row["track_id"])
        state[tid] = {
            "deleted": bool(row["deleted"]),
            "new_class_id": _safe_int(row, "new_class_id"),
        }
    return state


def load_track_detections(detections_csv_path: str, track_id: int) -> pd.DataFrame:
    """Загружает детекции одного трека из _detections.csv."""
    if not os.path.exists(detections_csv_path):
        return pd.DataFrame()

    chunks = []
    for chunk in pd.read_csv(detections_csv_path, chunksize=50000):
        mask = chunk["track_id"] == track_id
        if mask.any():
            chunks.append(chunk[mask])
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def verified_csv_path(tracks_csv_path: str) -> str:
    """Вычисляет путь к _verified.csv на основе пути _tracks.csv."""
    p = Path(tracks_csv_path)
    stem = p.stem
    if stem.endswith("_tracks"):
        stem = stem[:-7]
    return str(p.parent / f"{stem}_verified.csv")


def apply_changes(
    tracks: List[TrackInfo],
    tracks_csv_path: str,
    detections_csv_path: str,
) -> VerificationResult:
    """Применяет изменения: обновляет CSV и записывает _verified.csv."""
    deleted_ids = set(t.track_id for t in tracks if t.deleted)
    changed_map = {
        t.track_id: t.new_class_id
        for t in tracks
        if not t.deleted and t.new_class_id is not None
    }

    deleted_count = len(deleted_ids)
    changed_count = len(changed_map)

    # 1. Обновляем _detections.csv
    if os.path.exists(detections_csv_path):
        det_df = pd.read_csv(detections_csv_path)

        if deleted_ids:
            det_df = det_df[~det_df["track_id"].isin(deleted_ids)]

        for tid, new_cid in changed_map.items():
            mask = det_df["track_id"] == tid
            det_df.loc[mask, "class_id"] = new_cid
            det_df.loc[mask, "class_name"] = CLASS_NAMES.get(
                new_cid, f"unknown_{new_cid}"
            )

        det_df.to_csv(detections_csv_path, index=False)

    # 2. Обновляем _tracks.csv
    if os.path.exists(tracks_csv_path):
        tracks_df = pd.read_csv(tracks_csv_path)

        if deleted_ids:
            tracks_df = tracks_df[~tracks_df["track_id"].isin(deleted_ids)]

        for tid, new_cid in changed_map.items():
            mask = tracks_df["track_id"] == tid
            tracks_df.loc[mask, "class_id"] = new_cid
            tracks_df.loc[mask, "class_name"] = CLASS_NAMES.get(
                new_cid, f"unknown_{new_cid}"
            )

        tracks_df.to_csv(tracks_csv_path, index=False)

    # 3. Записываем _verified.csv
    vp = verified_csv_path(tracks_csv_path)
    now = datetime.now().isoformat()
    rows = []
    for t in tracks:
        rows.append(
            {
                "track_id": t.track_id,
                "original_class_id": t.class_id,
                "original_class_name": t.class_name,
                "deleted": t.deleted,
                "new_class_id": t.new_class_id,
                "verified_at": now,
            }
        )
    pd.DataFrame(rows).to_csv(vp, index=False)

    return VerificationResult(
        total_tracks=len(tracks),
        deleted_count=deleted_count,
        changed_class_count=changed_count,
        verified_csv_path=vp,
    )


def _safe_float(row: pd.Series, col: str) -> Optional[float]:
    if col in row and pd.notna(row[col]):
        return float(row[col])
    return None


def _safe_int(row: pd.Series, col: str) -> Optional[int]:
    if col in row and pd.notna(row[col]):
        return int(row[col])
    return None
