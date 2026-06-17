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
    confirmed: bool = False
    merged_into_track_id: Optional[int] = None

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
        return (
            self.deleted
            or self.new_class_id is not None
            or self.confirmed
            or self.merged_into_track_id is not None
        )

    @property
    def status_text(self) -> str:
        if self.deleted:
            return "Удалён"
        if self.merged_into_track_id is not None:
            return f"Объединён → {self.merged_into_track_id}"
        if self.confirmed:
            return "Подтверждён"
        if self.new_class_id is not None:
            return f"→ {CLASS_NAMES.get(self.new_class_id, str(self.new_class_id))}"
        return "OK"


@dataclass
class VerificationResult:
    """Результат применения изменений."""

    total_tracks: int
    deleted_count: int
    changed_class_count: int
    merged_count: int
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
        Словарь {track_id: {'deleted': bool, 'new_class_id': int | None,
        'confirmed': bool, 'merged_into_track_id': int | None}}
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
            "confirmed": bool(row["confirmed"]) if "confirmed" in row else False,
            "merged_into_track_id": _safe_int(row, "merged_into_track_id"),
        }
    return state


def save_verified(
    tracks: List[TrackInfo],
    tracks_csv_path: str,
) -> str:
    """Сохраняет текущее состояние верификации в _verified.csv.

    Не трогает _detections.csv и _tracks.csv.
    Returns:
        Путь к записанному файлу.
    """
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
                "confirmed": t.confirmed,
                "merged_into_track_id": t.merged_into_track_id,
                "verified_at": now,
            }
        )
    pd.DataFrame(rows).to_csv(vp, index=False)
    return vp


def get_task_verification_status(tracks_csv_path: str) -> str:
    """Возвращает статус проверки задачи: 'Подтверждено', 'В работе', или ''.

    'Подтверждено' — только если все треки разрешены (confirmed|deleted)
    и изменения применены к _tracks.csv (удалённых треков в нём уже нет).
    """
    vp = verified_csv_path(tracks_csv_path)

    if not os.path.exists(vp) or not os.path.exists(tracks_csv_path):
        return ""

    verified_df = pd.read_csv(vp)
    if verified_df.empty:
        return ""

    tracks_df = pd.read_csv(tracks_csv_path)
    track_ids_csv = set(tracks_df["track_id"].astype(int))

    all_resolved = True
    has_pending = False

    for _, row in verified_df.iterrows():
        tid = int(row["track_id"])
        is_deleted = bool(row["deleted"])
        is_confirmed = bool(row.get("confirmed", False))
        is_merged = pd.notna(row.get("merged_into_track_id"))

        if not is_deleted and not is_confirmed and not is_merged:
            all_resolved = False
            has_pending = True

        if (is_deleted or is_merged) and tid in track_ids_csv:
            all_resolved = False

    if not has_pending and all_resolved:
        return "Подтверждено"

    resolved_any = any(
        bool(row["deleted"])
        or bool(row.get("confirmed", False))
        or pd.notna(row.get("merged_into_track_id"))
        for _, row in verified_df.iterrows()
    )
    if resolved_any:
        return "В работе"

    return ""


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


def load_frame_detections(
    detections_csv_path: str,
    timestamp_s: float,
    tolerance_s: float = 0.15,
) -> pd.DataFrame:
    """Загружает все детекции на заданный момент времени из _detections.csv."""
    if not os.path.exists(detections_csv_path):
        return pd.DataFrame()

    t_min = timestamp_s - tolerance_s
    t_max = timestamp_s + tolerance_s

    chunks = []
    for chunk in pd.read_csv(detections_csv_path, chunksize=50000):
        if chunk["timestamp_s"].min() > t_max:
            break
        mask = (chunk["timestamp_s"] >= t_min) & (chunk["timestamp_s"] <= t_max)
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
    merge_map = {
        t.track_id: t.merged_into_track_id
        for t in tracks
        if not t.deleted and t.merged_into_track_id is not None
    }
    target_class_map = {
        t.track_id: t.effective_class_id
        for t in tracks
        if t.track_id in set(merge_map.values())
    }
    changed_map = {
        t.track_id: t.new_class_id
        for t in tracks
        if not t.deleted and t.merged_into_track_id is None and t.new_class_id is not None
    }

    deleted_count = len(deleted_ids)
    changed_count = len(changed_map)
    merged_count = len(merge_map)

    # 1. Обновляем _detections.csv
    if os.path.exists(detections_csv_path):
        det_df = pd.read_csv(detections_csv_path)

        if deleted_ids:
            det_df = det_df[~det_df["track_id"].isin(deleted_ids)]

        remaining_ids = set(t.track_id for t in tracks if not t.deleted)
        det_df = det_df[det_df["track_id"].isin(remaining_ids)]

        for source_id, target_id in merge_map.items():
            det_df.loc[det_df["track_id"] == source_id, "track_id"] = target_id

        for tid, new_cid in changed_map.items():
            mask = det_df["track_id"] == tid
            det_df.loc[mask, "class_id"] = new_cid
            det_df.loc[mask, "class_name"] = CLASS_NAMES.get(
                new_cid, f"unknown_{new_cid}"
            )

        for tid, cid in target_class_map.items():
            mask = det_df["track_id"] == tid
            det_df.loc[mask, "class_id"] = cid
            det_df.loc[mask, "class_name"] = CLASS_NAMES.get(cid, f"unknown_{cid}")

        det_df.to_csv(detections_csv_path, index=False)

    # 2. Обновляем _tracks.csv
    if os.path.exists(tracks_csv_path):
        tracks_df = pd.read_csv(tracks_csv_path)

        if deleted_ids:
            tracks_df = tracks_df[~tracks_df["track_id"].isin(deleted_ids)]

        if merge_map:
            tracks_df = tracks_df[~tracks_df["track_id"].isin(merge_map.keys())]

        for tid, new_cid in changed_map.items():
            mask = tracks_df["track_id"] == tid
            tracks_df.loc[mask, "class_id"] = new_cid
            tracks_df.loc[mask, "class_name"] = CLASS_NAMES.get(
                new_cid, f"unknown_{new_cid}"
            )

        for tid, cid in target_class_map.items():
            mask = tracks_df["track_id"] == tid
            tracks_df.loc[mask, "class_id"] = cid
            tracks_df.loc[mask, "class_name"] = CLASS_NAMES.get(cid, f"unknown_{cid}")

        if merge_map:
            track_by_id = {t.track_id: t for t in tracks}
            for target_id in sorted(set(merge_map.values())):
                if 'det_df' in locals():
                    _update_track_row_from_detections(tracks_df, det_df, target_id)
                else:
                    _update_track_row_from_track(tracks_df, track_by_id.get(target_id))

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
                "confirmed": t.confirmed,
                "merged_into_track_id": t.merged_into_track_id,
                "verified_at": now,
            }
        )
    pd.DataFrame(rows).to_csv(vp, index=False)

    return VerificationResult(
        total_tracks=len(tracks),
        deleted_count=deleted_count,
        changed_class_count=changed_count,
        merged_count=merged_count,
        verified_csv_path=vp,
    )


def _update_track_row_from_detections(
    tracks_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    track_id: int,
) -> None:
    mask = tracks_df["track_id"] == track_id
    det = detections_df[detections_df["track_id"] == track_id]
    if not mask.any() or det.empty:
        return

    det = det.sort_values("frame") if "frame" in det.columns else det
    first = det.iloc[0]
    last = det.iloc[-1]

    if "frame" in det.columns:
        tracks_df.loc[mask, "first_frame"] = int(det["frame"].min())
        tracks_df.loc[mask, "last_frame"] = int(det["frame"].max())
        tracks_df.loc[mask, "frame_span"] = int(det["frame"].max() - det["frame"].min() + 1)
    if "timestamp_s" in det.columns:
        first_ts = float(det["timestamp_s"].min())
        last_ts = float(det["timestamp_s"].max())
        tracks_df.loc[mask, "first_timestamp_s"] = round(first_ts, 2)
        tracks_df.loc[mask, "last_timestamp_s"] = round(last_ts, 2)
        tracks_df.loc[mask, "duration_s"] = round(last_ts - first_ts, 2)
    if "depth_m" in det.columns:
        first_depth = _safe_float(first, "depth_m")
        last_depth = _safe_float(last, "depth_m")
        tracks_df.loc[mask, "first_depth_m"] = first_depth
        tracks_df.loc[mask, "last_depth_m"] = last_depth
        if first_depth is not None and last_depth is not None:
            tracks_df.loc[mask, "depth_change_m"] = round(last_depth - first_depth, 2)
    tracks_df.loc[mask, "detections_count"] = len(det)
    if "confidence" in det.columns:
        tracks_df.loc[mask, "avg_confidence"] = round(float(det["confidence"].mean()), 3)


def _update_track_row_from_track(
    tracks_df: pd.DataFrame,
    track: Optional[TrackInfo],
) -> None:
    if track is None:
        return
    mask = tracks_df["track_id"] == track.track_id
    if not mask.any():
        return

    values = {
        "first_frame": track.first_frame,
        "last_frame": track.last_frame,
        "frame_span": track.frame_span,
        "duration_s": track.duration_s,
        "first_timestamp_s": track.first_timestamp_s,
        "last_timestamp_s": track.last_timestamp_s,
        "first_depth_m": track.first_depth_m,
        "last_depth_m": track.last_depth_m,
        "depth_change_m": track.depth_change_m,
        "detections_count": track.detections_count,
        "avg_confidence": track.avg_confidence,
    }
    for column, value in values.items():
        if column in tracks_df.columns:
            tracks_df.loc[mask, column] = value


def _safe_float(row: pd.Series, col: str) -> Optional[float]:
    if col in row and pd.notna(row[col]):
        return float(row[col])
    return None


def _safe_int(row: pd.Series, col: str) -> Optional[int]:
    if col in row and pd.notna(row[col]):
        return int(row[col])
    return None
