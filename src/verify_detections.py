"""
Модуль ручной проверки результатов детекции.

Позволяет загружать треки, детекции, сохранять состояние проверки,
применять изменения (удаление трека, смена класса) к CSV-файлам.
"""

import os
import hashlib
import json
from datetime import datetime
from dataclasses import dataclass, replace, asdict
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
    fragment_rows: tuple = ()
    fragment_signature: str = ""
    fragment_assigned: bool = False
    is_archive: bool = False
    merged_member_ids: tuple = ()
    group_parent_id: Optional[int] = None
    detached_from_track_id: Optional[int] = None
    detached_parent_id: Optional[int] = None
    detached_rows: tuple = ()
    detached_signature: str = ""
    detached_owner_ids: tuple = ()

    @property
    def is_candidate(self) -> bool:
        return bool(self.fragment_rows) and not self.fragment_assigned

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
            or bool(self.detached_rows)
        )

    @property
    def status_text(self) -> str:
        if self.is_candidate:
            return "Без ID"
        if self.deleted:
            return "Удалён"
        if self.merged_into_track_id is not None:
            return f"Объединён → {self.merged_into_track_id}"
        if self.detached_rows:
            return "Отсоединён (не применено)"
        if self.confirmed:
            return "Подтверждён"
        if self.new_class_id is not None:
            return f"→ {CLASS_NAMES.get(self.new_class_id, str(self.new_class_id))}"
        if self.fragment_rows:
            return "Новый трек"
        return "OK"


@dataclass
class VerificationResult:
    """Результат применения изменений."""

    total_tracks: int
    deleted_count: int
    changed_class_count: int
    merged_count: int
    verified_csv_path: str
    detached_count: int = 0


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
            "fragment_rows": tuple(json.loads(row["fragment_rows"]))
            if pd.notna(row.get("fragment_rows")) else (),
            "fragment_signature": str(row.get("fragment_signature", "")),
            "fragment_assigned": bool(row["fragment_assigned"])
            if pd.notna(row.get("fragment_assigned")) else False,
            "track_snapshot": json.loads(row["track_snapshot"])
            if pd.notna(row.get("track_snapshot")) else None,
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
                "fragment_rows": json.dumps(t.fragment_rows) if t.fragment_rows else None,
                "fragment_signature": t.fragment_signature or None,
                "fragment_assigned": t.fragment_assigned,
                "track_snapshot": json.dumps(asdict(t), ensure_ascii=False)
                if t.merged_into_track_id is not None or t.detached_rows or t.merged_member_ids else None,
                "verified_at": now,
            }
        )
    pd.DataFrame(rows, columns=[
        "track_id", "original_class_id", "original_class_name", "deleted",
        "new_class_id", "confirmed", "merged_into_track_id", "fragment_rows",
        "fragment_signature", "fragment_assigned", "track_snapshot", "verified_at",
    ]).to_csv(vp, index=False)
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
    if "track_snapshot" in verified_df and any(
        json.loads(value).get("detached_rows") for value in verified_df["track_snapshot"].dropna()
    ):
        return "В работе"
    if "fragment_rows" in verified_df:
        assigned = verified_df["fragment_assigned"].fillna(False).astype(bool)
        verified_df = verified_df[verified_df["fragment_rows"].isna() | assigned]
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
        if pd.notna(row.get("fragment_rows")):
            # Назначение ещё не применено к исходным CSV.
            all_resolved = False
            has_pending = True

    if not has_pending and all_resolved:
        return "Подтверждено"

    resolved_any = any(
        bool(row["deleted"])
        or bool(row.get("confirmed", False))
        or pd.notna(row.get("merged_into_track_id"))
        for _, row in verified_df.iterrows()
    )
    if resolved_any or ("fragment_rows" in verified_df and verified_df["fragment_rows"].notna().any()):
        return "В работе"

    return ""


def detection_file_signature(path: str) -> str:
    """Отпечаток исходного CSV для безопасного восстановления назначений строк."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_untracked_fragments(path: str, preserved_groups: Optional[dict] = None):
    """Собирает однозначные цепочки без ID в соседних кадрах (IoU >= 0.3)."""
    if not path or not os.path.exists(path):
        return [], pd.DataFrame()
    chunks = []
    for chunk in pd.read_csv(path, chunksize=50000):
        if "track_id" not in chunk:
            return [], pd.DataFrame()
        selected = chunk[chunk["track_id"].isna()]
        if not selected.empty:
            chunks.append(selected.copy())
    if not chunks:
        return [], pd.DataFrame()
    orphan = pd.concat(chunks)
    signature = detection_file_signature(path)
    protected = {idx: group for group, (rows, saved_signature) in enumerate((preserved_groups or {}).items())
                 if saved_signature == signature for idx in rows}
    groups = []
    previous = []
    previous_frame = None
    for frame, current in orphan.sort_values("frame").groupby("frame", sort=True):
        current_rows = list(current.iterrows())
        links = {}
        reverse = {}
        if previous_frame is not None and frame == previous_frame + 1:
            for idx, row in current_rows:
                matches = []
                for old_idx, old_row, group_idx in previous:
                    if protected.get(idx) != protected.get(old_idx) and (
                        idx in protected or old_idx in protected
                    ):
                        continue
                    if row["class_id"] != old_row["class_id"]:
                        continue
                    if _bbox_iou(row, old_row) >= 0.3:
                        matches.append((old_idx, group_idx))
                        reverse.setdefault(old_idx, []).append(idx)
                links[idx] = matches
        next_previous = []
        for idx, row in current_rows:
            matches = links.get(idx, [])
            if len(matches) == 1 and len(reverse[matches[0][0]]) == 1:
                group_idx = matches[0][1]
                groups[group_idx].append(int(idx))
            else:
                group_idx = len(groups)
                groups.append([int(idx)])
            next_previous.append((idx, row, group_idx))
        previous, previous_frame = next_previous, frame

    fragments = []
    for rows in groups:
        det = orphan.loc[rows].sort_values("frame")
        first, last = det.iloc[0], det.iloc[-1]
        first_depth, last_depth = _safe_float(first, "depth_m"), _safe_float(last, "depth_m")
        fragments.append(TrackInfo(
            track_id=-(rows[0] + 1), class_id=int(first["class_id"]),
            class_name=CLASS_NAMES.get(int(first["class_id"]), str(first["class_id"])),
            first_frame=int(first["frame"]), last_frame=int(last["frame"]),
            frame_span=int(last["frame"] - first["frame"] + 1),
            duration_s=round(float(last["timestamp_s"] - first["timestamp_s"]), 3),
            first_timestamp_s=float(first["timestamp_s"]),
            last_timestamp_s=float(last["timestamp_s"]),
            first_depth_m=first_depth, last_depth_m=last_depth,
            depth_change_m=round(last_depth - first_depth, 2)
            if first_depth is not None and last_depth is not None else None,
            detections_count=len(det), avg_confidence=float(det["confidence"].mean()),
            fragment_rows=tuple(rows), fragment_signature=signature,
        ))
    return fragments, orphan


def _bbox_iou(a, b) -> float:
    values = [float(row[col]) for row in (a, b)
              for col in ("x_center", "y_center", "width", "height")]
    if not np.isfinite(values).all() or any(values[i] <= 0 for i in (2, 3, 6, 7)):
        return 0.0
    ax, ay, aw, ah, bx, by, bw, bh = values
    w = max(0.0, min(ax + aw / 2, bx + bw / 2) - max(ax - aw / 2, bx - bw / 2))
    h = max(0.0, min(ay + ah / 2, by + bh / 2) - max(ay - ah / 2, by - bh / 2))
    intersection = w * h
    return intersection / (aw * ah + bw * bh - intersection)


def load_track_proximity(path: str, reference: pd.DataFrame,
                         track_ids: Dict[int, int], orphan_ids: Dict[int, int],
                         row_targets: Optional[dict] = None) -> dict:
    """Сравнивает концы выбранного фрагмента с ближайшими по времени рамками.

    Оценка: разрыв в секундах + расстояние в диагоналях средней рамки + (1 - IoU).
    Меньше — ближе. Это подсказка для ручного объединения, а не решение трекера.
    Словари переводят исходные ID и номера строк без ID в актуальные ID окна.
    """
    if reference.empty or not path or not os.path.exists(path):
        return {}
    reference = reference.sort_values("timestamp_s")
    anchors = reference.iloc[[0, -1]].drop_duplicates()
    best = [{} for _ in range(len(anchors))]
    columns = ["track_id", "timestamp_s", "x_center", "y_center", "width", "height"]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=50000):
        ids = chunk["track_id"].map(track_ids)
        missing = chunk["track_id"].isna()
        ids.loc[missing] = chunk.index.to_series().loc[missing].map(orphan_ids)
        if row_targets:
            override = chunk.index.isin(row_targets)
            ids.loc[override] = chunk.index.to_series().loc[override].map(row_targets)
        chunk = chunk.assign(_target_id=ids).dropna(subset=["_target_id"])
        geometry = chunk[["timestamp_s", "x_center", "y_center", "width", "height"]]
        chunk = chunk[np.isfinite(geometry).all(axis=1)
                      & (chunk["width"] > 0) & (chunk["height"] > 0)].copy()
        if chunk.empty:
            continue
        for anchor_idx, (_, anchor) in enumerate(anchors.iterrows()):
            values = anchor[["timestamp_s", "x_center", "y_center", "width", "height"]].astype(float)
            if not np.isfinite(values).all() or anchor["width"] <= 0 or anchor["height"] <= 0:
                continue
            chunk["_dt"] = (chunk["timestamp_s"] - anchor["timestamp_s"]).abs()
            nearest = chunk[chunk["_dt"] == chunk.groupby("_target_id")["_dt"].transform("min")]
            for tid, group in nearest.groupby("_target_id"):
                for _, row in group.iterrows():
                    distance = float(np.hypot(row["x_center"] - anchor["x_center"],
                                              row["y_center"] - anchor["y_center"]))
                    diagonal = float(np.hypot((row["width"] + anchor["width"]) / 2,
                                              (row["height"] + anchor["height"]) / 2))
                    iou = _bbox_iou(anchor, row)
                    dt = float(row["_dt"])
                    score = dt + distance / diagonal + (1 - iou)
                    result = {"score": score, "time_gap_s": dt,
                              "center_distance": distance, "iou": iou}
                    previous = best[anchor_idx].get(int(tid))
                    if previous is None or (dt, score) < (previous["time_gap_s"], previous["score"]):
                        best[anchor_idx][int(tid)] = result
    results = {}
    for endpoint in best:
        for tid, result in endpoint.items():
            if tid not in results or result["score"] < results[tid]["score"]:
                results[tid] = result
    return results


def load_track_detections(detections_csv_path: str, track_id: int,
                          source_ids: Optional[tuple] = None) -> pd.DataFrame:
    """Загружает детекции одного трека из _detections.csv."""
    if not os.path.exists(detections_csv_path):
        return pd.DataFrame()

    chunks = []
    for chunk in pd.read_csv(detections_csv_path, chunksize=50000):
        if source_ids is not None and "verification_source_track_id" in chunk:
            mask = chunk["verification_source_track_id"].isin(source_ids)
        else:
            mask = chunk["track_id"] == track_id
        if mask.any():
            chunks.append(chunk[mask])
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def load_verification_track_detections(path: str, tracks: List[TrackInfo], track_id: int) -> pd.DataFrame:
    """Представление детекций с учётом ещё не применённых объединений и отсоединений.

    Индекс сохраняет номера строк исходного CSV для обратимых назначений.
    """
    by_id = {t.track_id: t for t in tracks}
    if track_id not in by_id or not path or not os.path.exists(path):
        return pd.DataFrame()
    routes = {}
    for track in tracks:
        root = track
        visited = set()
        while root.merged_into_track_id in by_id and root.track_id not in visited:
            visited.add(root.track_id)
            root = by_id[root.merged_into_track_id]
        routes[track.track_id] = root.track_id
    row_sources = {idx: t.track_id for t in tracks for idx in t.fragment_rows}
    row_destinations = {idx: routes[t.track_id] for t in tracks
                        for idx in set(t.fragment_rows) | set(t.detached_rows)}
    children = {}
    for track in tracks:
        if track.merged_into_track_id is not None:
            parent = track.group_parent_id if track.group_parent_id in by_id else track.merged_into_track_id
            children.setdefault(parent, []).append(track)
    members = set()
    visited = set()

    def collect(track):
        if track.track_id in visited:
            return
        visited.add(track.track_id)
        members.update(track.merged_member_ids or (track.track_id,))
        for child in children.get(track.track_id, []):
            collect(child)

    selected = by_id[track_id]
    collect(selected)
    chunks = []
    for chunk in pd.read_csv(path, chunksize=50000):
        ids = chunk["track_id"].map(routes)
        overrides = chunk.index.to_series().map(row_destinations)
        ids = overrides.combine_first(ids)
        mask = ids == routes[track_id]
        if selected.merged_into_track_id is not None:
            origins = chunk.get("verification_source_track_id", chunk["track_id"]).fillna(chunk["track_id"])
            origins = chunk.index.to_series().map(row_sources).combine_first(origins)
            mask &= origins.isin(members)
        if mask.any():
            chunks.append(chunk[mask].copy())
    return pd.concat(chunks) if chunks else pd.DataFrame()


def refresh_track_statistics(track: TrackInfo, detections: pd.DataFrame) -> None:
    """Пересчитывает статистику по фактическому составу группы без суммирования копий."""
    if detections.empty:
        track.detections_count = 0
        return
    detections = detections.sort_values("frame")
    first, last = detections.iloc[0], detections.iloc[-1]
    track.first_frame, track.last_frame = int(first["frame"]), int(last["frame"])
    track.frame_span = track.last_frame - track.first_frame + 1
    track.first_timestamp_s = float(first["timestamp_s"])
    track.last_timestamp_s = float(last["timestamp_s"])
    track.duration_s = round(track.last_timestamp_s - track.first_timestamp_s, 3)
    track.first_depth_m, track.last_depth_m = _safe_float(first, "depth_m"), _safe_float(last, "depth_m")
    track.depth_change_m = (round(track.last_depth_m - track.first_depth_m, 2)
                            if track.first_depth_m is not None and track.last_depth_m is not None else None)
    track.detections_count = len(detections)
    track.avg_confidence = round(float(detections["confidence"].mean()), 3)


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
            selected = chunk[mask].copy()
            selected["_detection_row"] = selected.index
            chunks.append(selected)

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
    restorations = [t for t in tracks if t.detached_rows]
    fragments = [t for t in tracks if t.fragment_rows and t.fragment_assigned and not t.detached_rows]
    candidates = [t for t in tracks if t.is_candidate]
    # Неназначенные кандидаты остаются в CSV без ID и не становятся треками.
    tracks = [t for t in tracks if not t.is_candidate]
    tracks_df = pd.read_csv(tracks_csv_path) if os.path.exists(tracks_csv_path) else None
    if fragments or restorations:
        if tracks_df is None or not os.path.exists(detections_csv_path):
            raise ValueError("Для назначения фрагментов необходимы исходные CSV треков и детекций.")
        for track in fragments:
            if track.merged_into_track_id is None and not track.deleted:
                if track.track_id <= 0 or (tracks_df["track_id"] == track.track_id).any():
                    raise ValueError(f"ID нового трека {track.track_id} уже занят или некорректен.")
    deleted_ids = set(t.track_id for t in tracks if t.deleted)
    merge_map = {
        t.track_id: t.merged_into_track_id
        for t in tracks
        if not t.deleted and t.merged_into_track_id is not None
    }
    track_ids = {t.track_id for t in tracks}
    for source_id, target_id in list(merge_map.items()):
        visited = {source_id}
        while target_id in merge_map:
            if target_id in visited:
                raise ValueError("Обнаружена циклическая связь объединённых треков.")
            visited.add(target_id)
            target_id = merge_map[target_id]
        if target_id not in track_ids:
            raise ValueError(f"Целевой трек {target_id} отсутствует. Проверьте назначения.")
        merge_map[source_id] = target_id
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
    merged_count = sum(t.track_id in merge_map and not t.is_archive for t in tracks)

    # 1. Обновляем _detections.csv
    if os.path.exists(detections_csv_path):
        det_df = pd.read_csv(detections_csv_path)
        original_ids = det_df["track_id"].copy()
        if fragments or candidates or restorations:
            signature = detection_file_signature(detections_csv_path)
            if any(t.fragment_signature != signature for t in fragments + candidates):
                raise ValueError("Исходный CSV изменился. Откройте окно проверки заново.")
        uid_by_id = {}
        if "track_uid" in det_df:
            uid_by_id = (det_df.dropna(subset=["track_id", "track_uid"])
                          .groupby("track_id")["track_uid"].first().to_dict())

        restored_rows = set()
        for track in restorations:
            rows = list(track.detached_rows)
            if (track.detached_signature != signature or restored_rows.intersection(rows)
                    or not set(rows).issubset(det_df.index)):
                raise ValueError("Исходные детекции отсоединяемой части изменились. Откройте проверку заново.")
            restored_rows.update(rows)
            if track.is_candidate:
                det_df.loc[rows, "track_id"] = np.nan
                if "track_uid" in det_df:
                    det_df.loc[rows, "track_uid"] = None
                if "verification_source_track_id" in det_df:
                    det_df.loc[rows, "verification_source_track_id"] = np.nan
            else:
                if track.track_id > 0 and (original_ids == track.track_id).any():
                    raise ValueError(f"ID отсоединённого трека {track.track_id} уже занят.")
                det_df.loc[rows, "track_id"] = track.track_id
                if track.fragment_rows and track.track_id > 0 and "verification_source_track_id" in det_df:
                    det_df.loc[rows, "verification_source_track_id"] = track.track_id
            det_df.loc[rows, "class_id"] = track.effective_class_id
            det_df.loc[rows, "class_name"] = track.effective_class_name

        if fragments:
            used_rows = set()
            for track in fragments:
                rows = list(track.fragment_rows)
                if track.track_id > 0 and (original_ids == track.track_id).any():
                    raise ValueError(f"ID нового трека {track.track_id} уже есть в детекциях.")
                if (track.fragment_signature != signature
                        or used_rows.intersection(rows)
                        or not set(rows).issubset(det_df.index)
                        or not (original_ids.loc[rows].isna()
                                | original_ids.loc[rows].index.isin(restored_rows)).all()):
                    raise ValueError(
                        "Исходный CSV изменился или назначения пересекаются. "
                        "Откройте окно проверки заново и повторите назначение фрагментов."
                    )
                used_rows.update(rows)
                det_df.loc[rows, "track_id"] = track.track_id
                if "verification_source_track_id" in det_df:
                    det_df.loc[rows, "verification_source_track_id"] = track.track_id

        if merge_map:
            # Происхождение рамок сохраняется после замены ID: вложенные части
            # можно просматривать и после применения изменений к исходным CSV.
            if "verification_source_track_id" not in det_df:
                det_df["verification_source_track_id"] = det_df["track_id"]
            else:
                det_df["verification_source_track_id"] = det_df["verification_source_track_id"].fillna(
                    det_df["track_id"]
                )

        if deleted_ids:
            det_df = det_df[~det_df["track_id"].isin(deleted_ids)]

        remaining_ids = set(t.track_id for t in tracks if not t.deleted)
        det_df = det_df[det_df["track_id"].isin(remaining_ids) | det_df["track_id"].isna()].copy()

        for source_id, target_id in merge_map.items():
            det_df.loc[det_df["track_id"] == source_id, "track_id"] = target_id
        # Удаление целевого трека удаляет и уже присоединённые к нему детекции.
        det_df = det_df[~det_df["track_id"].isin(deleted_ids)].copy()

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

        for track in fragments:
            tid = merge_map.get(track.track_id, track.track_id)
            mask = det_df.index.isin(track.fragment_rows)
            cid = target_class_map.get(tid, track.effective_class_id)
            det_df.loc[mask, "class_id"] = cid
            det_df.loc[mask, "class_name"] = CLASS_NAMES.get(cid, f"unknown_{cid}")

        # После объединения UID должен соответствовать целевому треку.
        if "track_uid" in det_df:
            affected_ids = set(merge_map.values()) | {
                t.track_id for t in fragments + restorations if not t.deleted and not t.is_candidate
            }
            for tid in affected_ids - set(merge_map):
                mask = det_df["track_id"] == tid
                uid = uid_by_id.get(tid, f"manual_{tid}")
                det_df.loc[mask, "track_uid"] = uid

    # 2. Обновляем _tracks.csv
    if tracks_df is not None:

        for track in fragments + restorations:
            if track.is_candidate or track.deleted or track.merged_into_track_id is not None:
                continue
            if (tracks_df["track_id"] == track.track_id).any():
                raise ValueError(f"ID восстанавливаемого трека {track.track_id} уже занят в CSV треков.")
            values = {"track_id": track.track_id, "class_id": track.effective_class_id,
                      "class_name": track.effective_class_name}
            if "track_uid" in tracks_df:
                values["track_uid"] = f"manual_{track.track_id}"
            tracks_df = pd.concat([tracks_df, pd.DataFrame([values])], ignore_index=True)
            _update_track_row_from_detections(tracks_df, det_df, track.track_id)

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

        if merge_map or restorations:
            track_by_id = {t.track_id: t for t in tracks}
            affected = set(merge_map.values()) | {
                t.track_id for t in tracks if t.merged_into_track_id is None
            }
            for target_id in sorted(affected):
                if 'det_df' in locals():
                    _update_track_row_from_detections(tracks_df, det_df, target_id)
                else:
                    _update_track_row_from_track(tracks_df, track_by_id.get(target_id))

        if "track_uid" in tracks_df and 'det_df' in locals() and "track_uid" in det_df:
            for tid in set(merge_map.values()) | {t.track_id for t in fragments + restorations}:
                uids = det_df.loc[det_df["track_id"] == tid, "track_uid"].dropna()
                if not uids.empty:
                    tracks_df.loc[tracks_df["track_id"] == tid, "track_uid"] = uids.iloc[0]

    # Все проверки и пересчёты завершены до записи исходных файлов.
    if 'det_df' in locals():
        det_df.to_csv(detections_csv_path, index=False)
    if tracks_df is not None:
        tracks_df.to_csv(tracks_csv_path, index=False)

    # 3. Записываем _verified.csv
    # Назначения уже материализованы; ссылки на старые номера строк больше не нужны.
    saved_tracks = []
    for track in tracks:
        target_id = merge_map.get(track.track_id)
        if target_id is not None:
            if target_id in deleted_ids:
                continue
            saved_tracks.append(replace(
                track, fragment_rows=(), fragment_signature="", fragment_assigned=False,
                is_archive=True, merged_into_track_id=target_id,
                merged_member_ids=track.merged_member_ids or (track.track_id,),
                detached_from_track_id=None, detached_parent_id=None, detached_rows=(), detached_signature="",
                detached_owner_ids=(),
            ))
        elif not track.fragment_rows or not track.deleted:
            saved_tracks.append(replace(
                track, fragment_rows=(), fragment_signature="", fragment_assigned=False, is_archive=False,
                detached_from_track_id=None, detached_parent_id=None, detached_rows=(), detached_signature="",
                detached_owner_ids=(),
            ))
    if candidates and 'det_df' in locals():
        # После удаления строк CSV их номера меняются. Сохраняем также правки
        # класса ещё не назначенных фрагментов с актуальными ссылками на строки.
        row_numbers = {old: new for new, old in enumerate(det_df.index)}
        signature = detection_file_signature(detections_csv_path)
        saved_tracks.extend(replace(
            t, fragment_rows=tuple(row_numbers[idx] for idx in t.fragment_rows),
            fragment_signature=signature, is_archive=False,
            detached_from_track_id=None, detached_parent_id=None, detached_rows=(), detached_signature="",
            detached_owner_ids=(),
        ) for t in candidates)
    vp = save_verified(saved_tracks, tracks_csv_path)

    return VerificationResult(
        total_tracks=len(tracks_df) if tracks_df is not None else sum(
            not t.deleted and t.merged_into_track_id is None for t in tracks
        ),
        deleted_count=deleted_count,
        changed_class_count=changed_count,
        merged_count=merged_count,
        verified_csv_path=vp,
        detached_count=len(restorations),
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
    if "source_track_id" in tracks_df and "source_track_id" in det:
        source = det
        if "verification_source_track_id" in det:
            own = det[det["verification_source_track_id"] == track_id]
            if not own.empty:
                source = own
        values = source["source_track_id"].dropna()
        if not values.empty:
            tracks_df.loc[mask & tracks_df["source_track_id"].isna(), "source_track_id"] = values.iloc[0]

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
        else:
            tracks_df.loc[mask, "depth_change_m"] = None
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
