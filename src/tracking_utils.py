"""Утилиты для стабильных ID треков при параллельной обработке."""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Dict, Optional, Tuple

import pandas as pd


_TRACK_ID_LOCK = threading.Lock()


def patch_ultralytics_track_ids() -> None:
    """Делает счетчик ID Ultralytics безопасным для нескольких QThread.

    ByteTrack в Ultralytics использует глобальный BaseTrack._count. Новый трекер
    вызывает reset_id(), что в одном процессе может сбросить ID другому потоку.
    """
    try:
        from ultralytics.trackers.basetrack import BaseTrack
    except Exception:
        return

    if getattr(BaseTrack, "_yolo_jellyfish_patched", False):
        return

    def next_id() -> int:
        with _TRACK_ID_LOCK:
            BaseTrack._count += 1
            return BaseTrack._count

    def reset_id() -> None:
        return None

    BaseTrack.next_id = staticmethod(next_id)
    BaseTrack.reset_id = staticmethod(reset_id)
    BaseTrack._yolo_jellyfish_patched = True


def normalize_track_ids(
    df: pd.DataFrame,
    max_track_gap: int = 60,
    uid_prefix: Optional[str] = None,
) -> pd.DataFrame:
    """Сохраняет source_track_id и разделяет повторившиеся ID по разрывам."""
    if df.empty or "track_id" not in df.columns:
        return df

    df = df.copy()
    if "source_track_id" not in df.columns:
        df["source_track_id"] = df["track_id"]

    valid_mask = df["source_track_id"].notna()
    if not valid_mask.any():
        return df

    valid_ids = pd.to_numeric(df.loc[valid_mask, "source_track_id"], errors="coerce").dropna()
    next_track_id = int(valid_ids.max()) + 1 if not valid_ids.empty else 1
    normalized: Dict[int, int] = {}

    valid_df = df.loc[valid_mask].copy()
    valid_df["_source_track_id_int"] = pd.to_numeric(
        valid_df["source_track_id"], errors="coerce"
    ).astype("Int64")
    valid_df = valid_df.dropna(subset=["_source_track_id_int"]).sort_values(
        ["_source_track_id_int", "frame"]
    )

    for source_id, group in valid_df.groupby("_source_track_id_int", sort=True):
        current_track_id = int(source_id)
        prev_frame: Optional[int] = None

        for idx, row in group.iterrows():
            frame = int(row["frame"])
            if prev_frame is not None and frame - prev_frame > max_track_gap:
                current_track_id = next_track_id
                next_track_id += 1
            normalized[idx] = current_track_id
            prev_frame = frame

    if normalized:
        df.loc[list(normalized.keys()), "track_id"] = pd.Series(normalized)

    df["source_track_id"] = pd.to_numeric(df["source_track_id"], errors="coerce").astype("Int64")
    df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce").astype("Int64")

    if uid_prefix:
        uid_mask = df["track_id"].notna()
        df.loc[uid_mask, "track_uid"] = df.loc[uid_mask, "track_id"].map(
            lambda tid: f"{uid_prefix}_{int(tid)}"
        )
        df.loc[~uid_mask, "track_uid"] = None

    return df


def rebuild_track_state(
    df: pd.DataFrame,
    class_names: Dict[int, str],
) -> Tuple[Dict[int, dict], Dict[int, list]]:
    """Пересобирает track_info и голоса классов после нормализации ID."""
    track_info: Dict[int, dict] = {}
    track_class_votes = defaultdict(list)

    if df.empty or "track_id" not in df.columns:
        return track_info, track_class_votes

    valid_df = df[df["track_id"].notna()].copy().sort_values(["track_id", "frame"])
    for track_id, group in valid_df.groupby("track_id", sort=True):
        tid = int(track_id)
        first = group.iloc[0]
        last = group.iloc[-1]
        class_votes = [int(cid) for cid in group["class_id"].dropna().tolist()]
        class_id = class_votes[0] if class_votes else int(first["class_id"])

        track_class_votes[tid].extend(class_votes)
        track_info[tid] = {
            "class_id": class_id,
            "class_name": class_names.get(class_id, f"unknown_{class_id}"),
            "first_frame": int(first["frame"]),
            "last_frame": int(last["frame"]),
            "first_timestamp": float(first["timestamp_s"]),
            "last_timestamp": float(last["timestamp_s"]),
            "first_depth": None if pd.isna(first.get("depth_m")) else float(first["depth_m"]),
            "last_depth": None if pd.isna(last.get("depth_m")) else float(last["depth_m"]),
        }

    return track_info, track_class_votes
