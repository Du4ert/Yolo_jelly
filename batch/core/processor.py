"""
Processor - обёртка над detect_video.py для интеграции с batch processing.

Поддерживает:
- Callback для отслеживания прогресса
- Отмену выполнения
- Паузу
"""

import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from collections import defaultdict, Counter

# Добавляем путь к src для импорта
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))


@dataclass
class ProcessingResult:
    """Результат обработки видео."""
    success: bool
    detections_count: int = 0
    tracks_count: int = 0
    processing_time_s: float = 0.0
    output_video_path: Optional[str] = None
    output_csv_path: Optional[str] = None
    output_tracks_path: Optional[str] = None
    error_message: Optional[str] = None
    cancelled: bool = False


# Соответствие классов (копируем из detect_video.py)
CLASS_NAMES = {
    0: 'Aurelia aurita',
    1: 'Beroe ovata',
    2: 'Mnemiopsis leidyi',
    3: 'Pleurobrachia pileus',
    4: 'Rhizostoma pulmo'
}


def load_ctd_data(ctd_path: str) -> pd.DataFrame:
    """Загружает данные CTD из CSV файла."""
    with open(ctd_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
    
    for sep in ['|', ';', '\t', ',']:
        if sep in first_line:
            delimiter = sep
            break
    else:
        delimiter = ','
    
    df = pd.read_csv(ctd_path, sep=delimiter)
    df.columns = df.columns.str.lower()
    
    required_cols = ['time', 'depth']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Отсутствует обязательная колонка: {col}")
    
    return df


def interpolate_ctd(ctd_data: pd.DataFrame, timestamp: float, param: str) -> Optional[float]:
    """Интерполирует параметр CTD по времени."""
    if param not in ctd_data.columns:
        return None
    return float(np.interp(timestamp, ctd_data['time'], ctd_data[param]))


def generate_color_for_id(track_id: int) -> tuple:
    """Генерирует уникальный цвет для ID трека."""
    golden_ratio = 0.618033988749895
    hue = (track_id * golden_ratio) % 1.0
    h = hue * 6
    c = 1.0
    x = c * (1 - abs(h % 2 - 1))
    
    if h < 1:
        r, g, b = c, x, 0
    elif h < 2:
        r, g, b = x, c, 0
    elif h < 3:
        r, g, b = 0, c, x
    elif h < 4:
        r, g, b = 0, x, c
    elif h < 5:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    return (int(b * 255), int(g * 255), int(r * 255))


class Processor:
    """
    Обработчик видео с поддержкой прогресса и отмены.
    """

    def __init__(
        self,
        video_path: str,
        model_path: str,
        output_dir: str,
        ctd_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        enable_tracking: bool = True,
        tracker_type: str = "bytetrack.yaml",
        show_trails: bool = False,
        trail_length: int = 30,
        min_track_length: int = 3,
        depth_rate: Optional[float] = None,
        save_video: bool = True,
    ):
        self.video_path = video_path
        self.model_path = model_path
        self.output_dir = output_dir
        self.ctd_path = ctd_path
        self.conf_threshold = conf_threshold
        self.enable_tracking = enable_tracking
        self.tracker_type = tracker_type
        self.show_trails = show_trails
        self.trail_length = trail_length
        self.min_track_length = min_track_length
        self.depth_rate = depth_rate
        self.save_video = save_video
        
        # Callbacks
        self.progress_callback: Optional[Callable[[int, int, int, int], None]] = None
        
        # Флаги управления
        self._cancelled = False
        self._paused = False

    def cancel(self) -> None:
        """Отменяет выполнение."""
        self._cancelled = True

    def pause(self) -> None:
        """Ставит на паузу."""
        self._paused = True

    def resume(self) -> None:
        """Снимает с паузы."""
        self._paused = False

    def is_cancelled(self) -> bool:
        return self._cancelled

    def is_paused(self) -> bool:
        return self._paused

    def _generate_output_paths(self) -> Dict[str, str]:
        """Генерирует пути для выходных файлов."""
        video_name = Path(self.video_path).stem
        return {
            "video": os.path.join(self.output_dir, f"{video_name}_detected.mp4"),
            "csv": os.path.join(self.output_dir, f"{video_name}_detections.csv"),
            "tracks": os.path.join(self.output_dir, f"{video_name}_tracks.csv"),
        }

    def run(self) -> ProcessingResult:
        """Запускает обработку видео."""
        self._cancelled = False
        self._paused = False
        start_time = time.time()
        
        os.makedirs(self.output_dir, exist_ok=True)
        output_paths = self._generate_output_paths()
        
        try:
            from ultralytics import YOLO
            
            # Загрузка модели
            model = YOLO(self.model_path)
            
            # Загрузка CTD
            ctd_data = None
            if self.ctd_path and os.path.exists(self.ctd_path):
                ctd_data = load_ctd_data(self.ctd_path)
            
            # Открытие видео
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise ValueError(f"Не удалось открыть видео: {self.video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Подготовка выходного видео
            out = None
            if self.save_video:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_paths["video"], fourcc, fps, (width, height))
            
            # Данные
            detections = []
            track_history = defaultdict(list)
            track_info = {}
            track_class_votes = defaultdict(list)
            
            frame_count = 0
            
            while True:
                # Проверка отмены
                if self._cancelled:
                    cap.release()
                    if out:
                        out.release()
                    return ProcessingResult(
                        success=False,
                        cancelled=True,
                        processing_time_s=time.time() - start_time,
                        error_message="Отменено пользователем"
                    )
                
                # Пауза
                while self._paused and not self._cancelled:
                    time.sleep(0.1)
                
                if self._cancelled:
                    continue
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_count / fps if fps > 0 else 0
                
                # Глубина
                depth = None
                temperature = None
                salinity = None
                
                if ctd_data is not None:
                    depth = interpolate_ctd(ctd_data, timestamp, 'depth')
                    temperature = interpolate_ctd(ctd_data, timestamp, 'temperature')
                    salinity = interpolate_ctd(ctd_data, timestamp, 'salinity')
                elif self.depth_rate:
                    depth = timestamp * self.depth_rate
                
                # Детекция
                if self.enable_tracking:
                    results = model.track(
                        frame,
                        conf=self.conf_threshold,
                        persist=True,
                        tracker=self.tracker_type,
                        verbose=False
                    )[0]
                else:
                    results = model(frame, conf=self.conf_threshold, verbose=False)[0]
                
                # Обработка результатов
                boxes = results.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        class_id = int(box.cls)
                        confidence = float(box.conf)
                        
                        xywhn = box.xywhn[0]
                        x_center = float(xywhn[0])
                        y_center = float(xywhn[1])
                        bbox_width = float(xywhn[2])
                        bbox_height = float(xywhn[3])
                        
                        track_id = None
                        if self.enable_tracking and box.id is not None:
                            track_id = int(box.id)
                            
                            center_x = int(x_center * width)
                            center_y = int(y_center * height)
                            track_history[track_id].append((center_x, center_y))
                            
                            if len(track_history[track_id]) > self.trail_length:
                                track_history[track_id] = track_history[track_id][-self.trail_length:]
                            
                            track_class_votes[track_id].append(class_id)
                            
                            if track_id not in track_info:
                                track_info[track_id] = {
                                    'class_id': class_id,
                                    'class_name': CLASS_NAMES.get(class_id, f'unknown_{class_id}'),
                                    'first_frame': frame_count,
                                    'last_frame': frame_count,
                                    'first_timestamp': timestamp,
                                    'last_timestamp': timestamp,
                                    'first_depth': depth,
                                    'last_depth': depth
                                }
                            else:
                                track_info[track_id]['last_frame'] = frame_count
                                track_info[track_id]['last_timestamp'] = timestamp
                                track_info[track_id]['last_depth'] = depth
                        
                        det = {
                            'frame': frame_count,
                            'timestamp_s': round(timestamp, 3),
                            'depth_m': round(depth, 2) if depth is not None else None,
                            'temperature_c': round(temperature, 2) if temperature is not None else None,
                            'salinity_psu': round(salinity, 2) if salinity is not None else None,
                            'track_id': track_id,
                            'class_id': class_id,
                            'class_name': CLASS_NAMES.get(class_id, f'unknown_{class_id}'),
                            'confidence': round(confidence, 3),
                            'x_center': round(x_center, 4),
                            'y_center': round(y_center, 4),
                            'width': round(bbox_width, 4),
                            'height': round(bbox_height, 4),
                            'bbox_area_norm': round(bbox_width * bbox_height, 6)
                        }
                        detections.append(det)
                
                # Отрисовка
                if out is not None:
                    annotated_frame = results.plot(line_width=2, font_size=0.6, labels=True, conf=True)
                    
                    if self.enable_tracking and self.show_trails:
                        for tid, trail in track_history.items():
                            if len(trail) > 1:
                                color = generate_color_for_id(tid)
                                points = np.array(trail, dtype=np.int32).reshape((-1, 1, 2))
                                cv2.polylines(annotated_frame, [points], False, color, 2)
                    
                    info_text = f"Frame: {frame_count}/{total_frames}"
                    if depth is not None:
                        info_text += f" | Depth: {depth:.1f}m"
                    cv2.putText(annotated_frame, info_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    out.write(annotated_frame)
                
                frame_count += 1
                
                # Callback прогресса (каждые 10 кадров)
                if self.progress_callback and frame_count % 10 == 0:
                    self.progress_callback(frame_count, total_frames, len(detections), len(track_info))
            
            # Завершение
            cap.release()
            if out:
                out.release()
            
            # Постобработка треков
            df = pd.DataFrame(detections)
            
            if self.enable_tracking and len(df) > 0 and len(track_info) > 0:
                # Определяем доминирующий класс
                track_dominant_class = {}
                for tid, votes in track_class_votes.items():
                    class_counts = Counter(votes)
                    dominant_class_id = class_counts.most_common(1)[0][0]
                    track_dominant_class[tid] = dominant_class_id
                    track_info[tid]['class_id'] = dominant_class_id
                    track_info[tid]['class_name'] = CLASS_NAMES.get(dominant_class_id, f'unknown_{dominant_class_id}')
                
                # Фильтруем короткие треки
                valid_tracks = {tid for tid, votes in track_class_votes.items() 
                               if len(votes) >= self.min_track_length}
                short_tracks = set(track_info.keys()) - valid_tracks
                
                df = df[df['track_id'].isin(valid_tracks) | df['track_id'].isna()].copy()
                
                # Обновляем классы
                def update_class(row):
                    if pd.notna(row['track_id']) and row['track_id'] in track_dominant_class:
                        dominant_id = track_dominant_class[row['track_id']]
                        row['class_id'] = dominant_id
                        row['class_name'] = CLASS_NAMES.get(dominant_id, f'unknown_{dominant_id}')
                    return row
                
                df = df.apply(update_class, axis=1)
                
                for tid in short_tracks:
                    if tid in track_info:
                        del track_info[tid]
            
            # Сохранение CSV
            df.to_csv(output_paths["csv"], index=False)
            
            # Сохранение статистики треков
            tracks_path = None
            if self.enable_tracking and track_info:
                track_stats = []
                for tid, info in track_info.items():
                    duration = info['last_timestamp'] - info['first_timestamp']
                    frame_span = info['last_frame'] - info['first_frame'] + 1
                    track_detections = df[df['track_id'] == tid]
                    
                    stat = {
                        'track_id': tid,
                        'class_id': info['class_id'],
                        'class_name': info['class_name'],
                        'first_frame': info['first_frame'],
                        'last_frame': info['last_frame'],
                        'frame_span': frame_span,
                        'duration_s': round(duration, 2),
                        'first_timestamp_s': round(info['first_timestamp'], 2),
                        'last_timestamp_s': round(info['last_timestamp'], 2),
                        'first_depth_m': round(info['first_depth'], 2) if info['first_depth'] else None,
                        'last_depth_m': round(info['last_depth'], 2) if info['last_depth'] else None,
                        'detections_count': len(track_detections),
                        'avg_confidence': round(track_detections['confidence'].mean(), 3) if len(track_detections) > 0 else 0
                    }
                    
                    if info['first_depth'] is not None and info['last_depth'] is not None:
                        stat['depth_change_m'] = round(info['last_depth'] - info['first_depth'], 2)
                    else:
                        stat['depth_change_m'] = None
                    
                    track_stats.append(stat)
                
                track_df = pd.DataFrame(track_stats)
                track_df.to_csv(output_paths["tracks"], index=False)
                tracks_path = output_paths["tracks"]
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                detections_count=len(df),
                tracks_count=len(track_info),
                processing_time_s=processing_time,
                output_video_path=output_paths["video"] if self.save_video else None,
                output_csv_path=output_paths["csv"],
                output_tracks_path=tracks_path,
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return ProcessingResult(
                success=False,
                processing_time_s=time.time() - start_time,
                error_message=str(e),
            )


class ProcessorFactory:
    """Фабрика для создания Processor."""
    
    @staticmethod
    def from_task_data(
        video_path: str,
        model_path: str,
        dive_folder: str,
        ctd_path: Optional[str] = None,
        **task_params,
    ) -> Processor:
        output_dir = os.path.join(dive_folder, "output")
        
        return Processor(
            video_path=video_path,
            model_path=model_path,
            output_dir=output_dir,
            ctd_path=ctd_path,
            conf_threshold=task_params.get("conf_threshold", 0.25),
            enable_tracking=task_params.get("enable_tracking", True),
            tracker_type=task_params.get("tracker_type", "bytetrack.yaml"),
            show_trails=task_params.get("show_trails", False),
            trail_length=task_params.get("trail_length", 30),
            min_track_length=task_params.get("min_track_length", 3),
            depth_rate=task_params.get("depth_rate"),
            save_video=task_params.get("save_video", True),
        )
