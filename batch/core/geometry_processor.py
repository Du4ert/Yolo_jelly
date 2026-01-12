"""
GeometryProcessor - обёртка над camera_geometry.py для интеграции с batch processing.

Поддерживает:
- Оценку наклона камеры (FOE)
- Расчёт размеров объектов по трекам
- Расчёт осмотренного объёма воды
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

# Добавляем путь к src для импорта
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))


@dataclass
class GeometryResult:
    """Результат обработки геометрии камеры."""
    success: bool
    processing_time_s: float = 0.0
    output_csv_path: Optional[str] = None
    mean_tilt_deg: Optional[float] = None
    n_intervals: int = 0
    n_outliers: int = 0
    error_message: Optional[str] = None
    cancelled: bool = False


@dataclass
class SizeEstimationResult:
    """Результат оценки размеров объектов."""
    success: bool
    processing_time_s: float = 0.0
    output_csv_path: Optional[str] = None
    tracks_csv_path: Optional[str] = None
    tracks_with_regression: int = 0
    tracks_with_reference: int = 0
    tracks_with_typical: int = 0
    total_tracks: int = 0
    error_message: Optional[str] = None
    cancelled: bool = False


@dataclass
class VolumeEstimationResult:
    """Результат расчёта осмотренного объёма."""
    success: bool
    processing_time_s: float = 0.0
    output_csv_path: Optional[str] = None
    total_volume_m3: Optional[float] = None
    depth_min_m: Optional[float] = None
    depth_max_m: Optional[float] = None
    depth_traversed_m: Optional[float] = None
    detection_distance_m: Optional[float] = None
    counts_by_class: Optional[Dict[str, int]] = None
    density_by_class: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    cancelled: bool = False


class GeometryProcessor:
    """
    Обработчик геометрии камеры (FOE - Focus of Expansion).
    
    Анализирует видео и оценивает наклон камеры по движению частиц
    (морской снег).
    """
    
    def __init__(
        self,
        video_path: str,
        output_csv: str,
        frame_interval: int = 30,
        frame_width: int = 1920,
        frame_height: int = 1080,
    ):
        self.video_path = video_path
        self.output_csv = output_csv
        self.frame_interval = frame_interval
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Callbacks
        self.progress_callback: Optional[Callable[[int, int], None]] = None
        
        # Флаги управления
        self._cancelled = False
    
    def cancel(self) -> None:
        """Отменяет выполнение."""
        self._cancelled = True
    
    def is_cancelled(self) -> bool:
        return self._cancelled
    
    def run(self) -> GeometryResult:
        """Запускает анализ геометрии камеры."""
        self._cancelled = False
        start_time = time.time()
        
        try:
            from camera_geometry import (
                process_video_geometry,
                CameraCalibration
            )
            
            # Создаём выходную директорию
            os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
            
            # Калибровка
            calibration = CameraCalibration()
            calibration.frame_width = self.frame_width
            calibration.frame_height = self.frame_height
            
            # Обработка видео
            df = process_video_geometry(
                video_path=self.video_path,
                output_csv=self.output_csv,
                frame_interval=self.frame_interval,
                calibration=calibration,
                verbose=True
            )
            
            processing_time = time.time() - start_time
            
            if len(df) == 0:
                return GeometryResult(
                    success=False,
                    processing_time_s=processing_time,
                    error_message="Не удалось извлечь данные геометрии"
                )
            
            # Статистика (фильтруем выбросы)
            import numpy as np
            
            if 'confidence' in df.columns:
                valid_df = df[df['confidence'] >= 0.5]
            else:
                valid_df = df
            
            n_outliers = len(df) - len(valid_df)
            
            if len(valid_df) > 0:
                mean_tilt = np.sqrt(
                    valid_df['tilt_horizontal_deg'].mean()**2 + 
                    valid_df['tilt_vertical_deg'].mean()**2
                )
            else:
                mean_tilt = None
            
            return GeometryResult(
                success=True,
                processing_time_s=processing_time,
                output_csv_path=self.output_csv,
                mean_tilt_deg=round(mean_tilt, 2) if mean_tilt else None,
                n_intervals=len(df),
                n_outliers=n_outliers
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return GeometryResult(
                success=False,
                processing_time_s=time.time() - start_time,
                error_message=str(e)
            )


class SizeEstimationProcessor:
    """
    Обработчик оценки размеров объектов.
    
    Анализирует треки и оценивает реальные размеры объектов
    по динамике изменения размера bbox при погружении камеры.
    """
    
    def __init__(
        self,
        detections_csv: str,
        output_csv: str,
        tracks_output_csv: Optional[str] = None,
        geometry_csv: Optional[str] = None,
        frame_width: int = 1920,
        frame_height: int = 1080,
        min_depth_change: float = 0.3,
        min_track_points: int = 3,
        min_r_squared: float = 0.5,
        min_size_change_ratio: float = 0.3,
    ):
        self.detections_csv = detections_csv
        self.output_csv = output_csv
        self.tracks_output_csv = tracks_output_csv
        self.geometry_csv = geometry_csv
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.min_depth_change = min_depth_change
        self.min_track_points = min_track_points
        self.min_r_squared = min_r_squared
        self.min_size_change_ratio = min_size_change_ratio
        
        self._cancelled = False
    
    def cancel(self) -> None:
        self._cancelled = True
    
    def is_cancelled(self) -> bool:
        return self._cancelled
    
    def run(self) -> SizeEstimationResult:
        """Запускает оценку размеров."""
        self._cancelled = False
        start_time = time.time()
        
        try:
            from camera_geometry import (
                process_detections_with_size,
                CameraCalibration
            )
            
            # Создаём выходные директории
            os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
            if self.tracks_output_csv:
                os.makedirs(os.path.dirname(self.tracks_output_csv), exist_ok=True)
            
            # Калибровка
            calibration = CameraCalibration()
            calibration.frame_width = self.frame_width
            calibration.frame_height = self.frame_height
            
            # Обработка
            df, tracks_df = process_detections_with_size(
                detections_csv=self.detections_csv,
                output_csv=self.output_csv,
                tracks_output_csv=self.tracks_output_csv,
                geometry_csv=self.geometry_csv,
                calibration=calibration,
                frame_width=self.frame_width,
                frame_height=self.frame_height,
                min_depth_change=self.min_depth_change,
                min_track_points=self.min_track_points,
                min_r_squared=self.min_r_squared,
                min_size_change_ratio=self.min_size_change_ratio,
                verbose=True
            )
            
            processing_time = time.time() - start_time
            
            # Статистика по методам
            regression = 0
            reference = 0
            typical = 0
            
            if len(tracks_df) > 0 and 'method' in tracks_df.columns:
                method_counts = tracks_df['method'].value_counts()
                regression = method_counts.get('regression', 0)
                reference = method_counts.get('reference', 0)
                typical = method_counts.get('typical', 0)
            
            return SizeEstimationResult(
                success=True,
                processing_time_s=processing_time,
                output_csv_path=self.output_csv,
                tracks_csv_path=self.tracks_output_csv,
                tracks_with_regression=int(regression),
                tracks_with_reference=int(reference),
                tracks_with_typical=int(typical),
                total_tracks=len(tracks_df) if tracks_df is not None else 0
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return SizeEstimationResult(
                success=False,
                processing_time_s=time.time() - start_time,
                error_message=str(e)
            )


class VolumeEstimationProcessor:
    """
    Обработчик расчёта осмотренного объёма воды.
    
    Вычисляет объём воды, осмотренный камерой во время погружения,
    и плотность организмов по классам.
    """
    
    def __init__(
        self,
        detections_csv: str,
        output_csv: str,
        tracks_csv: Optional[str] = None,
        ctd_csv: Optional[str] = None,
        fov_horizontal: float = 100.0,
        near_distance: float = 0.3,
        detection_distance: Optional[float] = None,
        depth_min: Optional[float] = None,
        depth_max: Optional[float] = None,
        total_duration: Optional[float] = None,
        fps: float = 60.0,
        frame_width: int = 1920,
        frame_height: int = 1080,
    ):
        self.detections_csv = detections_csv
        self.output_csv = output_csv
        self.tracks_csv = tracks_csv
        self.ctd_csv = ctd_csv
        self.fov_horizontal = fov_horizontal
        self.near_distance = near_distance
        self.detection_distance = detection_distance
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.total_duration = total_duration
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        self._cancelled = False
    
    def cancel(self) -> None:
        self._cancelled = True
    
    def is_cancelled(self) -> bool:
        return self._cancelled
    
    def run(self) -> VolumeEstimationResult:
        """Запускает расчёт объёма."""
        self._cancelled = False
        start_time = time.time()
        
        try:
            from camera_geometry import process_volume_estimation
            
            # Создаём выходную директорию
            os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
            
            # Обработка
            result = process_volume_estimation(
                detections_csv=self.detections_csv,
                tracks_csv=self.tracks_csv,
                ctd_csv=self.ctd_csv,
                output_csv=self.output_csv,
                fov_horizontal=self.fov_horizontal,
                near_distance=self.near_distance,
                detection_distance=self.detection_distance,
                depth_min=self.depth_min,
                depth_max=self.depth_max,
                total_duration=self.total_duration,
                fps=self.fps,
                frame_width=self.frame_width,
                frame_height=self.frame_height,
                verbose=True
            )
            
            processing_time = time.time() - start_time
            
            return VolumeEstimationResult(
                success=True,
                processing_time_s=processing_time,
                output_csv_path=self.output_csv,
                total_volume_m3=result.total_volume_m3,
                depth_min_m=result.depth_range_m[0],
                depth_max_m=result.depth_range_m[1],
                depth_traversed_m=result.depth_traversed_m,
                detection_distance_m=result.detection_distance_m,
                counts_by_class=result.counts_by_class,
                density_by_class=result.density_by_class
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return VolumeEstimationResult(
                success=False,
                processing_time_s=time.time() - start_time,
                error_message=str(e)
            )
