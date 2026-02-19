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
    tracks_with_k_method: int = 0
    tracks_with_fixed: int = 0
    tracks_with_typical: int = 0
    total_tracks: int = 0
    tilt_correction_applied: bool = False
    error_message: Optional[str] = None
    cancelled: bool = False


@dataclass
class SizeVideoRenderResult:
    """Результат рендеринга видео с размерами."""
    success: bool
    processing_time_s: float = 0.0
    output_video_path: Optional[str] = None
    frames_processed: int = 0
    detections_rendered: int = 0
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
    
    def __init__(self):
        self._cancelled = False
    
    def cancel(self) -> None:
        """Отменяет выполнение."""
        self._cancelled = True
    
    def is_cancelled(self) -> bool:
        return self._cancelled
    
    def process(
        self,
        video_path: str,
        output_csv: str,
        frame_interval: int = 30,
        frame_width: int = 3840,
        frame_height: int = 2160,
    ) -> GeometryResult:
        """
        Запускает анализ геометрии камеры.
        
        Args:
            video_path: Путь к видеофайлу.
            output_csv: Путь к выходному CSV.
            frame_interval: Интервал между анализируемыми кадрами.
            frame_width: Ширина кадра.
            frame_height: Высота кадра.
            
        Returns:
            GeometryResult с результатами анализа.
        """
        self._cancelled = False
        start_time = time.time()
        
        try:
            from camera_geometry import (
                process_video_geometry,
                CameraCalibration
            )
            
            # Создаём выходную директорию
            os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
            
            # Калибровка
            calibration = CameraCalibration()
            calibration.frame_width = frame_width
            calibration.frame_height = frame_height
            
            # Обработка видео
            df = process_video_geometry(
                video_path=video_path,
                output_csv=output_csv,
                frame_interval=frame_interval,
                calibration=calibration,
                verbose=True
            )
            
            processing_time = time.time() - start_time
            
            if df is None or len(df) == 0:
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
                output_csv_path=output_csv,
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
    
    def __init__(self):
        self._cancelled = False
    
    def cancel(self) -> None:
        self._cancelled = True
    
    def is_cancelled(self) -> bool:
        return self._cancelled
    
    def process(
        self,
        detections_csv: str,
        output_csv: str,
        tracks_csv: Optional[str] = None,
        geometry_csv: Optional[str] = None,
        frame_width: int = 3840,
        frame_height: int = 2160,
        min_depth_change: float = 0.3,
        min_track_points: int = 3,
        min_r_squared: float = 0.5,
        min_size_change_ratio: float = 0.3,
        apply_tilt_correction: bool = True,
    ) -> SizeEstimationResult:
        """
        Запускает оценку размеров.
        
        Args:
            detections_csv: Путь к CSV с детекциями.
            output_csv: Путь к выходному CSV с размерами.
            tracks_csv: Путь к выходному CSV со статистикой треков.
            geometry_csv: Путь к CSV с геометрией камеры (опционально).
            frame_width: Ширина кадра.
            frame_height: Высота кадра.
            min_depth_change: Минимальное изменение глубины для регрессии.
            min_track_points: Минимальное количество точек в треке.
            min_r_squared: Минимальный R² для принятия регрессии.
            min_size_change_ratio: Минимальное относительное изменение размера.
            
        Returns:
            SizeEstimationResult с результатами.
        """
        self._cancelled = False
        start_time = time.time()
        
        try:
            from camera_geometry import (
                process_detections_with_size,
                CameraCalibration
            )
            
            # Создаём выходные директории
            os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
            if tracks_csv:
                os.makedirs(os.path.dirname(os.path.abspath(tracks_csv)), exist_ok=True)
            
            # Калибровка
            calibration = CameraCalibration()
            calibration.frame_width = frame_width
            calibration.frame_height = frame_height
            
            # Обработка
            df, tracks_df = process_detections_with_size(
                detections_csv=detections_csv,
                output_csv=output_csv,
                tracks_output_csv=tracks_csv,
                geometry_csv=geometry_csv,
                calibration=calibration,
                frame_width=frame_width,
                frame_height=frame_height,
                min_depth_change=min_depth_change,
                min_track_points=min_track_points,
                apply_tilt_correction=apply_tilt_correction,
                verbose=True
            )
            
            processing_time = time.time() - start_time
            
            # Статистика по методам
            k_method = 0
            fixed = 0
            typical = 0
            tilt_applied = False
            
            if tracks_df is not None and len(tracks_df) > 0:
                if 'method' in tracks_df.columns:
                    method_counts = tracks_df['method'].value_counts()
                    k_method = method_counts.get('k_method', 0)
                    fixed = method_counts.get('fixed', 0)
                    typical = method_counts.get('typical', 0)
                
                # Проверяем, была ли применена коррекция наклона
                if 'warnings' in tracks_df.columns:
                    tilt_warnings = tracks_df['warnings'].str.contains('tilt_corrected', na=False)
                    tilt_applied = tilt_warnings.any()
            
            return SizeEstimationResult(
                success=True,
                processing_time_s=processing_time,
                output_csv_path=output_csv,
                tracks_csv_path=tracks_csv,
                tracks_with_k_method=int(k_method),
                tracks_with_fixed=int(fixed),
                tracks_with_typical=int(typical),
                total_tracks=len(tracks_df) if tracks_df is not None else 0,
                tilt_correction_applied=tilt_applied
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
    
    def __init__(self):
        self._cancelled = False
    
    def cancel(self) -> None:
        self._cancelled = True
    
    def is_cancelled(self) -> bool:
        return self._cancelled
    
    def process(
        self,
        detections_csv: str,
        output_csv: str,
        tracks_csv: Optional[str] = None,
        ctd_csv: Optional[str] = None,
        fov: float = 156.0,
        near_distance: float = 0.3,
        detection_distance: Optional[float] = None,
        depth_min: Optional[float] = None,
        depth_max: Optional[float] = None,
        duration: Optional[float] = None,
        fps: float = 60.0,
        frame_width: int = 3840,
        frame_height: int = 2160,
    ) -> VolumeEstimationResult:
        """
        Запускает расчёт объёма.
        
        Args:
            detections_csv: Путь к CSV с детекциями.
            output_csv: Путь к выходному CSV.
            tracks_csv: Путь к CSV со статистикой треков (опционально).
            ctd_csv: Путь к CSV с данными CTD (опционально).
            fov: Горизонтальный угол обзора камеры (градусы).
            near_distance: Ближняя граница обнаружения (метры).
            detection_distance: Дистанция обнаружения (метры), None = авто.
            depth_min: Минимальная глубина (метры).
            depth_max: Максимальная глубина (метры).
            duration: Длительность записи (секунды).
            fps: Частота кадров.
            frame_width: Ширина кадра.
            frame_height: Высота кадра.
            
        Returns:
            VolumeEstimationResult с результатами.
        """
        self._cancelled = False
        start_time = time.time()
        
        try:
            from camera_geometry import process_volume_estimation
            
            # Создаём выходную директорию
            os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
            
            # Обработка
            result = process_volume_estimation(
                detections_csv=detections_csv,
                tracks_csv=tracks_csv,
                ctd_csv=ctd_csv,
                output_csv=output_csv,
                fov_horizontal=fov,
                near_distance=near_distance,
                detection_distance=detection_distance,
                depth_min=depth_min,
                depth_max=depth_max,
                total_duration=duration,
                fps=fps,
                frame_width=frame_width,
                frame_height=frame_height,
                verbose=True
            )
            
            processing_time = time.time() - start_time
            
            return VolumeEstimationResult(
                success=True,
                processing_time_s=processing_time,
                output_csv_path=output_csv,
                total_volume_m3=result.total_volume_m3,
                depth_min_m=result.depth_range_m[0] if result.depth_range_m else None,
                depth_max_m=result.depth_range_m[1] if result.depth_range_m else None,
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


class SizeVideoRenderProcessor:
    """
    Обработчик рендеринга видео с информацией о размерах.
    
    Добавляет на видео с детекциями:
    - Дистанцию до объекта и размер под рамками
    - Углы наклона камеры в левом нижнем углу
    """
    
    def __init__(self):
        self._cancelled = False
    
    def cancel(self) -> None:
        self._cancelled = True
    
    def is_cancelled(self) -> bool:
        return self._cancelled
    
    def process(
        self,
        input_video: str,
        detections_csv: str,
        size_csv: str,
        output_video: str,
        geometry_csv: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> SizeVideoRenderResult:
        """
        Запускает рендеринг видео с размерами.
        
        Args:
            input_video: Путь к видео с детекциями (detected.mp4).
            detections_csv: Путь к CSV с базовыми детекциями.
            size_csv: Путь к CSV с размерами (detections_with_size.csv).
            output_video: Путь к выходному видео.
            geometry_csv: Путь к CSV с геометрией (опционально).
            progress_callback: Коллбэк прогресса (current_frame, total_frames).
            
        Returns:
            SizeVideoRenderResult с результатами.
        """
        self._cancelled = False
        start_time = time.time()
        
        try:
            from render_size_video import render_size_video
            
            # Создаём выходную директорию
            os.makedirs(os.path.dirname(os.path.abspath(output_video)), exist_ok=True)
            
            # Запускаем рендеринг
            output_path = render_size_video(
                input_video=input_video,
                detections_csv=detections_csv,
                size_csv=size_csv,
                geometry_csv=geometry_csv,
                output_video=output_video,
                verbose=True,
                progress_callback=progress_callback,
            )
            
            processing_time = time.time() - start_time
            
            return SizeVideoRenderResult(
                success=True,
                processing_time_s=processing_time,
                output_video_path=output_path
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return SizeVideoRenderResult(
                success=False,
                processing_time_s=time.time() - start_time,
                error_message=str(e)
            )
