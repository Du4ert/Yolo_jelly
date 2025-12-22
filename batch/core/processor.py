"""
Processor - обёртка над detect_video.py для интеграции с batch processing.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

# Добавляем путь к src для импорта detect_video
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


class Processor:
    """
    Обработчик видео - обёртка над detect_on_video.
    
    Создаёт папку output в директории погружения и сохраняет туда результаты.
    Поддерживает callback для отслеживания прогресса и отмену выполнения.
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
        """
        Инициализация процессора.
        
        Args:
            video_path: Путь к входному видео.
            model_path: Путь к модели YOLO.
            output_dir: Директория для сохранения результатов.
            ctd_path: Путь к файлу CTD (опционально).
            conf_threshold: Порог уверенности.
            enable_tracking: Включить трекинг.
            tracker_type: Тип трекера.
            show_trails: Показывать траектории.
            trail_length: Длина траектории.
            min_track_length: Минимальная длина трека.
            depth_rate: Скорость погружения (если нет CTD).
            save_video: Сохранять видео с разметкой.
        """
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
        
        # Callback для прогресса: (current_frame, total_frames, detections_count)
        self.progress_callback: Optional[Callable[[int, int, int], None]] = None
        
        # Флаг отмены
        self._cancelled = False

    def cancel(self) -> None:
        """Отменяет выполнение."""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        """Проверяет, отменена ли задача."""
        return self._cancelled

    def _generate_output_paths(self) -> Dict[str, str]:
        """Генерирует пути для выходных файлов."""
        video_name = Path(self.video_path).stem
        
        paths = {
            "video": os.path.join(self.output_dir, f"{video_name}_detected.mp4"),
            "csv": os.path.join(self.output_dir, f"{video_name}_detections.csv"),
            "tracks": os.path.join(self.output_dir, f"{video_name}_tracks.csv"),
        }
        return paths

    def run(self) -> ProcessingResult:
        """
        Запускает обработку видео.
        
        Returns:
            ProcessingResult с результатами обработки.
        """
        self._cancelled = False
        start_time = time.time()
        
        # Создаём директорию для результатов
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Генерируем пути
        output_paths = self._generate_output_paths()
        
        try:
            # Импортируем detect_video
            from detect_video import detect_on_video, CLASS_NAMES
            
            # Запускаем детекцию
            # Примечание: detect_on_video не поддерживает отмену,
            # но мы можем добавить эту функциональность позже
            df = detect_on_video(
                video_path=self.video_path,
                model_path=self.model_path,
                output_video=output_paths["video"] if self.save_video else None,
                output_csv=output_paths["csv"],
                conf_threshold=self.conf_threshold,
                depth_rate=self.depth_rate,
                ctd_path=self.ctd_path,
                save_video=self.save_video,
                enable_tracking=self.enable_tracking,
                tracker_type=self.tracker_type,
                show_trails=self.show_trails,
                trail_length=self.trail_length,
                min_track_length=self.min_track_length,
                use_dominant_class=True,
            )
            
            processing_time = time.time() - start_time
            
            # Подсчитываем результаты
            detections_count = len(df)
            tracks_count = 0
            if self.enable_tracking and "track_id" in df.columns:
                tracks_count = df["track_id"].nunique()
            
            # Проверяем, создался ли файл треков
            tracks_path = None
            expected_tracks = Path(output_paths["csv"]).parent / (
                Path(output_paths["csv"]).stem.replace("_detections", "") + "_detections_tracks.csv"
            )
            if expected_tracks.exists():
                tracks_path = str(expected_tracks)
            elif Path(output_paths["tracks"]).exists():
                tracks_path = output_paths["tracks"]
            
            return ProcessingResult(
                success=True,
                detections_count=detections_count,
                tracks_count=tracks_count,
                processing_time_s=processing_time,
                output_video_path=output_paths["video"] if self.save_video and os.path.exists(output_paths["video"]) else None,
                output_csv_path=output_paths["csv"] if os.path.exists(output_paths["csv"]) else None,
                output_tracks_path=tracks_path,
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                success=False,
                processing_time_s=processing_time,
                error_message=str(e),
            )


class ProcessorFactory:
    """
    Фабрика для создания Processor из данных Task.
    """
    
    @staticmethod
    def from_task_data(
        video_path: str,
        model_path: str,
        dive_folder: str,
        ctd_path: Optional[str] = None,
        **task_params,
    ) -> Processor:
        """
        Создаёт Processor из данных задачи.
        
        Args:
            video_path: Путь к видео.
            model_path: Путь к модели.
            dive_folder: Папка погружения (для создания output).
            ctd_path: Путь к CTD файлу.
            **task_params: Параметры из Task.
        
        Returns:
            Настроенный Processor.
        """
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
