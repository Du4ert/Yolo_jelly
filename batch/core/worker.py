"""
Worker - QThread для выполнения задач в фоновом режиме.
"""

from typing import Optional
from PyQt6.QtCore import QThread, pyqtSignal

from ..database import Repository, Task, TaskStatus, OutputType
from .processor import Processor, ProcessorFactory, ProcessingResult


class Worker(QThread):
    """
    Воркер для выполнения задачи детекции в отдельном потоке.
    
    Signals:
        started_task: Задача начала выполняться (task_id).
        progress: Обновление прогресса (task_id, percent, current_frame, total_frames).
        finished_task: Задача завершена (task_id, success, error_message).
        all_finished: Все задачи в очереди выполнены.
    """
    
    # Сигналы
    started_task = pyqtSignal(int)  # task_id
    progress = pyqtSignal(int, float, int, int)  # task_id, percent, current_frame, total_frames
    finished_task = pyqtSignal(int, bool, str)  # task_id, success, error_message
    all_finished = pyqtSignal()

    def __init__(self, repository: Repository, parent=None):
        """
        Инициализация воркера.
        
        Args:
            repository: Репозиторий для работы с БД.
            parent: Родительский QObject.
        """
        super().__init__(parent)
        self.repo = repository
        self._stop_requested = False
        self._pause_requested = False
        self._current_processor: Optional[Processor] = None

    def run(self):
        """Основной цикл воркера."""
        self._stop_requested = False
        
        while not self._stop_requested:
            # Проверяем паузу
            if self._pause_requested:
                self.msleep(100)
                continue
            
            # Получаем следующую задачу
            pending_tasks = self.repo.get_pending_tasks()
            if not pending_tasks:
                break
            
            task = pending_tasks[0]
            self._execute_task(task)
        
        self.all_finished.emit()

    def _execute_task(self, task: Task) -> None:
        """
        Выполняет одну задачу.
        
        Args:
            task: Задача для выполнения.
        """
        task_id = task.id
        
        # Обновляем статус на RUNNING
        self.repo.update_task_status(task_id, TaskStatus.RUNNING)
        self.started_task.emit(task_id)
        
        try:
            # Получаем данные для процессора
            # Нужно заново получить task с загруженными связями
            with self.repo.get_session() as session:
                from ..database import Task as TaskModel, VideoFile, CTDFile, Model, Dive
                from sqlalchemy.orm import joinedload
                from sqlalchemy import select
                
                stmt = (
                    select(TaskModel)
                    .options(
                        joinedload(TaskModel.video_file).joinedload(VideoFile.dive),
                        joinedload(TaskModel.ctd_file),
                        joinedload(TaskModel.model),
                    )
                    .where(TaskModel.id == task_id)
                )
                task_with_relations = session.scalar(stmt)
                
                if not task_with_relations:
                    raise ValueError(f"Task {task_id} not found")
                
                video_path = task_with_relations.video_file.filepath
                model_path = task_with_relations.model.filepath
                dive_folder = task_with_relations.video_file.dive.folder_path
                ctd_path = task_with_relations.ctd_file.filepath if task_with_relations.ctd_file else None
                
                # Параметры задачи
                task_params = {
                    "conf_threshold": task_with_relations.conf_threshold,
                    "enable_tracking": task_with_relations.enable_tracking,
                    "tracker_type": task_with_relations.tracker_type,
                    "show_trails": task_with_relations.show_trails,
                    "trail_length": task_with_relations.trail_length,
                    "min_track_length": task_with_relations.min_track_length,
                    "depth_rate": task_with_relations.depth_rate,
                    "save_video": task_with_relations.save_video,
                }
            
            # Создаём процессор
            processor = ProcessorFactory.from_task_data(
                video_path=video_path,
                model_path=model_path,
                dive_folder=dive_folder,
                ctd_path=ctd_path,
                **task_params,
            )
            self._current_processor = processor
            
            # Запускаем обработку
            result = processor.run()
            
            self._current_processor = None
            
            if self._stop_requested:
                # Задача была отменена
                self.repo.update_task_status(task_id, TaskStatus.CANCELLED)
                self.finished_task.emit(task_id, False, "Cancelled by user")
                return
            
            if result.success:
                # Обновляем задачу с результатами
                self.repo.update_task(
                    task_id,
                    detections_count=result.detections_count,
                    tracks_count=result.tracks_count,
                    processing_time_s=result.processing_time_s,
                    progress_percent=100.0,
                )
                self.repo.update_task_status(task_id, TaskStatus.DONE)
                
                # Сохраняем пути к выходным файлам
                if result.output_video_path:
                    self.repo.add_task_output(task_id, OutputType.VIDEO, result.output_video_path)
                if result.output_csv_path:
                    self.repo.add_task_output(task_id, OutputType.CSV, result.output_csv_path)
                if result.output_tracks_path:
                    self.repo.add_task_output(task_id, OutputType.TRACKS_CSV, result.output_tracks_path)
                
                self.finished_task.emit(task_id, True, "")
            else:
                self.repo.update_task_status(task_id, TaskStatus.ERROR, result.error_message)
                self.finished_task.emit(task_id, False, result.error_message or "Unknown error")
                
        except Exception as e:
            self._current_processor = None
            error_msg = str(e)
            self.repo.update_task_status(task_id, TaskStatus.ERROR, error_msg)
            self.finished_task.emit(task_id, False, error_msg)

    def stop(self) -> None:
        """Останавливает воркер после завершения текущей задачи."""
        self._stop_requested = True
        if self._current_processor:
            self._current_processor.cancel()

    def pause(self) -> None:
        """Приостанавливает воркер."""
        self._pause_requested = True

    def resume(self) -> None:
        """Возобновляет воркер."""
        self._pause_requested = False

    def is_paused(self) -> bool:
        """Возвращает True, если воркер на паузе."""
        return self._pause_requested
