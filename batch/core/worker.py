"""
Worker - QThread для выполнения задач в фоновом режиме.
"""

from typing import Optional
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition

from ..database import Repository, Task, TaskStatus, OutputType
from .processor import Processor, ProcessorFactory, ProcessingResult


class Worker(QThread):
    """
    Воркер для выполнения задачи детекции в отдельном потоке.
    
    Signals:
        started_task: Задача начала выполняться (task_id).
        progress: Обновление прогресса (task_id, percent, current_frame, total_frames, detections, tracks).
        finished_task: Задача завершена (task_id, success, error_message).
        all_finished: Все задачи в очереди выполнены.
    """
    
    started_task = pyqtSignal(int)
    progress = pyqtSignal(int, float, int, int, int, int)  # task_id, percent, frame, total, detections, tracks
    finished_task = pyqtSignal(int, bool, str)
    all_finished = pyqtSignal()

    def __init__(self, repository: Repository, parent=None):
        super().__init__(parent)
        self.repo = repository
        self._stop_requested = False
        self._pause_requested = False
        self._current_processor: Optional[Processor] = None
        self._current_task_id: Optional[int] = None
        
        # Для синхронизации паузы
        self._mutex = QMutex()
        self._pause_condition = QWaitCondition()

    def run(self):
        """Основной цикл воркера."""
        self._stop_requested = False
        
        while not self._stop_requested:
            # Проверяем паузу
            self._mutex.lock()
            while self._pause_requested and not self._stop_requested:
                self._pause_condition.wait(self._mutex)
            self._mutex.unlock()
            
            if self._stop_requested:
                break
            
            # Получаем следующую задачу
            pending_tasks = self.repo.get_pending_tasks()
            if not pending_tasks:
                break
            
            task = pending_tasks[0]
            self._execute_task(task)
        
        self.all_finished.emit()

    def _execute_task(self, task: Task) -> None:
        """Выполняет одну задачу."""
        task_id = task.id
        self._current_task_id = task_id
        
        # Обновляем статус
        self.repo.update_task_status(task_id, TaskStatus.RUNNING)
        self.started_task.emit(task_id)
        
        try:
            # Получаем данные
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
                task_data = session.scalar(stmt)
                
                if not task_data:
                    raise ValueError(f"Task {task_id} not found")
                
                video_path = task_data.video_file.filepath
                model_path = task_data.model.filepath
                dive_folder = task_data.video_file.dive.folder_path
                ctd_path = task_data.ctd_file.filepath if task_data.ctd_file else None
                
                task_params = {
                    "conf_threshold": task_data.conf_threshold,
                    "enable_tracking": task_data.enable_tracking,
                    "tracker_type": task_data.tracker_type,
                    "show_trails": task_data.show_trails,
                    "trail_length": task_data.trail_length,
                    "min_track_length": task_data.min_track_length,
                    "depth_rate": task_data.depth_rate,
                    "save_video": task_data.save_video,
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
            
            # Устанавливаем callback прогресса
            def on_progress(current_frame: int, total_frames: int, detections: int, tracks: int):
                if total_frames > 0:
                    percent = (current_frame / total_frames) * 100
                else:
                    percent = 0
                
                self.progress.emit(task_id, percent, current_frame, total_frames, detections, tracks)
                self.repo.update_task_progress(task_id, percent, current_frame)
                
                # Проверяем паузу
                if self._pause_requested:
                    processor.pause()
                else:
                    processor.resume()
                
                # Проверяем отмену
                if self._stop_requested:
                    processor.cancel()
            
            processor.progress_callback = on_progress
            
            # Запускаем
            result = processor.run()
            
            self._current_processor = None
            self._current_task_id = None
            
            if result.cancelled or self._stop_requested:
                self.repo.update_task_status(task_id, TaskStatus.CANCELLED, "Отменено пользователем")
                self.finished_task.emit(task_id, False, "Отменено пользователем")
                return
            
            if result.success:
                self.repo.update_task(
                    task_id,
                    detections_count=result.detections_count,
                    tracks_count=result.tracks_count,
                    processing_time_s=result.processing_time_s,
                    progress_percent=100.0,
                )
                self.repo.update_task_status(task_id, TaskStatus.DONE)
                
                # Сохраняем выходные файлы
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
            import traceback
            traceback.print_exc()
            self._current_processor = None
            self._current_task_id = None
            error_msg = str(e)
            self.repo.update_task_status(task_id, TaskStatus.ERROR, error_msg)
            self.finished_task.emit(task_id, False, error_msg)

    def stop(self) -> None:
        """Останавливает воркер."""
        self._stop_requested = True
        
        # Снимаем с паузы, чтобы поток мог завершиться
        self._mutex.lock()
        self._pause_requested = False
        self._pause_condition.wakeAll()
        self._mutex.unlock()
        
        # Отменяем текущий процессор
        if self._current_processor:
            self._current_processor.cancel()

    def pause(self) -> None:
        """Приостанавливает воркер."""
        self._mutex.lock()
        self._pause_requested = True
        self._mutex.unlock()
        
        if self._current_processor:
            self._current_processor.pause()

    def resume(self) -> None:
        """Возобновляет воркер."""
        self._mutex.lock()
        self._pause_requested = False
        self._pause_condition.wakeAll()
        self._mutex.unlock()
        
        if self._current_processor:
            self._current_processor.resume()

    def is_paused(self) -> bool:
        return self._pause_requested
    
    def get_current_task_id(self) -> Optional[int]:
        return self._current_task_id
