"""
TaskManager - управление очередью задач.
"""

from typing import Optional, List, Callable
from PyQt6.QtCore import QObject, pyqtSignal

from ..database import Repository, Task, TaskStatus
from .worker import Worker


class TaskManager(QObject):
    """
    Менеджер задач - управляет очередью и выполнением.
    
    Signals:
        queue_changed: Очередь изменилась (нужно обновить UI).
        task_started: Задача начала выполняться (task_id).
        task_progress: Прогресс задачи (task_id, percent).
        task_finished: Задача завершена (task_id, success).
        queue_finished: Все задачи выполнены.
        queue_state_changed: Состояние очереди изменилось (is_running, is_paused).
    """
    
    # Сигналы
    queue_changed = pyqtSignal()
    task_started = pyqtSignal(int)  # task_id
    task_progress = pyqtSignal(int, float)  # task_id, percent
    task_finished = pyqtSignal(int, bool, str)  # task_id, success, error_message
    queue_finished = pyqtSignal()
    queue_state_changed = pyqtSignal(bool, bool)  # is_running, is_paused

    def __init__(self, repository: Repository, parent=None):
        """
        Инициализация менеджера задач.
        
        Args:
            repository: Репозиторий для работы с БД.
            parent: Родительский QObject.
        """
        super().__init__(parent)
        self.repo = repository
        self._worker: Optional[Worker] = None
        self._is_running = False
        self._is_paused = False

    # ========== CRUD операции ==========

    def add_task(
        self,
        video_id: int,
        model_id: int,
        ctd_id: Optional[int] = None,
        **params,
    ) -> Optional[Task]:
        """
        Добавляет задачу в очередь.
        
        Args:
            video_id: ID видеофайла.
            model_id: ID модели.
            ctd_id: ID файла CTD (опционально).
            **params: Параметры детекции.
        
        Returns:
            Созданная задача или None.
        """
        task = self.repo.create_task(video_id, model_id, ctd_id, **params)
        if task:
            self.queue_changed.emit()
        return task

    def remove_task(self, task_id: int) -> bool:
        """
        Удаляет задачу из очереди.
        
        Args:
            task_id: ID задачи.
        
        Returns:
            True, если удалено успешно.
        """
        task = self.repo.get_task(task_id)
        if not task:
            return False
        
        # Нельзя удалить выполняющуюся задачу
        if task.status == TaskStatus.RUNNING:
            return False
        
        result = self.repo.delete_task(task_id)
        if result:
            self.queue_changed.emit()
        return result

    def move_task_up(self, task_id: int) -> bool:
        """Перемещает задачу вверх в очереди."""
        task = self.repo.get_task(task_id)
        if not task or task.position <= 1:
            return False
        
        result = self.repo.move_task(task_id, task.position - 1)
        if result:
            self.queue_changed.emit()
        return result

    def move_task_down(self, task_id: int) -> bool:
        """Перемещает задачу вниз в очереди."""
        task = self.repo.get_task(task_id)
        if not task:
            return False
        
        result = self.repo.move_task(task_id, task.position + 1)
        if result:
            self.queue_changed.emit()
        return result

    def retry_task(self, task_id: int) -> bool:
        """
        Повторяет задачу с ошибкой.
        
        Args:
            task_id: ID задачи.
        
        Returns:
            True, если задача поставлена на повтор.
        """
        task = self.repo.get_task(task_id)
        if not task or task.status not in (TaskStatus.ERROR, TaskStatus.CANCELLED):
            return False
        
        self.repo.update_task(
            task_id,
            status=TaskStatus.PENDING,
            progress_percent=0.0,
            current_frame=0,
            error_message=None,
            started_at=None,
            completed_at=None,
            detections_count=None,
            tracks_count=None,
            processing_time_s=None,
        )
        self.queue_changed.emit()
        return True

    # ========== Получение данных ==========

    def get_all_tasks(self) -> List[Task]:
        """Возвращает все задачи."""
        return self.repo.get_all_tasks()

    def get_pending_tasks(self) -> List[Task]:
        """Возвращает задачи в ожидании."""
        return self.repo.get_pending_tasks()

    def get_task(self, task_id: int) -> Optional[Task]:
        """Возвращает задачу по ID."""
        return self.repo.get_task(task_id)

    def get_current_task(self) -> Optional[Task]:
        """Возвращает текущую выполняющуюся задачу."""
        tasks = self.repo.get_tasks_by_status(TaskStatus.RUNNING)
        return tasks[0] if tasks else None

    # ========== Управление очередью ==========

    def start_queue(self) -> bool:
        """
        Запускает выполнение очереди.
        
        Returns:
            True, если очередь запущена.
        """
        if self._is_running:
            return False
        
        pending = self.get_pending_tasks()
        if not pending:
            return False
        
        self._worker = Worker(self.repo)
        self._worker.started_task.connect(self._on_task_started)
        self._worker.progress.connect(self._on_task_progress)
        self._worker.finished_task.connect(self._on_task_finished)
        self._worker.all_finished.connect(self._on_queue_finished)
        
        self._is_running = True
        self._is_paused = False
        self._worker.start()
        
        self.queue_state_changed.emit(True, False)
        return True

    def stop_queue(self) -> None:
        """Останавливает выполнение очереди."""
        if self._worker and self._is_running:
            self._worker.stop()

    def pause_queue(self) -> None:
        """Приостанавливает выполнение очереди."""
        if self._worker and self._is_running and not self._is_paused:
            self._worker.pause()
            self._is_paused = True
            self.queue_state_changed.emit(True, True)

    def resume_queue(self) -> None:
        """Возобновляет выполнение очереди."""
        if self._worker and self._is_running and self._is_paused:
            self._worker.resume()
            self._is_paused = False
            self.queue_state_changed.emit(True, False)

    def is_running(self) -> bool:
        """Возвращает True, если очередь выполняется."""
        return self._is_running

    def is_paused(self) -> bool:
        """Возвращает True, если очередь на паузе."""
        return self._is_paused

    # ========== Обработчики сигналов воркера ==========

    def _on_task_started(self, task_id: int) -> None:
        """Обработчик начала задачи."""
        self.task_started.emit(task_id)
        self.queue_changed.emit()

    def _on_task_progress(self, task_id: int, percent: float, current_frame: int, total_frames: int) -> None:
        """Обработчик прогресса задачи."""
        self.repo.update_task_progress(task_id, percent, current_frame)
        self.task_progress.emit(task_id, percent)

    def _on_task_finished(self, task_id: int, success: bool, error_message: str) -> None:
        """Обработчик завершения задачи."""
        self.task_finished.emit(task_id, success, error_message)
        self.queue_changed.emit()

    def _on_queue_finished(self) -> None:
        """Обработчик завершения всех задач."""
        self._is_running = False
        self._is_paused = False
        self._worker = None
        self.queue_finished.emit()
        self.queue_state_changed.emit(False, False)
