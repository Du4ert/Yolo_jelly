"""
Виджет статусной строки - отображение общей статистики и текущей задачи.
"""

from typing import Optional

from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QLabel,
    QProgressBar,
)
from PyQt6.QtCore import Qt, QTimer

from ...database import Repository, TaskStatus
from ...core import TaskManager


class StatusBarWidget(QWidget):
    """
    Виджет для статусной строки с общей статистикой.
    """

    def __init__(self, repository: Repository, task_manager: TaskManager, parent=None):
        super().__init__(parent)
        self.repo = repository
        self.task_manager = task_manager
        self._current_task_id: Optional[int] = None
        
        self._setup_ui()
        self.update_stats()
        
        # Подключаем сигналы
        self.task_manager.task_progress.connect(self._on_progress)
        self.task_manager.queue_changed.connect(self.update_stats)
        self.task_manager.queue_state_changed.connect(self._on_state_changed)

    def _setup_ui(self):
        """Настройка интерфейса."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(16)
        
        # Статус
        self.label_status = QLabel("Готов")
        layout.addWidget(self.label_status)
        
        layout.addWidget(self._create_separator())
        
        # Статистика задач
        self.label_tasks = QLabel("Задач: 0")
        layout.addWidget(self.label_tasks)
        
        layout.addWidget(self._create_separator())
        
        self.label_pending = QLabel("В очереди: 0")
        layout.addWidget(self.label_pending)
        
        layout.addWidget(self._create_separator())
        
        self.label_done = QLabel("Выполнено: 0")
        layout.addWidget(self.label_done)
        
        layout.addWidget(self._create_separator())
        
        self.label_errors = QLabel("Ошибок: 0")
        layout.addWidget(self.label_errors)
        
        layout.addStretch()
        
        # Информация о текущей задаче
        self.label_current = QLabel("")
        layout.addWidget(self.label_current)
        
        # Детали прогресса
        self.label_details = QLabel("")
        self.label_details.setStyleSheet("color: gray;")
        layout.addWidget(self.label_details)
        
        # Прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

    def _create_separator(self) -> QLabel:
        """Создаёт разделитель."""
        sep = QLabel("|")
        sep.setStyleSheet("color: gray;")
        return sep

    def update_stats(self):
        """Обновляет статистику."""
        stats = self.repo.get_statistics()
        
        self.label_tasks.setText(f"Задач: {stats['tasks_total']}")
        self.label_pending.setText(f"В очереди: {stats['tasks_pending']}")
        self.label_done.setText(f"Выполнено: {stats['tasks_done']}")
        self.label_errors.setText(f"Ошибок: {stats['tasks_error']}")
        
        if stats['tasks_error'] > 0:
            self.label_errors.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.label_errors.setStyleSheet("")

    def _on_state_changed(self, is_running: bool, is_paused: bool):
        """Обновление при изменении состояния очереди."""
        if is_running:
            if is_paused:
                self.label_status.setText("⏸ Пауза")
                self.label_status.setStyleSheet("color: orange; font-weight: bold;")
            else:
                self.label_status.setText("▶ Выполнение")
                self.label_status.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.label_status.setText("Готов")
            self.label_status.setStyleSheet("")
            self.clear_current_task()

    def set_current_task(self, task_id: int):
        """Устанавливает текущую задачу."""
        self._current_task_id = task_id
        
        task = self.task_manager.get_task(task_id)
        if task:
            video = self.repo.get_video_file(task.video_id)
            video_name = video.filename if video else f"Task #{task_id}"
            self.label_current.setText(f"📹 {video_name}")
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            self.label_details.setText("")

    def clear_current_task(self):
        """Очищает текущую задачу."""
        self._current_task_id = None
        self.label_current.setText("")
        self.label_details.setText("")
        self.progress_bar.setVisible(False)
        self.update_stats()

    def _on_progress(self, task_id: int, percent: float, current_frame: int, total_frames: int, detections: int, tracks: int):
        """Обработка обновления прогресса."""
        if task_id == self._current_task_id:
            self.progress_bar.setValue(int(percent))
            
            # Формируем детали
            details = f"{current_frame}/{total_frames} кадров"
            if detections > 0:
                details += f" | {detections} дет."
            if tracks > 0:
                details += f" | {tracks} тр."
            
            self.label_details.setText(details)
