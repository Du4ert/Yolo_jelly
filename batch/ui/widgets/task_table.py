"""
Таблица задач - отображение и управление очередью задач.
"""

import os
import subprocess
import platform
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QMenu,
    QMessageBox,
    QHeaderView,
    QGroupBox,
    QProgressBar,
    QAbstractItemView,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QBrush

from ...database import Repository, Task, TaskStatus, VideoFile, Model
from ...core import TaskManager
from ..dialogs import EditTaskDialog


class TaskTable(QWidget):
    """
    Таблица для отображения и управления задачами.
    """

    # Цвета статусов
    STATUS_COLORS = {
        TaskStatus.PENDING: QColor(200, 200, 200),
        TaskStatus.RUNNING: QColor(100, 180, 255),
        TaskStatus.PAUSED: QColor(255, 220, 100),
        TaskStatus.DONE: QColor(100, 220, 100),
        TaskStatus.ERROR: QColor(255, 120, 120),
        TaskStatus.CANCELLED: QColor(180, 180, 180),
    }
    
    STATUS_ICONS = {
        TaskStatus.PENDING: "○",
        TaskStatus.RUNNING: "▶",
        TaskStatus.PAUSED: "⏸",
        TaskStatus.DONE: "✓",
        TaskStatus.ERROR: "✗",
        TaskStatus.CANCELLED: "⊘",
    }

    def __init__(self, repository: Repository, task_manager: TaskManager, parent=None):
        super().__init__(parent)
        self.repo = repository
        self.task_manager = task_manager
        self._setup_ui()
        self._connect_signals()
        self.refresh()

    def _connect_signals(self):
        """Подключает сигналы."""
        self.task_manager.task_progress.connect(self._on_task_progress)
        self.task_manager.queue_changed.connect(self.refresh)

    def _setup_ui(self):
        """Настройка интерфейса."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Группа
        group = QGroupBox("Очередь задач")
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(4, 4, 4, 4)
        group_layout.setSpacing(4)
        
        # Таблица
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "#", "Видео", "Модель", "Статус", "Прогресс", "Результат"
        ])
        
        # Настройка колонок
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        
        self.table.setColumnWidth(0, 40)
        self.table.setColumnWidth(4, 100)
        
        # Настройка поведения
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_context_menu)
        self.table.itemDoubleClicked.connect(self._on_double_click)
        self.table.verticalHeader().setVisible(False)
        
        group_layout.addWidget(self.table)
        
        # Кнопки управления
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)
        
        self.btn_move_up = QPushButton("▲")
        self.btn_move_up.setFixedWidth(30)
        self.btn_move_up.setToolTip("Переместить вверх")
        self.btn_move_up.clicked.connect(self._move_up)
        btn_layout.addWidget(self.btn_move_up)
        
        self.btn_move_down = QPushButton("▼")
        self.btn_move_down.setFixedWidth(30)
        self.btn_move_down.setToolTip("Переместить вниз")
        self.btn_move_down.clicked.connect(self._move_down)
        btn_layout.addWidget(self.btn_move_down)
        
        self.btn_delete = QPushButton("🗑")
        self.btn_delete.setFixedWidth(30)
        self.btn_delete.setToolTip("Удалить задачу")
        self.btn_delete.clicked.connect(self._delete_selected)
        btn_layout.addWidget(self.btn_delete)
        
        btn_layout.addStretch()
        
        self.btn_retry = QPushButton("↻ Повторить")
        self.btn_retry.setToolTip("Повторить задачу с ошибкой")
        self.btn_retry.clicked.connect(self._retry_selected)
        btn_layout.addWidget(self.btn_retry)
        
        group_layout.addLayout(btn_layout)
        
        layout.addWidget(group)

    def refresh(self):
        """Обновляет таблицу."""
        self.table.setRowCount(0)
        
        tasks = self.task_manager.get_all_tasks()
        
        for task in tasks:
            self._add_task_row(task)

    def _add_task_row(self, task: Task):
        """Добавляет строку задачи."""
        row = self.table.rowCount()
        self.table.insertRow(row)
        
        # Получаем связанные данные
        video = self.repo.get_video_file(task.video_id)
        model = self.repo.get_model(task.model_id)
        
        # ID
        id_item = QTableWidgetItem(str(task.id))
        id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        id_item.setData(Qt.ItemDataRole.UserRole, task.id)
        self.table.setItem(row, 0, id_item)
        
        # Видео
        video_name = video.filename if video else "???"
        video_item = QTableWidgetItem(video_name)
        video_item.setToolTip(video.filepath if video else "")
        self.table.setItem(row, 1, video_item)
        
        # Модель
        model_name = model.name if model else "???"
        model_item = QTableWidgetItem(model_name)
        self.table.setItem(row, 2, model_item)
        
        # Статус
        status_icon = self.STATUS_ICONS.get(task.status, "?")
        status_text = f"{status_icon} {task.status.value}"
        status_item = QTableWidgetItem(status_text)
        status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        
        color = self.STATUS_COLORS.get(task.status, QColor(255, 255, 255))
        status_item.setBackground(QBrush(color))
        
        self.table.setItem(row, 3, status_item)
        
        # Прогресс
        progress_item = QTableWidgetItem(f"{task.progress_percent:.0f}%")
        progress_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 4, progress_item)
        
        # Результат
        result_text = ""
        if task.status == TaskStatus.DONE:
            result_text = f"{task.detections_count or 0} дет."
            if task.tracks_count:
                result_text += f" / {task.tracks_count} тр."
        elif task.status == TaskStatus.ERROR:
            result_text = task.error_message or "Ошибка"
        
        result_item = QTableWidgetItem(result_text)
        if task.status == TaskStatus.ERROR:
            result_item.setToolTip(task.error_message or "")
        self.table.setItem(row, 5, result_item)
        
        # Окрашиваем всю строку
        for col in range(self.table.columnCount()):
            item = self.table.item(row, col)
            if item and col != 3:
                if task.status == TaskStatus.DONE:
                    item.setForeground(QBrush(QColor(0, 100, 0)))
                elif task.status == TaskStatus.ERROR:
                    item.setForeground(QBrush(QColor(150, 0, 0)))

    def _on_double_click(self, item):
        """Двойной клик - открыть диалог редактирования."""
        task_id = self._get_selected_task_id()
        if task_id:
            self._edit_task(task_id)

    def _edit_task(self, task_id: int):
        """Открывает диалог редактирования задачи."""
        try:
            dialog = EditTaskDialog(self.repo, task_id, parent=self)
            if dialog.exec():
                self.refresh()
        except ValueError as e:
            QMessageBox.warning(self, "Ошибка", str(e))

    def _get_selected_task_id(self) -> Optional[int]:
        """Возвращает ID выбранной задачи."""
        items = self.table.selectedItems()
        if not items:
            return None
        row = items[0].row()
        id_item = self.table.item(row, 0)
        return id_item.data(Qt.ItemDataRole.UserRole)

    def _on_context_menu(self, position):
        """Контекстное меню."""
        item = self.table.itemAt(position)
        if not item:
            return
        
        task_id = self._get_selected_task_id()
        if not task_id:
            return
        
        task = self.task_manager.get_task(task_id)
        if not task:
            return
        
        menu = QMenu(self)
        
        # Просмотр/редактирование
        action_edit = menu.addAction("✏ Просмотр/редактировать...")
        action_edit.triggered.connect(lambda: self._edit_task(task_id))
        
        menu.addSeparator()
        
        # Открыть результаты
        if task.status == TaskStatus.DONE:
            action_open = menu.addAction("📂 Открыть папку с результатами")
            action_open.triggered.connect(lambda: self._open_output_folder(task_id))
            menu.addSeparator()
        
        # Перемещение (только для pending)
        if task.status == TaskStatus.PENDING:
            action_up = menu.addAction("▲ Переместить вверх")
            action_up.triggered.connect(self._move_up)
            
            action_down = menu.addAction("▼ Переместить вниз")
            action_down.triggered.connect(self._move_down)
            
            menu.addSeparator()
        
        # Повтор (для error/cancelled)
        if task.status in (TaskStatus.ERROR, TaskStatus.CANCELLED):
            action_retry = menu.addAction("↻ Повторить")
            action_retry.triggered.connect(self._retry_selected)
            menu.addSeparator()
        
        # Удаление (кроме running)
        if task.status != TaskStatus.RUNNING:
            action_delete = menu.addAction("🗑 Удалить")
            action_delete.triggered.connect(self._delete_selected)
        
        menu.exec(self.table.viewport().mapToGlobal(position))

    def _move_up(self):
        """Перемещает задачу вверх."""
        task_id = self._get_selected_task_id()
        if task_id:
            self.task_manager.move_task_up(task_id)

    def _move_down(self):
        """Перемещает задачу вниз."""
        task_id = self._get_selected_task_id()
        if task_id:
            self.task_manager.move_task_down(task_id)

    def _delete_selected(self):
        """Удаляет выбранную задачу."""
        task_id = self._get_selected_task_id()
        if not task_id:
            return
        
        task = self.task_manager.get_task(task_id)
        if task and task.status == TaskStatus.RUNNING:
            QMessageBox.warning(
                self,
                "Невозможно удалить",
                "Нельзя удалить выполняющуюся задачу.\nСначала остановите очередь."
            )
            return
        
        if self.task_manager.remove_task(task_id):
            pass

    def _retry_selected(self):
        """Повторяет выбранную задачу."""
        task_id = self._get_selected_task_id()
        if task_id:
            self.task_manager.retry_task(task_id)

    def _open_output_folder(self, task_id: int):
        """Открывает папку с результатами задачи."""
        task = self.task_manager.get_task(task_id)
        if not task:
            return
        
        outputs = self.repo.get_task_outputs(task_id)
        if outputs:
            output_dir = str(Path(outputs[0].filepath).parent)
            self._open_folder(output_dir)
        else:
            video = self.repo.get_video_file(task.video_id)
            if video:
                dive = self.repo.get_dive(video.dive_id)
                if dive:
                    output_dir = os.path.join(dive.folder_path, "output")
                    if os.path.exists(output_dir):
                        self._open_folder(output_dir)

    def _open_folder(self, path: str):
        """Открывает папку в проводнике."""
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])

    def _on_task_progress(self, task_id: int, percent: float, current_frame: int, 
                          total_frames: int, detections: int, tracks: int):
        """Обновляет прогресс задачи в таблице."""
        # Ищем строку с этой задачей
        for row in range(self.table.rowCount()):
            id_item = self.table.item(row, 0)
            if id_item and id_item.data(Qt.ItemDataRole.UserRole) == task_id:
                # Обновляем прогресс
                progress_item = self.table.item(row, 4)
                if progress_item:
                    progress_item.setText(f"{percent:.0f}%")
                
                # Обновляем результат (текущие детекции/треки)
                result_item = self.table.item(row, 5)
                if result_item:
                    result_text = f"{detections} дет."
                    if tracks > 0:
                        result_text += f" / {tracks} тр."
                    result_item.setText(result_text)
                
                break
