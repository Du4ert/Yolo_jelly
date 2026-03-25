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
    QTreeWidget,
    QTreeWidgetItem,
    QPushButton,
    QMenu,
    QMessageBox,
    QHeaderView,
    QGroupBox,
    QAbstractItemView,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QFont

from ...database import Repository, Task, SubTask, SubTaskType, TaskStatus, VideoFile, Model
from ...core import TaskManager
from ..dialogs import EditTaskDialog, PostProcessDialog


class TaskTable(QWidget):
    """
    Таблица для отображения и управления задачами.
    Использует QTreeWidget для отображения задач и подзадач.
    """

    # Цвета статусов
    STATUS_COLORS = {
        TaskStatus.PENDING: QColor(220, 220, 220),
        TaskStatus.RUNNING: QColor(100, 180, 255),
        TaskStatus.PAUSED: QColor(255, 220, 100),
        TaskStatus.DONE: QColor(120, 200, 120),
        TaskStatus.ERROR: QColor(255, 140, 140),
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
    
    SUBTASK_ICONS = {
        SubTaskType.GEOMETRY: "📐",
        SubTaskType.SIZE: "📏",
        SubTaskType.VOLUME: "📦",
        SubTaskType.ANALYSIS: "📊",
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
        self.task_manager.subtask_progress.connect(self._on_subtask_progress)
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
        
        # Дерево задач
        self.tree = QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(["#", "Задача", "Статус", "Прогресс", "Результат"])
        
        # Настройка колонок
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        
        self.tree.setColumnWidth(0, 50)
        self.tree.setColumnWidth(3, 80)
        
        # Настройка поведения
        self.tree.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        self.tree.setRootIsDecorated(True)
        self.tree.setAnimated(True)
        
        group_layout.addWidget(self.tree)
        
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
        
        self.btn_postprocess = QPushButton("📊 Постобработка")
        self.btn_postprocess.setToolTip("Добавить постобработку к задаче")
        self.btn_postprocess.clicked.connect(self._postprocess_selected)
        btn_layout.addWidget(self.btn_postprocess)
        
        self.btn_retry = QPushButton("↻ Повторить")
        self.btn_retry.setToolTip("Повторить задачу с ошибкой")
        self.btn_retry.clicked.connect(self._retry_selected)
        btn_layout.addWidget(self.btn_retry)
        
        group_layout.addLayout(btn_layout)
        
        layout.addWidget(group)

    def refresh(self):
        """Обновляет дерево задач."""
        # Сохраняем состояние раскрытия
        expanded_tasks = set()
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item and item.isExpanded():
                task_id = item.data(0, Qt.ItemDataRole.UserRole)
                if task_id:
                    expanded_tasks.add(task_id)
        
        self.tree.clear()
        
        tasks = self.task_manager.get_all_tasks()
        
        for task in tasks:
            item = self._create_task_item(task)
            self.tree.addTopLevelItem(item)
            
            # Добавляем подзадачи
            subtasks = self.repo.get_subtasks_for_task(task.id)
            for subtask in subtasks:
                sub_item = self._create_subtask_item(subtask)
                item.addChild(sub_item)
            
            # Восстанавливаем раскрытие
            if task.id in expanded_tasks or len(subtasks) > 0:
                item.setExpanded(True)

    def _create_task_item(self, task: Task) -> QTreeWidgetItem:
        """Создаёт элемент дерева для задачи."""
        video = self.repo.get_video_file(task.video_id)
        model = self.repo.get_model(task.model_id)
        
        item = QTreeWidgetItem()
        
        # ID
        item.setText(0, str(task.id))
        item.setData(0, Qt.ItemDataRole.UserRole, task.id)
        item.setData(0, Qt.ItemDataRole.UserRole + 1, "task")
        item.setTextAlignment(0, Qt.AlignmentFlag.AlignCenter)
        
        # Видео + модель
        model_name = model.name if model else "???"
        if video:
            dive = self.repo.get_dive(video.dive_id)
            dive_name = dive.name if dive else Path(video.filepath).parent.name
            video_label = f"{dive_name} / {video.filename}"
        else:
            video_label = "???"
        item.setText(1, f"🎬 {video_label}")
        item.setToolTip(1, f"Видео: {video.filepath if video else '???'}\nМодель: {model_name}")
        
        # Статус
        status_icon = self.STATUS_ICONS.get(task.status, "?")
        item.setText(2, f"{status_icon} {task.status.value}")
        item.setTextAlignment(2, Qt.AlignmentFlag.AlignCenter)
        
        color = self.STATUS_COLORS.get(task.status, QColor(255, 255, 255))
        item.setBackground(2, QBrush(color))
        
        # Прогресс
        item.setText(3, f"{task.progress_percent:.0f}%")
        item.setTextAlignment(3, Qt.AlignmentFlag.AlignCenter)
        
        # Результат
        result_text = ""
        if task.status == TaskStatus.DONE:
            result_text = f"{task.detections_count or 0} дет."
            if task.tracks_count:
                result_text += f" / {task.tracks_count} тр."
        elif task.status == TaskStatus.ERROR:
            result_text = task.error_message[:30] + "..." if task.error_message and len(task.error_message) > 30 else (task.error_message or "Ошибка")
            item.setToolTip(4, task.error_message or "")
        
        item.setText(4, result_text)
        
        # Стиль текста
        if task.status == TaskStatus.DONE:
            for col in range(5):
                item.setForeground(col, QBrush(QColor(0, 100, 0)))
        elif task.status == TaskStatus.ERROR:
            for col in range(5):
                item.setForeground(col, QBrush(QColor(150, 0, 0)))
        
        # Жирный шрифт для основных задач
        font = item.font(1)
        font.setBold(True)
        item.setFont(1, font)
        
        return item

    def _create_subtask_item(self, subtask: SubTask) -> QTreeWidgetItem:
        """Создаёт элемент дерева для подзадачи."""
        item = QTreeWidgetItem()
        
        # ID подзадачи (пустой для ID)
        item.setText(0, "")
        item.setData(0, Qt.ItemDataRole.UserRole, subtask.id)
        item.setData(0, Qt.ItemDataRole.UserRole + 1, "subtask")
        item.setData(0, Qt.ItemDataRole.UserRole + 2, subtask.parent_task_id)
        
        # Название с иконкой
        icon = self.SUBTASK_ICONS.get(subtask.subtask_type, "?")
        item.setText(1, f"  {icon} {subtask.type_name}")
        
        # Статус
        status_icon = self.STATUS_ICONS.get(subtask.status, "?")
        item.setText(2, f"{status_icon} {subtask.status.value}")
        item.setTextAlignment(2, Qt.AlignmentFlag.AlignCenter)
        
        color = self.STATUS_COLORS.get(subtask.status, QColor(255, 255, 255))
        item.setBackground(2, QBrush(color))
        
        # Прогресс
        item.setText(3, f"{subtask.progress_percent:.0f}%")
        item.setTextAlignment(3, Qt.AlignmentFlag.AlignCenter)
        
        # Результат
        result_text = ""
        if subtask.status == TaskStatus.DONE:
            if subtask.result_text:
                result_text = subtask.result_text
            elif subtask.result_value is not None:
                if subtask.subtask_type == SubTaskType.GEOMETRY:
                    result_text = f"{subtask.result_value:.1f}°"
                elif subtask.subtask_type == SubTaskType.VOLUME:
                    result_text = f"{subtask.result_value:.2f} м³"
                else:
                    result_text = f"{subtask.result_value:.0f}"
        elif subtask.status == TaskStatus.ERROR:
            result_text = subtask.error_message[:25] + "..." if subtask.error_message and len(subtask.error_message) > 25 else (subtask.error_message or "Ошибка")
            item.setToolTip(4, subtask.error_message or "")
        
        item.setText(4, result_text)
        
        # Стиль для подзадач - чуть светлее
        if subtask.status == TaskStatus.DONE:
            for col in range(5):
                item.setForeground(col, QBrush(QColor(60, 130, 60)))
        elif subtask.status == TaskStatus.ERROR:
            for col in range(5):
                item.setForeground(col, QBrush(QColor(180, 60, 60)))
        else:
            for col in range(5):
                item.setForeground(col, QBrush(QColor(80, 80, 80)))
        
        return item

    def _on_double_click(self, item: QTreeWidgetItem, column: int):
        """Двойной клик."""
        item_type = item.data(0, Qt.ItemDataRole.UserRole + 1)
        
        if item_type == "task":
            task_id = item.data(0, Qt.ItemDataRole.UserRole)
            task = self.task_manager.get_task(task_id)
            if task and task.status == TaskStatus.DONE:
                self._postprocess_task(task_id)
            else:
                self._edit_task(task_id)

    def _edit_task(self, task_id: int):
        """Открывает диалог редактирования задачи."""
        try:
            dialog = EditTaskDialog(self.repo, task_id, parent=self)
            if dialog.exec():
                self.refresh()
        except ValueError as e:
            QMessageBox.warning(self, "Ошибка", str(e))

    def _postprocess_task(self, task_id: int):
        """Открывает диалог постобработки."""
        try:
            dialog = PostProcessDialog(self.repo, self.task_manager, task_id, parent=self)
            dialog.exec()
            self.refresh()
        except ValueError as e:
            QMessageBox.warning(self, "Ошибка", str(e))

    def _postprocess_selected(self):
        """Открывает постобработку для выбранной задачи."""
        item = self.tree.currentItem()
        if not item:
            QMessageBox.information(self, "Не выбрано", "Выберите задачу")
            return
        
        # Если выбрана подзадача, берём родительскую задачу
        item_type = item.data(0, Qt.ItemDataRole.UserRole + 1)
        if item_type == "subtask":
            task_id = item.data(0, Qt.ItemDataRole.UserRole + 2)
        else:
            task_id = item.data(0, Qt.ItemDataRole.UserRole)
        
        if not task_id:
            return
        
        task = self.task_manager.get_task(task_id)
        if not task or task.status != TaskStatus.DONE:
            QMessageBox.warning(
                self, "Недоступно",
                "Постобработка доступна только для завершённых задач детекции."
            )
            return
        
        self._postprocess_task(task_id)

    def _get_selected_task_id(self) -> Optional[int]:
        """Возвращает ID выбранной задачи."""
        item = self.tree.currentItem()
        if not item:
            return None
        
        item_type = item.data(0, Qt.ItemDataRole.UserRole + 1)
        if item_type == "subtask":
            return item.data(0, Qt.ItemDataRole.UserRole + 2)
        return item.data(0, Qt.ItemDataRole.UserRole)

    def _on_context_menu(self, position):
        """Контекстное меню."""
        item = self.tree.itemAt(position)
        if not item:
            return
        
        item_type = item.data(0, Qt.ItemDataRole.UserRole + 1)
        
        if item_type == "subtask":
            self._show_subtask_context_menu(item, position)
        else:
            self._show_task_context_menu(item, position)

    def _show_task_context_menu(self, item: QTreeWidgetItem, position):
        """Контекстное меню для задачи."""
        task_id = item.data(0, Qt.ItemDataRole.UserRole)
        task = self.task_manager.get_task(task_id)
        if not task:
            return
        
        menu = QMenu(self)
        
        # Постобработка (для завершённых)
        if task.status == TaskStatus.DONE:
            action_postprocess = menu.addAction("📊 Добавить постобработку...")
            action_postprocess.triggered.connect(lambda: self._postprocess_task(task_id))
            menu.addSeparator()
        
        # Редактирование
        action_edit = menu.addAction("✏ Редактировать...")
        action_edit.triggered.connect(lambda: self._edit_task(task_id))
        
        menu.addSeparator()
        
        # Открыть результаты
        if task.status == TaskStatus.DONE:
            action_open = menu.addAction("📂 Открыть папку с результатами")
            action_open.triggered.connect(lambda: self._open_output_folder(task_id))
            report_path = self._find_report_path(task_id)
            if report_path:
                action_report = menu.addAction("📄 Открыть отчёт")
                action_report.triggered.connect(lambda: self._open_file(report_path))
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
        
        menu.exec(self.tree.viewport().mapToGlobal(position))

    def _show_subtask_context_menu(self, item: QTreeWidgetItem, position):
        """Контекстное меню для подзадачи."""
        subtask_id = item.data(0, Qt.ItemDataRole.UserRole)
        subtask = self.repo.get_subtask(subtask_id)
        if not subtask:
            return
        
        menu = QMenu(self)
        
        # Повтор (для error/cancelled)
        if subtask.status in (TaskStatus.ERROR, TaskStatus.CANCELLED):
            action_retry = menu.addAction("↻ Повторить")
            action_retry.triggered.connect(lambda: self._retry_subtask(subtask_id))
            menu.addSeparator()
        
        # Удаление (кроме running)
        if subtask.status != TaskStatus.RUNNING:
            action_delete = menu.addAction("🗑 Удалить подзадачу")
            action_delete.triggered.connect(lambda: self._delete_subtask(subtask_id))
        
        menu.exec(self.tree.viewport().mapToGlobal(position))

    def _retry_subtask(self, subtask_id: int):
        """Повторяет подзадачу."""
        self.repo.update_subtask_status(subtask_id, TaskStatus.PENDING)
        self.refresh()

    def _delete_subtask(self, subtask_id: int):
        """Удаляет подзадачу."""
        subtask = self.repo.get_subtask(subtask_id)
        if subtask and subtask.status == TaskStatus.RUNNING:
            QMessageBox.warning(self, "Невозможно удалить", "Подзадача выполняется")
            return
        
        self.repo.delete_subtask(subtask_id)
        self.refresh()

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
        
        self.task_manager.remove_task(task_id)

    def _retry_selected(self):
        """Повторяет выбранную задачу."""
        task_id = self._get_selected_task_id()
        if task_id:
            self.task_manager.retry_task(task_id)

    def _find_report_path(self, task_id: int) -> Optional[str]:
        """Возвращает путь к report.txt задачи, если он существует."""
        from ...database import OutputType
        outputs = self.repo.get_task_outputs(task_id)
        for out in outputs:
            if out.output_type == OutputType.ANALYSIS_REPORT and os.path.exists(out.filepath):
                return out.filepath
        return None

    def _open_file(self, path: str):
        """Открывает файл в системном приложении по умолчанию."""
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])

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
        # Ищем элемент с этой задачей
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item and item.data(0, Qt.ItemDataRole.UserRole) == task_id:
                item.setText(3, f"{percent:.0f}%")
                result_text = f"{detections} дет."
                if tracks > 0:
                    result_text += f" / {tracks} тр."
                item.setText(4, result_text)
                break

    def _on_subtask_progress(self, subtask_id: int, percent: float):
        """Обновляет прогресс подзадачи."""
        # Ищем подзадачу в дереве
        for i in range(self.tree.topLevelItemCount()):
            task_item = self.tree.topLevelItem(i)
            if task_item:
                for j in range(task_item.childCount()):
                    sub_item = task_item.child(j)
                    if sub_item and sub_item.data(0, Qt.ItemDataRole.UserRole) == subtask_id:
                        sub_item.setText(3, f"{percent:.0f}%")
                        return
