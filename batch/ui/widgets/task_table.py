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
import json

from PyQt6.QtCore import Qt, QModelIndex, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QFont, QKeySequence, QShortcut

from ...database import Repository, Task, SubTask, SubTaskType, TaskStatus, VideoFile, Model
from ...core import TaskManager, get_config, save_config
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
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        self.tree.setRootIsDecorated(True)
        self.tree.setAnimated(True)
        self.tree.itemExpanded.connect(lambda _: self._save_expanded_to_config())
        self.tree.itemCollapsed.connect(lambda _: self._save_expanded_to_config())

        delete_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), self.tree)
        delete_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        delete_shortcut.activated.connect(self._delete_selected)
        
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
        is_first_load = self.tree.topLevelItemCount() == 0
        if is_first_load:
            expanded_tasks, expanded_groups = self._load_expanded_from_config()
        else:
            expanded_tasks, expanded_groups = self._collect_expanded_state()

        self.tree.blockSignals(True)
        self.tree.clear()

        tasks = self.task_manager.get_all_tasks()

        # Группируем задачи по каталогам
        task_groups: dict = {}  # catalog_id (or None) -> [task, ...]
        for task in tasks:
            catalog_id = self._get_task_catalog_id(task)
            if catalog_id not in task_groups:
                task_groups[catalog_id] = []
            task_groups[catalog_id].append(task)

        # Упорядоченный список групп: сначала каталоги в порядке БД, потом «Без экспедиции»
        catalogs = self.repo.get_all_catalogs()
        ordered_groups = []
        for cat in catalogs:
            if cat.id in task_groups:
                ordered_groups.append((cat.id, task_groups[cat.id]))
        if None in task_groups:
            ordered_groups.append((None, task_groups[None]))

        for catalog_id, task_list in ordered_groups:
            group_item = self._create_group_item(catalog_id, len(task_list))
            self.tree.addTopLevelItem(group_item)
            self.tree.setFirstColumnSpanned(
                self.tree.topLevelItemCount() - 1, QModelIndex(), True
            )
            if catalog_id in expanded_groups:
                group_item.setExpanded(True)

            for task in task_list:
                task_item = self._create_task_item(task)
                group_item.addChild(task_item)

                subtasks = self.repo.get_subtasks_for_task(task.id)
                for subtask in subtasks:
                    sub_item = self._create_subtask_item(subtask)
                    task_item.addChild(sub_item)

                if task.id in expanded_tasks:
                    task_item.setExpanded(True)

        self.tree.blockSignals(False)

    def _collect_expanded_state(self):
        """Собирает текущее состояние развёрнутости из дерева."""
        expanded_tasks = set()
        expanded_groups = set()
        for i in range(self.tree.topLevelItemCount()):
            group_item = self.tree.topLevelItem(i)
            if not group_item:
                continue
            if group_item.isExpanded():
                expanded_groups.add(group_item.data(0, Qt.ItemDataRole.UserRole))
            for j in range(group_item.childCount()):
                task_item = group_item.child(j)
                if task_item and task_item.isExpanded():
                    task_id = task_item.data(0, Qt.ItemDataRole.UserRole)
                    if task_id:
                        expanded_tasks.add(task_id)
        return expanded_tasks, expanded_groups

    def _save_expanded_to_config(self):
        """Сохраняет состояние развёрнутости в конфиг."""
        try:
            config = get_config()
            expanded_tasks, expanded_groups = self._collect_expanded_state()
            config.ui.task_table_expanded = list(expanded_tasks)
            config.ui.task_table_expanded_groups = list(expanded_groups)
            save_config()
        except Exception:
            pass

    def _load_expanded_from_config(self):
        """Загружает состояние развёрнутости из конфига."""
        try:
            config = get_config()
            tasks_data = config.ui.task_table_expanded
            groups_data = config.ui.task_table_expanded_groups
            expanded_tasks = set(tasks_data) if tasks_data else set()
            expanded_groups = set(groups_data) if groups_data else set()
            return expanded_tasks, expanded_groups
        except Exception:
            return set(), set()

    def _get_task_catalog_id(self, task: Task) -> Optional[int]:
        """Возвращает catalog_id задачи (или None если без экспедиции)."""
        video = self.repo.get_video_file(task.video_id)
        if not video:
            return None
        dive = self.repo.get_dive(video.dive_id)
        if not dive:
            return None
        return dive.catalog_id

    def _create_group_item(self, catalog_id: Optional[int], task_count: int) -> QTreeWidgetItem:
        """Создаёт элемент группы (экспедиции)."""
        item = QTreeWidgetItem()
        if catalog_id is None:
            item.setText(0, f"📂 Без экспедиции  ({task_count} задач)")
            item.setForeground(0, QBrush(QColor(128, 128, 128)))
        else:
            catalog = self.repo.get_catalog(catalog_id)
            name = catalog.name if catalog else f"#{catalog_id}"
            item.setText(0, f"🗂 {name}  ({task_count} задач)")
            if catalog and catalog.color:
                item.setForeground(0, QBrush(QColor(catalog.color)))
        item.setData(0, Qt.ItemDataRole.UserRole, catalog_id)
        item.setData(0, Qt.ItemDataRole.UserRole + 1, "group")
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        return item

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
        skip_prefix = "⏭ " if task.is_skipped else ""
        item.setText(2, f"{skip_prefix}{status_icon} {task.status.value}")
        item.setTextAlignment(2, Qt.AlignmentFlag.AlignCenter)

        color = self.STATUS_COLORS.get(task.status, QColor(255, 255, 255))
        item.setBackground(2, QBrush(color))
        
        # Прогресс
        item.setText(3, f"{task.progress_percent:.0f}%")
        item.setTextAlignment(3, Qt.AlignmentFlag.AlignCenter)
        
        # Результат
        result_text = ""
        if task.status == TaskStatus.DONE:
            result_text = f"{task.detections_count or 0} дет. / {task.tracks_count or 0} тр."
            if task.class_stats_json:
                try:
                    class_stats = json.loads(task.class_stats_json)
                    # Все 5 классов в фиксированном порядке, нули для отсутствующих
                    ALL_CLASSES = [
                        ("Aurelia aurita", "A. aurita"),
                        ("Beroe ovata", "B. ovata"),
                        ("Mnemiopsis leidyi", "M. leidyi"),
                        ("Pleurobrachia pileus", "P. pileus"),
                        ("Rhizostoma pulmo", "R. pulmo"),
                    ]
                    parts = []
                    tooltip_parts = []
                    for full_name, short_name in ALL_CLASSES:
                        counts = class_stats.get(full_name, {"detections": 0, "tracks": 0})
                        trks = counts.get("tracks", 0)
                        dets = counts.get("detections", 0)
                        parts.append(f"{short_name} {trks:03d}")
                        tooltip_parts.append(f"{full_name}: {dets} дет. / {trks} тр.")
                    result_text = " | ".join(parts)
                    item.setToolTip(4, "\n".join(tooltip_parts))
                except (json.JSONDecodeError, AttributeError):
                    pass
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
        if task.is_skipped:
            font.setItalic(True)
        item.setFont(1, font)

        # Пропущенные задачи — приглушённый цвет
        if task.is_skipped and task.status == TaskStatus.PENDING:
            muted = QBrush(QColor(160, 160, 160))
            for col in range(5):
                item.setForeground(col, muted)

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

    def _postprocess_task(self, task_ids):
        """Открывает диалог постобработки для одной или нескольких задач.

        Принимает int или list[int] для обратной совместимости с прежними
        вызовами по одному ID.
        """
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        if not task_ids:
            return
        try:
            dialog = PostProcessDialog(
                self.repo, self.task_manager, task_ids, parent=self,
            )
            dialog.exec()
            self.refresh()
        except ValueError as e:
            QMessageBox.warning(self, "Ошибка", str(e))

    def _export_label_studio(self, task_ids: list[int]):
        """Открывает диалог экспорта в Label Studio для одной или нескольких задач."""
        try:
            from ..dialogs.export_ls_dialog import ExportLabelStudioDialog
            dialog = ExportLabelStudioDialog(
                self.repo, self.task_manager, task_ids, parent=self,
            )
            dialog.exec()
            self.refresh()
        except ValueError as e:
            QMessageBox.warning(self, "Ошибка", str(e))

    def _postprocess_selected(self):
        """Открывает постобработку для выбранных задач (одной или нескольких)."""
        task_ids = self._get_selected_task_ids()
        if not task_ids:
            QMessageBox.information(self, "Не выбрано", "Выберите задачу")
            return
        self._postprocess_task(task_ids)

    def _get_selected_task_ids(self) -> list:
        """Возвращает список ID всех выбранных задач (группы и подзадачи игнорируются)."""
        result = []
        for item in self.tree.selectedItems():
            if item.data(0, Qt.ItemDataRole.UserRole + 1) == "task":
                result.append(item.data(0, Qt.ItemDataRole.UserRole))
        return result

    def _get_selected_task_id(self) -> Optional[int]:
        """Возвращает ID выбранной задачи."""
        item = self.tree.currentItem()
        if not item:
            return None
        item_type = item.data(0, Qt.ItemDataRole.UserRole + 1)
        if item_type == "group":
            return None
        if item_type == "subtask":
            return item.data(0, Qt.ItemDataRole.UserRole + 2)
        return item.data(0, Qt.ItemDataRole.UserRole)

    def _on_context_menu(self, position):
        """Контекстное меню."""
        item = self.tree.itemAt(position)
        if not item:
            return
        item_type = item.data(0, Qt.ItemDataRole.UserRole + 1)
        if item_type == "group":
            catalog_id = item.data(0, Qt.ItemDataRole.UserRole)
            self._show_group_context_menu(item, catalog_id, position)
            return

        # Множественный выбор задач
        selected_task_ids = self._get_selected_task_ids()
        if len(selected_task_ids) > 1:
            self._show_multi_task_context_menu(selected_task_ids, position)
            return

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
        
        # Постобработка и экспорт (для завершённых)
        if task.status == TaskStatus.DONE:
            action_postprocess = menu.addAction("📊 Добавить постобработку...")
            action_postprocess.triggered.connect(lambda: self._postprocess_task(task_id))
            action_export_ls = menu.addAction("🏷 Экспорт в Label Studio...")
            action_export_ls.triggered.connect(lambda: self._export_label_studio([task_id]))
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
            plot_path = self._find_interactive_plot_path(task_id)
            if plot_path:
                action_plot = menu.addAction("📈 Открыть интерактивный график")
                action_plot.triggered.connect(lambda: self._open_file(plot_path))
            menu.addSeparator()
        
        # Пропуск (только для pending)
        if task.status == TaskStatus.PENDING:
            if task.is_skipped:
                action_unskip = menu.addAction("↩ Снять пропуск")
                action_unskip.triggered.connect(lambda: self.task_manager.unskip_task(task_id))
            else:
                action_skip = menu.addAction("⏭ Пропустить")
                action_skip.triggered.connect(lambda: self.task_manager.skip_task(task_id))
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

    def _show_group_context_menu(self, item, catalog_id, position):
        if catalog_id is None:
            return

        menu = QMenu(self)
        action_export = menu.addAction("📊 Экспорт данных экспедиции")
        action_export.triggered.connect(
            lambda: self._export_expedition_data(catalog_id)
        )
        menu.exec(self.tree.viewport().mapToGlobal(position))

    def _export_expedition_data(self, catalog_id: int):
        from ..dialogs.expedition_export_dialog import export_expedition_data
        export_expedition_data(self, self.repo, catalog_id)

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
            self._restore_selection(task_id)

    def _move_down(self):
        """Перемещает задачу вниз."""
        task_id = self._get_selected_task_id()
        if task_id:
            self.task_manager.move_task_down(task_id)
            self._restore_selection(task_id)

    def _restore_selection(self, task_id: int):
        """Восстанавливает выделение задачи после перестройки дерева."""
        for i in range(self.tree.topLevelItemCount()):
            group_item = self.tree.topLevelItem(i)
            if not group_item:
                continue
            for j in range(group_item.childCount()):
                item = group_item.child(j)
                if item and item.data(0, Qt.ItemDataRole.UserRole) == task_id:
                    self.tree.setCurrentItem(item)
                    self.tree.scrollToItem(item)
                    self.tree.setFocus()
                    return

    def _delete_selected(self):
        """Удаляет выбранные задачи."""
        task_ids = self._get_selected_task_ids()
        if not task_ids:
            return

        running_ids = [
            tid for tid in task_ids
            if (t := self.task_manager.get_task(tid)) and t.status == TaskStatus.RUNNING
        ]
        deletable_ids = [tid for tid in task_ids if tid not in running_ids]

        if running_ids and not deletable_ids:
            QMessageBox.warning(
                self,
                "Невозможно удалить",
                "Нельзя удалить выполняющуюся задачу.\nСначала остановите очередь."
            )
            return

        if running_ids:
            QMessageBox.warning(
                self,
                "Часть задач пропущена",
                f"{len(running_ids)} выполняющихся задач не будут удалены."
            )

        if len(deletable_ids) > 1:
            reply = QMessageBox.question(
                self, f"Удалить {len(deletable_ids)} задач?",
                f"Удалить {len(deletable_ids)} задач из очереди?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        for tid in deletable_ids:
            self.task_manager.remove_task(tid)

    def _retry_selected(self):
        """Повторяет выбранную задачу."""
        task_id = self._get_selected_task_id()
        if task_id:
            self.task_manager.retry_task(task_id)

    def _show_multi_task_context_menu(self, task_ids: list, position):
        """Контекстное меню для группового выделения задач."""
        tasks = [t for tid in task_ids if (t := self.task_manager.get_task(tid))]

        pending_tasks = [t for t in tasks if t.status == TaskStatus.PENDING]
        unskippable = [t.id for t in pending_tasks if not t.is_skipped]
        skipped_ids = [t.id for t in pending_tasks if t.is_skipped]
        done_ids = [t.id for t in tasks if t.status == TaskStatus.DONE]
        error_cancelled = [t.id for t in tasks if t.status in (TaskStatus.ERROR, TaskStatus.CANCELLED)]
        deletable_ids = [t.id for t in tasks if t.status != TaskStatus.RUNNING]
        non_running_ids = [t.id for t in tasks if t.status != TaskStatus.RUNNING]

        menu = QMenu(self)
        has_items = False

        if non_running_ids:
            action = menu.addAction(
                f"📊 Постобработка {len(non_running_ids)} задач..."
            )
            action.triggered.connect(
                lambda: self._postprocess_task(non_running_ids)
            )
            has_items = True

        if done_ids:
            action = menu.addAction(f"🏷 Экспорт в Label Studio ({len(done_ids)})...")
            action.triggered.connect(lambda: self._export_label_studio(done_ids))
            has_items = True

        if unskippable:
            action = menu.addAction(f"⏭ Пропустить {len(unskippable)} задач")
            action.triggered.connect(lambda: self._skip_tasks(unskippable))
            has_items = True

        if skipped_ids:
            action = menu.addAction(f"↩ Снять пропуск у {len(skipped_ids)} задач")
            action.triggered.connect(lambda: self._unskip_tasks(skipped_ids))
            has_items = True

        if error_cancelled:
            if has_items:
                menu.addSeparator()
            action = menu.addAction(f"↻ Повторить {len(error_cancelled)} задач")
            action.triggered.connect(lambda: self._retry_tasks(error_cancelled))
            has_items = True

        if deletable_ids:
            if has_items:
                menu.addSeparator()
            action = menu.addAction(f"🗑 Удалить {len(deletable_ids)} задач")
            action.triggered.connect(self._delete_selected)

        if has_items or deletable_ids:
            menu.exec(self.tree.viewport().mapToGlobal(position))

    def _skip_tasks(self, task_ids: list):
        """Пропускает несколько задач."""
        for tid in task_ids:
            self.task_manager.skip_task(tid)

    def _unskip_tasks(self, task_ids: list):
        """Снимает пропуск у нескольких задач."""
        for tid in task_ids:
            self.task_manager.unskip_task(tid)

    def _retry_tasks(self, task_ids: list):
        """Повторяет несколько задач."""
        for tid in task_ids:
            self.task_manager.retry_task(tid)

    def _find_interactive_plot_path(self, task_id: int) -> Optional[str]:
        """Возвращает путь к depth_interactive.html, если он существует."""
        from ...database import OutputType
        outputs = self.repo.get_task_outputs(task_id)
        for out in outputs:
            if out.output_type == OutputType.INTERACTIVE_PLOT and os.path.exists(out.filepath):
                return out.filepath
        return None

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
        for i in range(self.tree.topLevelItemCount()):
            group_item = self.tree.topLevelItem(i)
            if not group_item:
                continue
            for j in range(group_item.childCount()):
                item = group_item.child(j)
                if item and item.data(0, Qt.ItemDataRole.UserRole) == task_id:
                    item.setText(3, f"{percent:.0f}%")
                    result_text = f"{detections} дет."
                    if tracks > 0:
                        result_text += f" / {tracks} тр."
                    item.setText(4, result_text)
                    return

    def _on_subtask_progress(self, subtask_id: int, percent: float):
        """Обновляет прогресс подзадачи."""
        for i in range(self.tree.topLevelItemCount()):
            group_item = self.tree.topLevelItem(i)
            if not group_item:
                continue
            for j in range(group_item.childCount()):
                task_item = group_item.child(j)
                if not task_item:
                    continue
                for k in range(task_item.childCount()):
                    sub_item = task_item.child(k)
                    if sub_item and sub_item.data(0, Qt.ItemDataRole.UserRole) == subtask_id:
                        sub_item.setText(3, f"{percent:.0f}%")
                        return
