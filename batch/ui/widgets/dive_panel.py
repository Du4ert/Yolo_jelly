"""
Панель погружений - отображение и управление погружениями и их файлами.
"""

import os
import subprocess
import platform
from pathlib import Path
from typing import Optional, List

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTreeWidget,
    QTreeWidgetItem,
    QPushButton,
    QMenu,
    QMessageBox,
    QFileDialog,
    QInputDialog,
    QLabel,
    QGroupBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction

from ...database import Repository, Dive, VideoFile, CTDFile
from ...core import get_config, save_config
from ..dialogs import AddDiveDialog


class DivePanel(QWidget):
    """
    Панель для отображения погружений и их файлов.
    
    Signals:
        dive_selected: Выбрано погружение (dive_id).
        video_selected: Выбрано видео (video_id).
        add_to_queue_requested: Запрос на добавление в очередь (video_id, ctd_id).
    """
    
    dive_selected = pyqtSignal(int)
    video_selected = pyqtSignal(int)
    add_to_queue_requested = pyqtSignal(int, object)  # video_id, ctd_id (может быть None)
    quick_add_to_queue_requested = pyqtSignal(int, object)  # Быстрое добавление без диалога

    # Типы элементов в дереве
    TYPE_DIVE = 0
    TYPE_VIDEO = 1
    TYPE_CTD = 2

    def __init__(self, repository: Repository, parent=None):
        """
        Инициализация панели.
        
        Args:
            repository: Репозиторий для работы с БД.
            parent: Родительский виджет.
        """
        super().__init__(parent)
        self.repo = repository
        self._setup_ui()
        self._load_data()

    def _setup_ui(self):
        """Настройка интерфейса."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Группа
        group = QGroupBox("Погружения")
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(4, 4, 4, 4)
        group_layout.setSpacing(4)
        
        # Дерево погружений
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Название", "Инфо"])
        self.tree.setColumnWidth(0, 200)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        group_layout.addWidget(self.tree)
        
        # Кнопки
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)
        
        self.btn_add = QPushButton("+ Добавить папку")
        self.btn_add.clicked.connect(self.add_dive)
        btn_layout.addWidget(self.btn_add)
        
        self.btn_refresh = QPushButton("↻")
        self.btn_refresh.setFixedWidth(30)
        self.btn_refresh.setToolTip("Обновить")
        self.btn_refresh.clicked.connect(self._load_data)
        btn_layout.addWidget(self.btn_refresh)
        
        group_layout.addLayout(btn_layout)
        
        layout.addWidget(group)

    def _load_data(self):
        """Загружает данные из БД."""
        self.tree.clear()
        
        dives = self.repo.get_all_dives()
        
        for dive in dives:
            dive_item = self._create_dive_item(dive)
            self.tree.addTopLevelItem(dive_item)
            
            # Загружаем видеофайлы
            videos = self.repo.get_videos_by_dive(dive.id)
            for video in videos:
                video_item = self._create_video_item(video)
                dive_item.addChild(video_item)
            
            # Загружаем CTD файлы
            ctd_files = self.repo.get_ctd_by_dive(dive.id)
            for ctd in ctd_files:
                ctd_item = self._create_ctd_item(ctd)
                dive_item.addChild(ctd_item)
            
            # Разворачиваем если есть файлы
            if videos or ctd_files:
                dive_item.setExpanded(True)

    def _create_dive_item(self, dive: Dive) -> QTreeWidgetItem:
        """Создаёт элемент дерева для погружения."""
        item = QTreeWidgetItem()
        item.setText(0, f"📁 {dive.name}")
        item.setText(1, dive.location or "")
        item.setData(0, Qt.ItemDataRole.UserRole, dive.id)
        item.setData(0, Qt.ItemDataRole.UserRole + 1, self.TYPE_DIVE)
        return item

    def _create_video_item(self, video: VideoFile) -> QTreeWidgetItem:
        """Создаёт элемент дерева для видеофайла."""
        item = QTreeWidgetItem()
        item.setText(0, f"📹 {video.filename}")
        
        info_parts = []
        if video.duration_s:
            mins = int(video.duration_s // 60)
            secs = int(video.duration_s % 60)
            info_parts.append(f"{mins}:{secs:02d}")
        if video.width and video.height:
            info_parts.append(f"{video.width}×{video.height}")
        
        item.setText(1, " | ".join(info_parts))
        item.setData(0, Qt.ItemDataRole.UserRole, video.id)
        item.setData(0, Qt.ItemDataRole.UserRole + 1, self.TYPE_VIDEO)
        return item

    def _create_ctd_item(self, ctd: CTDFile) -> QTreeWidgetItem:
        """Создаёт элемент дерева для CTD файла."""
        item = QTreeWidgetItem()
        item.setText(0, f"📊 {ctd.filename}")
        
        info_parts = []
        if ctd.max_depth:
            info_parts.append(f"до {ctd.max_depth:.1f}м")
        if ctd.records_count:
            info_parts.append(f"{ctd.records_count} записей")
        
        item.setText(1, " | ".join(info_parts))
        item.setData(0, Qt.ItemDataRole.UserRole, ctd.id)
        item.setData(0, Qt.ItemDataRole.UserRole + 1, self.TYPE_CTD)
        return item

    def _get_item_type(self, item: QTreeWidgetItem) -> int:
        """Возвращает тип элемента."""
        return item.data(0, Qt.ItemDataRole.UserRole + 1)

    def _get_item_id(self, item: QTreeWidgetItem) -> int:
        """Возвращает ID элемента."""
        return item.data(0, Qt.ItemDataRole.UserRole)

    def _on_selection_changed(self):
        """Обработка изменения выбора."""
        items = self.tree.selectedItems()
        if not items:
            return
        
        item = items[0]
        item_type = self._get_item_type(item)
        item_id = self._get_item_id(item)
        
        if item_type == self.TYPE_DIVE:
            self.dive_selected.emit(item_id)
        elif item_type == self.TYPE_VIDEO:
            self.video_selected.emit(item_id)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Двойной клик на элементе."""
        item_type = self._get_item_type(item)
        item_id = self._get_item_id(item)
        
        if item_type == self.TYPE_VIDEO:
            # Добавляем в очередь
            self._add_video_to_queue(item)

    def _on_context_menu(self, position):
        """Контекстное меню."""
        item = self.tree.itemAt(position)
        if not item:
            return
        
        menu = QMenu(self)
        item_type = self._get_item_type(item)
        item_id = self._get_item_id(item)
        
        if item_type == self.TYPE_DIVE:
            # Меню для погружения
            action_open = menu.addAction("📂 Открыть в проводнике")
            action_open.triggered.connect(lambda: self._open_dive_folder(item_id))
            
            action_open_output = menu.addAction("📂 Открыть папку output")
            action_open_output.triggered.connect(lambda: self._open_output_folder(item_id))
            
            menu.addSeparator()
            
            action_add_all = menu.addAction("📋 Добавить все видео в очередь")
            action_add_all.triggered.connect(lambda: self._add_all_videos_to_queue(item_id))
            
            action_scan = menu.addAction("🔍 Сканировать папку")
            action_scan.triggered.connect(lambda: self._scan_dive_folder(item_id))
            
            menu.addSeparator()
            
            action_delete = menu.addAction("🗑 Удалить из базы")
            action_delete.triggered.connect(lambda: self._delete_dive(item_id))
            
        elif item_type == self.TYPE_VIDEO:
            # Меню для видео
            action_add = menu.addAction("📋 Добавить в очередь...")
            action_add.triggered.connect(lambda: self._add_video_to_queue(item))
            
            action_quick_add = menu.addAction("⚡ Быстро добавить (параметры по умолчанию)")
            action_quick_add.triggered.connect(lambda: self._quick_add_video_to_queue(item))
            
            menu.addSeparator()
            
            action_delete = menu.addAction("🗑 Удалить из базы")
            action_delete.triggered.connect(lambda: self._delete_video(item_id))
            
        elif item_type == self.TYPE_CTD:
            # Меню для CTD
            action_delete = menu.addAction("🗑 Удалить из базы")
            action_delete.triggered.connect(lambda: self._delete_ctd(item_id))
        
        menu.exec(self.tree.viewport().mapToGlobal(position))

    def add_dive(self):
        """Добавляет новое погружение."""
        dialog = AddDiveDialog(self.repo, parent=self)
        
        if dialog.exec():
            data = dialog.get_dive_data()
            dive = self.repo.create_dive(**data)
            
            if dive:
                # Сканируем папку на наличие файлов
                if dialog.should_scan():
                    self._scan_dive_folder(dive.id)
                self._load_data()

    def _scan_dive_folder(self, dive_id: int):
        """Сканирует папку погружения на наличие видео и CTD файлов."""
        dive = self.repo.get_dive(dive_id)
        if not dive:
            return
        
        folder = Path(dive.folder_path)
        if not folder.exists():
            QMessageBox.warning(self, "Ошибка", f"Папка не найдена: {folder}")
            return
        
        # Расширения видео
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        # Расширения CTD
        ctd_extensions = {'.csv', '.txt', '.dat'}
        
        videos_added = 0
        ctd_added = 0
        
        for file_path in folder.iterdir():
            if file_path.is_file():
                ext = file_path.suffix.lower()
                
                if ext in video_extensions:
                    result = self.repo.add_video_file(dive_id, str(file_path))
                    if result:
                        videos_added += 1
                        
                elif ext in ctd_extensions:
                    # Пропускаем файлы, которые выглядят как результаты детекции
                    if '_detections' in file_path.name or '_tracks' in file_path.name:
                        continue
                    result = self.repo.add_ctd_file(dive_id, str(file_path))
                    if result:
                        ctd_added += 1
        
        if videos_added or ctd_added:
            self._load_data()
            self.statusBar_message(f"Добавлено: {videos_added} видео, {ctd_added} CTD файлов")

    def statusBar_message(self, message: str):
        """Показывает сообщение в статусбаре (если доступен)."""
        main_window = self.window()
        if hasattr(main_window, 'statusBar'):
            main_window.statusBar().showMessage(message, 3000)

    def _open_dive_folder(self, dive_id: int):
        """Открывает папку погружения в проводнике."""
        dive = self.repo.get_dive(dive_id)
        if dive and os.path.exists(dive.folder_path):
            self._open_folder(dive.folder_path)

    def _open_output_folder(self, dive_id: int):
        """Открывает папку output погружения."""
        dive = self.repo.get_dive(dive_id)
        if dive:
            output_path = os.path.join(dive.folder_path, "output")
            if os.path.exists(output_path):
                self._open_folder(output_path)
            else:
                QMessageBox.information(
                    self,
                    "Папка не найдена",
                    "Папка output ещё не создана.\nОна появится после обработки видео."
                )

    def _open_folder(self, path: str):
        """Открывает папку в проводнике."""
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", path])
        else:  # Linux
            subprocess.run(["xdg-open", path])

    def _get_video_ctd_ids(self, item: QTreeWidgetItem):
        """Получает video_id и ctd_id для элемента видео."""
        video_id = self._get_item_id(item)
        
        # Ищем CTD файл в том же погружении
        parent = item.parent()
        if parent:
            dive_id = self._get_item_id(parent)
            ctd_files = self.repo.get_ctd_by_dive(dive_id)
            ctd_id = ctd_files[0].id if ctd_files else None
        else:
            ctd_id = None
        
        return video_id, ctd_id

    def _add_video_to_queue(self, item: QTreeWidgetItem):
        """Добавляет видео в очередь (с диалогом)."""
        video_id, ctd_id = self._get_video_ctd_ids(item)
        self.add_to_queue_requested.emit(video_id, ctd_id)

    def _quick_add_video_to_queue(self, item: QTreeWidgetItem):
        """Быстро добавляет видео в очередь (без диалога)."""
        video_id, ctd_id = self._get_video_ctd_ids(item)
        self.quick_add_to_queue_requested.emit(video_id, ctd_id)

    def _add_all_videos_to_queue(self, dive_id: int):
        """Добавляет все видео погружения в очередь."""
        videos = self.repo.get_videos_by_dive(dive_id)
        ctd_files = self.repo.get_ctd_by_dive(dive_id)
        ctd_id = ctd_files[0].id if ctd_files else None
        
        for video in videos:
            self.add_to_queue_requested.emit(video.id, ctd_id)

    def _delete_dive(self, dive_id: int):
        """Удаляет погружение из базы."""
        reply = QMessageBox.question(
            self,
            "Удалить погружение?",
            "Удалить погружение и все связанные данные из базы?\n"
            "(Файлы на диске не будут удалены)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.repo.delete_dive(dive_id)
            self._load_data()

    def _delete_video(self, video_id: int):
        """Удаляет видео из базы."""
        self.repo.delete_video_file(video_id)
        self._load_data()

    def _delete_ctd(self, ctd_id: int):
        """Удаляет CTD файл из базы."""
        self.repo.delete_ctd_file(ctd_id)
        self._load_data()

    def refresh(self):
        """Обновляет отображение."""
        self._load_data()
