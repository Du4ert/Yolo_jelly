"""
Панель погружений с поддержкой каталогов (экспедиций).
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
    QGroupBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QDropEvent


from ...database import Repository, Catalog, Dive, VideoFile, CTDFile
from ...core import get_config, save_config
from ..dialogs import AddDiveDialog, select_multiple_directories, EditDiveDialog, CatalogDialog


class _DiveTree(QTreeWidget):
    """QTreeWidget с перехватом drag-drop для сохранения перемещений в БД."""

    item_moved = pyqtSignal(object, object, object)  # item, old_parent, new_parent

    def dropEvent(self, event: QDropEvent):
        item = self.currentItem()
        old_parent = item.parent() if item else None
        super().dropEvent(event)
        new_parent = item.parent() if item else None
        if item is not None and old_parent is not new_parent:
            self.item_moved.emit(item, old_parent, new_parent)


class DivePanel(QWidget):
    """
    Панель для отображения каталогов, погружений и их файлов.
    """
    
    dive_selected = pyqtSignal(int)
    video_selected = pyqtSignal(int)
    add_to_queue_requested = pyqtSignal(int, object)
    quick_add_to_queue_requested = pyqtSignal(int, object)
    batch_add_to_queue_requested = pyqtSignal(list)  # list of (video_id, ctd_id|None)

    # Типы элементов в дереве
    TYPE_CATALOG = 0
    TYPE_DIVE = 1
    TYPE_VIDEO = 2
    TYPE_CTD = 3
    TYPE_UNCATEGORIZED = 4  # Виртуальный узел "Без категории"

    def __init__(self, repository: Repository, parent=None):
        super().__init__(parent)
        self.repo = repository
        self._setup_ui()
        self._load_data()

    def _setup_ui(self):
        """Настройка интерфейса."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        group = QGroupBox("Погружения")
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(4, 4, 4, 4)
        group_layout.setSpacing(4)
        
        # Дерево
        self.tree = _DiveTree()
        self.tree.setHeaderLabels(["Название", "Инфо"])
        self.tree.setColumnWidth(0, 220)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree.setDragDropMode(QTreeWidget.DragDropMode.InternalMove)
        self.tree.setAcceptDrops(True)
        self.tree.item_moved.connect(self._on_item_moved)
        group_layout.addWidget(self.tree)
        
        # Кнопки
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)
        
        self.btn_add_catalog = QPushButton("+ Экспедиция")
        self.btn_add_catalog.setToolTip("Создать новую экспедицию")
        self.btn_add_catalog.clicked.connect(self._add_catalog)
        btn_layout.addWidget(self.btn_add_catalog)
        
        self.btn_add_dive = QPushButton("+ Папка")
        self.btn_add_dive.setToolTip("Добавить папку с погружением")
        self.btn_add_dive.clicked.connect(self.add_dive)
        btn_layout.addWidget(self.btn_add_dive)

        self.btn_add_multiple_dives = QPushButton("+ Папки")
        self.btn_add_multiple_dives.setToolTip("Добавить несколько папок с погружениями сразу")
        self.btn_add_multiple_dives.clicked.connect(self.add_multiple_dives)
        btn_layout.addWidget(self.btn_add_multiple_dives)

        self.btn_refresh = QPushButton("↻")
        self.btn_refresh.setFixedWidth(30)
        self.btn_refresh.setToolTip("Обновить")
        self.btn_refresh.clicked.connect(self._load_data)
        btn_layout.addWidget(self.btn_refresh)
        
        group_layout.addLayout(btn_layout)
        layout.addWidget(group)

    def _save_expanded_state(self) -> set:
        """Возвращает множество (item_type, item_id) развёрнутых узлов."""
        expanded = set()
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            top = root.child(i)
            if top.isExpanded():
                expanded.add((self._get_item_type(top), self._get_item_id(top)))
            for j in range(top.childCount()):
                child = top.child(j)
                if child.isExpanded():
                    expanded.add((self._get_item_type(child), self._get_item_id(child)))
        return expanded

    def _restore_expanded_state(self, expanded: set):
        """Восстанавливает развёрнутость узлов по сохранённому множеству."""
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            top = root.child(i)
            top.setExpanded((self._get_item_type(top), self._get_item_id(top)) in expanded)
            for j in range(top.childCount()):
                child = top.child(j)
                child.setExpanded((self._get_item_type(child), self._get_item_id(child)) in expanded)

    def _load_data(self):
        """Загружает данные из БД."""
        # Сохраняем состояние разворота перед перестройкой
        first_load = self.tree.topLevelItemCount() == 0
        if not first_load:
            saved_expanded = self._save_expanded_state()

        self.tree.clear()

        # Загружаем каталоги
        catalogs = self.repo.get_all_catalogs()

        for catalog in catalogs:
            catalog_item = self._create_catalog_item(catalog)
            self.tree.addTopLevelItem(catalog_item)

            # Погружения в каталоге
            dives = self.repo.get_dives_by_catalog(catalog.id)
            for dive in dives:
                dive_item = self._create_dive_item(dive)
                catalog_item.addChild(dive_item)
                self._add_dive_children(dive_item, dive.id)

            if first_load and dives:
                catalog_item.setExpanded(True)

        # Погружения без каталога
        uncategorized_dives = self.repo.get_dives_by_catalog(None)

        if uncategorized_dives:
            # Создаём виртуальный узел
            uncat_item = QTreeWidgetItem()
            uncat_item.setText(0, "📂 Без экспедиции")
            uncat_item.setText(1, f"{len(uncategorized_dives)} погр.")
            uncat_item.setData(0, Qt.ItemDataRole.UserRole, None)
            uncat_item.setData(0, Qt.ItemDataRole.UserRole + 1, self.TYPE_UNCATEGORIZED)
            uncat_item.setForeground(0, QBrush(QColor(128, 128, 128)))
            # Принимаем drop, но сам узел не перетаскиваем
            uncat_item.setFlags(
                (uncat_item.flags() | Qt.ItemFlag.ItemIsDropEnabled)
                & ~Qt.ItemFlag.ItemIsDragEnabled
            )
            self.tree.addTopLevelItem(uncat_item)

            for dive in uncategorized_dives:
                dive_item = self._create_dive_item(dive)
                uncat_item.addChild(dive_item)
                self._add_dive_children(dive_item, dive.id)

            if first_load:
                uncat_item.setExpanded(True)

        if not first_load:
            self._restore_expanded_state(saved_expanded)

    def _create_catalog_item(self, catalog: Catalog) -> QTreeWidgetItem:
        """Создаёт элемент для каталога."""
        item = QTreeWidgetItem()
        item.setText(0, f"🗂 {catalog.name}")

        # Считаем погружения
        dives = self.repo.get_dives_by_catalog(catalog.id)
        item.setText(1, f"{len(dives)} погр.")

        item.setData(0, Qt.ItemDataRole.UserRole, catalog.id)
        item.setData(0, Qt.ItemDataRole.UserRole + 1, self.TYPE_CATALOG)

        # Каталоги не перетаскиваем, но принимаем drop
        item.setFlags(
            (item.flags() | Qt.ItemFlag.ItemIsDropEnabled)
            & ~Qt.ItemFlag.ItemIsDragEnabled
        )

        # Цвет
        if catalog.color:
            item.setForeground(0, QBrush(QColor(catalog.color)))

        return item

    def _create_dive_item(self, dive: Dive) -> QTreeWidgetItem:
        """Создаёт элемент для погружения."""
        item = QTreeWidgetItem()
        item.setText(0, f"📁 {dive.name}")
        item.setText(1, dive.location or "")
        item.setData(0, Qt.ItemDataRole.UserRole, dive.id)
        item.setData(0, Qt.ItemDataRole.UserRole + 1, self.TYPE_DIVE)
        # Погружения можно перетаскивать, но нельзя принимать drop внутрь
        item.setFlags(
            (item.flags() | Qt.ItemFlag.ItemIsDragEnabled)
            & ~Qt.ItemFlag.ItemIsDropEnabled
        )
        return item

    def _create_video_item(self, video: VideoFile) -> QTreeWidgetItem:
        """Создаёт элемент для видео."""
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
        item.setFlags(item.flags() & ~(Qt.ItemFlag.ItemIsDragEnabled | Qt.ItemFlag.ItemIsDropEnabled))
        return item

    def _create_ctd_item(self, ctd: CTDFile) -> QTreeWidgetItem:
        """Создаёт элемент для CTD."""
        item = QTreeWidgetItem()
        item.setText(0, f"📊 {ctd.filename}")

        info_parts = []
        if ctd.max_depth:
            info_parts.append(f"до {ctd.max_depth:.1f}м")
        if ctd.records_count:
            info_parts.append(f"{ctd.records_count} зап.")

        item.setText(1, " | ".join(info_parts))
        item.setData(0, Qt.ItemDataRole.UserRole, ctd.id)
        item.setData(0, Qt.ItemDataRole.UserRole + 1, self.TYPE_CTD)
        item.setFlags(item.flags() & ~(Qt.ItemFlag.ItemIsDragEnabled | Qt.ItemFlag.ItemIsDropEnabled))
        return item

    def _add_dive_children(self, dive_item: QTreeWidgetItem, dive_id: int):
        """Добавляет дочерние элементы погружения."""
        videos = self.repo.get_videos_by_dive(dive_id)
        for video in videos:
            dive_item.addChild(self._create_video_item(video))
        
        ctd_files = self.repo.get_ctd_by_dive(dive_id)
        for ctd in ctd_files:
            dive_item.addChild(self._create_ctd_item(ctd))
        
        if videos or ctd_files:
            dive_item.setExpanded(True)

    def _get_item_type(self, item: QTreeWidgetItem) -> int:
        return item.data(0, Qt.ItemDataRole.UserRole + 1)

    def _get_item_id(self, item: QTreeWidgetItem) -> Optional[int]:
        return item.data(0, Qt.ItemDataRole.UserRole)

    def _on_selection_changed(self):
        """Обработка выбора."""
        items = self.tree.selectedItems()
        if not items:
            return
        
        item = items[0]
        item_type = self._get_item_type(item)
        item_id = self._get_item_id(item)
        
        if item_type == self.TYPE_DIVE and item_id:
            self.dive_selected.emit(item_id)
        elif item_type == self.TYPE_VIDEO and item_id:
            self.video_selected.emit(item_id)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Двойной клик."""
        item_type = self._get_item_type(item)
        item_id = self._get_item_id(item)
        
        if item_type == self.TYPE_CATALOG and item_id:
            self._edit_catalog(item_id)
        elif item_type == self.TYPE_DIVE and item_id:
            self._edit_dive(item_id)
        elif item_type == self.TYPE_VIDEO and item_id:
            self._add_video_to_queue(item)

    def _on_item_moved(self, item, old_parent, new_parent):
        """Обработка перетаскивания — сохраняет перемещение погружения в БД."""
        item_type = self._get_item_type(item)
        if item_type != self.TYPE_DIVE:
            # Каталоги и файлы перетаскивать нельзя — откатываем визуально
            self._load_data()
            return

        dive_id = self._get_item_id(item)
        if not dive_id:
            self._load_data()
            return

        # Определяем новый catalog_id по новому родителю
        if new_parent is None:
            catalog_id = None  # top-level → без экспедиции
        else:
            parent_type = self._get_item_type(new_parent)
            if parent_type == self.TYPE_CATALOG:
                catalog_id = self._get_item_id(new_parent)
            elif parent_type == self.TYPE_UNCATEGORIZED:
                catalog_id = None
            else:
                # Вложили в погружение или видео — некорректно, откатываем
                self._load_data()
                return

        self.repo.move_dive_to_catalog(dive_id, catalog_id)
        self._load_data()

    def _on_context_menu(self, position):
        """Контекстное меню."""
        item = self.tree.itemAt(position)
        menu = QMenu(self)
        
        if not item:
            # Клик на пустом месте
            action_add_cat = menu.addAction("🗂 Новая экспедиция...")
            action_add_cat.triggered.connect(self._add_catalog)

            action_add_dive = menu.addAction("📁 Добавить папку...")
            action_add_dive.triggered.connect(self.add_dive)

            action_add_multi = menu.addAction("📁 Добавить несколько папок...")
            action_add_multi.triggered.connect(self.add_multiple_dives)
        else:
            item_type = self._get_item_type(item)
            item_id = self._get_item_id(item)
            
            if item_type == self.TYPE_CATALOG:
                self._build_catalog_menu(menu, item_id)
            elif item_type == self.TYPE_UNCATEGORIZED:
                action_add = menu.addAction("📁 Добавить папку сюда...")
                action_add.triggered.connect(self.add_dive)

                action_add_multi = menu.addAction("📁 Добавить несколько папок...")
                action_add_multi.triggered.connect(self.add_multiple_dives)
            elif item_type == self.TYPE_DIVE:
                self._build_dive_menu(menu, item, item_id)
            elif item_type == self.TYPE_VIDEO:
                self._build_video_menu(menu, item, item_id)
            elif item_type == self.TYPE_CTD:
                self._build_ctd_menu(menu, item_id)
        
        menu.exec(self.tree.viewport().mapToGlobal(position))

    def _build_catalog_menu(self, menu: QMenu, catalog_id: int):
        """Меню для каталога."""
        action_edit = menu.addAction("✏ Редактировать...")
        action_edit.triggered.connect(lambda: self._edit_catalog(catalog_id))
        
        menu.addSeparator()
        
        action_add = menu.addAction("📁 Добавить папку в экспедицию...")
        action_add.triggered.connect(lambda: self._add_dive_to_catalog(catalog_id))

        action_add_multi = menu.addAction("📁 Добавить несколько папок...")
        action_add_multi.triggered.connect(lambda: self._add_multiple_dives_to_catalog(catalog_id))

        action_add_all = menu.addAction("📋 Добавить все видео в очередь")
        action_add_all.triggered.connect(lambda: self._add_all_catalog_videos(catalog_id))
        
        menu.addSeparator()
        
        action_delete = menu.addAction("🗑 Удалить экспедицию")
        action_delete.triggered.connect(lambda: self._delete_catalog(catalog_id))

    def _build_dive_menu(self, menu: QMenu, item: QTreeWidgetItem, dive_id: int):
        """Меню для погружения."""
        action_edit = menu.addAction("✏ Редактировать...")
        action_edit.triggered.connect(lambda: self._edit_dive(dive_id))
        
        menu.addSeparator()
        
        action_open = menu.addAction("📂 Открыть в проводнике")
        action_open.triggered.connect(lambda: self._open_dive_folder(dive_id))
        
        action_output = menu.addAction("📂 Открыть папку output")
        action_output.triggered.connect(lambda: self._open_output_folder(dive_id))
        
        menu.addSeparator()
        
        # Подменю перемещения в каталог
        move_menu = menu.addMenu("📦 Переместить в...")
        
        catalogs = self.repo.get_all_catalogs()
        dive = self.repo.get_dive(dive_id)
        
        for cat in catalogs:
            if dive and dive.catalog_id != cat.id:
                action = move_menu.addAction(f"🗂 {cat.name}")
                action.triggered.connect(lambda checked, cid=cat.id: self._move_dive_to_catalog(dive_id, cid))
        
        if dive and dive.catalog_id is not None:
            move_menu.addSeparator()
            action_uncat = move_menu.addAction("📂 Без экспедиции")
            action_uncat.triggered.connect(lambda: self._move_dive_to_catalog(dive_id, None))
        
        menu.addSeparator()
        
        action_add_all = menu.addAction("📋 Добавить все видео в очередь")
        action_add_all.triggered.connect(lambda: self._add_all_videos_to_queue(dive_id))
        
        action_scan = menu.addAction("🔍 Сканировать папку")
        action_scan.triggered.connect(lambda: self._scan_dive_folder(dive_id))
        
        menu.addSeparator()
        
        action_delete = menu.addAction("🗑 Удалить из базы")
        action_delete.triggered.connect(lambda: self._delete_dive(dive_id))

    def _build_video_menu(self, menu: QMenu, item: QTreeWidgetItem, video_id: int):
        """Меню для видео."""
        action_add = menu.addAction("📋 Добавить в очередь...")
        action_add.triggered.connect(lambda: self._add_video_to_queue(item))
        
        action_quick = menu.addAction("⚡ Быстро добавить")
        action_quick.triggered.connect(lambda: self._quick_add_video_to_queue(item))
        
        menu.addSeparator()
        
        action_delete = menu.addAction("🗑 Удалить из базы")
        action_delete.triggered.connect(lambda: self._delete_video(video_id))

    def _build_ctd_menu(self, menu: QMenu, ctd_id: int):
        """Меню для CTD."""
        action_delete = menu.addAction("🗑 Удалить из базы")
        action_delete.triggered.connect(lambda: self._delete_ctd(ctd_id))

    # ========== ДЕЙСТВИЯ ==========

    def _add_catalog(self):
        """Создаёт новый каталог."""
        dialog = CatalogDialog(self.repo, parent=self)
        if dialog.exec():
            self._load_data()

    def _edit_catalog(self, catalog_id: int):
        """Редактирует каталог."""
        try:
            dialog = CatalogDialog(self.repo, catalog_id, parent=self)
            if dialog.exec():
                self._load_data()
        except ValueError as e:
            QMessageBox.warning(self, "Ошибка", str(e))

    def _delete_catalog(self, catalog_id: int):
        """Удаляет каталог."""
        catalog = self.repo.get_catalog(catalog_id)
        if not catalog:
            return
        
        dives = self.repo.get_dives_by_catalog(catalog_id)
        
        msg = f"Удалить экспедицию «{catalog.name}»?"
        if dives:
            msg += f"\n\n{len(dives)} погружений будут перемещены в «Без экспедиции»."
        
        reply = QMessageBox.question(
            self, "Удалить экспедицию?", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.repo.delete_catalog(catalog_id)
            self._load_data()

    def _add_all_catalog_videos(self, catalog_id: int):
        """Добавляет все видео каталога в очередь (один диалог на всё)."""
        pairs = []
        for dive in self.repo.get_dives_by_catalog(catalog_id):
            ctd_files = self.repo.get_ctd_by_dive(dive.id)
            ctd_id = ctd_files[0].id if ctd_files else None
            for video in self.repo.get_videos_by_dive(dive.id):
                pairs.append((video.id, ctd_id))
        if pairs:
            self.batch_add_to_queue_requested.emit(pairs)

    def add_dive(self):
        """Добавляет погружение."""
        dialog = AddDiveDialog(self.repo, parent=self)
        if dialog.exec():
            data = dialog.get_dive_data()
            dive = self.repo.create_dive(**data)
            if dive and dialog.should_scan():
                self._scan_dive_folder(dive.id)
            self._load_data()

    def _add_dive_to_catalog(self, catalog_id: int):
        """Добавляет погружение в каталог."""
        dialog = AddDiveDialog(self.repo, parent=self)
        if dialog.exec():
            data = dialog.get_dive_data()
            dive = self.repo.create_dive(**data)
            if dive:
                self.repo.move_dive_to_catalog(dive.id, catalog_id)
                if dialog.should_scan():
                    self._scan_dive_folder(dive.id)
            self._load_data()

    def add_multiple_dives(self):
        """Добавляет несколько погружений сразу."""
        self._add_multiple_dives_to_catalog(None)

    def _add_multiple_dives_to_catalog(self, catalog_id):
        """Открывает диалог множественного выбора папок и добавляет их."""
        config = get_config()
        start_path = config.ui.last_browse_path or ""

        folders = select_multiple_directories(
            parent=self,
            title="Выберите папки погружений",
            start_path=start_path,
        )
        if not folders:
            return

        config.ui.last_browse_path = str(Path(folders[0]).parent)
        save_config()

        added = 0
        skipped = 0
        for folder in folders:
            existing = self.repo.get_dive_by_path(folder)
            if existing:
                skipped += 1
                continue

            name = Path(folder).name
            dive = self.repo.create_dive(name=name, folder_path=folder)
            if dive:
                if catalog_id is not None:
                    self.repo.move_dive_to_catalog(dive.id, catalog_id)
                self._scan_dive_folder(dive.id)
                added += 1

        self._load_data()

        msg = f"Добавлено погружений: {added}"
        if skipped:
            msg += f", пропущено (уже есть): {skipped}"
        self._show_status(msg)

    def _edit_dive(self, dive_id: int):
        """Редактирует погружение."""
        try:
            dialog = EditDiveDialog(self.repo, dive_id, parent=self)
            if dialog.exec():
                self._load_data()
        except ValueError as e:
            QMessageBox.warning(self, "Ошибка", str(e))

    def _move_dive_to_catalog(self, dive_id: int, catalog_id: Optional[int]):
        """Перемещает погружение в каталог."""
        self.repo.move_dive_to_catalog(dive_id, catalog_id)
        self._load_data()

    def _delete_dive(self, dive_id: int):
        """Удаляет погружение."""
        reply = QMessageBox.question(
            self, "Удалить погружение?",
            "Удалить погружение из базы?\n(Файлы на диске не будут удалены)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.repo.delete_dive(dive_id)
            self._load_data()

    def _scan_dive_folder(self, dive_id: int):
        """Сканирует папку погружения."""
        dive = self.repo.get_dive(dive_id)
        if not dive:
            return
        
        folder = Path(dive.folder_path)
        if not folder.exists():
            QMessageBox.warning(self, "Ошибка", f"Папка не найдена: {folder}")
            return
        
        video_ext = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        ctd_ext = {'.csv', '.txt', '.dat'}
        
        videos_added = ctd_added = 0
        
        for fp in folder.iterdir():
            if fp.is_file():
                ext = fp.suffix.lower()
                if ext in video_ext:
                    if self.repo.add_video_file(dive_id, str(fp)):
                        videos_added += 1
                elif ext in ctd_ext:
                    if '_detections' not in fp.name and '_tracks' not in fp.name:
                        if self.repo.add_ctd_file(dive_id, str(fp)):
                            ctd_added += 1
        
        if videos_added or ctd_added:
            self._load_data()
            self._show_status(f"Добавлено: {videos_added} видео, {ctd_added} CTD")

    def _open_dive_folder(self, dive_id: int):
        """Открывает папку погружения."""
        dive = self.repo.get_dive(dive_id)
        if dive and os.path.exists(dive.folder_path):
            self._open_folder(dive.folder_path)

    def _open_output_folder(self, dive_id: int):
        """Открывает папку output."""
        dive = self.repo.get_dive(dive_id)
        if dive:
            output = os.path.join(dive.folder_path, "output")
            if os.path.exists(output):
                self._open_folder(output)
            else:
                QMessageBox.information(self, "Папка не найдена",
                    "Папка output ещё не создана.")

    def _open_folder(self, path: str):
        """Открывает папку в проводнике."""
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])

    def _get_video_ctd_ids(self, item: QTreeWidgetItem):
        """Получает video_id и ctd_id."""
        video_id = self._get_item_id(item)
        
        parent = item.parent()
        if parent and self._get_item_type(parent) == self.TYPE_DIVE:
            dive_id = self._get_item_id(parent)
            ctd_files = self.repo.get_ctd_by_dive(dive_id)
            ctd_id = ctd_files[0].id if ctd_files else None
        else:
            ctd_id = None
        
        return video_id, ctd_id

    def _add_video_to_queue(self, item: QTreeWidgetItem):
        """Добавляет видео в очередь с диалогом."""
        video_id, ctd_id = self._get_video_ctd_ids(item)
        self.add_to_queue_requested.emit(video_id, ctd_id)

    def _quick_add_video_to_queue(self, item: QTreeWidgetItem):
        """Быстро добавляет видео."""
        video_id, ctd_id = self._get_video_ctd_ids(item)
        self.quick_add_to_queue_requested.emit(video_id, ctd_id)

    def _add_all_videos_to_queue(self, dive_id: int):
        """Добавляет все видео погружения (один диалог на всё)."""
        videos = self.repo.get_videos_by_dive(dive_id)
        ctd_files = self.repo.get_ctd_by_dive(dive_id)
        ctd_id = ctd_files[0].id if ctd_files else None
        pairs = [(video.id, ctd_id) for video in videos]
        if pairs:
            self.batch_add_to_queue_requested.emit(pairs)

    def _delete_video(self, video_id: int):
        """Удаляет видео."""
        self.repo.delete_video_file(video_id)
        self._load_data()

    def _delete_ctd(self, ctd_id: int):
        """Удаляет CTD."""
        self.repo.delete_ctd_file(ctd_id)
        self._load_data()

    def _show_status(self, message: str):
        """Показывает сообщение в статусбаре."""
        main = self.window()
        if hasattr(main, 'statusBar'):
            main.statusBar().showMessage(message, 3000)

    def refresh(self):
        """Обновляет данные."""
        self._load_data()

    def get_selected_dive_id(self) -> Optional[int]:
        """
        Возвращает ID выбранного погружения.
        
        Если выбрано видео или CTD, возвращает ID их родительского погружения.
        """
        items = self.tree.selectedItems()
        if not items:
            return None
        
        item = items[0]
        item_type = self._get_item_type(item)
        item_id = self._get_item_id(item)
        
        if item_type == self.TYPE_DIVE:
            return item_id
        elif item_type in (self.TYPE_VIDEO, self.TYPE_CTD):
            # Поднимаемся к родителю
            parent = item.parent()
            if parent and self._get_item_type(parent) == self.TYPE_DIVE:
                return self._get_item_id(parent)
        
        return None
