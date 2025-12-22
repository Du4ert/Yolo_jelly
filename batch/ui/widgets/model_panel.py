"""
Панель моделей - отображение и управление моделями YOLO.
"""

import os
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QMenu,
    QMessageBox,
    QFileDialog,
    QInputDialog,
    QGroupBox,
)
from PyQt6.QtCore import Qt, pyqtSignal

from ...database import Repository, Model
from ...core import get_config, save_config
from ..dialogs import AddModelDialog


class ModelPanel(QWidget):
    """
    Панель для отображения и управления моделями.
    
    Signals:
        model_selected: Выбрана модель (model_id).
    """
    
    model_selected = pyqtSignal(int)

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
        group = QGroupBox("Модели")
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(4, 4, 4, 4)
        group_layout.setSpacing(4)
        
        # Список моделей
        self.list = QListWidget()
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._on_context_menu)
        self.list.itemSelectionChanged.connect(self._on_selection_changed)
        group_layout.addWidget(self.list)
        
        # Кнопки
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)
        
        self.btn_add = QPushButton("+ Добавить модель")
        self.btn_add.clicked.connect(self.add_model)
        btn_layout.addWidget(self.btn_add)
        
        group_layout.addLayout(btn_layout)
        
        layout.addWidget(group)

    def _load_data(self):
        """Загружает данные из БД."""
        self.list.clear()
        
        models = self.repo.get_all_models()
        config = get_config()
        
        for model in models:
            item = self._create_model_item(model)
            self.list.addItem(item)
            
            # Выбираем последнюю использованную модель
            if config.ui.last_model_id == model.id:
                item.setSelected(True)
        
        # Если ничего не выбрано, выбираем первую
        if not self.list.selectedItems() and self.list.count() > 0:
            self.list.item(0).setSelected(True)

    def _create_model_item(self, model: Model) -> QListWidgetItem:
        """Создаёт элемент списка для модели."""
        item = QListWidgetItem()
        
        # Формируем текст
        text = f"🧠 {model.name}"
        if model.base_model:
            text += f" ({model.base_model})"
        
        item.setText(text)
        item.setToolTip(model.filepath)
        item.setData(Qt.ItemDataRole.UserRole, model.id)
        
        return item

    def _on_selection_changed(self):
        """Обработка изменения выбора."""
        items = self.list.selectedItems()
        if not items:
            return
        
        model_id = items[0].data(Qt.ItemDataRole.UserRole)
        self.model_selected.emit(model_id)
        
        # Сохраняем выбор
        config = get_config()
        config.ui.last_model_id = model_id
        save_config()

    def _on_context_menu(self, position):
        """Контекстное меню."""
        item = self.list.itemAt(position)
        if not item:
            return
        
        menu = QMenu(self)
        model_id = item.data(Qt.ItemDataRole.UserRole)
        
        action_edit = menu.addAction("✏ Переименовать")
        action_edit.triggered.connect(lambda: self._rename_model(model_id))
        
        menu.addSeparator()
        
        action_delete = menu.addAction("🗑 Удалить")
        action_delete.triggered.connect(lambda: self._delete_model(model_id))
        
        menu.exec(self.list.viewport().mapToGlobal(position))

    def add_model(self):
        """Добавляет новую модель."""
        dialog = AddModelDialog(self.repo, parent=self)
        
        if dialog.exec():
            data = dialog.get_model_data()
            model = self.repo.add_model(**data)
            
            if model:
                self._load_data()
                # Выбираем новую модель
                for i in range(self.list.count()):
                    item = self.list.item(i)
                    if item.data(Qt.ItemDataRole.UserRole) == model.id:
                        item.setSelected(True)
                        break

    def _rename_model(self, model_id: int):
        """Переименовывает модель."""
        model = self.repo.get_model(model_id)
        if not model:
            return
        
        name, ok = QInputDialog.getText(
            self,
            "Переименовать модель",
            "Новое название:",
            text=model.name,
        )
        
        if ok and name:
            self.repo.update_model(model_id, name=name)
            self._load_data()

    def _delete_model(self, model_id: int):
        """Удаляет модель."""
        reply = QMessageBox.question(
            self,
            "Удалить модель?",
            "Удалить модель из базы?\n(Файл на диске не будет удалён)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.repo.delete_model(model_id)
            self._load_data()

    def get_selected_model_id(self) -> Optional[int]:
        """Возвращает ID выбранной модели."""
        items = self.list.selectedItems()
        if not items:
            return None
        return items[0].data(Qt.ItemDataRole.UserRole)

    def refresh(self):
        """Обновляет отображение."""
        self._load_data()
