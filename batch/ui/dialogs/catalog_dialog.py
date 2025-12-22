"""
Диалог создания/редактирования каталога (экспедиции).
"""

from typing import Optional

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QTextEdit,
    QDialogButtonBox,
    QPushButton,
    QColorDialog,
    QHBoxLayout,
    QLabel,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from ...database import Repository, Catalog


class CatalogDialog(QDialog):
    """
    Диалог создания/редактирования каталога.
    """

    # Предустановленные цвета
    PRESET_COLORS = [
        "#3498db",  # Синий
        "#2ecc71",  # Зелёный
        "#e74c3c",  # Красный
        "#f39c12",  # Оранжевый
        "#9b59b6",  # Фиолетовый
        "#1abc9c",  # Бирюзовый
        "#e91e63",  # Розовый
        "#607d8b",  # Серый
    ]

    def __init__(
        self, 
        repository: Repository, 
        catalog_id: Optional[int] = None,
        parent=None
    ):
        """
        Args:
            repository: Репозиторий.
            catalog_id: ID каталога для редактирования (None = создание нового).
            parent: Родительский виджет.
        """
        super().__init__(parent)
        self.repo = repository
        self.catalog_id = catalog_id
        self.catalog = None
        self.selected_color = self.PRESET_COLORS[0]
        
        if catalog_id:
            self.catalog = self.repo.get_catalog(catalog_id)
            if not self.catalog:
                raise ValueError(f"Catalog {catalog_id} not found")
        
        self._setup_ui()
        self._load_data()

    def _setup_ui(self):
        """Настройка интерфейса."""
        if self.catalog_id:
            self.setWindowTitle("Редактировать экспедицию")
        else:
            self.setWindowTitle("Новая экспедиция")
        
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        # Информация
        info_group = QGroupBox("Информация")
        info_layout = QFormLayout(info_group)
        
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("Название экспедиции")
        info_layout.addRow("Название:", self.edit_name)
        
        self.edit_description = QTextEdit()
        self.edit_description.setMaximumHeight(80)
        self.edit_description.setPlaceholderText("Описание (опционально)")
        info_layout.addRow("Описание:", self.edit_description)
        
        layout.addWidget(info_group)
        
        # Цвет
        color_group = QGroupBox("Цвет")
        color_layout = QVBoxLayout(color_group)
        
        # Предустановленные цвета
        presets_layout = QHBoxLayout()
        self.color_buttons = []
        
        for color in self.PRESET_COLORS:
            btn = QPushButton()
            btn.setFixedSize(30, 30)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    border: 2px solid transparent;
                    border-radius: 4px;
                }}
                QPushButton:hover {{
                    border: 2px solid #333;
                }}
            """)
            btn.clicked.connect(lambda checked, c=color: self._select_color(c))
            presets_layout.addWidget(btn)
            self.color_buttons.append((btn, color))
        
        presets_layout.addStretch()
        
        # Кнопка выбора своего цвета
        self.btn_custom_color = QPushButton("Другой...")
        self.btn_custom_color.clicked.connect(self._choose_custom_color)
        presets_layout.addWidget(self.btn_custom_color)
        
        color_layout.addLayout(presets_layout)
        
        # Превью выбранного цвета
        preview_layout = QHBoxLayout()
        preview_layout.addWidget(QLabel("Выбранный цвет:"))
        self.color_preview = QLabel()
        self.color_preview.setFixedSize(60, 25)
        self.color_preview.setStyleSheet(f"background-color: {self.selected_color}; border-radius: 4px;")
        preview_layout.addWidget(self.color_preview)
        preview_layout.addStretch()
        
        color_layout.addLayout(preview_layout)
        layout.addWidget(color_group)
        
        # Кнопки
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._save_and_accept)
        button_box.rejected.connect(self.reject)
        
        layout.addWidget(button_box)

    def _load_data(self):
        """Загружает данные каталога."""
        if self.catalog:
            self.edit_name.setText(self.catalog.name)
            self.edit_description.setText(self.catalog.description or "")
            if self.catalog.color:
                self._select_color(self.catalog.color)

    def _select_color(self, color: str):
        """Выбирает цвет."""
        self.selected_color = color
        self.color_preview.setStyleSheet(f"background-color: {color}; border-radius: 4px;")
        
        # Обновляем границы кнопок
        for btn, btn_color in self.color_buttons:
            if btn_color == color:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {btn_color};
                        border: 3px solid #333;
                        border-radius: 4px;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {btn_color};
                        border: 2px solid transparent;
                        border-radius: 4px;
                    }}
                    QPushButton:hover {{
                        border: 2px solid #333;
                    }}
                """)

    def _choose_custom_color(self):
        """Открывает диалог выбора цвета."""
        color = QColorDialog.getColor(
            QColor(self.selected_color),
            self,
            "Выберите цвет"
        )
        if color.isValid():
            self._select_color(color.name())

    def _save_and_accept(self):
        """Сохраняет и закрывает."""
        name = self.edit_name.text().strip()
        if not name:
            self.edit_name.setFocus()
            return
        
        description = self.edit_description.toPlainText().strip() or None
        
        if self.catalog_id:
            # Редактирование
            self.repo.update_catalog(
                self.catalog_id,
                name=name,
                description=description,
                color=self.selected_color,
            )
        else:
            # Создание
            self.repo.create_catalog(
                name=name,
                description=description,
                color=self.selected_color,
            )
        
        self.accept()
