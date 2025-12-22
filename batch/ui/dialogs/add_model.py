"""
Диалог добавления модели.
"""

from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QDialogButtonBox,
    QFileDialog,
    QComboBox,
    QTextEdit,
)
from PyQt6.QtCore import Qt

from ...database import Repository
from ...core import get_config, save_config


class AddModelDialog(QDialog):
    """
    Диалог добавления новой модели.
    """

    def __init__(self, repository: Repository, filepath: Optional[str] = None, parent=None):
        """
        Инициализация диалога.
        
        Args:
            repository: Репозиторий для работы с БД.
            filepath: Предварительно выбранный путь к файлу (опционально).
            parent: Родительский виджет.
        """
        super().__init__(parent)
        self.repo = repository
        self.filepath = filepath
        
        self._setup_ui()
        
        if filepath:
            self._set_file(filepath)

    def _setup_ui(self):
        """Настройка интерфейса."""
        self.setWindowTitle("Добавить модель")
        self.setMinimumWidth(450)
        
        layout = QVBoxLayout(self)
        
        # === Файл ===
        file_group = QGroupBox("Файл модели")
        file_layout = QHBoxLayout(file_group)
        
        self.edit_file = QLineEdit()
        self.edit_file.setReadOnly(True)
        self.edit_file.setPlaceholderText("Выберите .pt файл...")
        file_layout.addWidget(self.edit_file)
        
        self.btn_browse = QPushButton("Обзор...")
        self.btn_browse.clicked.connect(self._browse_file)
        file_layout.addWidget(self.btn_browse)
        
        layout.addWidget(file_group)
        
        # === Информация ===
        info_group = QGroupBox("Информация о модели")
        info_layout = QFormLayout(info_group)
        
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("Название модели")
        info_layout.addRow("Название:", self.edit_name)
        
        self.combo_base = QComboBox()
        self.combo_base.addItem("Не указано", None)
        self.combo_base.addItem("YOLOv8n (nano)", "yolov8n")
        self.combo_base.addItem("YOLOv8s (small)", "yolov8s")
        self.combo_base.addItem("YOLOv8m (medium)", "yolov8m")
        self.combo_base.addItem("YOLOv8l (large)", "yolov8l")
        self.combo_base.addItem("YOLOv8x (xlarge)", "yolov8x")
        info_layout.addRow("Базовая модель:", self.combo_base)
        
        self.edit_description = QTextEdit()
        self.edit_description.setMaximumHeight(80)
        self.edit_description.setPlaceholderText("Описание модели (опционально)...")
        info_layout.addRow("Описание:", self.edit_description)
        
        layout.addWidget(info_group)
        
        # === Кнопки ===
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._validate_and_accept)
        button_box.rejected.connect(self.reject)
        
        self.btn_ok = button_box.button(QDialogButtonBox.StandardButton.Ok)
        self.btn_ok.setText("Добавить")
        self.btn_ok.setEnabled(False)
        
        layout.addWidget(button_box)

    def _browse_file(self):
        """Выбор файла."""
        config = get_config()
        start_path = config.ui.last_browse_path or ""
        
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл модели",
            start_path,
            "YOLO Models (*.pt);;All Files (*.*)",
        )
        
        if filepath:
            self._set_file(filepath)
            
            # Сохраняем путь
            config.ui.last_browse_path = str(Path(filepath).parent)
            save_config()

    def _set_file(self, filepath: str):
        """Устанавливает выбранный файл."""
        self.filepath = filepath
        self.edit_file.setText(filepath)
        
        # Название по умолчанию - имя файла без расширения
        filename = Path(filepath).stem
        if not self.edit_name.text():
            self.edit_name.setText(filename)
        
        # Пытаемся определить базовую модель из имени файла
        self._detect_base_model(filename)
        
        self.btn_ok.setEnabled(True)

    def _detect_base_model(self, filename: str):
        """Пытается определить базовую модель из имени файла."""
        filename_lower = filename.lower()
        
        for i in range(self.combo_base.count()):
            base = self.combo_base.itemData(i)
            if base and base in filename_lower:
                self.combo_base.setCurrentIndex(i)
                return

    def _validate_and_accept(self):
        """Валидация и принятие."""
        if not self.filepath:
            return
        
        if not self.edit_name.text().strip():
            self.edit_name.setFocus()
            return
        
        # Проверяем, не добавлена ли уже
        from sqlalchemy import select
        from ...database import Model
        
        with self.repo.get_session() as session:
            stmt = select(Model).where(Model.filepath == str(Path(self.filepath).resolve()))
            existing = session.scalar(stmt)
            
        if existing:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Уже добавлено",
                f"Эта модель уже есть в базе:\n{existing.name}"
            )
            return
        
        self.accept()

    def get_model_data(self) -> dict:
        """Возвращает данные для создания модели."""
        return {
            "name": self.edit_name.text().strip(),
            "filepath": self.filepath,
            "base_model": self.combo_base.currentData(),
            "description": self.edit_description.toPlainText().strip() or None,
        }
