"""
Диалог добавления погружения.
"""

from pathlib import Path
from typing import Optional
from datetime import date

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
    QDateEdit,
    QTextEdit,
    QCheckBox,
)
from PyQt6.QtCore import Qt, QDate

from ...database import Repository
from ...core import get_config, save_config


class AddDiveDialog(QDialog):
    """
    Диалог добавления нового погружения.
    """

    def __init__(self, repository: Repository, folder_path: Optional[str] = None, parent=None):
        """
        Инициализация диалога.
        
        Args:
            repository: Репозиторий для работы с БД.
            folder_path: Предварительно выбранный путь (опционально).
            parent: Родительский виджет.
        """
        super().__init__(parent)
        self.repo = repository
        self.folder_path = folder_path
        
        self._setup_ui()
        
        if folder_path:
            self._set_folder(folder_path)

    def _setup_ui(self):
        """Настройка интерфейса."""
        self.setWindowTitle("Добавить погружение")
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        # === Папка ===
        folder_group = QGroupBox("Папка с данными")
        folder_layout = QHBoxLayout(folder_group)
        
        self.edit_folder = QLineEdit()
        self.edit_folder.setReadOnly(True)
        self.edit_folder.setPlaceholderText("Выберите папку...")
        folder_layout.addWidget(self.edit_folder)
        
        self.btn_browse = QPushButton("Обзор...")
        self.btn_browse.clicked.connect(self._browse_folder)
        folder_layout.addWidget(self.btn_browse)
        
        layout.addWidget(folder_group)
        
        # === Информация ===
        info_group = QGroupBox("Информация о погружении")
        info_layout = QFormLayout(info_group)
        
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("Название погружения")
        info_layout.addRow("Название:", self.edit_name)
        
        self.edit_date = QDateEdit()
        self.edit_date.setCalendarPopup(True)
        self.edit_date.setDate(QDate.currentDate())
        self.edit_date.setDisplayFormat("dd.MM.yyyy")
        info_layout.addRow("Дата:", self.edit_date)
        
        self.edit_location = QLineEdit()
        self.edit_location.setPlaceholderText("Место проведения")
        info_layout.addRow("Место:", self.edit_location)
        
        self.edit_notes = QTextEdit()
        self.edit_notes.setMaximumHeight(80)
        self.edit_notes.setPlaceholderText("Дополнительные заметки...")
        info_layout.addRow("Заметки:", self.edit_notes)
        
        layout.addWidget(info_group)
        
        # === Опции ===
        options_group = QGroupBox("Опции")
        options_layout = QVBoxLayout(options_group)
        
        self.check_scan = QCheckBox("Автоматически сканировать папку на наличие видео и CTD файлов")
        self.check_scan.setChecked(True)
        options_layout.addWidget(self.check_scan)
        
        layout.addWidget(options_group)
        
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

    def _browse_folder(self):
        """Выбор папки."""
        config = get_config()
        start_path = config.ui.last_browse_path or ""
        
        folder = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку погружения",
            start_path,
        )
        
        if folder:
            self._set_folder(folder)
            
            # Сохраняем путь
            config.ui.last_browse_path = str(Path(folder).parent)
            save_config()

    def _set_folder(self, folder: str):
        """Устанавливает выбранную папку."""
        self.folder_path = folder
        self.edit_folder.setText(folder)
        
        # Название по умолчанию - имя папки
        folder_name = Path(folder).name
        if not self.edit_name.text():
            self.edit_name.setText(folder_name)
        
        # Пытаемся извлечь дату из имени папки (форматы: YYYY-MM-DD, YYYYMMDD, DD.MM.YYYY)
        self._try_parse_date(folder_name)
        
        self.btn_ok.setEnabled(True)

    def _try_parse_date(self, text: str):
        """Пытается извлечь дату из текста."""
        import re
        
        # YYYY-MM-DD или YYYY_MM_DD
        match = re.search(r'(\d{4})[-_](\d{2})[-_](\d{2})', text)
        if match:
            try:
                year, month, day = map(int, match.groups())
                self.edit_date.setDate(QDate(year, month, day))
                return
            except:
                pass
        
        # DD.MM.YYYY
        match = re.search(r'(\d{2})\.(\d{2})\.(\d{4})', text)
        if match:
            try:
                day, month, year = map(int, match.groups())
                self.edit_date.setDate(QDate(year, month, day))
                return
            except:
                pass

    def _validate_and_accept(self):
        """Валидация и принятие."""
        if not self.folder_path:
            return
        
        if not self.edit_name.text().strip():
            self.edit_name.setFocus()
            return
        
        # Проверяем, не добавлено ли уже
        existing = self.repo.get_dive_by_path(self.folder_path)
        if existing:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Уже добавлено",
                f"Это погружение уже есть в базе:\n{existing.name}"
            )
            return
        
        self.accept()

    def get_dive_data(self) -> dict:
        """Возвращает данные для создания погружения."""
        qdate = self.edit_date.date()
        dive_date = date(qdate.year(), qdate.month(), qdate.day())
        
        return {
            "name": self.edit_name.text().strip(),
            "folder_path": self.folder_path,
            "date": dive_date,
            "location": self.edit_location.text().strip() or None,
            "notes": self.edit_notes.toPlainText().strip() or None,
        }

    def should_scan(self) -> bool:
        """Возвращает True, если нужно сканировать папку."""
        return self.check_scan.isChecked()
