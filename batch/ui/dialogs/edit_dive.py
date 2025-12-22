"""
Диалог просмотра и редактирования погружения.
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
    QDateEdit,
    QTextEdit,
)
from PyQt6.QtCore import Qt, QDate

from ...database import Repository, Dive


class EditDiveDialog(QDialog):
    """
    Диалог редактирования погружения.
    """

    def __init__(self, repository: Repository, dive_id: int, parent=None):
        super().__init__(parent)
        self.repo = repository
        self.dive_id = dive_id
        self.dive = self.repo.get_dive(dive_id)
        
        if not self.dive:
            raise ValueError(f"Dive {dive_id} not found")
        
        self._setup_ui()
        self._load_data()

    def _setup_ui(self):
        """Настройка интерфейса."""
        self.setWindowTitle("Редактировать погружение")
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        # === Папка (только чтение) ===
        folder_group = QGroupBox("Папка с данными")
        folder_layout = QHBoxLayout(folder_group)
        
        self.label_folder = QLabel()
        self.label_folder.setWordWrap(True)
        self.label_folder.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        folder_layout.addWidget(self.label_folder)
        
        layout.addWidget(folder_group)
        
        # === Информация ===
        info_group = QGroupBox("Информация о погружении")
        info_layout = QFormLayout(info_group)
        
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("Название погружения")
        info_layout.addRow("Название:", self.edit_name)
        
        self.edit_date = QDateEdit()
        self.edit_date.setCalendarPopup(True)
        self.edit_date.setDisplayFormat("dd.MM.yyyy")
        # Добавляем возможность очистить дату
        date_row = QHBoxLayout()
        date_row.addWidget(self.edit_date)
        self.btn_clear_date = QPushButton("✕")
        self.btn_clear_date.setFixedWidth(30)
        self.btn_clear_date.setToolTip("Очистить дату")
        self.btn_clear_date.clicked.connect(self._clear_date)
        date_row.addWidget(self.btn_clear_date)
        date_row.addStretch()
        info_layout.addRow("Дата:", date_row)
        
        self.edit_location = QLineEdit()
        self.edit_location.setPlaceholderText("Место проведения")
        info_layout.addRow("Место:", self.edit_location)
        
        self.edit_notes = QTextEdit()
        self.edit_notes.setMaximumHeight(100)
        self.edit_notes.setPlaceholderText("Дополнительные заметки...")
        info_layout.addRow("Заметки:", self.edit_notes)
        
        layout.addWidget(info_group)
        
        # === Статистика ===
        stats_group = QGroupBox("Статистика")
        stats_layout = QFormLayout(stats_group)
        
        self.label_videos = QLabel()
        stats_layout.addRow("Видеофайлов:", self.label_videos)
        
        self.label_ctd = QLabel()
        stats_layout.addRow("CTD файлов:", self.label_ctd)
        
        self.label_tasks = QLabel()
        stats_layout.addRow("Задач:", self.label_tasks)
        
        self.label_created = QLabel()
        stats_layout.addRow("Создано:", self.label_created)
        
        layout.addWidget(stats_group)
        
        # === Кнопки ===
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._save_and_accept)
        button_box.rejected.connect(self.reject)
        
        layout.addWidget(button_box)
        
        self._date_cleared = False

    def _load_data(self):
        """Загружает данные."""
        dive = self.dive
        
        # Папка
        self.label_folder.setText(dive.folder_path)
        
        # Информация
        self.edit_name.setText(dive.name)
        
        if dive.date:
            if isinstance(dive.date, date):
                self.edit_date.setDate(QDate(dive.date.year, dive.date.month, dive.date.day))
            else:
                self.edit_date.setDate(QDate.currentDate())
            self._date_cleared = False
        else:
            self.edit_date.setDate(QDate.currentDate())
            self._date_cleared = True
            self.edit_date.setStyleSheet("color: gray;")
        
        self.edit_location.setText(dive.location or "")
        self.edit_notes.setText(dive.notes or "")
        
        # Статистика
        videos = self.repo.get_videos_by_dive(dive.id)
        ctd_files = self.repo.get_ctd_by_dive(dive.id)
        
        self.label_videos.setText(str(len(videos)))
        self.label_ctd.setText(str(len(ctd_files)))
        
        # Подсчёт задач
        task_count = 0
        for video in videos:
            # Простой подсчёт через репозиторий
            with self.repo.get_session() as session:
                from sqlalchemy import select, func
                from ...database import Task
                stmt = select(func.count()).select_from(Task).where(Task.video_id == video.id)
                task_count += session.scalar(stmt) or 0
        
        self.label_tasks.setText(str(task_count))
        self.label_created.setText(dive.created_at.strftime("%d.%m.%Y %H:%M"))

    def _clear_date(self):
        """Очищает дату."""
        self._date_cleared = True
        self.edit_date.setStyleSheet("color: gray;")

    def _save_and_accept(self):
        """Сохраняет изменения."""
        name = self.edit_name.text().strip()
        if not name:
            self.edit_name.setFocus()
            return
        
        # Дата
        dive_date = None
        if not self._date_cleared:
            qdate = self.edit_date.date()
            dive_date = date(qdate.year(), qdate.month(), qdate.day())
        
        # Обновляем
        self.repo.update_dive(
            self.dive_id,
            name=name,
            date=dive_date,
            location=self.edit_location.text().strip() or None,
            notes=self.edit_notes.toPlainText().strip() or None,
        )
        
        self.accept()
