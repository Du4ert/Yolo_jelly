"""
Диалог экспорта кадров и предразметки в Label Studio.

Поддерживает выбор нескольких завершённых задач.
Создаёт подзадачи LABEL_STUDIO_EXPORT, которые выполняются в общей очереди.
"""

import json

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QCheckBox,
    QSpinBox,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QWidget,
)
from PyQt6.QtCore import Qt

from ...database import Repository, TaskStatus, SubTaskType
from ...core import TaskManager, get_config, save_config


class ExportLabelStudioDialog(QDialog):
    """
    Диалог экспорта в Label Studio для одной или нескольких завершённых задач.

    Создаёт подзадачи LABEL_STUDIO_EXPORT, которые будут выполнены в общей очереди.
    """

    def __init__(
        self,
        repo: Repository,
        task_manager: TaskManager,
        task_ids: list[int],
        parent=None,
    ):
        super().__init__(parent)
        self.repo = repo
        self.task_manager = task_manager

        # Фильтруем: только завершённые задачи с CSV детекций
        self.tasks = []
        for tid in task_ids:
            task = repo.get_task_with_outputs(tid)
            if task and task.status == TaskStatus.DONE and task.detections_csv_path:
                self.tasks.append(task)

        if not self.tasks:
            raise ValueError("Нет подходящих завершённых задач с детекциями")

        count = len(self.tasks)
        if count == 1:
            title = f"Экспорт в Label Studio — задача #{self.tasks[0].id}"
        else:
            title = f"Экспорт в Label Studio — {count} задач"
        self.setWindowTitle(title)
        self.setMinimumWidth(500)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # === Информация о задачах ===
        info_group = QGroupBox("Задачи")
        info_layout = QVBoxLayout(info_group)

        if len(self.tasks) == 1:
            task = self.tasks[0]
            form = QFormLayout()
            video = self.repo.get_video_file(task.video_id)
            form.addRow("Видео:", QLabel(video.filename if video else "???"))
            det_text = f"{task.detections_count or 0} детекций"
            if task.tracks_count:
                det_text += f", {task.tracks_count} треков"
            form.addRow("Результат:", QLabel(det_text))
            info_layout.addLayout(form)
        else:
            total_det = sum(t.detections_count or 0 for t in self.tasks)
            info_layout.addWidget(QLabel(
                f"Выбрано задач: {len(self.tasks)}, "
                f"всего детекций: {total_det}"
            ))

        layout.addWidget(info_group)

        # === Параметры экспорта ===
        params_group = QGroupBox("Параметры")
        params_layout = QVBoxLayout(params_group)

        # Интервал кадров
        interval_widget = QWidget()
        interval_layout = QHBoxLayout(interval_widget)
        interval_layout.setContentsMargins(0, 0, 0, 0)
        interval_layout.addWidget(QLabel("Интервал кадров:"))
        self.spin_interval = QSpinBox()
        self.spin_interval.setRange(1, 300)
        self.spin_interval.setValue(15)
        self.spin_interval.setMaximumWidth(80)
        self.spin_interval.setToolTip(
            "Сохранять каждый N-й кадр с детекциями.\n"
            "1 = каждый кадр, 15 = каждый 15-й."
        )
        interval_layout.addWidget(self.spin_interval)
        interval_layout.addStretch()
        params_layout.addWidget(interval_widget)

        # Классы
        classes_label = QLabel("Классы для экспорта:")
        params_layout.addWidget(classes_label)
        self.class_checks = {}
        ls_classes_ordered = [
            'Rhizostoma pulmo', 'Beroe ovata', 'Mnemiopsis leidyi',
            'Pleurobrachia pileus', 'Aurelia aurita',
        ]
        for class_name in ls_classes_ordered:
            chk = QCheckBox(class_name)
            chk.setChecked(True)
            self.class_checks[class_name] = chk
            params_layout.addWidget(chk)

        layout.addWidget(params_group)

        # === Папка экспорта ===
        dir_group = QGroupBox("Папка экспорта")
        dir_layout = QHBoxLayout(dir_group)
        self.edit_dir = QLineEdit()
        self.edit_dir.setPlaceholderText("По умолчанию — папка погружения")
        self.edit_dir.setText(get_config().ui.label_studio_dir or "")
        self.edit_dir.textChanged.connect(self._on_dir_changed)
        dir_layout.addWidget(self.edit_dir)
        btn_browse = QPushButton("Обзор...")
        btn_browse.setMaximumWidth(80)
        btn_browse.clicked.connect(self._browse_dir)
        dir_layout.addWidget(btn_browse)
        layout.addWidget(dir_group)

        # === Кнопки ===
        self.btn_add = QPushButton("Добавить в очередь")
        self.btn_add.clicked.connect(self._on_add)
        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(self.close)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

    def _browse_dir(self):
        current = self.edit_dir.text() or ""
        path = QFileDialog.getExistingDirectory(self, "Папка экспорта Label Studio", current)
        if path:
            self.edit_dir.setText(path)

    def _on_dir_changed(self, text: str):
        config = get_config()
        config.ui.label_studio_dir = text or None
        save_config()

    def _get_export_classes(self) -> list | None:
        """Возвращает список выбранных классов или None если все выбраны."""
        selected = [name for name, chk in self.class_checks.items() if chk.isChecked()]
        if len(selected) == len(self.class_checks):
            return None  # все выбраны
        return selected

    def _on_add(self):
        """Создаёт подзадачи LABEL_STUDIO_EXPORT для каждой задачи."""
        export_classes = self._get_export_classes()
        if export_classes is not None and not export_classes:
            QMessageBox.warning(self, "Нет классов", "Выберите хотя бы один класс")
            return

        params = {
            "frame_interval": self.spin_interval.value(),
            "export_classes": export_classes,
            "image_quality": 95,
        }

        # Если указана папка — передаём в параметры
        custom_dir = self.edit_dir.text().strip()
        if custom_dir:
            params["output_dir"] = custom_dir

        params_json = json.dumps(params, ensure_ascii=False)

        created = 0
        for task in self.tasks:
            st = self.repo.create_subtask(
                parent_task_id=task.id,
                subtask_type=SubTaskType.LABEL_STUDIO_EXPORT,
                position=0,
                params_json=params_json,
            )
            if st:
                created += 1

        if created:
            QMessageBox.information(
                self, "Добавлено",
                f"Добавлено {created} подзадач экспорта в очередь.\n\n"
                "Подзадачи будут выполнены автоматически\n"
                "при запуске очереди."
            )
            self.task_manager.queue_changed.emit()
            self.accept()
        else:
            QMessageBox.warning(self, "Ошибка", "Не удалось создать подзадачи")
