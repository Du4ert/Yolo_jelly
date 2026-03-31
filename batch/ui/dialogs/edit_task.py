"""
Диалог просмотра и редактирования задачи.
"""

from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QComboBox,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QDialogButtonBox,
    QTabWidget,
    QWidget,
    QTextEdit,
)
from PyQt6.QtCore import Qt

from ...database import Repository, Task, TaskStatus


class EditTaskDialog(QDialog):
    """
    Диалог просмотра и редактирования задачи.
    """

    def __init__(self, repository: Repository, task_id: int, parent=None):
        super().__init__(parent)
        self.repo = repository
        self.task_id = task_id
        self.task = self.repo.get_task(task_id)
        
        if not self.task:
            raise ValueError(f"Task {task_id} not found")
        
        self._setup_ui()
        self._load_data()

    def _setup_ui(self):
        """Настройка интерфейса."""
        self.setWindowTitle(f"Задача #{self.task_id}")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        
        layout = QVBoxLayout(self)
        
        # Табы
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # === Вкладка "Информация" ===
        info_tab = QWidget()
        info_layout = QVBoxLayout(info_tab)
        
        # Статус
        status_group = QGroupBox("Статус")
        status_layout = QFormLayout(status_group)
        
        self.label_status = QLabel()
        status_layout.addRow("Статус:", self.label_status)
        
        self.label_progress = QLabel()
        status_layout.addRow("Прогресс:", self.label_progress)
        
        self.label_created = QLabel()
        status_layout.addRow("Создана:", self.label_created)
        
        self.label_started = QLabel()
        status_layout.addRow("Начата:", self.label_started)
        
        self.label_completed = QLabel()
        status_layout.addRow("Завершена:", self.label_completed)
        
        self.label_time = QLabel()
        status_layout.addRow("Время обработки:", self.label_time)
        
        info_layout.addWidget(status_group)
        
        # Файлы
        files_group = QGroupBox("Файлы")
        files_layout = QFormLayout(files_group)
        
        self.label_video = QLabel()
        self.label_video.setWordWrap(True)
        self.label_video.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        files_layout.addRow("Видео:", self.label_video)
        
        self.label_ctd = QLabel()
        self.label_ctd.setWordWrap(True)
        self.label_ctd.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        files_layout.addRow("CTD:", self.label_ctd)
        
        self.label_model = QLabel()
        self.label_model.setWordWrap(True)
        self.label_model.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        files_layout.addRow("Модель:", self.label_model)
        
        info_layout.addWidget(files_group)
        
        # Результаты
        results_group = QGroupBox("Результаты")
        results_layout = QFormLayout(results_group)
        
        self.label_detections = QLabel()
        results_layout.addRow("Детекций:", self.label_detections)
        
        self.label_tracks = QLabel()
        results_layout.addRow("Треков:", self.label_tracks)
        
        self.label_error = QLabel()
        self.label_error.setWordWrap(True)
        self.label_error.setStyleSheet("color: red;")
        results_layout.addRow("Ошибка:", self.label_error)
        
        info_layout.addWidget(results_group)
        info_layout.addStretch()
        
        tabs.addTab(info_tab, "Информация")
        
        # === Вкладка "Параметры" ===
        params_tab = QWidget()
        params_layout = QVBoxLayout(params_tab)
        
        # Можно редактировать только pending задачи
        self._is_editable = self.task.status == TaskStatus.PENDING
        
        # Детекция
        detection_group = QGroupBox("Параметры детекции")
        detection_layout = QFormLayout(detection_group)
        
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 0.99)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setEnabled(self._is_editable)
        detection_layout.addRow("Порог уверенности:", self.spin_conf)
        
        self.spin_depth_rate = QDoubleSpinBox()
        self.spin_depth_rate.setRange(0.0, 10.0)
        self.spin_depth_rate.setSingleStep(0.1)
        self.spin_depth_rate.setDecimals(2)
        self.spin_depth_rate.setSpecialValueText("Не задано")
        self.spin_depth_rate.setEnabled(self._is_editable)
        detection_layout.addRow("Скорость погружения (м/с):", self.spin_depth_rate)
        
        params_layout.addWidget(detection_group)
        
        # Трекинг
        tracking_group = QGroupBox("Трекинг")
        tracking_layout = QVBoxLayout(tracking_group)
        
        self.check_tracking = QCheckBox("Включить трекинг")
        self.check_tracking.setEnabled(self._is_editable)
        tracking_layout.addWidget(self.check_tracking)
        
        tracking_form = QFormLayout()
        tracking_form.setContentsMargins(20, 0, 0, 0)
        
        self.combo_tracker = QComboBox()
        self.combo_tracker.addItem("ByteTrack", "bytetrack.yaml")
        self.combo_tracker.addItem("BoT-SORT", "botsort.yaml")
        self.combo_tracker.setEnabled(self._is_editable)
        tracking_form.addRow("Трекер:", self.combo_tracker)
        
        self.check_trails = QCheckBox("Показывать траектории")
        self.check_trails.setEnabled(self._is_editable)
        tracking_form.addRow("", self.check_trails)
        
        self.spin_trail_length = QSpinBox()
        self.spin_trail_length.setRange(10, 200)
        self.spin_trail_length.setEnabled(self._is_editable)
        tracking_form.addRow("Длина траектории:", self.spin_trail_length)
        
        self.spin_min_track = QSpinBox()
        self.spin_min_track.setRange(1, 50)
        self.spin_min_track.setEnabled(self._is_editable)
        tracking_form.addRow("Мин. длина трека:", self.spin_min_track)
        
        tracking_layout.addLayout(tracking_form)
        params_layout.addWidget(tracking_group)
        
        # GPU / Ускорение
        gpu_group = QGroupBox("GPU / Ускорение")
        gpu_layout = QFormLayout(gpu_group)

        self.combo_device = QComboBox()
        self.combo_device.addItem("Автоматически", "auto")
        self.combo_device.addItem("CPU", "cpu")
        self.combo_device.addItem("GPU 0", "0")
        self.combo_device.setEnabled(self._is_editable)
        gpu_layout.addRow("Устройство:", self.combo_device)

        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(320, 3840)
        self.spin_imgsz.setSingleStep(32)
        self.spin_imgsz.setEnabled(self._is_editable)
        gpu_layout.addRow("Размер изображения (imgsz):", self.spin_imgsz)

        self.check_half = QCheckBox("FP16 (half precision)")
        self.check_half.setEnabled(self._is_editable)
        gpu_layout.addRow("", self.check_half)

        params_layout.addWidget(gpu_group)

        # Выход
        output_group = QGroupBox("Выход")
        output_layout = QVBoxLayout(output_group)

        self.check_save_video = QCheckBox("Сохранять видео с разметкой")
        self.check_save_video.setEnabled(self._is_editable)
        output_layout.addWidget(self.check_save_video)

        self.check_export_label_studio = QCheckBox("Экспортировать кадры и разметку для Label Studio")
        self.check_export_label_studio.setEnabled(self._is_editable)
        self.check_export_label_studio.toggled.connect(self._on_export_ls_toggled)
        output_layout.addWidget(self.check_export_label_studio)

        # Интервал кадров для Label Studio
        self.ls_interval_widget = QWidget()
        ls_interval_layout = QHBoxLayout(self.ls_interval_widget)
        ls_interval_layout.setContentsMargins(20, 0, 0, 0)
        ls_interval_label = QLabel("Интервал кадров:")
        ls_interval_layout.addWidget(ls_interval_label)
        self.spin_ls_interval = QSpinBox()
        self.spin_ls_interval.setRange(1, 300)
        self.spin_ls_interval.setValue(15)
        self.spin_ls_interval.setMaximumWidth(80)
        self.spin_ls_interval.setEnabled(self._is_editable)
        self.spin_ls_interval.setToolTip(
            "Сохранять каждый N-й кадр с детекциями.\n"
            "1 = каждый кадр, 15 = каждый 15-й.\n"
            "Уменьшает количество похожих кадров."
        )
        ls_interval_layout.addWidget(self.spin_ls_interval)
        ls_interval_layout.addStretch()
        self.ls_interval_widget.setVisible(False)
        output_layout.addWidget(self.ls_interval_widget)

        # Выбор классов для экспорта Label Studio
        self.ls_classes_widget = QWidget()
        ls_classes_layout = QVBoxLayout(self.ls_classes_widget)
        ls_classes_layout.setContentsMargins(20, 0, 0, 0)
        ls_classes_layout.setSpacing(2)
        ls_classes_label = QLabel("Классы для экспорта:")
        ls_classes_layout.addWidget(ls_classes_label)
        self.ls_class_checks = {}
        ls_classes_ordered = [
            'Rhizostoma pulmo', 'Beroe ovata', 'Mnemiopsis leidyi',
            'Pleurobrachia pileus', 'Aurelia aurita',
        ]
        for class_name in ls_classes_ordered:
            chk = QCheckBox(class_name)
            chk.setChecked(True)
            chk.setEnabled(self._is_editable)
            self.ls_class_checks[class_name] = chk
            ls_classes_layout.addWidget(chk)
        self.ls_classes_widget.setVisible(False)
        output_layout.addWidget(self.ls_classes_widget)

        params_layout.addWidget(output_group)
        params_layout.addStretch()
        
        tabs.addTab(params_tab, "Параметры")
        
        # === Вкладка "Выходные файлы" ===
        outputs_tab = QWidget()
        outputs_layout = QVBoxLayout(outputs_tab)
        
        self.outputs_text = QTextEdit()
        self.outputs_text.setReadOnly(True)
        outputs_layout.addWidget(self.outputs_text)
        
        tabs.addTab(outputs_tab, "Выходные файлы")
        
        # === Кнопки ===
        button_box = QDialogButtonBox()
        
        if self._is_editable:
            button_box.setStandardButtons(
                QDialogButtonBox.StandardButton.Save | 
                QDialogButtonBox.StandardButton.Cancel
            )
            button_box.accepted.connect(self._save_and_accept)
        else:
            button_box.setStandardButtons(QDialogButtonBox.StandardButton.Close)
        
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _load_data(self):
        """Загружает данные."""
        task = self.task
        
        # Статус
        status_text = {
            TaskStatus.PENDING: "⏳ Ожидание",
            TaskStatus.RUNNING: "▶ Выполняется",
            TaskStatus.PAUSED: "⏸ Пауза",
            TaskStatus.DONE: "✓ Завершено",
            TaskStatus.ERROR: "✗ Ошибка",
            TaskStatus.CANCELLED: "⊘ Отменено",
        }
        self.label_status.setText(status_text.get(task.status, str(task.status)))
        
        self.label_progress.setText(f"{task.progress_percent:.1f}% (кадр {task.current_frame})")
        
        self.label_created.setText(task.created_at.strftime("%d.%m.%Y %H:%M:%S") if task.created_at else "-")
        self.label_started.setText(task.started_at.strftime("%d.%m.%Y %H:%M:%S") if task.started_at else "-")
        self.label_completed.setText(task.completed_at.strftime("%d.%m.%Y %H:%M:%S") if task.completed_at else "-")
        
        if task.processing_time_s:
            mins = int(task.processing_time_s // 60)
            secs = int(task.processing_time_s % 60)
            self.label_time.setText(f"{mins}:{secs:02d}")
        else:
            self.label_time.setText("-")
        
        # Файлы
        video = self.repo.get_video_file(task.video_id)
        if video:
            self.label_video.setText(f"{video.filename}\n{video.filepath}")
        else:
            self.label_video.setText("Не найден")
        
        if task.ctd_id:
            ctd = self.repo.get_ctd_file(task.ctd_id)
            if ctd:
                self.label_ctd.setText(f"{ctd.filename}\n{ctd.filepath}")
            else:
                self.label_ctd.setText("Не найден")
        else:
            self.label_ctd.setText("Не используется")
        
        model = self.repo.get_model(task.model_id)
        if model:
            self.label_model.setText(f"{model.name}\n{model.filepath}")
        else:
            self.label_model.setText("Не найдена")
        
        # Результаты
        self.label_detections.setText(str(task.detections_count) if task.detections_count else "-")
        self.label_tracks.setText(str(task.tracks_count) if task.tracks_count else "-")
        self.label_error.setText(task.error_message if task.error_message else "-")
        
        # Параметры
        self.spin_conf.setValue(task.conf_threshold)
        self.spin_depth_rate.setValue(task.depth_rate or 0)
        self.check_tracking.setChecked(task.enable_tracking)
        
        for i in range(self.combo_tracker.count()):
            if self.combo_tracker.itemData(i) == task.tracker_type:
                self.combo_tracker.setCurrentIndex(i)
                break
        
        self.check_trails.setChecked(task.show_trails)
        self.spin_trail_length.setValue(task.trail_length)
        self.spin_min_track.setValue(task.min_track_length)
        self.check_save_video.setChecked(task.save_video)
        self.check_export_label_studio.setChecked(task.export_label_studio)
        self.spin_ls_interval.setValue(task.export_ls_interval if task.export_ls_interval else 15)
        # Восстанавливаем выбор классов
        if task.export_ls_classes:
            import json
            selected = set(json.loads(task.export_ls_classes))
            for name, chk in self.ls_class_checks.items():
                chk.setChecked(name in selected)
        self._on_export_ls_toggled(task.export_label_studio)

        # GPU / Ускорение
        device_val = task.device or "auto"
        for i in range(self.combo_device.count()):
            if self.combo_device.itemData(i) == device_val:
                self.combo_device.setCurrentIndex(i)
                break
        self.spin_imgsz.setValue(task.imgsz or 1280)
        self.check_half.setChecked(task.half if task.half is not None else True)

        # Выходные файлы
        outputs = self.repo.get_task_outputs(task.id)
        if outputs:
            text = ""
            for out in outputs:
                size_str = f" ({out.filesize_mb:.1f} MB)" if out.filesize_mb else ""
                text += f"• {out.output_type.value}: {out.filepath}{size_str}\n"
            self.outputs_text.setText(text)
        else:
            self.outputs_text.setText("Нет выходных файлов")

    def _on_export_ls_toggled(self, enabled: bool):
        """Показывает/скрывает настройки Label Studio."""
        self.ls_interval_widget.setVisible(enabled)
        self.ls_classes_widget.setVisible(enabled)

    def _get_ls_classes_json(self) -> str:
        """Возвращает JSON со списком выбранных классов для LS-экспорта. Пустая строка = все."""
        import json
        selected = [name for name, chk in self.ls_class_checks.items() if chk.isChecked()]
        if len(selected) == len(self.ls_class_checks):
            return ""
        return json.dumps(selected, ensure_ascii=False)

    def _save_and_accept(self):
        """Сохраняет изменения."""
        params = {
            "conf_threshold": self.spin_conf.value(),
            "depth_rate": self.spin_depth_rate.value() if self.spin_depth_rate.value() > 0 else None,
            "enable_tracking": self.check_tracking.isChecked(),
            "tracker_type": self.combo_tracker.currentData(),
            "show_trails": self.check_trails.isChecked(),
            "trail_length": self.spin_trail_length.value(),
            "min_track_length": self.spin_min_track.value(),
            "save_video": self.check_save_video.isChecked(),
            "export_label_studio": self.check_export_label_studio.isChecked(),
            "export_ls_interval": self.spin_ls_interval.value(),
            "export_ls_classes": self._get_ls_classes_json() or None,
            # GPU / Ускорение
            "device": self.combo_device.currentData(),
            "imgsz": self.spin_imgsz.value(),
            "half": self.check_half.isChecked(),
        }
        
        self.repo.update_task(self.task_id, **params)
        self.accept()
