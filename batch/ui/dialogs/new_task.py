"""
Диалог создания новой задачи с параметрами детекции.
"""

from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QComboBox,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QDialogButtonBox,
)
from PyQt6.QtCore import Qt

from ...database import Repository, VideoFile, CTDFile, Model
from ...core import get_config


class NewTaskDialog(QDialog):
    """
    Диалог создания новой задачи с настройками детекции.
    """

    def __init__(
        self,
        repository: Repository,
        video_id: int,
        ctd_id: Optional[int] = None,
        model_id: Optional[int] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.repo = repository
        self.video_id = video_id
        self.ctd_id = ctd_id
        self.preselected_model_id = model_id
        
        self._setup_ui()
        self._load_data()
        self._apply_defaults()

    def _setup_ui(self):
        """Настройка интерфейса."""
        self.setWindowTitle("Новая задача")
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        # === Информация о файлах ===
        files_group = QGroupBox("Файлы")
        files_layout = QFormLayout(files_group)
        
        self.label_video = QLabel()
        self.label_video.setWordWrap(True)
        files_layout.addRow("Видео:", self.label_video)
        
        # Выбор CTD
        ctd_row = QHBoxLayout()
        self.combo_ctd = QComboBox()
        self.combo_ctd.setMinimumWidth(300)
        self.combo_ctd.currentIndexChanged.connect(self._on_ctd_changed)
        ctd_row.addWidget(self.combo_ctd)
        ctd_row.addStretch()
        files_layout.addRow("CTD:", ctd_row)
        
        # Выбор модели
        self.combo_model = QComboBox()
        files_layout.addRow("Модель:", self.combo_model)
        
        layout.addWidget(files_group)
        
        # === Параметры детекции ===
        detection_group = QGroupBox("Параметры детекции")
        detection_layout = QFormLayout(detection_group)
        
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 0.99)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setDecimals(2)
        detection_layout.addRow("Порог уверенности:", self.spin_conf)
        
        self.spin_depth_rate = QDoubleSpinBox()
        self.spin_depth_rate.setRange(0.0, 10.0)
        self.spin_depth_rate.setSingleStep(0.1)
        self.spin_depth_rate.setDecimals(2)
        self.spin_depth_rate.setSpecialValueText("Не задано")
        detection_layout.addRow("Скорость погружения (м/с):", self.spin_depth_rate)
        
        layout.addWidget(detection_group)
        
        # === Параметры трекинга ===
        tracking_group = QGroupBox("Трекинг")
        tracking_layout = QVBoxLayout(tracking_group)
        
        self.check_tracking = QCheckBox("Включить трекинг объектов")
        self.check_tracking.toggled.connect(self._on_tracking_toggled)
        tracking_layout.addWidget(self.check_tracking)
        
        tracking_form = QFormLayout()
        tracking_form.setContentsMargins(20, 0, 0, 0)
        
        self.combo_tracker = QComboBox()
        self.combo_tracker.addItem("ByteTrack (быстрый)", "bytetrack.yaml")
        self.combo_tracker.addItem("BoT-SORT (точный)", "botsort.yaml")
        tracking_form.addRow("Трекер:", self.combo_tracker)
        
        self.check_trails = QCheckBox("Показывать траектории")
        tracking_form.addRow("", self.check_trails)
        
        self.spin_trail_length = QSpinBox()
        self.spin_trail_length.setRange(10, 200)
        tracking_form.addRow("Длина траектории (кадров):", self.spin_trail_length)
        
        self.spin_min_track = QSpinBox()
        self.spin_min_track.setRange(1, 50)
        tracking_form.addRow("Мин. длина трека (кадров):", self.spin_min_track)
        
        tracking_layout.addLayout(tracking_form)
        layout.addWidget(tracking_group)
        
        # === Выход ===
        output_group = QGroupBox("Выход")
        output_layout = QVBoxLayout(output_group)
        
        self.check_save_video = QCheckBox("Сохранять видео с разметкой")
        output_layout.addWidget(self.check_save_video)
        
        layout.addWidget(output_group)
        
        # === Кнопки ===
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Добавить в очередь")
        
        layout.addWidget(button_box)

    def _load_data(self):
        """Загружает данные."""
        # Видео
        video = self.repo.get_video_file(self.video_id)
        if video:
            info = video.filename
            if video.duration_s:
                mins = int(video.duration_s // 60)
                secs = int(video.duration_s % 60)
                info += f" ({mins}:{secs:02d})"
            self.label_video.setText(info)
            
            # Загружаем CTD файлы из того же погружения
            dive_id = video.dive_id
            ctd_files = self.repo.get_ctd_by_dive(dive_id)
            
            self.combo_ctd.clear()
            self.combo_ctd.addItem("— Не использовать CTD —", None)
            
            for ctd in ctd_files:
                display = ctd.filename
                if ctd.max_depth:
                    display += f" (до {ctd.max_depth:.1f}м)"
                self.combo_ctd.addItem(display, ctd.id)
            
            # Выбираем предустановленный CTD или первый доступный
            if self.ctd_id:
                for i in range(self.combo_ctd.count()):
                    if self.combo_ctd.itemData(i) == self.ctd_id:
                        self.combo_ctd.setCurrentIndex(i)
                        break
            elif ctd_files:
                self.combo_ctd.setCurrentIndex(1)
        else:
            self.label_video.setText("Файл не найден")
        
        # Модели
        models = self.repo.get_all_models()
        for model in models:
            display_name = model.name
            if model.base_model:
                display_name += f" ({model.base_model})"
            self.combo_model.addItem(display_name, model.id)
        
        if self.preselected_model_id:
            for i in range(self.combo_model.count()):
                if self.combo_model.itemData(i) == self.preselected_model_id:
                    self.combo_model.setCurrentIndex(i)
                    break

    def _apply_defaults(self):
        """Применяет значения по умолчанию."""
        config = get_config()
        params = config.default_detection_params
        
        self.spin_conf.setValue(params.conf_threshold)
        self.check_tracking.setChecked(params.enable_tracking)
        
        for i in range(self.combo_tracker.count()):
            if self.combo_tracker.itemData(i) == params.tracker_type:
                self.combo_tracker.setCurrentIndex(i)
                break
        
        self.check_trails.setChecked(params.show_trails)
        self.spin_trail_length.setValue(params.trail_length)
        self.spin_min_track.setValue(params.min_track_length)
        self.check_save_video.setChecked(params.save_video)
        
        self._on_tracking_toggled(params.enable_tracking)
        self._update_depth_rate_state()

    def _on_ctd_changed(self, index: int):
        """Обработка изменения выбора CTD."""
        self._update_depth_rate_state()

    def _update_depth_rate_state(self):
        """Обновляет состояние поля depth_rate."""
        has_ctd = self.combo_ctd.currentData() is not None
        self.spin_depth_rate.setEnabled(not has_ctd)
        if has_ctd:
            self.spin_depth_rate.setValue(0)

    def _on_tracking_toggled(self, enabled: bool):
        """Обработка переключения трекинга."""
        self.combo_tracker.setEnabled(enabled)
        self.check_trails.setEnabled(enabled)
        self.spin_trail_length.setEnabled(enabled and self.check_trails.isChecked())
        self.spin_min_track.setEnabled(enabled)

    def get_task_params(self) -> Dict[str, Any]:
        """Возвращает параметры для создания задачи."""
        params = {
            "conf_threshold": self.spin_conf.value(),
            "enable_tracking": self.check_tracking.isChecked(),
            "tracker_type": self.combo_tracker.currentData(),
            "show_trails": self.check_trails.isChecked(),
            "trail_length": self.spin_trail_length.value(),
            "min_track_length": self.spin_min_track.value(),
            "save_video": self.check_save_video.isChecked(),
        }
        
        # Скорость погружения только если нет CTD
        if self.combo_ctd.currentData() is None and self.spin_depth_rate.value() > 0:
            params["depth_rate"] = self.spin_depth_rate.value()
        
        return params

    def get_model_id(self) -> Optional[int]:
        """Возвращает ID выбранной модели."""
        return self.combo_model.currentData()

    def get_ctd_id(self) -> Optional[int]:
        """Возвращает ID CTD файла."""
        return self.combo_ctd.currentData()
