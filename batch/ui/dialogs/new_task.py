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
    QFrame,
    QWidget,
    QScrollArea,
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
        self.setMinimumWidth(520)
        self.setMinimumHeight(400)
        self.resize(520, 600)
        
        # Основной layout диалога
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 10)
        
        # Создаём область прокрутки
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Контейнер для содержимого
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
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
        
        self.check_auto_postprocess = QCheckBox("Автоматическая постобработка после детекции")
        self.check_auto_postprocess.setToolTip(
            "После завершения детекции автоматически запустить выбранные операции постобработки"
        )
        self.check_auto_postprocess.toggled.connect(self._on_auto_postprocess_toggled)
        output_layout.addWidget(self.check_auto_postprocess)
        
        # Настройки автопостобработки (скрыты по умолчанию)
        self.postprocess_widget = self._create_postprocess_settings()
        output_layout.addWidget(self.postprocess_widget)
        
        layout.addWidget(output_group)
        
        # Устанавливаем содержимое в область прокрутки
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        
        # === Кнопки (вне области прокрутки) ===
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Добавить в очередь")
        
        # Отступ для кнопок
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(10, 0, 10, 0)
        button_layout.addWidget(button_box)
        main_layout.addWidget(button_container)

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
        
        # По умолчанию автопостобработка выключена
        self.check_auto_postprocess.setChecked(False)
        
        self._on_tracking_toggled(params.enable_tracking)
        self._update_depth_rate_state()
        self._on_auto_postprocess_toggled(False)

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
        import json
        
        params = {
            "conf_threshold": self.spin_conf.value(),
            "enable_tracking": self.check_tracking.isChecked(),
            "tracker_type": self.combo_tracker.currentData(),
            "show_trails": self.check_trails.isChecked(),
            "trail_length": self.spin_trail_length.value(),
            "min_track_length": self.spin_min_track.value(),
            "save_video": self.check_save_video.isChecked(),
            "auto_postprocess": self.check_auto_postprocess.isChecked(),
        }
        
        # Скорость погружения только если нет CTD
        if self.combo_ctd.currentData() is None and self.spin_depth_rate.value() > 0:
            params["depth_rate"] = self.spin_depth_rate.value()
        
        # Параметры автопостобработки
        if self.check_auto_postprocess.isChecked():
            postprocess_params = {
                # Выбранные операции
                "geometry": self.pp_chk_geometry.isChecked(),
                "size": self.pp_chk_size.isChecked(),
                "size_use_geometry": self.pp_chk_size_use_geometry.isChecked(),
                "size_video": self.pp_chk_size_video.isChecked(),
                "video_use_geometry": self.pp_chk_video_use_geometry.isChecked(),
                "volume": self.pp_chk_volume.isChecked(),
                "analysis": self.pp_chk_analysis.isChecked(),
                # Параметры
                "fov": self.pp_spin_fov.value(),
                "near_distance": self.pp_spin_near.value(),
                "depth_bin": self.pp_spin_depth_bin.value(),
                "ctd_columns": self.pp_edit_ctd_columns.text().strip() or "6",
            }
            params["auto_postprocess_params"] = json.dumps(postprocess_params)
        
        return params

    def get_model_id(self) -> Optional[int]:
        """Возвращает ID выбранной модели."""
        return self.combo_model.currentData()

    def get_ctd_id(self) -> Optional[int]:
        """Возвращает ID CTD файла."""
        return self.combo_ctd.currentData()

    def _create_postprocess_settings(self) -> QWidget:
        """Создаёт виджет с настройками постобработки."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 5, 0, 5)
        layout.setSpacing(4)
        
        # === Выбор операций ===
        ops_label = QLabel("Операции:")
        ops_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
        layout.addWidget(ops_label)
        
        # Геометрия
        self.pp_chk_geometry = QCheckBox("📐 Геометрия камеры (FOE)")
        self.pp_chk_geometry.setToolTip("Оценка наклона камеры по Focus of Expansion")
        self.pp_chk_geometry.setChecked(True)
        self.pp_chk_geometry.toggled.connect(self._update_postprocess_dependencies)
        layout.addWidget(self.pp_chk_geometry)
        
        # Разделитель
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.HLine)
        separator1.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator1)
        
        # Размеры объектов
        self.pp_chk_size = QCheckBox("📏 Размеры объектов")
        self.pp_chk_size.setToolTip("Расчёт реальных размеров по k-методу")
        self.pp_chk_size.setChecked(True)
        self.pp_chk_size.toggled.connect(self._update_postprocess_dependencies)
        layout.addWidget(self.pp_chk_size)
        
        # Опция коррекции наклона
        size_indent = QWidget()
        size_indent_layout = QHBoxLayout(size_indent)
        size_indent_layout.setContentsMargins(20, 0, 0, 0)
        self.pp_chk_size_use_geometry = QCheckBox("С коррекцией наклона камеры")
        self.pp_chk_size_use_geometry.setToolTip(
            "Коррекция k-значений с учётом угла наклона камеры.\n"
            "Требует расчёта геометрии."
        )
        self.pp_chk_size_use_geometry.setChecked(True)
        size_indent_layout.addWidget(self.pp_chk_size_use_geometry)
        size_indent_layout.addStretch()
        layout.addWidget(size_indent)
        
        # Видео с размерами
        self.pp_chk_size_video = QCheckBox("🎬 Видео с размерами")
        self.pp_chk_size_video.setToolTip(
            "Рендеринг видео с отображением дистанции и размера.\n"
            "Требует расчёта размеров."
        )
        self.pp_chk_size_video.setChecked(True)
        self.pp_chk_size_video.toggled.connect(self._update_postprocess_dependencies)
        layout.addWidget(self.pp_chk_size_video)
        
        # Опция отображения геометрии на видео
        video_indent = QWidget()
        video_indent_layout = QHBoxLayout(video_indent)
        video_indent_layout.setContentsMargins(20, 0, 0, 0)
        self.pp_chk_video_use_geometry = QCheckBox("Показывать углы наклона")
        self.pp_chk_video_use_geometry.setToolTip(
            "Отображать информацию об углах наклона камеры.\n"
            "Требует расчёта геометрии."
        )
        self.pp_chk_video_use_geometry.setChecked(True)
        video_indent_layout.addWidget(self.pp_chk_video_use_geometry)
        video_indent_layout.addStretch()
        layout.addWidget(video_indent)
        
        # Объём
        self.pp_chk_volume = QCheckBox("📦 Объём воды")
        self.pp_chk_volume.setToolTip("Расчёт осмотренного объёма воды и плотности организмов")
        self.pp_chk_volume.setChecked(True)
        layout.addWidget(self.pp_chk_volume)
        
        # Разделитель
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        separator2.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator2)
        
        # Анализ
        self.pp_chk_analysis = QCheckBox("📊 Анализ и графики")
        self.pp_chk_analysis.setToolTip("Генерация графиков вертикального распределения и отчётов")
        self.pp_chk_analysis.setChecked(True)
        layout.addWidget(self.pp_chk_analysis)

        # Колонки CTD для интерактивного графика
        ctd_col_indent = QWidget()
        ctd_col_indent_layout = QHBoxLayout(ctd_col_indent)
        ctd_col_indent_layout.setContentsMargins(20, 0, 0, 0)
        ctd_col_label = QLabel("Колонки CTD:")
        self.pp_edit_ctd_columns = QLineEdit("6")
        self.pp_edit_ctd_columns.setMaximumWidth(120)
        self.pp_edit_ctd_columns.setToolTip(
            "Колонки CTD для интерактивного графика (0-based индексы), через запятую.\n"
            "Например: 6 или 5,6,7\n"
            "Используется только если к задаче привязан CTD-файл."
        )
        ctd_col_indent_layout.addWidget(ctd_col_label)
        ctd_col_indent_layout.addWidget(self.pp_edit_ctd_columns)
        ctd_col_indent_layout.addStretch()
        layout.addWidget(ctd_col_indent)

        # === Параметры ===
        params_label = QLabel("Параметры:")
        params_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(params_label)
        
        params_form = QFormLayout()
        params_form.setContentsMargins(0, 0, 0, 0)
        
        self.pp_spin_fov = QDoubleSpinBox()
        self.pp_spin_fov.setRange(60, 180)
        self.pp_spin_fov.setValue(156.0)
        self.pp_spin_fov.setSuffix("°")
        self.pp_spin_fov.setToolTip("Горизонтальный угол обзора камеры (GoPro 12 Wide 4K ~156°)")
        params_form.addRow("FOV камеры:", self.pp_spin_fov)
        
        self.pp_spin_near = QDoubleSpinBox()
        self.pp_spin_near.setRange(0.1, 2.0)
        self.pp_spin_near.setValue(0.3)
        self.pp_spin_near.setSingleStep(0.1)
        self.pp_spin_near.setSuffix(" м")
        self.pp_spin_near.setToolTip("Ближняя граница обнаружения (мёртвая зона)")
        params_form.addRow("Ближняя дистанция:", self.pp_spin_near)
        
        self.pp_spin_depth_bin = QDoubleSpinBox()
        self.pp_spin_depth_bin.setRange(0.5, 10.0)
        self.pp_spin_depth_bin.setValue(2.0)
        self.pp_spin_depth_bin.setSingleStep(0.5)
        self.pp_spin_depth_bin.setSuffix(" м")
        self.pp_spin_depth_bin.setToolTip("Шаг биннинга по глубине для графиков распределения")
        params_form.addRow("Бин глубины:", self.pp_spin_depth_bin)
        
        layout.addLayout(params_form)
        
        return widget

    def _on_auto_postprocess_toggled(self, enabled: bool):
        """Обработка переключения автопостобработки."""
        self.postprocess_widget.setVisible(enabled)
        if enabled:
            self._update_postprocess_dependencies()

    def _update_postprocess_dependencies(self):
        """Обновляет состояние зависимых элементов постобработки."""
        geometry_selected = self.pp_chk_geometry.isChecked()
        size_selected = self.pp_chk_size.isChecked()
        size_video_selected = self.pp_chk_size_video.isChecked()
        
        # Опции зависящие от геометрии
        self.pp_chk_size_use_geometry.setEnabled(geometry_selected)
        self.pp_chk_video_use_geometry.setEnabled(geometry_selected)
        
        if not geometry_selected:
            self.pp_chk_size_use_geometry.setChecked(False)
            self.pp_chk_video_use_geometry.setChecked(False)
        
        # Видео с размерами требует размеров
        if size_video_selected and not size_selected:
            self.pp_chk_size.setChecked(True)
