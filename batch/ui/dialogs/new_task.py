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
    QFileDialog,
)
from PyQt6.QtCore import Qt

from ...database import Repository, VideoFile, CTDFile, Model
from ...core import get_calibration_defaults, get_config, save_config


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
        
        # === GPU / Ускорение ===
        gpu_group = QGroupBox("GPU / Ускорение")
        gpu_layout = QFormLayout(gpu_group)

        self.combo_device = QComboBox()
        self.combo_device.addItem("Автоматически", "auto")
        self.combo_device.addItem("CPU", "cpu")
        self.combo_device.addItem("GPU 0", "0")
        self.combo_device.setToolTip(
            "Устройство для инференса YOLO.\n"
            "auto — GPU если доступен, иначе CPU."
        )
        gpu_layout.addRow("Устройство:", self.combo_device)

        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(320, 3840)
        self.spin_imgsz.setSingleStep(32)
        self.spin_imgsz.setToolTip(
            "Размер изображения для YOLO-инференса.\n"
            "Должен совпадать с imgsz при обучении модели (обычно 1280).\n"
            "Больше = точнее, но медленнее и больше памяти GPU."
        )
        gpu_layout.addRow("Размер изображения (imgsz):", self.spin_imgsz)

        self.check_half = QCheckBox("FP16 (half precision)")
        self.check_half.setToolTip(
            "Использовать половинную точность для инференса.\n"
            "Ускоряет обработку в ~2 раза на GPU с минимальным влиянием на точность.\n"
            "Не поддерживается на CPU."
        )
        gpu_layout.addRow("", self.check_half)

        layout.addWidget(gpu_group)

        # === Выход ===
        output_group = QGroupBox("Выход")
        output_layout = QVBoxLayout(output_group)
        
        self.check_save_video = QCheckBox("Сохранять видео с разметкой")
        output_layout.addWidget(self.check_save_video)

        self.check_export_label_studio = QCheckBox("Экспортировать кадры и разметку для Label Studio")
        self.check_export_label_studio.setToolTip(
            "Сохранить кадры с детекциями в папки по классам и JSON-файл предразметки для импорта в Label Studio"
        )
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
        # Классы в порядке убывания приоритета
        ls_classes_ordered = [
            'Rhizostoma pulmo', 'Beroe ovata', 'Mnemiopsis leidyi',
            'Pleurobrachia pileus', 'Aurelia aurita',
        ]
        for class_name in ls_classes_ordered:
            chk = QCheckBox(class_name)
            chk.setChecked(True)
            self.ls_class_checks[class_name] = chk
            ls_classes_layout.addWidget(chk)
        self.ls_classes_widget.setVisible(False)
        output_layout.addWidget(self.ls_classes_widget)

        # Папка экспорта Label Studio
        self.ls_dir_widget = QWidget()
        ls_dir_layout = QHBoxLayout(self.ls_dir_widget)
        ls_dir_layout.setContentsMargins(20, 0, 0, 0)
        ls_dir_layout.addWidget(QLabel("Папка экспорта:"))
        self.edit_ls_dir = QLineEdit()
        self.edit_ls_dir.setPlaceholderText("По умолчанию — папка погружения")
        self.edit_ls_dir.setText(get_config().ui.label_studio_dir or "")
        self.edit_ls_dir.setToolTip(
            "Общая папка для экспорта кадров и предразметки.\n"
            "Если пусто — экспорт в папку погружения."
        )
        self.edit_ls_dir.textChanged.connect(self._on_ls_dir_changed)
        ls_dir_layout.addWidget(self.edit_ls_dir)
        btn_ls_browse = QPushButton("Обзор...")
        btn_ls_browse.setMaximumWidth(80)
        btn_ls_browse.clicked.connect(self._browse_ls_dir)
        ls_dir_layout.addWidget(btn_ls_browse)
        self.ls_dir_widget.setVisible(False)
        output_layout.addWidget(self.ls_dir_widget)

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
        
        # Модели — в том же порядке, что и в панели моделей
        models = self.repo.get_all_models()
        order = get_config().ui.models_order
        if order:
            models.sort(key=lambda m: order.index(m.id) if m.id in order else len(order))
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

        # GPU / Ускорение
        for i in range(self.combo_device.count()):
            if self.combo_device.itemData(i) == params.device:
                self.combo_device.setCurrentIndex(i)
                break
        self.spin_imgsz.setValue(params.imgsz)
        self.check_half.setChecked(params.half)

        # По умолчанию автопостобработка выключена
        self.check_auto_postprocess.setChecked(False)
        self._populate_calibration_defaults()
        
        self._on_tracking_toggled(params.enable_tracking)
        self._update_depth_rate_state()
        self._on_export_ls_toggled(False)
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

    def _on_export_ls_toggled(self, enabled: bool):
        """Показывает/скрывает настройки Label Studio."""
        self.ls_interval_widget.setVisible(enabled)
        self.ls_classes_widget.setVisible(enabled)
        self.ls_dir_widget.setVisible(enabled)

    def _browse_ls_dir(self):
        """Диалог выбора папки для экспорта Label Studio."""
        current = self.edit_ls_dir.text() or ""
        path = QFileDialog.getExistingDirectory(self, "Папка экспорта Label Studio", current)
        if path:
            self.edit_ls_dir.setText(path)

    def _on_ls_dir_changed(self, text: str):
        """Сохраняет выбранную папку LS в глобальный конфиг."""
        config = get_config()
        config.ui.label_studio_dir = text or None
        save_config()

    def _get_ls_classes_json(self) -> str:
        """Возвращает JSON со списком выбранных классов для LS-экспорта. Пустая строка = все."""
        import json
        selected = [name for name, chk in self.ls_class_checks.items() if chk.isChecked()]
        if len(selected) == len(self.ls_class_checks):
            return ""  # все выбраны — null в БД
        return json.dumps(selected, ensure_ascii=False)

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
            "export_label_studio": self.check_export_label_studio.isChecked(),
            "export_ls_interval": self.spin_ls_interval.value(),
            "export_ls_classes": self._get_ls_classes_json() or None,
            "export_ls_dir": self.edit_ls_dir.text() or None,
            "auto_postprocess": self.check_auto_postprocess.isChecked(),
            # GPU / Ускорение
            "device": self.combo_device.currentData(),
            "imgsz": self.spin_imgsz.value(),
            "half": self.check_half.isChecked(),
        }
        
        # Скорость погружения только если нет CTD
        if self.combo_ctd.currentData() is None and self.spin_depth_rate.value() > 0:
            params["depth_rate"] = self.spin_depth_rate.value()
        
        # Параметры автопостобработки
        if self.check_auto_postprocess.isChecked():
            effective_distance_auto = self.pp_chk_effective_distance_auto.isChecked()
            try:
                config = get_config()
                config.ui.min_reliable_distance = self.pp_spin_min_reliable.value()
                config.ui.max_reliable_distance = self.pp_spin_max_reliable.value()
                config.ui.effective_distance_auto = effective_distance_auto
                config.ui.effective_distance = self.pp_spin_effective_distance.value()
                save_config()
            except Exception:
                pass

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
                "fov_horizontal": self.pp_spin_fov.value(),
                "fov_vertical": self.pp_spin_fov_vertical.value(),
                "min_reliable_distance": self.pp_spin_min_reliable.value(),
                "max_reliable_distance": self.pp_spin_max_reliable.value(),
                "effective_distance_auto": effective_distance_auto,
                "detection_distance": (
                    None if effective_distance_auto else self.pp_spin_effective_distance.value()
                ),
                "depth_bin": self.pp_spin_depth_bin.value(),
                "ctd_columns": self.pp_edit_ctd_columns.text().strip() or "6,11,12",
                "frame_step": self.pp_spin_frame_step.value(),
            }
            try:
                calib_path = get_config().ui.calibration_json
                if calib_path:
                    postprocess_params["calibration_json"] = calib_path
            except Exception:
                pass
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
        self.pp_chk_geometry.setToolTip(
            "Оценка наклона камеры по Focus of Expansion.\n\n"
            "Если выбрано — геометрия пересчитывается и используется для всей\n"
            "последующей постобработки.\n"
            "Если не выбрано, но в папке output уже есть *_geometry.csv —\n"
            "постобработка использует существующий файл."
        )
        self.pp_chk_geometry.setChecked(True)
        layout.addWidget(self.pp_chk_geometry)

        # Опция frame_step для геометрии
        geom_step_indent = QWidget()
        geom_step_layout = QHBoxLayout(geom_step_indent)
        geom_step_layout.setContentsMargins(20, 0, 0, 0)
        geom_step_label = QLabel("Шаг кадров:")
        geom_step_layout.addWidget(geom_step_label)
        self.pp_spin_frame_step = QSpinBox()
        self.pp_spin_frame_step.setRange(1, 5)
        self.pp_spin_frame_step.setValue(1)
        self.pp_spin_frame_step.setMaximumWidth(60)
        self.pp_spin_frame_step.setToolTip(
            "Шаг чтения кадров для optical flow.\n"
            "1 = каждый кадр (максимальная точность)\n"
            "2 = через кадр (~2x быстрее)\n"
            "3 = каждый 3-й (~3x быстрее)"
        )
        geom_step_layout.addWidget(self.pp_spin_frame_step)
        geom_step_layout.addStretch()
        layout.addWidget(geom_step_indent)

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
            "Коррекция k-значений с учётом угла наклона камеры:\n"
            "k_real = k_measured / cos(θ)\n\n"
            "Применяется, если в папке output уже есть файл *_geometry.csv\n"
            "или если выбран пункт «Геометрия камеры (FOE)» (он пересчитает\n"
            "геометрию заново). Если геометрии нет — опция игнорируется,\n"
            "размеры считаются как для вертикальной камеры."
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
            "Отображать информацию об углах наклона камеры на видео.\n\n"
            "Применяется, если в папке output уже есть файл *_geometry.csv\n"
            "или если выбран пункт «Геометрия камеры (FOE)».\n"
            "Если геометрии нет — опция игнорируется."
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
        self.pp_edit_ctd_columns = QLineEdit("6,11,12")
        self.pp_edit_ctd_columns.setMaximumWidth(120)
        self.pp_edit_ctd_columns.setToolTip(
            "Колонки CTD для интерактивного графика (0-based индексы), через запятую.\n"
            "Например: 6 или 6,11,12\n"
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
        self.pp_spin_fov.setValue(95.0)
        self.pp_spin_fov.setSuffix("°")
        self.pp_spin_fov.setToolTip("Горизонтальный угол обзора камеры")
        params_form.addRow("Горизонтальный FOV:", self.pp_spin_fov)

        self.pp_spin_fov_vertical = QDoubleSpinBox()
        self.pp_spin_fov_vertical.setRange(30, 180)
        self.pp_spin_fov_vertical.setValue(55.0)
        self.pp_spin_fov_vertical.setSuffix("°")
        self.pp_spin_fov_vertical.setToolTip("Вертикальный угол обзора камеры")
        params_form.addRow("Вертикальный FOV:", self.pp_spin_fov_vertical)
        
        self.pp_spin_min_reliable = QDoubleSpinBox()
        self.pp_spin_min_reliable.setRange(0.05, 2.0)
        self.pp_spin_min_reliable.setValue(0.1)
        self.pp_spin_min_reliable.setSingleStep(0.05)
        self.pp_spin_min_reliable.setSuffix(" м")
        self.pp_spin_min_reliable.setToolTip("Ближняя дистанция: граница обнаружения для объёма и минимум для оценки размеров")
        params_form.addRow("Ближняя дистанция:", self.pp_spin_min_reliable)

        self.pp_spin_max_reliable = QDoubleSpinBox()
        self.pp_spin_max_reliable.setRange(0.1, 20.0)
        self.pp_spin_max_reliable.setValue(3.0)
        self.pp_spin_max_reliable.setSingleStep(0.5)
        self.pp_spin_max_reliable.setSuffix(" м")
        self.pp_spin_max_reliable.setToolTip("Верхняя граница автооценки effective distance и надёжности размеров")
        params_form.addRow("Дальняя надёжная дистанция:", self.pp_spin_max_reliable)

        self.pp_chk_effective_distance_auto = QCheckBox("Рассчитать автоматически")
        self.pp_chk_effective_distance_auto.setChecked(True)
        self.pp_chk_effective_distance_auto.toggled.connect(
            self._on_effective_distance_auto_toggled
        )
        params_form.addRow("Effective distance:", self.pp_chk_effective_distance_auto)

        self.pp_spin_effective_distance = QDoubleSpinBox()
        self.pp_spin_effective_distance.setRange(0.05, 20.0)
        self.pp_spin_effective_distance.setValue(1.0)
        self.pp_spin_effective_distance.setSingleStep(0.5)
        self.pp_spin_effective_distance.setSuffix(" м")
        self.pp_spin_effective_distance.setEnabled(False)
        self.pp_spin_effective_distance.setToolTip("Ручная effective distance для площади основания эллипса цилиндра")
        params_form.addRow("Ручная effective distance:", self.pp_spin_effective_distance)

        self.pp_spin_depth_bin = QDoubleSpinBox()
        self.pp_spin_depth_bin.setRange(0.5, 10.0)
        self.pp_spin_depth_bin.setValue(2.0)
        self.pp_spin_depth_bin.setSingleStep(0.5)
        self.pp_spin_depth_bin.setSuffix(" м")
        self.pp_spin_depth_bin.setToolTip("Шаг биннинга по глубине для графиков распределения")
        params_form.addRow("Бин глубины:", self.pp_spin_depth_bin)
        
        layout.addLayout(params_form)
        
        return widget

    def _populate_calibration_defaults(self):
        try:
            path = get_config().ui.calibration_json
        except Exception:
            path = None
        defaults = get_calibration_defaults(path)
        self.pp_spin_min_reliable.setValue(defaults["min_reliable_distance"])
        self.pp_spin_max_reliable.setValue(defaults["max_reliable_distance"])
        auto = defaults.get("effective_distance_auto", True)
        self.pp_chk_effective_distance_auto.setChecked(auto)
        if defaults.get("effective_distance") is not None:
            self.pp_spin_effective_distance.setValue(defaults["effective_distance"])
        self._on_effective_distance_auto_toggled(auto)

    def _on_effective_distance_auto_toggled(self, enabled: bool):
        self.pp_spin_effective_distance.setEnabled(not enabled)

    def _on_auto_postprocess_toggled(self, enabled: bool):
        """Обработка переключения автопостобработки."""
        self.postprocess_widget.setVisible(enabled)
        if enabled:
            self._update_postprocess_dependencies()

    def _update_postprocess_dependencies(self):
        """Обновляет состояние зависимых элементов постобработки.

        Чекбоксы «С коррекцией наклона камеры» и «Показывать углы наклона»
        остаются всегда активными независимо от чекбокса «Геометрия камеры (FOE)».
        Во время выполнения они применяются, только если в папке output
        есть файл *_geometry.csv (либо рассчитанный этой же задачей, либо
        существовавший ранее); иначе опция молча игнорируется.
        """
        size_selected = self.pp_chk_size.isChecked()
        size_video_selected = self.pp_chk_size_video.isChecked()

        # Видео с размерами требует размеров
        if size_video_selected and not size_selected:
            self.pp_chk_size.setChecked(True)
