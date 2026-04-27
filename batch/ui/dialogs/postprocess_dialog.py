"""
Диалог добавления постобработки к задаче (или группе задач).

Поддерживает:
- повторный инференс (с другой моделью или другими параметрами) первым шагом;
- per-операция выбор: «Не делать / Использовать существующее / Пересчитать»;
- групповое применение к нескольким выделенным задачам.
"""

import json
import os
from typing import List, Optional, Set, Tuple

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QComboBox,
    QPushButton,
    QMessageBox,
    QFrame,
    QWidget,
    QRadioButton,
    QButtonGroup,
    QScrollArea,
)
from PyQt6.QtCore import Qt

from ...database import (
    Repository,
    Task,
    TaskStatus,
    SubTaskType,
    OutputType,
    SUBTASK_OUTPUT_TYPES,
)
from ...core import TaskManager, get_config


# Радио-режимы для каждой операции.
ACTION_NONE = "none"
ACTION_KEEP = "keep"
ACTION_RECOMPUTE = "recompute"


class PostProcessDialog(QDialog):
    """
    Диалог постобработки одной или нескольких задач.

    Принимает список task_ids: для одной задачи — точечный режим, для группы —
    сводные бейджи и общие параметры.
    """

    def __init__(
        self,
        repo: Repository,
        task_manager: TaskManager,
        task_ids: List[int],
        parent=None,
    ):
        super().__init__(parent)
        self.repo = repo
        self.task_manager = task_manager
        self.task_ids: List[int] = list(task_ids)

        self.tasks: List[Task] = [
            t for t in (repo.get_task(tid) for tid in self.task_ids) if t is not None
        ]
        if not self.tasks:
            raise ValueError("Не выбрано ни одной задачи")

        # Группы радио-кнопок per операция.
        self._radio_groups: dict = {}

        if len(self.tasks) == 1:
            self.setWindowTitle(f"Постобработка задачи #{self.tasks[0].id}")
        else:
            self.setWindowTitle(f"Постобработка {len(self.tasks)} задач")
        self.setMinimumWidth(640)
        self.setMinimumHeight(680)

        self._setup_ui()
        self._populate_models()
        self._populate_inference_defaults()
        self._populate_existing_state()
        self._on_inference_mode_changed()

    # ------------------------------------------------------------------ UI

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # === Заголовок-сводка ===
        info_group = QGroupBox("Задачи")
        info_layout = QVBoxLayout(info_group)
        info_layout.addWidget(self._build_summary_label())
        layout.addWidget(info_group)

        # === Повторная детекция ===
        layout.addWidget(self._build_inference_group())

        # === Операции постобработки ===
        ops_group = QGroupBox("Операции постобработки")
        ops_layout = QVBoxLayout(ops_group)
        ops_layout.setSpacing(6)

        ops_layout.addWidget(self._build_op_row(
            SubTaskType.GEOMETRY, "📐 Геометрия камеры (FOE)",
            "Оценка наклона камеры по Focus of Expansion.",
        ))
        self.spin_frame_step = QSpinBox()
        self.spin_frame_step.setRange(1, 5)
        self.spin_frame_step.setValue(1)
        self.spin_frame_step.setMaximumWidth(60)
        self.spin_frame_step.setToolTip(
            "Шаг чтения кадров для optical flow.\n"
            "1 = каждый кадр (максимальная точность)\n"
            "2 = через кадр (~2x быстрее)\n"
            "3 = каждый 3-й (~3x быстрее)"
        )
        geom_row = QHBoxLayout()
        geom_row.setContentsMargins(40, 0, 0, 0)
        geom_row.addWidget(QLabel("Шаг кадров:"))
        geom_row.addWidget(self.spin_frame_step)
        geom_row.addStretch()
        ops_layout.addLayout(geom_row)

        ops_layout.addWidget(self._build_separator())

        ops_layout.addWidget(self._build_op_row(
            SubTaskType.SIZE, "📏 Размеры объектов",
            "Расчёт реальных размеров по k-методу.",
        ))
        self.chk_size_use_geometry = QCheckBox("С коррекцией наклона камеры")
        self.chk_size_use_geometry.setChecked(True)
        self.chk_size_use_geometry.setToolTip(
            "Применяется, если есть файл *_geometry.csv (рассчитанный или существующий)."
        )
        size_geom_row = QHBoxLayout()
        size_geom_row.setContentsMargins(40, 0, 0, 0)
        size_geom_row.addWidget(self.chk_size_use_geometry)
        size_geom_row.addStretch()
        ops_layout.addLayout(size_geom_row)

        self.label_calibration = QLabel()
        self.label_calibration.setStyleSheet("color: gray; padding-left: 40px;")
        ops_layout.addWidget(self.label_calibration)
        self._update_calibration_label()

        ops_layout.addWidget(self._build_op_row(
            SubTaskType.SIZE_VIDEO_RENDER, "🎬 Видео с размерами",
            "Рендер видео с дистанцией и размером под рамками.",
        ))
        self.chk_video_use_geometry = QCheckBox("Показывать углы наклона")
        self.chk_video_use_geometry.setChecked(True)
        self.chk_video_use_geometry.setToolTip(
            "Отображать информацию об углах наклона камеры на видео."
        )
        video_geom_row = QHBoxLayout()
        video_geom_row.setContentsMargins(40, 0, 0, 0)
        video_geom_row.addWidget(self.chk_video_use_geometry)
        video_geom_row.addStretch()
        ops_layout.addLayout(video_geom_row)

        ops_layout.addWidget(self._build_op_row(
            SubTaskType.VOLUME, "📦 Объём воды",
            "Расчёт осмотренного объёма воды и плотности организмов.",
        ))

        ops_layout.addWidget(self._build_separator())

        ops_layout.addWidget(self._build_op_row(
            SubTaskType.ANALYSIS, "📊 Анализ и графики",
            "Графики вертикального распределения и текстовый отчёт.",
        ))
        ctd_row = QHBoxLayout()
        ctd_row.setContentsMargins(40, 0, 0, 0)
        ctd_row.addWidget(QLabel("Колонки CTD:"))
        self.edit_ctd_columns = QLineEdit("6")
        self.edit_ctd_columns.setMaximumWidth(120)
        self.edit_ctd_columns.setToolTip(
            "Колонки CTD для интерактивного графика (через запятую).\n"
            "Используется только если к задаче привязан CTD-файл."
        )
        ctd_row.addWidget(self.edit_ctd_columns)
        ctd_row.addStretch()
        ops_layout.addLayout(ctd_row)

        layout.addWidget(ops_group)

        # === Общие параметры ===
        params_group = QGroupBox("Параметры (общие для постобработки)")
        params_layout = QFormLayout(params_group)

        self.spin_fov = QDoubleSpinBox()
        self.spin_fov.setRange(60, 180)
        self.spin_fov.setValue(156.0)
        self.spin_fov.setSuffix("°")
        params_layout.addRow("FOV камеры:", self.spin_fov)

        self.spin_min_reliable = QDoubleSpinBox()
        self.spin_min_reliable.setRange(0.05, 2.0)
        self.spin_min_reliable.setValue(0.1)
        self.spin_min_reliable.setSingleStep(0.05)
        self.spin_min_reliable.setSuffix(" м")
        params_layout.addRow("Ближняя дистанция:", self.spin_min_reliable)

        self.spin_depth_bin = QDoubleSpinBox()
        self.spin_depth_bin.setRange(0.5, 10.0)
        self.spin_depth_bin.setValue(2.0)
        self.spin_depth_bin.setSingleStep(0.5)
        self.spin_depth_bin.setSuffix(" м")
        params_layout.addRow("Бин глубины:", self.spin_depth_bin)

        layout.addWidget(params_group)

        # Footer-предупреждение
        self.label_warning = QLabel()
        self.label_warning.setStyleSheet("color: #b58900;")
        self.label_warning.setWordWrap(True)
        layout.addWidget(self.label_warning)

        scroll.setWidget(content)
        outer.addWidget(scroll)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(10, 0, 10, 0)
        self.btn_add = QPushButton("➕ Добавить в очередь")
        self.btn_add.clicked.connect(self._on_add)
        btn_row.addWidget(self.btn_add)
        btn_row.addStretch()
        self.btn_close = QPushButton("Закрыть")
        self.btn_close.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_close)
        outer.addLayout(btn_row)

    def _build_separator(self) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        return sep

    def _build_summary_label(self) -> QLabel:
        if len(self.tasks) == 1:
            t = self.tasks[0]
            video = self.repo.get_video_file(t.video_id)
            text = f"Видео: {video.filename if video else '???'}"
            if t.detections_count is not None:
                text += f"\nДетекций: {t.detections_count}, треков: {t.tracks_count or 0}"
            text += f"\nСтатус задачи: {t.status.value}"
        else:
            statuses: dict = {}
            for t in self.tasks:
                statuses[t.status] = statuses.get(t.status, 0) + 1
            parts = [f"{count} × {st.value}" for st, count in statuses.items()]
            text = f"{len(self.tasks)} задач — " + ", ".join(parts)
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        return lbl

    def _build_inference_group(self) -> QGroupBox:
        group = QGroupBox("🎯 Повторная детекция (инференс)")
        layout = QVBoxLayout(group)

        radio_row = QHBoxLayout()
        self.rb_infer_none = QRadioButton("Не делать")
        self.rb_infer_none.setChecked(True)
        self.rb_infer_other = QRadioButton("С другими параметрами")
        self.rb_infer_none.setToolTip(
            "Не запускать инференс — использовать существующие детекции."
        )
        self.rb_infer_other.setToolTip(
            "Запустить детекцию с переопределёнными моделью/параметрами."
        )
        bg = QButtonGroup(self)
        bg.addButton(self.rb_infer_none)
        bg.addButton(self.rb_infer_other)
        for rb in (self.rb_infer_none, self.rb_infer_other):
            radio_row.addWidget(rb)
            rb.toggled.connect(self._on_inference_mode_changed)
        radio_row.addStretch()
        layout.addLayout(radio_row)

        self.lbl_infer_existing = QLabel()
        self.lbl_infer_existing.setStyleSheet("color: gray;")
        layout.addWidget(self.lbl_infer_existing)

        # Блок параметров детекции (скрыт по умолчанию)
        self.infer_fields = QWidget()
        f_outer = QVBoxLayout(self.infer_fields)
        f_outer.setContentsMargins(10, 5, 10, 5)
        f_outer.setSpacing(8)

        # --- Модель + базовые параметры ---
        detect_form = QFormLayout()
        detect_form.setContentsMargins(0, 0, 0, 0)

        self.infer_combo_model = QComboBox()
        detect_form.addRow("Модель:", self.infer_combo_model)

        self.infer_spin_conf = QDoubleSpinBox()
        self.infer_spin_conf.setRange(0.01, 0.99)
        self.infer_spin_conf.setSingleStep(0.05)
        self.infer_spin_conf.setDecimals(2)
        detect_form.addRow("Порог уверенности:", self.infer_spin_conf)

        self.infer_spin_depth_rate = QDoubleSpinBox()
        self.infer_spin_depth_rate.setRange(0.0, 10.0)
        self.infer_spin_depth_rate.setSingleStep(0.1)
        self.infer_spin_depth_rate.setDecimals(2)
        self.infer_spin_depth_rate.setSpecialValueText("Не задано")
        detect_form.addRow("Скорость погружения (м/с):", self.infer_spin_depth_rate)

        f_outer.addLayout(detect_form)

        # --- Трекинг ---
        tracking_group = QGroupBox("Трекинг")
        tracking_layout = QVBoxLayout(tracking_group)

        self.infer_check_tracking = QCheckBox("Включить трекинг объектов")
        self.infer_check_tracking.toggled.connect(self._on_infer_tracking_toggled)
        tracking_layout.addWidget(self.infer_check_tracking)

        tracking_form = QFormLayout()
        tracking_form.setContentsMargins(20, 0, 0, 0)

        self.infer_combo_tracker = QComboBox()
        self.infer_combo_tracker.addItem("ByteTrack (быстрый)", "bytetrack.yaml")
        self.infer_combo_tracker.addItem("BoT-SORT (точный)", "botsort.yaml")
        tracking_form.addRow("Трекер:", self.infer_combo_tracker)

        self.infer_check_trails = QCheckBox("Показывать траектории")
        self.infer_check_trails.toggled.connect(
            lambda checked: self.infer_spin_trail_length.setEnabled(
                self.infer_check_tracking.isChecked() and checked
            )
        )
        tracking_form.addRow("", self.infer_check_trails)

        self.infer_spin_trail_length = QSpinBox()
        self.infer_spin_trail_length.setRange(10, 200)
        tracking_form.addRow("Длина траектории (кадров):", self.infer_spin_trail_length)

        self.infer_spin_min_track = QSpinBox()
        self.infer_spin_min_track.setRange(1, 50)
        tracking_form.addRow("Мин. длина трека (кадров):", self.infer_spin_min_track)

        tracking_layout.addLayout(tracking_form)
        f_outer.addWidget(tracking_group)

        # --- GPU / Ускорение ---
        gpu_group = QGroupBox("GPU / Ускорение")
        gpu_form = QFormLayout(gpu_group)

        self.infer_combo_device = QComboBox()
        self.infer_combo_device.addItem("Автоматически", "auto")
        self.infer_combo_device.addItem("CPU", "cpu")
        self.infer_combo_device.addItem("GPU 0", "0")
        self.infer_combo_device.setToolTip(
            "Устройство для инференса YOLO.\nauto — GPU если доступен, иначе CPU."
        )
        gpu_form.addRow("Устройство:", self.infer_combo_device)

        self.infer_spin_imgsz = QSpinBox()
        self.infer_spin_imgsz.setRange(320, 3840)
        self.infer_spin_imgsz.setSingleStep(32)
        self.infer_spin_imgsz.setToolTip(
            "Размер изображения для YOLO-инференса.\n"
            "Должен совпадать с imgsz при обучении модели.\n"
            "Больше = точнее, но медленнее и больше памяти GPU."
        )
        gpu_form.addRow("Размер изображения (imgsz):", self.infer_spin_imgsz)

        self.infer_check_half = QCheckBox("FP16 (half precision)")
        self.infer_check_half.setToolTip(
            "Использовать половинную точность для инференса.\n"
            "Ускоряет обработку в ~2 раза на GPU с минимальным влиянием на точность.\n"
            "Не поддерживается на CPU."
        )
        gpu_form.addRow("", self.infer_check_half)

        f_outer.addWidget(gpu_group)

        # --- Выход ---
        self.infer_check_save_video = QCheckBox("Сохранять видео с разметкой")
        f_outer.addWidget(self.infer_check_save_video)

        layout.addWidget(self.infer_fields)

        self.chk_cascade = QCheckBox("Удалить устаревшие результаты постобработки")
        self.chk_cascade.setChecked(True)
        self.chk_cascade.setToolTip(
            "После успешного инференса автоматически удалить старые результаты\n"
            "GEOMETRY/SIZE/VOLUME/ANALYSIS — их нужно будет пересчитать."
        )
        layout.addWidget(self.chk_cascade)

        return group

    def _build_op_row(
        self, st_type: SubTaskType, label: str, tooltip: str,
    ) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)

        title = QLabel(label)
        title.setMinimumWidth(210)
        title.setToolTip(tooltip)
        h.addWidget(title)

        checkmark = QLabel("✓")
        checkmark.setStyleSheet(
            "color: #27ae60; font-weight: bold; font-size: 13px; padding-right: 2px;"
        )
        checkmark.setToolTip("Результат уже посчитан")
        checkmark.setVisible(False)
        h.addWidget(checkmark)

        rb_none = QRadioButton("Не делать")
        rb_keep = QRadioButton("Исп. текущий")
        rb_recompute = QRadioButton("Пересчитать")
        rb_none.setChecked(True)

        _f = rb_none.font()
        _f.setPointSize(max(7, _f.pointSize() - 1))
        for rb in (rb_none, rb_keep, rb_recompute):
            rb.setFont(_f)

        bg = QButtonGroup(w)
        bg.addButton(rb_none)
        bg.addButton(rb_keep)
        bg.addButton(rb_recompute)

        h.addWidget(rb_none)
        h.addWidget(rb_keep)
        h.addWidget(rb_recompute)

        badge = QLabel("")
        badge.setStyleSheet("color: gray; padding-left: 6px;")
        badge.setFont(_f)
        h.addWidget(badge)
        h.addStretch()

        self._radio_groups[st_type] = {
            "none": rb_none,
            "keep": rb_keep,
            "recompute": rb_recompute,
            "badge": badge,
            "checkmark": checkmark,
        }
        return w

    # ------------------------------------------------------------- helpers

    def _populate_models(self):
        models = self.repo.get_all_models()
        order = get_config().ui.models_order
        if order:
            models.sort(
                key=lambda m: order.index(m.id) if m.id in order else len(order)
            )
        for m in models:
            display = m.name + (f" ({m.base_model})" if m.base_model else "")
            self.infer_combo_model.addItem(display, m.id)

    def _populate_inference_defaults(self):
        """Заполняет поля инференса значениями задачи (одна) или дефолтами (группа)."""
        config = get_config()
        params = config.default_detection_params

        if len(self.tasks) == 1:
            t = self.tasks[0]

            for i in range(self.infer_combo_model.count()):
                if self.infer_combo_model.itemData(i) == t.model_id:
                    self.infer_combo_model.setCurrentIndex(i)
                    break

            conf = t.conf_threshold if t.conf_threshold is not None else params.conf_threshold
            self.infer_spin_conf.setValue(conf)

            dr = getattr(t, "depth_rate", None)
            self.infer_spin_depth_rate.setValue(dr if dr else 0.0)

            et = getattr(t, "enable_tracking", None)
            self.infer_check_tracking.setChecked(
                et if et is not None else params.enable_tracking
            )

            tt = getattr(t, "tracker_type", None) or params.tracker_type
            for i in range(self.infer_combo_tracker.count()):
                if self.infer_combo_tracker.itemData(i) == tt:
                    self.infer_combo_tracker.setCurrentIndex(i)
                    break

            st = getattr(t, "show_trails", None)
            self.infer_check_trails.setChecked(
                st if st is not None else params.show_trails
            )

            tl = getattr(t, "trail_length", None)
            self.infer_spin_trail_length.setValue(tl if tl else params.trail_length)

            mtl = getattr(t, "min_track_length", None)
            self.infer_spin_min_track.setValue(mtl if mtl else params.min_track_length)

            dev = getattr(t, "device", None) or params.device
            for i in range(self.infer_combo_device.count()):
                if self.infer_combo_device.itemData(i) == dev:
                    self.infer_combo_device.setCurrentIndex(i)
                    break

            imgsz = getattr(t, "imgsz", None)
            self.infer_spin_imgsz.setValue(imgsz if imgsz else params.imgsz)

            half = getattr(t, "half", None)
            self.infer_check_half.setChecked(
                half if half is not None else params.half
            )

            sv = getattr(t, "save_video", None)
            self.infer_check_save_video.setChecked(
                sv if sv is not None else params.save_video
            )
        else:
            # Группа задач — используем дефолтные параметры из конфига.
            self.infer_spin_conf.setValue(params.conf_threshold)
            self.infer_spin_depth_rate.setValue(0.0)
            self.infer_check_tracking.setChecked(params.enable_tracking)
            for i in range(self.infer_combo_tracker.count()):
                if self.infer_combo_tracker.itemData(i) == params.tracker_type:
                    self.infer_combo_tracker.setCurrentIndex(i)
                    break
            self.infer_check_trails.setChecked(params.show_trails)
            self.infer_spin_trail_length.setValue(params.trail_length)
            self.infer_spin_min_track.setValue(params.min_track_length)
            for i in range(self.infer_combo_device.count()):
                if self.infer_combo_device.itemData(i) == params.device:
                    self.infer_combo_device.setCurrentIndex(i)
                    break
            self.infer_spin_imgsz.setValue(params.imgsz)
            self.infer_check_half.setChecked(params.half)
            self.infer_check_save_video.setChecked(params.save_video)

        self._on_infer_tracking_toggled(self.infer_check_tracking.isChecked())

    def _on_infer_tracking_toggled(self, enabled: bool):
        self.infer_combo_tracker.setEnabled(enabled)
        self.infer_check_trails.setEnabled(enabled)
        self.infer_spin_trail_length.setEnabled(
            enabled and self.infer_check_trails.isChecked()
        )
        self.infer_spin_min_track.setEnabled(enabled)

    def _update_calibration_label(self):
        try:
            path = get_config().ui.calibration_json
        except Exception:
            path = None
        if path and os.path.exists(path):
            self.label_calibration.setText(
                f"📏 Калибровка: {os.path.basename(path)}"
            )
        else:
            self.label_calibration.setText(
                "📏 Калибровка: дефолтные коэффициенты"
            )

    def _has_output_for_task(self, task_id: int, st_type: SubTaskType) -> bool:
        output_types = SUBTASK_OUTPUT_TYPES.get(st_type, [])
        if not output_types:
            return False
        outs = self.repo.get_task_outputs(task_id)
        return any(o.output_type in output_types for o in outs)

    def _has_pending_or_running(self, task_id: int, st_type: SubTaskType) -> bool:
        for st in self.repo.get_subtasks_for_task(task_id):
            if st.subtask_type == st_type and st.status in (
                TaskStatus.PENDING, TaskStatus.RUNNING,
            ):
                return True
        return False

    def _populate_existing_state(self):
        op_types = [
            SubTaskType.GEOMETRY,
            SubTaskType.SIZE,
            SubTaskType.SIZE_VIDEO_RENDER,
            SubTaskType.VOLUME,
            SubTaskType.ANALYSIS,
        ]
        N = len(self.tasks)

        for st_type in op_types:
            done_count = sum(
                1 for t in self.tasks
                if self._has_output_for_task(t.id, st_type)
            )
            queued_count = sum(
                1 for t in self.tasks
                if self._has_pending_or_running(t.id, st_type)
            )
            grp = self._radio_groups[st_type]

            grp["checkmark"].setVisible(done_count > 0)

            badge_parts = []
            if done_count > 0 and N > 1:
                badge_parts.append(f"{done_count}/{N}")
            if queued_count > 0:
                badge_parts.append(f"в очереди: {queued_count}")
            grp["badge"].setText(" | ".join(badge_parts))

            grp["keep"].setEnabled(done_count > 0)
            if done_count > 0:
                grp["keep"].setChecked(True)
            else:
                grp["recompute"].setChecked(True)

        done = sum(1 for t in self.tasks if t.status == TaskStatus.DONE)
        non_done = N - done
        parts = [f"DONE: {done}/{N}"]
        if non_done > 0:
            parts.append(f"не завершены: {non_done}")
        self.lbl_infer_existing.setText(" | ".join(parts))

        self._update_warning()

    def _on_inference_mode_changed(self):
        other = self.rb_infer_other.isChecked()
        self.infer_fields.setEnabled(other)
        self.infer_fields.setVisible(other)
        self.chk_cascade.setEnabled(not self.rb_infer_none.isChecked())
        self._update_warning()

    def _update_warning(self):
        non_done = [t for t in self.tasks if t.status != TaskStatus.DONE]
        infer_active = self.rb_infer_other.isChecked()
        if non_done and not infer_active:
            self.label_warning.setText(
                f"⚠ {len(non_done)} задач не в статусе DONE — "
                "для них постобработка не будет добавлена. "
                "Выберите «Повторная детекция» чтобы включить их."
            )
            self.label_warning.setVisible(True)
        elif non_done and infer_active:
            self.label_warning.setText(
                f"ℹ {len(non_done)} задач не в статусе DONE — "
                "они будут сброшены в очередь с новыми параметрами; "
                "выбранная постобработка запустится автоматически после детекции."
            )
            self.label_warning.setVisible(True)
        else:
            self.label_warning.setVisible(False)

    # ------------------------------------------------------------- on_add

    def _collect_common_params(self) -> dict:
        return {
            "fov": self.spin_fov.value(),
            "min_reliable_distance": self.spin_min_reliable.value(),
            "depth_bin": self.spin_depth_bin.value(),
            "ctd_columns": self.edit_ctd_columns.text().strip() or "6",
        }

    def _build_inference_params(self) -> Optional[dict]:
        """params_json для INFERENCE-подзадачи или None если не запускаем."""
        if self.rb_infer_none.isChecked():
            return None

        params: dict = {
            "cascade_invalidate": self.chk_cascade.isChecked(),
            "model_id": self.infer_combo_model.currentData(),
            "conf_threshold": self.infer_spin_conf.value(),
            "enable_tracking": self.infer_check_tracking.isChecked(),
            "tracker_type": self.infer_combo_tracker.currentData(),
            "show_trails": self.infer_check_trails.isChecked(),
            "trail_length": self.infer_spin_trail_length.value(),
            "min_track_length": self.infer_spin_min_track.value(),
            "device": self.infer_combo_device.currentData(),
            "imgsz": self.infer_spin_imgsz.value(),
            "half": self.infer_check_half.isChecked(),
            "save_video": self.infer_check_save_video.isChecked(),
        }

        depth_rate = self.infer_spin_depth_rate.value()
        if depth_rate > 0:
            params["depth_rate"] = depth_rate

        return params

    def _selected_action(self, st_type: SubTaskType) -> str:
        grp = self._radio_groups[st_type]
        if grp["recompute"].isChecked():
            return ACTION_RECOMPUTE
        if grp["keep"].isChecked():
            return ACTION_KEEP
        return ACTION_NONE

    def _on_add(self):
        common = self._collect_common_params()
        infer_params = self._build_inference_params()

        per_op_params: dict = {}
        for st_type in (
            SubTaskType.GEOMETRY,
            SubTaskType.SIZE,
            SubTaskType.SIZE_VIDEO_RENDER,
            SubTaskType.VOLUME,
            SubTaskType.ANALYSIS,
        ):
            action = self._selected_action(st_type)
            if action == ACTION_NONE or action == ACTION_KEEP:
                continue
            p = dict(common)
            if st_type == SubTaskType.GEOMETRY:
                p["frame_step"] = self.spin_frame_step.value()
            elif st_type == SubTaskType.SIZE:
                p["use_geometry"] = self.chk_size_use_geometry.isChecked()
                try:
                    cp = get_config().ui.calibration_json
                    if cp and os.path.exists(cp):
                        p["calibration_json"] = cp
                except Exception:
                    pass
            elif st_type == SubTaskType.SIZE_VIDEO_RENDER:
                p["use_geometry"] = self.chk_video_use_geometry.isChecked()
            per_op_params[st_type] = p

        if infer_params is None and not per_op_params:
            QMessageBox.warning(
                self, "Нет операций",
                "Не выбрано ни одной операции для запуска.",
            )
            return

        applied = 0
        skipped_non_done = 0
        errors: List[str] = []
        created_total = 0

        for task in self.tasks:
            if infer_params is not None:
                # Сбрасываем задачу в PENDING с новыми параметрами
                # (без создания INFERENCE-подзадачи)
                try:
                    self.repo.reset_task_for_reinference(task.id, infer_params)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    errors.append(f"#{task.id}: {e}")
                    continue

                # Создаём подзадачи постобработки (запустятся после детекции)
                ops: List[Tuple[SubTaskType, str]] = []
                force: Set[SubTaskType] = set()
                for st_type, p in per_op_params.items():
                    ops.append((st_type, json.dumps(p)))
                    if self._task_has_finished_subtask(task.id, st_type):
                        force.add(st_type)
                if ops:
                    try:
                        created = self.repo.create_postprocess_subtasks_v2(
                            task.id, ops, force_overwrite_types=force,
                        )
                        created_total += len(created)
                    except Exception as e:
                        import traceback; traceback.print_exc()
                        errors.append(f"#{task.id} (постобработка): {e}")
                applied += 1
                continue

            # Только постобработка — задача должна быть DONE
            if task.status != TaskStatus.DONE:
                skipped_non_done += 1
                continue

            ops = []
            force = set()
            for st_type, p in per_op_params.items():
                ops.append((st_type, json.dumps(p)))
                if self._task_has_finished_subtask(task.id, st_type):
                    force.add(st_type)

            if not ops:
                continue

            try:
                created = self.repo.create_postprocess_subtasks_v2(
                    task.id, ops, force_overwrite_types=force,
                )
                created_total += len(created)
                applied += 1
            except RuntimeError as e:
                errors.append(f"#{task.id}: {e}")
            except Exception as e:
                import traceback; traceback.print_exc()
                errors.append(f"#{task.id}: {e}")

        if errors:
            QMessageBox.warning(
                self, "Часть задач не обработана",
                "\n".join(errors[:10]),
            )

        if created_total > 0:
            self.task_manager.queue_changed.emit()

        msg_parts = [
            f"Задач обработано: {applied}",
            f"Создано подзадач: {created_total}",
        ]
        if skipped_non_done:
            msg_parts.append(
                f"Пропущено (не DONE, без инференса): {skipped_non_done}"
            )
        QMessageBox.information(self, "Готово", "\n".join(msg_parts))
        self.accept()

    def _task_has_finished_subtask(
        self, task_id: int, st_type: SubTaskType,
    ) -> bool:
        for st in self.repo.get_subtasks_for_task(task_id):
            if st.subtask_type == st_type and st.status in (
                TaskStatus.DONE, TaskStatus.ERROR, TaskStatus.CANCELLED,
            ):
                return True
        return False
