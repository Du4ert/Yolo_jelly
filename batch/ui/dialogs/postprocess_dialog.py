"""
Диалог добавления постобработки к задаче.

Позволяет выбрать и добавить в очередь:
- Оценку наклона камеры (FOE)
- Расчёт размеров объектов (с/без коррекции наклона)
- Рендеринг видео с размерами (с/без отображения углов)
- Расчёт объёма воды
- Анализ и графики
"""

import json
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QDoubleSpinBox,
    QCheckBox,
    QPushButton,
    QMessageBox,
    QFrame,
    QWidget,
)
from PyQt6.QtCore import Qt

from ...database import Repository, Task, TaskStatus, SubTaskType, OutputType
from ...core import TaskManager


class PostProcessDialog(QDialog):
    """
    Диалог добавления постобработки к задаче.
    Создаёт подзадачи, которые будут выполнены в общей очереди.
    """
    
    def __init__(
        self, 
        repo: Repository, 
        task_manager: TaskManager,
        task_id: int, 
        parent=None
    ):
        super().__init__(parent)
        self.repo = repo
        self.task_manager = task_manager
        self.task_id = task_id
        self.task = repo.get_task(task_id)
        
        if not self.task:
            raise ValueError(f"Задача {task_id} не найдена")
        
        if self.task.status != TaskStatus.DONE:
            raise ValueError("Постобработка доступна только для завершённых задач детекции")
        
        self.setWindowTitle(f"Постобработка задачи #{task_id}")
        self.setMinimumWidth(520)
        
        self._setup_ui()
        self._load_existing_subtasks()
        self._connect_signals()
    
    def _setup_ui(self):
        """Настройка интерфейса."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # === Информация о задаче ===
        info_group = QGroupBox("Задача детекции")
        info_layout = QFormLayout(info_group)
        
        video = self.repo.get_video_file(self.task.video_id)
        info_layout.addRow("Видео:", QLabel(video.filename if video else "???"))
        
        detections_info = f"{self.task.detections_count or 0} детекций"
        if self.task.tracks_count:
            detections_info += f", {self.task.tracks_count} треков"
        info_layout.addRow("Результат:", QLabel(detections_info))
        
        # Показываем статус существующих файлов
        status_parts = []
        if self._has_geometry_output():
            status_parts.append("📐 геометрия")
        if self._has_size_output():
            status_parts.append("📏 размеры")
        if self._has_size_video_output():
            status_parts.append("🎬 видео")
        if self._has_volume_output():
            status_parts.append("📦 объём")
        
        if status_parts:
            info_layout.addRow("Уже есть:", QLabel(" | ".join(status_parts)))
        
        layout.addWidget(info_group)
        
        # === Выбор операций ===
        ops_group = QGroupBox("Добавить в очередь")
        ops_layout = QVBoxLayout(ops_group)
        
        # Геометрия
        self.chk_geometry = QCheckBox("📐 Геометрия камеры (FOE)")
        self.chk_geometry.setToolTip("Оценка наклона камеры по Focus of Expansion.\nРекомендуется для коррекции размеров.")
        ops_layout.addWidget(self.chk_geometry)
        
        # Разделитель
        ops_layout.addSpacing(5)
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.HLine)
        separator1.setFrameShadow(QFrame.Shadow.Sunken)
        ops_layout.addWidget(separator1)
        ops_layout.addSpacing(5)
        
        # Размеры объектов
        self.chk_size = QCheckBox("📏 Размеры объектов")
        self.chk_size.setToolTip("Расчёт реальных размеров по k-методу (динамике изменения bbox)")
        ops_layout.addWidget(self.chk_size)
        
        # Опция использования геометрии для размеров
        indent_widget_size = QWidget()
        indent_layout_size = QHBoxLayout(indent_widget_size)
        indent_layout_size.setContentsMargins(20, 0, 0, 0)
        
        self.chk_size_use_geometry = QCheckBox("С коррекцией наклона камеры")
        self.chk_size_use_geometry.setToolTip(
            "Коррекция k-значений с учётом угла наклона камеры:\n"
            "k_real = k_measured / cos(θ)\n\n"
            "Без коррекции при наклоне 30° размеры занижаются на ~15%.\n"
            "Требует предварительного расчёта геометрии."
        )
        self.chk_size_use_geometry.setChecked(True)
        indent_layout_size.addWidget(self.chk_size_use_geometry)
        indent_layout_size.addStretch()
        ops_layout.addWidget(indent_widget_size)
        
        # Видео с размерами
        self.chk_size_video = QCheckBox("🎬 Видео с размерами")
        self.chk_size_video.setToolTip(
            "Рендеринг видео с отображением:\n"
            "- Дистанции до объекта и размера под рамками\n"
            "- Углов наклона камеры в левом нижнем углу\n\n"
            "Требует предварительного расчёта размеров."
        )
        ops_layout.addWidget(self.chk_size_video)
        
        # Опция отображения геометрии на видео
        indent_widget_video = QWidget()
        indent_layout_video = QHBoxLayout(indent_widget_video)
        indent_layout_video.setContentsMargins(20, 0, 0, 0)
        
        self.chk_video_use_geometry = QCheckBox("Показывать углы наклона")
        self.chk_video_use_geometry.setToolTip(
            "Отображать информацию об углах наклона камеры\n"
            "в левом нижнем углу видео.\n\n"
            "Требует предварительного расчёта геометрии."
        )
        self.chk_video_use_geometry.setChecked(True)
        indent_layout_video.addWidget(self.chk_video_use_geometry)
        indent_layout_video.addStretch()
        ops_layout.addWidget(indent_widget_video)
        
        # Объём
        self.chk_volume = QCheckBox("📦 Объём воды")
        self.chk_volume.setToolTip(
            "Расчёт осмотренного объёма воды и плотности организмов.\n"
            "Использует данные CTD для полного диапазона глубин."
        )
        ops_layout.addWidget(self.chk_volume)
        
        # Разделитель
        ops_layout.addSpacing(5)
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        separator2.setFrameShadow(QFrame.Shadow.Sunken)
        ops_layout.addWidget(separator2)
        ops_layout.addSpacing(5)
        
        # Анализ
        self.chk_analysis = QCheckBox("📊 Анализ и графики")
        self.chk_analysis.setToolTip("Генерация графиков вертикального распределения и отчётов")
        ops_layout.addWidget(self.chk_analysis)
        
        layout.addWidget(ops_group)
        
        # === Параметры ===
        params_group = QGroupBox("Параметры")
        params_layout = QFormLayout(params_group)
        
        self.spin_fov = QDoubleSpinBox()
        self.spin_fov.setRange(60, 180)
        self.spin_fov.setValue(100.0)
        self.spin_fov.setSuffix("°")
        self.spin_fov.setToolTip("Горизонтальный угол обзора камеры (GoPro Wide ~100°)")
        params_layout.addRow("FOV камеры:", self.spin_fov)
        
        self.spin_near = QDoubleSpinBox()
        self.spin_near.setRange(0.1, 2.0)
        self.spin_near.setValue(0.3)
        self.spin_near.setSingleStep(0.1)
        self.spin_near.setSuffix(" м")
        self.spin_near.setToolTip("Ближняя граница обнаружения (мёртвая зона)")
        params_layout.addRow("Ближняя дистанция:", self.spin_near)
        
        self.spin_depth_bin = QDoubleSpinBox()
        self.spin_depth_bin.setRange(0.5, 10.0)
        self.spin_depth_bin.setValue(2.0)
        self.spin_depth_bin.setSingleStep(0.5)
        self.spin_depth_bin.setSuffix(" м")
        self.spin_depth_bin.setToolTip("Шаг биннинга по глубине для графиков распределения")
        params_layout.addRow("Бин глубины:", self.spin_depth_bin)
        
        layout.addWidget(params_group)
        
        # === Кнопки ===
        btn_layout = QHBoxLayout()
        
        self.btn_add = QPushButton("➕ Добавить в очередь")
        self.btn_add.clicked.connect(self._on_add)
        btn_layout.addWidget(self.btn_add)
        
        btn_layout.addStretch()
        
        self.btn_close = QPushButton("Закрыть")
        self.btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_close)
        
        layout.addLayout(btn_layout)
    
    def _connect_signals(self):
        """Подключение сигналов для взаимозависимостей."""
        # Если выбрана геометрия - можно использовать её в других операциях
        self.chk_geometry.toggled.connect(self._update_geometry_dependencies)
        
        # Если выбраны размеры - можно делать видео с размерами
        self.chk_size.toggled.connect(self._update_size_dependencies)
        
        # Начальное состояние
        self._update_geometry_dependencies()
        self._update_size_dependencies()
    
    def _update_geometry_dependencies(self):
        """Обновляет состояние элементов, зависящих от геометрии."""
        geometry_selected = self.chk_geometry.isChecked() and self.chk_geometry.isEnabled()
        geometry_exists = self._has_geometry_output()
        geometry_available = geometry_selected or geometry_exists
        
        # Опция "с учётом наклона" активна если:
        # - Геометрия будет рассчитана (выбрана в чекбоксе) ИЛИ
        # - Геометрия уже существует
        self.chk_size_use_geometry.setEnabled(geometry_available)
        self.chk_video_use_geometry.setEnabled(geometry_available)
        
        if not geometry_available:
            self.chk_size_use_geometry.setChecked(False)
            self.chk_video_use_geometry.setChecked(False)
    
    def _update_size_dependencies(self):
        """Обновляет состояние элементов, зависящих от размеров."""
        size_selected = self.chk_size.isChecked() and self.chk_size.isEnabled()
        size_exists = self._has_size_output()
        size_available = size_selected or size_exists
        
        # Видео с размерами требует расчёта размеров
        if not size_available and not self.chk_size_video.isEnabled():
            return
        
        # Если размеры ещё не выбраны и не существуют - предлагаем выбрать
        if self.chk_size_video.isChecked() and not size_available:
            self.chk_size.setChecked(True)
    
    def _has_geometry_output(self) -> bool:
        """Проверяет, есть ли уже рассчитанная геометрия."""
        outputs = self.repo.get_task_outputs(self.task_id)
        return any(o.output_type == OutputType.GEOMETRY_CSV for o in outputs)
    
    def _has_size_output(self) -> bool:
        """Проверяет, есть ли уже рассчитанные размеры."""
        outputs = self.repo.get_task_outputs(self.task_id)
        return any(o.output_type == OutputType.SIZE_CSV for o in outputs)
    
    def _has_size_video_output(self) -> bool:
        """Проверяет, есть ли уже рендеренное видео с размерами."""
        outputs = self.repo.get_task_outputs(self.task_id)
        return any(o.output_type == OutputType.SIZE_VIDEO for o in outputs)
    
    def _has_volume_output(self) -> bool:
        """Проверяет, есть ли уже рассчитанный объём."""
        outputs = self.repo.get_task_outputs(self.task_id)
        return any(o.output_type == OutputType.VOLUME_CSV for o in outputs)
    
    def _load_existing_subtasks(self):
        """Загружает информацию о существующих подзадачах."""
        subtasks = self.repo.get_subtasks_for_task(self.task_id)
        
        existing_types = {st.subtask_type for st in subtasks}
        
        # Отключаем чекбоксы для уже существующих подзадач
        if SubTaskType.GEOMETRY in existing_types:
            self.chk_geometry.setChecked(False)
            self.chk_geometry.setEnabled(False)
            self.chk_geometry.setText("📐 Геометрия камеры (уже в очереди)")
        else:
            self.chk_geometry.setChecked(True)
        
        if SubTaskType.SIZE in existing_types:
            self.chk_size.setChecked(False)
            self.chk_size.setEnabled(False)
            self.chk_size.setText("📏 Размеры объектов (уже в очереди)")
        else:
            self.chk_size.setChecked(True)
        
        if SubTaskType.SIZE_VIDEO_RENDER in existing_types:
            self.chk_size_video.setChecked(False)
            self.chk_size_video.setEnabled(False)
            self.chk_size_video.setText("🎬 Видео с размерами (уже в очереди)")
        else:
            self.chk_size_video.setChecked(True)
        
        if SubTaskType.VOLUME in existing_types:
            self.chk_volume.setChecked(False)
            self.chk_volume.setEnabled(False)
            self.chk_volume.setText("📦 Объём воды (уже в очереди)")
        else:
            self.chk_volume.setChecked(True)
        
        if SubTaskType.ANALYSIS in existing_types:
            self.chk_analysis.setChecked(False)
            self.chk_analysis.setEnabled(False)
            self.chk_analysis.setText("📊 Анализ и графики (уже в очереди)")
        else:
            self.chk_analysis.setChecked(True)
        
        # Проверяем, есть ли уже файлы геометрии/размеров
        if self._has_geometry_output():
            if self.chk_geometry.isEnabled():
                self.chk_geometry.setChecked(False)
                self.chk_geometry.setText("📐 Геометрия камеры (уже рассчитана)")
        
        if self._has_size_output():
            if self.chk_size.isEnabled():
                self.chk_size.setChecked(False)
                self.chk_size.setText("📏 Размеры объектов (уже рассчитаны)")
        
        if self._has_size_video_output():
            if self.chk_size_video.isEnabled():
                self.chk_size_video.setChecked(False)
                self.chk_size_video.setText("🎬 Видео с размерами (уже создано)")
        
        if self._has_volume_output():
            if self.chk_volume.isEnabled():
                self.chk_volume.setChecked(False)
                self.chk_volume.setText("📦 Объём воды (уже рассчитан)")
    
    def _on_add(self):
        """Добавляет выбранные подзадачи в очередь."""
        geometry = self.chk_geometry.isChecked() and self.chk_geometry.isEnabled()
        size = self.chk_size.isChecked() and self.chk_size.isEnabled()
        size_video = self.chk_size_video.isChecked() and self.chk_size_video.isEnabled()
        volume = self.chk_volume.isChecked() and self.chk_volume.isEnabled()
        analysis = self.chk_analysis.isChecked() and self.chk_analysis.isEnabled()
        
        if not any([geometry, size, size_video, volume, analysis]):
            QMessageBox.warning(self, "Нет операций", "Выберите хотя бы одну операцию")
            return
        
        # Проверяем зависимости
        size_use_geometry = self.chk_size_use_geometry.isChecked()
        video_use_geometry = self.chk_video_use_geometry.isChecked()
        
        # Если хотим использовать геометрию, но она не выбрана и не существует
        geometry_exists = self._has_geometry_output()
        if (size_use_geometry or video_use_geometry) and not geometry and not geometry_exists:
            reply = QMessageBox.question(
                self,
                "Добавить геометрию?",
                "Для коррекции наклона требуется расчёт геометрии камеры.\n\n"
                "Добавить расчёт геометрии в очередь?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Yes:
                geometry = True
            else:
                # Отключаем использование геометрии
                size_use_geometry = False
                video_use_geometry = False
        
        # Если выбрано видео с размерами, но размеры не выбраны и не существуют
        size_exists = self._has_size_output()
        if size_video and not size and not size_exists:
            reply = QMessageBox.question(
                self,
                "Добавить размеры?",
                "Для рендеринга видео с размерами требуется расчёт размеров объектов.\n\n"
                "Добавить расчёт размеров в очередь?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Yes:
                size = True
            else:
                size_video = False
        
        # Собираем параметры в JSON
        params = {
            "fov": self.spin_fov.value(),
            "near_distance": self.spin_near.value(),
            "depth_bin": self.spin_depth_bin.value(),
        }
        
        # Создаём подзадачи в правильном порядке
        created = []
        position = 0
        
        # Сначала геометрия (если нужна)
        if geometry:
            st = self.repo.create_subtask(
                parent_task_id=self.task_id,
                subtask_type=SubTaskType.GEOMETRY,
                position=position,
                params_json=json.dumps(params),
            )
            if st:
                created.append(st)
                position += 1
        
        # Затем размеры (с указанием использовать ли геометрию)
        if size:
            size_params = params.copy()
            size_params["use_geometry"] = size_use_geometry
            st = self.repo.create_subtask(
                parent_task_id=self.task_id,
                subtask_type=SubTaskType.SIZE,
                position=position,
                params_json=json.dumps(size_params),
            )
            if st:
                created.append(st)
                position += 1
        
        # Видео с размерами (после размеров, с указанием использовать ли геометрию)
        if size_video:
            video_params = params.copy()
            video_params["use_geometry"] = video_use_geometry
            st = self.repo.create_subtask(
                parent_task_id=self.task_id,
                subtask_type=SubTaskType.SIZE_VIDEO_RENDER,
                position=position,
                params_json=json.dumps(video_params),
            )
            if st:
                created.append(st)
                position += 1
        
        # Объём
        if volume:
            st = self.repo.create_subtask(
                parent_task_id=self.task_id,
                subtask_type=SubTaskType.VOLUME,
                position=position,
                params_json=json.dumps(params),
            )
            if st:
                created.append(st)
                position += 1
        
        # Анализ в конце
        if analysis:
            st = self.repo.create_subtask(
                parent_task_id=self.task_id,
                subtask_type=SubTaskType.ANALYSIS,
                position=position,
                params_json=json.dumps(params),
            )
            if st:
                created.append(st)
                position += 1
        
        if created:
            count = len(created)
            QMessageBox.information(
                self, "Добавлено",
                f"Добавлено {count} подзадач в очередь.\n\n"
                "Подзадачи будут выполнены автоматически\n"
                "при запуске очереди."
            )
            # Обновляем очередь
            self.task_manager.queue_changed.emit()
            self.accept()
        else:
            QMessageBox.warning(self, "Ошибка", "Не удалось создать подзадачи")
