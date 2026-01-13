"""
Диалог добавления постобработки к задаче.

Позволяет выбрать и добавить в очередь:
- Оценку наклона камеры (FOE)
- Расчёт размеров объектов
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
)
from PyQt6.QtCore import Qt

from ...database import Repository, Task, TaskStatus, SubTaskType
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
        self.setMinimumWidth(450)
        
        self._setup_ui()
        self._load_existing_subtasks()
    
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
        
        layout.addWidget(info_group)
        
        # === Выбор операций ===
        ops_group = QGroupBox("Добавить в очередь")
        ops_layout = QVBoxLayout(ops_group)
        
        self.chk_geometry = QCheckBox("📐 Геометрия камеры (FOE)")
        self.chk_geometry.setToolTip("Оценка наклона камеры по Focus of Expansion")
        ops_layout.addWidget(self.chk_geometry)
        
        self.chk_size = QCheckBox("📏 Размеры объектов")
        self.chk_size.setToolTip("Расчёт реальных размеров по динамике bbox")
        ops_layout.addWidget(self.chk_size)
        
        self.chk_volume = QCheckBox("📦 Объём воды")
        self.chk_volume.setToolTip("Расчёт осмотренного объёма и плотности")
        ops_layout.addWidget(self.chk_volume)
        
        self.chk_analysis = QCheckBox("📊 Анализ и графики")
        self.chk_analysis.setToolTip("Генерация графиков распределения")
        ops_layout.addWidget(self.chk_analysis)
        
        layout.addWidget(ops_group)
        
        # === Параметры ===
        params_group = QGroupBox("Параметры")
        params_layout = QFormLayout(params_group)
        
        self.spin_fov = QDoubleSpinBox()
        self.spin_fov.setRange(60, 180)
        self.spin_fov.setValue(100.0)
        self.spin_fov.setSuffix("°")
        self.spin_fov.setToolTip("Горизонтальный угол обзора камеры")
        params_layout.addRow("FOV камеры:", self.spin_fov)
        
        self.spin_near = QDoubleSpinBox()
        self.spin_near.setRange(0.1, 2.0)
        self.spin_near.setValue(0.3)
        self.spin_near.setSingleStep(0.1)
        self.spin_near.setSuffix(" м")
        self.spin_near.setToolTip("Ближняя граница обнаружения")
        params_layout.addRow("Ближняя дистанция:", self.spin_near)
        
        self.spin_depth_bin = QDoubleSpinBox()
        self.spin_depth_bin.setRange(0.5, 10.0)
        self.spin_depth_bin.setValue(2.0)
        self.spin_depth_bin.setSingleStep(0.5)
        self.spin_depth_bin.setSuffix(" м")
        self.spin_depth_bin.setToolTip("Шаг биннинга по глубине для графиков")
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
    
    def _on_add(self):
        """Добавляет выбранные подзадачи в очередь."""
        geometry = self.chk_geometry.isChecked() and self.chk_geometry.isEnabled()
        size = self.chk_size.isChecked() and self.chk_size.isEnabled()
        volume = self.chk_volume.isChecked() and self.chk_volume.isEnabled()
        analysis = self.chk_analysis.isChecked() and self.chk_analysis.isEnabled()
        
        if not any([geometry, size, volume, analysis]):
            QMessageBox.warning(self, "Нет операций", "Выберите хотя бы одну операцию")
            return
        
        # Собираем параметры в JSON
        params = {
            "fov": self.spin_fov.value(),
            "near_distance": self.spin_near.value(),
            "depth_bin": self.spin_depth_bin.value(),
        }
        params_json = json.dumps(params)
        
        # Создаём подзадачи
        created = self.repo.create_postprocess_subtasks(
            task_id=self.task_id,
            geometry=geometry,
            size=size,
            volume=volume,
            analysis=analysis,
            params_json=params_json,
        )
        
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
