"""
Диалог для расчёта геометрии камеры и оценки размеров.
"""

import os
from pathlib import Path
from typing import Optional

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
    QPushButton,
    QComboBox,
    QFileDialog,
    QProgressBar,
    QMessageBox,
    QTabWidget,
    QWidget,
    QTextEdit,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ...database import Repository, Dive, VideoFile


class GeometryWorker(QThread):
    """Воркер для обработки геометрии в отдельном потоке."""
    
    finished = pyqtSignal(object)  # GeometryResult
    progress = pyqtSignal(int, int)  # current, total
    
    def __init__(
        self,
        video_path: str,
        output_csv: str,
        frame_interval: int = 30,
        frame_width: int = 1920,
        frame_height: int = 1080,
    ):
        super().__init__()
        self.video_path = video_path
        self.output_csv = output_csv
        self.frame_interval = frame_interval
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._processor = None
    
    def run(self):
        from ...core import GeometryProcessor
        
        self._processor = GeometryProcessor(
            video_path=self.video_path,
            output_csv=self.output_csv,
            frame_interval=self.frame_interval,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
        )
        
        result = self._processor.run()
        self.finished.emit(result)
    
    def cancel(self):
        if self._processor:
            self._processor.cancel()


class SizeWorker(QThread):
    """Воркер для оценки размеров в отдельном потоке."""
    
    finished = pyqtSignal(object)  # SizeEstimationResult
    
    def __init__(
        self,
        detections_csv: str,
        output_csv: str,
        tracks_csv: Optional[str] = None,
        geometry_csv: Optional[str] = None,
        frame_width: int = 1920,
        frame_height: int = 1080,
        min_depth_change: float = 0.3,
        min_track_points: int = 3,
        min_r_squared: float = 0.5,
        min_size_change: float = 0.3,
    ):
        super().__init__()
        self.detections_csv = detections_csv
        self.output_csv = output_csv
        self.tracks_csv = tracks_csv
        self.geometry_csv = geometry_csv
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.min_depth_change = min_depth_change
        self.min_track_points = min_track_points
        self.min_r_squared = min_r_squared
        self.min_size_change = min_size_change
        self._processor = None
    
    def run(self):
        from ...core import SizeEstimationProcessor
        
        self._processor = SizeEstimationProcessor(
            detections_csv=self.detections_csv,
            output_csv=self.output_csv,
            tracks_output_csv=self.tracks_csv,
            geometry_csv=self.geometry_csv,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            min_depth_change=self.min_depth_change,
            min_track_points=self.min_track_points,
            min_r_squared=self.min_r_squared,
            min_size_change_ratio=self.min_size_change,
        )
        
        result = self._processor.run()
        self.finished.emit(result)
    
    def cancel(self):
        if self._processor:
            self._processor.cancel()


class VolumeWorker(QThread):
    """Воркер для расчёта объёма в отдельном потоке."""
    
    finished = pyqtSignal(object)  # VolumeEstimationResult
    
    def __init__(
        self,
        detections_csv: str,
        output_csv: str,
        tracks_csv: Optional[str] = None,
        ctd_csv: Optional[str] = None,
        fov_horizontal: float = 100.0,
        near_distance: float = 0.3,
        detection_distance: Optional[float] = None,
        depth_min: Optional[float] = None,
        depth_max: Optional[float] = None,
        total_duration: Optional[float] = None,
        fps: float = 60.0,
        frame_width: int = 1920,
        frame_height: int = 1080,
    ):
        super().__init__()
        self.detections_csv = detections_csv
        self.output_csv = output_csv
        self.tracks_csv = tracks_csv
        self.ctd_csv = ctd_csv
        self.fov_horizontal = fov_horizontal
        self.near_distance = near_distance
        self.detection_distance = detection_distance
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.total_duration = total_duration
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._processor = None
    
    def run(self):
        from ...core import VolumeEstimationProcessor
        
        self._processor = VolumeEstimationProcessor(
            detections_csv=self.detections_csv,
            output_csv=self.output_csv,
            tracks_csv=self.tracks_csv,
            ctd_csv=self.ctd_csv,
            fov_horizontal=self.fov_horizontal,
            near_distance=self.near_distance,
            detection_distance=self.detection_distance,
            depth_min=self.depth_min,
            depth_max=self.depth_max,
            total_duration=self.total_duration,
            fps=self.fps,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
        )
        
        result = self._processor.run()
        self.finished.emit(result)
    
    def cancel(self):
        if self._processor:
            self._processor.cancel()


class GeometryDialog(QDialog):
    """
    Диалог для расчёта геометрии камеры, оценки размеров и объёма.
    """
    
    def __init__(
        self,
        repository: Repository,
        dive_id: Optional[int] = None,
        parent=None
    ):
        super().__init__(parent)
        
        self.repo = repository
        self.dive_id = dive_id
        self._worker = None
        
        self._setup_ui()
        self._load_data()
    
    def _setup_ui(self):
        """Настройка интерфейса."""
        self.setWindowTitle("Геометрия камеры и размеры")
        self.setMinimumSize(600, 500)
        
        layout = QVBoxLayout(self)
        
        # Вкладки
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # === Вкладка: Наклон камеры (FOE) ===
        self.tab_geometry = QWidget()
        self.tabs.addTab(self.tab_geometry, "Наклон камеры")
        self._setup_geometry_tab()
        
        # === Вкладка: Оценка размеров ===
        self.tab_size = QWidget()
        self.tabs.addTab(self.tab_size, "Размеры объектов")
        self._setup_size_tab()
        
        # === Вкладка: Объём воды ===
        self.tab_volume = QWidget()
        self.tabs.addTab(self.tab_volume, "Объём воды")
        self._setup_volume_tab()
        
        # Прогресс
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Результаты
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        layout.addWidget(self.result_text)
        
        # Кнопки
        btn_layout = QHBoxLayout()
        
        self.btn_run = QPushButton("Запустить")
        self.btn_run.clicked.connect(self._on_run)
        btn_layout.addWidget(self.btn_run)
        
        self.btn_cancel = QPushButton("Отмена")
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_cancel.setEnabled(False)
        btn_layout.addWidget(self.btn_cancel)
        
        btn_layout.addStretch()
        
        self.btn_close = QPushButton("Закрыть")
        self.btn_close.clicked.connect(self.close)
        btn_layout.addWidget(self.btn_close)
        
        layout.addLayout(btn_layout)
    
    def _setup_geometry_tab(self):
        """Настройка вкладки геометрии."""
        layout = QVBoxLayout(self.tab_geometry)
        
        # Группа: Входные данные
        input_group = QGroupBox("Входные данные")
        input_layout = QFormLayout(input_group)
        
        # Видео
        video_layout = QHBoxLayout()
        self.geom_video_edit = QLineEdit()
        self.geom_video_edit.setPlaceholderText("Путь к видеофайлу...")
        video_layout.addWidget(self.geom_video_edit)
        
        self.geom_video_btn = QPushButton("...")
        self.geom_video_btn.setMaximumWidth(30)
        self.geom_video_btn.clicked.connect(lambda: self._browse_video(self.geom_video_edit))
        video_layout.addWidget(self.geom_video_btn)
        
        input_layout.addRow("Видео:", video_layout)
        
        layout.addWidget(input_group)
        
        # Группа: Параметры
        params_group = QGroupBox("Параметры")
        params_layout = QFormLayout(params_group)
        
        self.geom_interval = QSpinBox()
        self.geom_interval.setRange(10, 300)
        self.geom_interval.setValue(30)
        self.geom_interval.setSuffix(" кадров")
        params_layout.addRow("Интервал анализа:", self.geom_interval)
        
        self.geom_width = QSpinBox()
        self.geom_width.setRange(640, 7680)
        self.geom_width.setValue(1920)
        params_layout.addRow("Ширина кадра:", self.geom_width)
        
        self.geom_height = QSpinBox()
        self.geom_height.setRange(480, 4320)
        self.geom_height.setValue(1080)
        params_layout.addRow("Высота кадра:", self.geom_height)
        
        layout.addWidget(params_group)
        
        # Группа: Выход
        output_group = QGroupBox("Выходные данные")
        output_layout = QFormLayout(output_group)
        
        output_path_layout = QHBoxLayout()
        self.geom_output_edit = QLineEdit()
        self.geom_output_edit.setPlaceholderText("Путь для сохранения geometry.csv...")
        output_path_layout.addWidget(self.geom_output_edit)
        
        self.geom_output_btn = QPushButton("...")
        self.geom_output_btn.setMaximumWidth(30)
        self.geom_output_btn.clicked.connect(lambda: self._browse_save_csv(self.geom_output_edit))
        output_path_layout.addWidget(self.geom_output_btn)
        
        output_layout.addRow("Выходной CSV:", output_path_layout)
        
        layout.addWidget(output_group)
        
        layout.addStretch()
    
    def _setup_size_tab(self):
        """Настройка вкладки размеров."""
        layout = QVBoxLayout(self.tab_size)
        
        # Группа: Входные данные
        input_group = QGroupBox("Входные данные")
        input_layout = QFormLayout(input_group)
        
        # Детекции
        det_layout = QHBoxLayout()
        self.size_det_edit = QLineEdit()
        self.size_det_edit.setPlaceholderText("Путь к detections.csv...")
        det_layout.addWidget(self.size_det_edit)
        
        self.size_det_btn = QPushButton("...")
        self.size_det_btn.setMaximumWidth(30)
        self.size_det_btn.clicked.connect(lambda: self._browse_csv(self.size_det_edit))
        det_layout.addWidget(self.size_det_btn)
        
        input_layout.addRow("CSV детекций:", det_layout)
        
        # Геометрия (опционально)
        geom_layout = QHBoxLayout()
        self.size_geom_edit = QLineEdit()
        self.size_geom_edit.setPlaceholderText("(опционально) Путь к geometry.csv...")
        geom_layout.addWidget(self.size_geom_edit)
        
        self.size_geom_btn = QPushButton("...")
        self.size_geom_btn.setMaximumWidth(30)
        self.size_geom_btn.clicked.connect(lambda: self._browse_csv(self.size_geom_edit))
        geom_layout.addWidget(self.size_geom_btn)
        
        input_layout.addRow("Геометрия:", geom_layout)
        
        layout.addWidget(input_group)
        
        # Группа: Параметры
        params_group = QGroupBox("Параметры регрессии")
        params_layout = QFormLayout(params_group)
        
        self.size_min_depth = QDoubleSpinBox()
        self.size_min_depth.setRange(0.1, 5.0)
        self.size_min_depth.setValue(0.3)
        self.size_min_depth.setSingleStep(0.1)
        self.size_min_depth.setSuffix(" м")
        params_layout.addRow("Мин. изменение глубины:", self.size_min_depth)
        
        self.size_min_points = QSpinBox()
        self.size_min_points.setRange(3, 50)
        self.size_min_points.setValue(3)
        params_layout.addRow("Мин. точек в треке:", self.size_min_points)
        
        self.size_min_r2 = QDoubleSpinBox()
        self.size_min_r2.setRange(0.1, 0.99)
        self.size_min_r2.setValue(0.5)
        self.size_min_r2.setSingleStep(0.05)
        params_layout.addRow("Мин. R²:", self.size_min_r2)
        
        self.size_min_change = QDoubleSpinBox()
        self.size_min_change.setRange(0.1, 1.0)
        self.size_min_change.setValue(0.3)
        self.size_min_change.setSingleStep(0.05)
        params_layout.addRow("Мин. изменение размера:", self.size_min_change)
        
        layout.addWidget(params_group)
        
        # Группа: Выход
        output_group = QGroupBox("Выходные данные")
        output_layout = QFormLayout(output_group)
        
        out_layout = QHBoxLayout()
        self.size_output_edit = QLineEdit()
        self.size_output_edit.setPlaceholderText("Путь для detections_with_size.csv...")
        out_layout.addWidget(self.size_output_edit)
        
        self.size_output_btn = QPushButton("...")
        self.size_output_btn.setMaximumWidth(30)
        self.size_output_btn.clicked.connect(lambda: self._browse_save_csv(self.size_output_edit))
        out_layout.addWidget(self.size_output_btn)
        
        output_layout.addRow("Детекции с размерами:", out_layout)
        
        layout.addWidget(output_group)
        
        layout.addStretch()
    
    def _setup_volume_tab(self):
        """Настройка вкладки объёма."""
        layout = QVBoxLayout(self.tab_volume)
        
        # Группа: Входные данные
        input_group = QGroupBox("Входные данные")
        input_layout = QFormLayout(input_group)
        
        # Детекции
        det_layout = QHBoxLayout()
        self.vol_det_edit = QLineEdit()
        self.vol_det_edit.setPlaceholderText("Путь к detections.csv...")
        det_layout.addWidget(self.vol_det_edit)
        
        self.vol_det_btn = QPushButton("...")
        self.vol_det_btn.setMaximumWidth(30)
        self.vol_det_btn.clicked.connect(lambda: self._browse_csv(self.vol_det_edit))
        det_layout.addWidget(self.vol_det_btn)
        
        input_layout.addRow("CSV детекций:", det_layout)
        
        # Треки (опционально)
        tracks_layout = QHBoxLayout()
        self.vol_tracks_edit = QLineEdit()
        self.vol_tracks_edit.setPlaceholderText("(опционально) Путь к track_sizes.csv...")
        tracks_layout.addWidget(self.vol_tracks_edit)
        
        self.vol_tracks_btn = QPushButton("...")
        self.vol_tracks_btn.setMaximumWidth(30)
        self.vol_tracks_btn.clicked.connect(lambda: self._browse_csv(self.vol_tracks_edit))
        tracks_layout.addWidget(self.vol_tracks_btn)
        
        input_layout.addRow("CSV треков:", tracks_layout)
        
        # CTD (опционально)
        ctd_layout = QHBoxLayout()
        self.vol_ctd_edit = QLineEdit()
        self.vol_ctd_edit.setPlaceholderText("(опционально) Путь к CTD.csv...")
        ctd_layout.addWidget(self.vol_ctd_edit)
        
        self.vol_ctd_btn = QPushButton("...")
        self.vol_ctd_btn.setMaximumWidth(30)
        self.vol_ctd_btn.clicked.connect(lambda: self._browse_csv(self.vol_ctd_edit))
        ctd_layout.addWidget(self.vol_ctd_btn)
        
        input_layout.addRow("CTD данные:", ctd_layout)
        
        layout.addWidget(input_group)
        
        # Группа: Параметры камеры
        camera_group = QGroupBox("Параметры камеры")
        camera_layout = QFormLayout(camera_group)
        
        self.vol_fov = QDoubleSpinBox()
        self.vol_fov.setRange(50.0, 180.0)
        self.vol_fov.setValue(100.0)
        self.vol_fov.setSingleStep(5.0)
        self.vol_fov.setSuffix("°")
        camera_layout.addRow("Горизонтальный FOV:", self.vol_fov)
        
        self.vol_near = QDoubleSpinBox()
        self.vol_near.setRange(0.1, 2.0)
        self.vol_near.setValue(0.3)
        self.vol_near.setSingleStep(0.1)
        self.vol_near.setSuffix(" м")
        camera_layout.addRow("Ближняя граница:", self.vol_near)
        
        self.vol_det_dist = QDoubleSpinBox()
        self.vol_det_dist.setRange(0.0, 10.0)
        self.vol_det_dist.setValue(0.0)
        self.vol_det_dist.setSingleStep(0.5)
        self.vol_det_dist.setSuffix(" м")
        self.vol_det_dist.setSpecialValueText("Авто")
        camera_layout.addRow("Дистанция обнаружения:", self.vol_det_dist)
        
        self.vol_fps = QDoubleSpinBox()
        self.vol_fps.setRange(1.0, 240.0)
        self.vol_fps.setValue(60.0)
        self.vol_fps.setSingleStep(1.0)
        camera_layout.addRow("FPS видео:", self.vol_fps)
        
        layout.addWidget(camera_group)
        
        # Группа: Выход
        output_group = QGroupBox("Выходные данные")
        output_layout = QFormLayout(output_group)
        
        out_layout = QHBoxLayout()
        self.vol_output_edit = QLineEdit()
        self.vol_output_edit.setPlaceholderText("Путь для volume.csv...")
        out_layout.addWidget(self.vol_output_edit)
        
        self.vol_output_btn = QPushButton("...")
        self.vol_output_btn.setMaximumWidth(30)
        self.vol_output_btn.clicked.connect(lambda: self._browse_save_csv(self.vol_output_edit))
        out_layout.addWidget(self.vol_output_btn)
        
        output_layout.addRow("Результат:", out_layout)
        
        layout.addWidget(output_group)
        
        layout.addStretch()
    
    def _load_data(self):
        """Загрузка данных из погружения."""
        if not self.dive_id:
            return
        
        dive = self.repo.get_dive(self.dive_id)
        if not dive:
            return
        
        output_dir = os.path.join(dive.folder_path, "output")
        
        # Пробуем найти видео
        video_files = self.repo.get_video_files_for_dive(self.dive_id)
        if video_files:
            self.geom_video_edit.setText(video_files[0].filepath)
            
            # Устанавливаем разрешение
            if video_files[0].width:
                self.geom_width.setValue(video_files[0].width)
            if video_files[0].height:
                self.geom_height.setValue(video_files[0].height)
            if video_files[0].fps:
                self.vol_fps.setValue(video_files[0].fps)
        
        # Устанавливаем пути по умолчанию
        self.geom_output_edit.setText(os.path.join(output_dir, "geometry.csv"))
        
        # Ищем существующие файлы детекций
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                if f.endswith("_detections.csv"):
                    det_path = os.path.join(output_dir, f)
                    self.size_det_edit.setText(det_path)
                    self.vol_det_edit.setText(det_path)
                    
                    # Генерируем имена выходных файлов
                    base = f.replace("_detections.csv", "")
                    self.size_output_edit.setText(os.path.join(output_dir, f"{base}_detections_with_size.csv"))
                    self.vol_output_edit.setText(os.path.join(output_dir, f"{base}_volume.csv"))
                    break
        
        # Ищем CTD
        ctd_files = self.repo.get_ctd_files_for_dive(self.dive_id)
        if ctd_files:
            self.vol_ctd_edit.setText(ctd_files[0].filepath)
    
    def _browse_video(self, line_edit: QLineEdit):
        """Выбор видеофайла."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать видео",
            "",
            "Видео (*.mp4 *.avi *.mov *.mkv);;Все файлы (*.*)"
        )
        if path:
            line_edit.setText(path)
    
    def _browse_csv(self, line_edit: QLineEdit):
        """Выбор CSV файла."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать CSV",
            "",
            "CSV файлы (*.csv);;Все файлы (*.*)"
        )
        if path:
            line_edit.setText(path)
    
    def _browse_save_csv(self, line_edit: QLineEdit):
        """Выбор пути для сохранения CSV."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить CSV",
            "",
            "CSV файлы (*.csv);;Все файлы (*.*)"
        )
        if path:
            if not path.endswith(".csv"):
                path += ".csv"
            line_edit.setText(path)
    
    def _on_run(self):
        """Запуск обработки."""
        current_tab = self.tabs.currentIndex()
        
        if current_tab == 0:
            self._run_geometry()
        elif current_tab == 1:
            self._run_size()
        elif current_tab == 2:
            self._run_volume()
    
    def _run_geometry(self):
        """Запуск расчёта геометрии."""
        video_path = self.geom_video_edit.text().strip()
        output_csv = self.geom_output_edit.text().strip()
        
        if not video_path or not os.path.exists(video_path):
            QMessageBox.warning(self, "Ошибка", "Выберите видеофайл.")
            return
        
        if not output_csv:
            QMessageBox.warning(self, "Ошибка", "Укажите путь для сохранения.")
            return
        
        self._set_running(True)
        self.result_text.clear()
        self.result_text.append("Запуск анализа наклона камеры...")
        
        self._worker = GeometryWorker(
            video_path=video_path,
            output_csv=output_csv,
            frame_interval=self.geom_interval.value(),
            frame_width=self.geom_width.value(),
            frame_height=self.geom_height.value(),
        )
        self._worker.finished.connect(self._on_geometry_finished)
        self._worker.start()
    
    def _run_size(self):
        """Запуск оценки размеров."""
        det_path = self.size_det_edit.text().strip()
        output_csv = self.size_output_edit.text().strip()
        
        if not det_path or not os.path.exists(det_path):
            QMessageBox.warning(self, "Ошибка", "Выберите файл детекций.")
            return
        
        if not output_csv:
            QMessageBox.warning(self, "Ошибка", "Укажите путь для сохранения.")
            return
        
        geom_path = self.size_geom_edit.text().strip()
        if geom_path and not os.path.exists(geom_path):
            geom_path = None
        
        # Путь для треков
        tracks_csv = output_csv.replace(".csv", "_track_sizes.csv")
        
        self._set_running(True)
        self.result_text.clear()
        self.result_text.append("Запуск оценки размеров...")
        
        self._worker = SizeWorker(
            detections_csv=det_path,
            output_csv=output_csv,
            tracks_csv=tracks_csv,
            geometry_csv=geom_path if geom_path else None,
            frame_width=self.geom_width.value(),
            frame_height=self.geom_height.value(),
            min_depth_change=self.size_min_depth.value(),
            min_track_points=self.size_min_points.value(),
            min_r_squared=self.size_min_r2.value(),
            min_size_change=self.size_min_change.value(),
        )
        self._worker.finished.connect(self._on_size_finished)
        self._worker.start()
    
    def _run_volume(self):
        """Запуск расчёта объёма."""
        det_path = self.vol_det_edit.text().strip()
        output_csv = self.vol_output_edit.text().strip()
        
        if not det_path or not os.path.exists(det_path):
            QMessageBox.warning(self, "Ошибка", "Выберите файл детекций.")
            return
        
        if not output_csv:
            QMessageBox.warning(self, "Ошибка", "Укажите путь для сохранения.")
            return
        
        tracks_path = self.vol_tracks_edit.text().strip()
        if tracks_path and not os.path.exists(tracks_path):
            tracks_path = None
        
        ctd_path = self.vol_ctd_edit.text().strip()
        if ctd_path and not os.path.exists(ctd_path):
            ctd_path = None
        
        det_dist = self.vol_det_dist.value()
        if det_dist == 0.0:
            det_dist = None
        
        self._set_running(True)
        self.result_text.clear()
        self.result_text.append("Запуск расчёта объёма...")
        
        self._worker = VolumeWorker(
            detections_csv=det_path,
            output_csv=output_csv,
            tracks_csv=tracks_path if tracks_path else None,
            ctd_csv=ctd_path if ctd_path else None,
            fov_horizontal=self.vol_fov.value(),
            near_distance=self.vol_near.value(),
            detection_distance=det_dist,
            fps=self.vol_fps.value(),
            frame_width=self.geom_width.value(),
            frame_height=self.geom_height.value(),
        )
        self._worker.finished.connect(self._on_volume_finished)
        self._worker.start()
    
    def _on_geometry_finished(self, result):
        """Обработка результата геометрии."""
        self._set_running(False)
        
        if result.success:
            self.result_text.append(f"\n✅ Готово за {result.processing_time_s:.1f} сек")
            self.result_text.append(f"Интервалов: {result.n_intervals}")
            if result.mean_tilt_deg is not None:
                self.result_text.append(f"Средний наклон: {result.mean_tilt_deg:.1f}°")
            if result.n_outliers > 0:
                self.result_text.append(f"Отфильтровано выбросов: {result.n_outliers}")
            self.result_text.append(f"\nФайл сохранён: {result.output_csv_path}")
        else:
            self.result_text.append(f"\n❌ Ошибка: {result.error_message}")
    
    def _on_size_finished(self, result):
        """Обработка результата оценки размеров."""
        self._set_running(False)
        
        if result.success:
            self.result_text.append(f"\n✅ Готово за {result.processing_time_s:.1f} сек")
            self.result_text.append(f"Всего треков с размерами: {result.total_tracks}")
            self.result_text.append(f"  - регрессия: {result.tracks_with_regression}")
            self.result_text.append(f"  - референс: {result.tracks_with_reference}")
            self.result_text.append(f"  - типичный: {result.tracks_with_typical}")
            self.result_text.append(f"\nФайл сохранён: {result.output_csv_path}")
            if result.tracks_csv_path:
                self.result_text.append(f"Треки: {result.tracks_csv_path}")
        else:
            self.result_text.append(f"\n❌ Ошибка: {result.error_message}")
    
    def _on_volume_finished(self, result):
        """Обработка результата расчёта объёма."""
        self._set_running(False)
        
        if result.success:
            self.result_text.append(f"\n✅ Готово за {result.processing_time_s:.1f} сек")
            self.result_text.append(f"Общий объём: {result.total_volume_m3:.1f} м³")
            self.result_text.append(f"Глубины: {result.depth_min_m:.1f} - {result.depth_max_m:.1f} м")
            self.result_text.append(f"Дистанция обнаружения: {result.detection_distance_m:.2f} м")
            
            if result.counts_by_class:
                self.result_text.append("\nПодсчёт по классам:")
                for cls, count in result.counts_by_class.items():
                    density = result.density_by_class.get(cls, 0)
                    self.result_text.append(f"  {cls}: {count} ({density:.4f} ос./м³)")
            
            self.result_text.append(f"\nФайл сохранён: {result.output_csv_path}")
        else:
            self.result_text.append(f"\n❌ Ошибка: {result.error_message}")
    
    def _on_cancel(self):
        """Отмена обработки."""
        if self._worker:
            self._worker.cancel()
            self._worker.wait()
            self._set_running(False)
            self.result_text.append("\n⚠️ Отменено")
    
    def _set_running(self, running: bool):
        """Установка состояния работы."""
        self.btn_run.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        self.tabs.setEnabled(not running)
        self.progress_bar.setVisible(running)
        
        if running:
            self.progress_bar.setRange(0, 0)  # Indeterminate
        else:
            self.progress_bar.setRange(0, 100)
