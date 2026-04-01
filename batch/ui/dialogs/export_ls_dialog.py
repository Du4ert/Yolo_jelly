"""
Диалог экспорта кадров и предразметки в Label Studio из завершённой задачи.

Читает CSV с детекциями + оригинальное видео, извлекает кадры
и формирует preannotations.json через LabelStudioExporter.
"""

import sys
from pathlib import Path

import cv2
import pandas as pd
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
    QDialogButtonBox,
    QProgressBar,
    QFileDialog,
    QMessageBox,
    QWidget,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ...database import Repository, TaskStatus
from ...core import get_config, save_config

# Путь к src для импорта label_studio_export и constants
ROOT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))


class _ExportWorker(QThread):
    """Фоновый поток экспорта кадров из видео по CSV детекций."""

    progress = pyqtSignal(int)       # процент 0-100
    finished = pyqtSignal(str)       # путь к JSON
    error = pyqtSignal(str)          # сообщение об ошибке

    def __init__(
        self,
        video_path: str,
        csv_path: str,
        output_dir: str,
        video_name: str,
        frame_interval: int,
        export_classes: set | None,
        image_quality: int = 95,
    ):
        super().__init__()
        self.video_path = video_path
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.video_name = video_name
        self.frame_interval = frame_interval
        self.export_classes = export_classes
        self.image_quality = image_quality
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            from label_studio_export import LabelStudioExporter

            df = pd.read_csv(self.csv_path)
            if df.empty:
                self.error.emit("CSV детекций пуст")
                return

            exporter = LabelStudioExporter(
                output_dir=self.output_dir,
                video_name=self.video_name,
                frame_interval=self.frame_interval,
                export_classes=self.export_classes,
                image_quality=self.image_quality,
            )

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit(f"Не удалось открыть видео: {self.video_path}")
                return

            grouped = df.groupby("frame")
            total = len(grouped)
            prev_frame_num = -1

            for i, (frame_num, group) in enumerate(grouped):
                if self._cancelled:
                    cap.release()
                    return

                # Перемотка к нужному кадру
                frame_num = int(frame_num)
                if frame_num != prev_frame_num + 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    prev_frame_num = frame_num
                    continue
                prev_frame_num = frame_num

                timestamp = float(group.iloc[0].get("timestamp_s", 0))

                detections = []
                for _, row in group.iterrows():
                    det = {
                        "class_name": row["class_name"],
                        "confidence": float(row["confidence"]),
                        "x_center": float(row["x_center"]),
                        "y_center": float(row["y_center"]),
                        "width": float(row["width"]),
                        "height": float(row["height"]),
                        "track_id": (
                            int(row["track_id"])
                            if pd.notna(row.get("track_id"))
                            else None
                        ),
                    }
                    detections.append(det)

                exporter.process_frame(frame, frame_num, timestamp, detections)

                self.progress.emit(int((i + 1) / total * 100))

            cap.release()
            json_path = exporter.finalize()
            self.finished.emit(json_path)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class ExportLabelStudioDialog(QDialog):
    """Диалог экспорта кадров и предразметки в Label Studio из завершённой задачи."""

    def __init__(self, repo: Repository, task_id: int, parent=None):
        super().__init__(parent)
        self.repo = repo
        self.task_id = task_id
        self.task = repo.get_task_with_outputs(task_id)
        self._worker = None

        if not self.task:
            raise ValueError(f"Задача {task_id} не найдена")
        if self.task.status != TaskStatus.DONE:
            raise ValueError("Экспорт доступен только для завершённых задач")
        if not self.task.detections_csv_path:
            raise ValueError("CSV с детекциями не найден")

        self.setWindowTitle(f"Экспорт в Label Studio — задача #{task_id}")
        self.setMinimumWidth(500)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # === Информация о задаче ===
        info_group = QGroupBox("Задача")
        info_layout = QFormLayout(info_group)

        video = self.repo.get_video_file(self.task.video_id)
        info_layout.addRow("Видео:", QLabel(video.filename if video else "???"))

        det_text = f"{self.task.detections_count or 0} детекций"
        if self.task.tracks_count:
            det_text += f", {self.task.tracks_count} треков"
        info_layout.addRow("Результат:", QLabel(det_text))
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

        # === Прогресс ===
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.label_status = QLabel()
        self.label_status.setVisible(False)
        layout.addWidget(self.label_status)

        # === Кнопки ===
        self.btn_export = QPushButton("Экспортировать")
        self.btn_export.clicked.connect(self._start_export)
        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(self.close)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_export)
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

    def _get_export_classes(self) -> set | None:
        selected = {name for name, chk in self.class_checks.items() if chk.isChecked()}
        if len(selected) == len(self.class_checks):
            return None  # все выбраны
        return selected

    def _get_output_dir(self) -> str:
        custom = self.edit_dir.text().strip()
        if custom:
            return custom
        # Fallback — папка погружения
        video = self.repo.get_video_file(self.task.video_id)
        if video:
            dive = self.repo.get_dive(video.dive_id)
            if dive and dive.folder_path:
                return str(Path(dive.folder_path) / "label_studio_export")
        raise ValueError("Не удалось определить папку экспорта")

    def _start_export(self):
        try:
            output_dir = self._get_output_dir()
        except ValueError as e:
            QMessageBox.warning(self, "Ошибка", str(e))
            return

        video = self.repo.get_video_file(self.task.video_id)
        if not video:
            QMessageBox.warning(self, "Ошибка", "Видеофайл не найден")
            return

        self.btn_export.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.label_status.setText("Экспорт...")
        self.label_status.setVisible(True)

        self._worker = _ExportWorker(
            video_path=video.filepath,
            csv_path=self.task.detections_csv_path,
            output_dir=output_dir,
            video_name=Path(video.filepath).stem,
            frame_interval=self.spin_interval.value(),
            export_classes=self._get_export_classes(),
        )
        self._worker.progress.connect(self.progress_bar.setValue)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, json_path: str):
        self.progress_bar.setValue(100)
        self.label_status.setText(f"Готово: {json_path}")
        self.btn_export.setEnabled(True)
        QMessageBox.information(
            self, "Экспорт завершён",
            f"Данные экспортированы.\n\nJSON: {json_path}"
        )

    def _on_error(self, message: str):
        self.label_status.setText(f"Ошибка: {message}")
        self.btn_export.setEnabled(True)
        QMessageBox.critical(self, "Ошибка экспорта", message)

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
        super().closeEvent(event)
