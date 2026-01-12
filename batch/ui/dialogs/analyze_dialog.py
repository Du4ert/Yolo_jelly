"""
Диалог для анализа результатов детекции.
"""

import os
from pathlib import Path
from typing import Optional, List

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QDoubleSpinBox,
    QCheckBox,
    QPushButton,
    QFileDialog,
    QProgressBar,
    QMessageBox,
    QTextEdit,
    QListWidget,
    QListWidgetItem,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ...database import Repository


class AnalyzeWorker(QThread):
    """Воркер для анализа в отдельном потоке."""
    
    finished = pyqtSignal(object)  # AnalyzeResult
    progress = pyqtSignal(int, int)  # current, total
    
    def __init__(
        self,
        csv_path: str,
        output_dir: str,
        depth_bin: float = 1.0,
        time_bin: float = 10.0,
        report_name: str = "report.txt",
        generate_vertical: bool = True,
        generate_timeline: bool = True,
        generate_summary: bool = True,
        generate_report: bool = True,
        video_name: Optional[str] = None,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.depth_bin = depth_bin
        self.time_bin = time_bin
        self.report_name = report_name
        self.generate_vertical = generate_vertical
        self.generate_timeline = generate_timeline
        self.generate_summary = generate_summary
        self.generate_report = generate_report
        self.video_name = video_name
        self._processor = None
    
    def run(self):
        from ...core import AnalyzeProcessor
        
        self._processor = AnalyzeProcessor(
            csv_path=self.csv_path,
            output_dir=self.output_dir,
            depth_bin=self.depth_bin,
            time_bin=self.time_bin,
            report_name=self.report_name,
            generate_vertical_distribution=self.generate_vertical,
            generate_timeline=self.generate_timeline,
            generate_species_summary=self.generate_summary,
            generate_report=self.generate_report,
            video_name=self.video_name,
        )
        
        result = self._processor.run()
        self.finished.emit(result)
    
    def cancel(self):
        if self._processor:
            self._processor.cancel()


class BatchAnalyzeWorker(QThread):
    """Воркер для пакетного анализа."""
    
    finished = pyqtSignal(list)  # List[AnalyzeResult]
    progress = pyqtSignal(int, int)  # current, total
    file_finished = pyqtSignal(str, bool)  # filename, success
    
    def __init__(
        self,
        csv_paths: List[str],
        output_base_dir: str,
        depth_bin: float = 1.0,
        time_bin: float = 10.0,
    ):
        super().__init__()
        self.csv_paths = csv_paths
        self.output_base_dir = output_base_dir
        self.depth_bin = depth_bin
        self.time_bin = time_bin
        self._processor = None
    
    def run(self):
        from ...core import BatchAnalyzeProcessor
        
        self._processor = BatchAnalyzeProcessor(
            csv_paths=self.csv_paths,
            output_base_dir=self.output_base_dir,
            depth_bin=self.depth_bin,
            time_bin=self.time_bin,
        )
        
        def on_progress(current, total):
            self.progress.emit(current, total)
            filename = Path(self.csv_paths[current - 1]).stem
            # Будет вызван после обработки файла
        
        self._processor.progress_callback = on_progress
        
        results = self._processor.run()
        
        for i, result in enumerate(results):
            filename = Path(self.csv_paths[i]).stem
            self.file_finished.emit(filename, result.success)
        
        self.finished.emit(results)
    
    def cancel(self):
        if self._processor:
            self._processor.cancel()


class AnalyzeDialog(QDialog):
    """
    Диалог для анализа результатов детекции.
    
    Генерирует графики и отчёты по данным детекции.
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
        self.setWindowTitle("Анализ детекций")
        self.setMinimumSize(600, 550)
        
        layout = QVBoxLayout(self)
        
        # Группа: Входные данные
        input_group = QGroupBox("Входные данные")
        input_layout = QFormLayout(input_group)
        
        # Список CSV файлов
        files_layout = QVBoxLayout()
        
        files_btn_layout = QHBoxLayout()
        
        self.btn_add_file = QPushButton("Добавить файл...")
        self.btn_add_file.clicked.connect(self._on_add_file)
        files_btn_layout.addWidget(self.btn_add_file)
        
        self.btn_add_folder = QPushButton("Добавить из папки...")
        self.btn_add_folder.clicked.connect(self._on_add_folder)
        files_btn_layout.addWidget(self.btn_add_folder)
        
        self.btn_remove_file = QPushButton("Удалить")
        self.btn_remove_file.clicked.connect(self._on_remove_file)
        files_btn_layout.addWidget(self.btn_remove_file)
        
        files_btn_layout.addStretch()
        
        files_layout.addLayout(files_btn_layout)
        
        self.files_list = QListWidget()
        self.files_list.setMinimumHeight(100)
        files_layout.addWidget(self.files_list)
        
        input_layout.addRow("CSV файлы:", files_layout)
        
        layout.addWidget(input_group)
        
        # Группа: Параметры
        params_group = QGroupBox("Параметры анализа")
        params_layout = QFormLayout(params_group)
        
        self.depth_bin = QDoubleSpinBox()
        self.depth_bin.setRange(0.1, 10.0)
        self.depth_bin.setValue(1.0)
        self.depth_bin.setSingleStep(0.5)
        self.depth_bin.setSuffix(" м")
        params_layout.addRow("Шаг биннинга по глубине:", self.depth_bin)
        
        self.time_bin = QDoubleSpinBox()
        self.time_bin.setRange(1.0, 120.0)
        self.time_bin.setValue(10.0)
        self.time_bin.setSingleStep(5.0)
        self.time_bin.setSuffix(" сек")
        params_layout.addRow("Шаг биннинга по времени:", self.time_bin)
        
        layout.addWidget(params_group)
        
        # Группа: Что генерировать
        output_group = QGroupBox("Генерируемые выходные данные")
        output_layout = QVBoxLayout(output_group)
        
        self.chk_vertical = QCheckBox("Вертикальное распределение (vertical_distribution.png)")
        self.chk_vertical.setChecked(True)
        output_layout.addWidget(self.chk_vertical)
        
        self.chk_timeline = QCheckBox("Временная шкала (detection_timeline.png)")
        self.chk_timeline.setChecked(True)
        output_layout.addWidget(self.chk_timeline)
        
        self.chk_summary = QCheckBox("Сводка по видам (species_summary.png)")
        self.chk_summary.setChecked(True)
        output_layout.addWidget(self.chk_summary)
        
        self.chk_report = QCheckBox("Текстовый отчёт (report.txt)")
        self.chk_report.setChecked(True)
        output_layout.addWidget(self.chk_report)
        
        layout.addWidget(output_group)
        
        # Группа: Выходная папка
        dir_group = QGroupBox("Выходная директория")
        dir_layout = QHBoxLayout(dir_group)
        
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Папка для сохранения результатов...")
        dir_layout.addWidget(self.output_dir_edit)
        
        self.output_dir_btn = QPushButton("...")
        self.output_dir_btn.setMaximumWidth(30)
        self.output_dir_btn.clicked.connect(self._on_browse_output_dir)
        dir_layout.addWidget(self.output_dir_btn)
        
        layout.addWidget(dir_group)
        
        # Прогресс
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Результаты
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(120)
        layout.addWidget(self.result_text)
        
        # Кнопки
        btn_layout = QHBoxLayout()
        
        self.btn_run = QPushButton("Запустить анализ")
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
    
    def _load_data(self):
        """Загрузка данных из погружения."""
        if not self.dive_id:
            return
        
        dive = self.repo.get_dive(self.dive_id)
        if not dive:
            return
        
        output_dir = os.path.join(dive.folder_path, "output")
        self.output_dir_edit.setText(output_dir)
        
        # Ищем файлы детекций
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                if f.endswith("_detections.csv"):
                    filepath = os.path.join(output_dir, f)
                    self.files_list.addItem(filepath)
    
    def _on_add_file(self):
        """Добавление CSV файла."""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Выбрать CSV файлы",
            "",
            "CSV файлы (*.csv);;Все файлы (*.*)"
        )
        
        for path in paths:
            # Проверяем, не добавлен ли уже
            exists = False
            for i in range(self.files_list.count()):
                if self.files_list.item(i).text() == path:
                    exists = True
                    break
            
            if not exists:
                self.files_list.addItem(path)
    
    def _on_add_folder(self):
        """Добавление всех CSV из папки."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Выбрать папку с CSV"
        )
        
        if folder:
            for f in os.listdir(folder):
                if f.endswith(".csv") and "detection" in f.lower():
                    filepath = os.path.join(folder, f)
                    
                    # Проверяем, не добавлен ли уже
                    exists = False
                    for i in range(self.files_list.count()):
                        if self.files_list.item(i).text() == filepath:
                            exists = True
                            break
                    
                    if not exists:
                        self.files_list.addItem(filepath)
    
    def _on_remove_file(self):
        """Удаление выбранного файла из списка."""
        current = self.files_list.currentRow()
        if current >= 0:
            self.files_list.takeItem(current)
    
    def _on_browse_output_dir(self):
        """Выбор выходной директории."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Выбрать выходную папку"
        )
        
        if folder:
            self.output_dir_edit.setText(folder)
    
    def _on_run(self):
        """Запуск анализа."""
        # Собираем файлы
        csv_paths = []
        for i in range(self.files_list.count()):
            csv_paths.append(self.files_list.item(i).text())
        
        if not csv_paths:
            QMessageBox.warning(self, "Ошибка", "Добавьте CSV файлы для анализа.")
            return
        
        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Ошибка", "Укажите выходную директорию.")
            return
        
        self._set_running(True)
        self.result_text.clear()
        self.result_text.append(f"Запуск анализа {len(csv_paths)} файл(ов)...")
        
        # Пакетный или одиночный анализ
        if len(csv_paths) == 1:
            video_name = Path(csv_paths[0]).stem.replace("_detections", "")
            
            self._worker = AnalyzeWorker(
                csv_path=csv_paths[0],
                output_dir=output_dir,
                depth_bin=self.depth_bin.value(),
                time_bin=self.time_bin.value(),
                generate_vertical=self.chk_vertical.isChecked(),
                generate_timeline=self.chk_timeline.isChecked(),
                generate_summary=self.chk_summary.isChecked(),
                generate_report=self.chk_report.isChecked(),
                video_name=video_name,
            )
            self._worker.finished.connect(self._on_single_finished)
        else:
            self._worker = BatchAnalyzeWorker(
                csv_paths=csv_paths,
                output_base_dir=output_dir,
                depth_bin=self.depth_bin.value(),
                time_bin=self.time_bin.value(),
            )
            self._worker.progress.connect(self._on_batch_progress)
            self._worker.file_finished.connect(self._on_file_finished)
            self._worker.finished.connect(self._on_batch_finished)
        
        self._worker.start()
    
    def _on_single_finished(self, result):
        """Обработка результата одиночного анализа."""
        self._set_running(False)
        
        if result.success:
            self.result_text.append(f"\n✅ Готово за {result.processing_time_s:.1f} сек")
            self.result_text.append(f"Всего детекций: {result.total_detections}")
            self.result_text.append(f"Видов: {result.unique_species}")
            
            if result.species_counts:
                self.result_text.append("\nПо видам:")
                for species, count in result.species_counts.items():
                    self.result_text.append(f"  {species}: {count}")
            
            self.result_text.append(f"\nРезультаты сохранены: {result.output_dir}")
        else:
            self.result_text.append(f"\n❌ Ошибка: {result.error_message}")
    
    def _on_batch_progress(self, current, total):
        """Прогресс пакетной обработки."""
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
    
    def _on_file_finished(self, filename, success):
        """Завершение обработки одного файла."""
        status = "✅" if success else "❌"
        self.result_text.append(f"{status} {filename}")
    
    def _on_batch_finished(self, results):
        """Обработка результатов пакетного анализа."""
        self._set_running(False)
        
        success_count = sum(1 for r in results if r.success)
        total = len(results)
        
        self.result_text.append(f"\n{'='*40}")
        self.result_text.append(f"Обработано: {success_count}/{total} файлов")
        
        total_detections = sum(r.total_detections for r in results if r.success)
        self.result_text.append(f"Всего детекций: {total_detections}")
    
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
        self.btn_add_file.setEnabled(not running)
        self.btn_add_folder.setEnabled(not running)
        self.btn_remove_file.setEnabled(not running)
        self.progress_bar.setVisible(running)
        
        if running:
            self.progress_bar.setRange(0, 0)  # Indeterminate
        else:
            self.progress_bar.setRange(0, 100)
