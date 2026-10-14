"""Параметры и фоновое построение общей панели графиков выбранных задач."""

import sys
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLabel, QLineEdit, QDoubleSpinBox,
    QSpinBox, QComboBox, QPushButton, QFileDialog, QProgressBar, QMessageBox,
    QPlainTextEdit,
)

from ...database import OutputType


class PlotPanelWorker(QThread):
    """Строит лист без блокировки интерфейса."""

    progress = pyqtSignal(int, int)
    result_ready = pyqtSignal(str, str)

    def __init__(self, tasks, path, parameters, parent=None):
        super().__init__(parent)
        self.tasks = tasks
        self.path = path
        self.parameters = parameters

    def run(self):
        try:
            from plot_panel import create_plot_panel
            path = create_plot_panel(self.tasks, self.path, progress=self.progress.emit, **self.parameters)
            self.result_ready.emit(path, '')
        except Exception as error:
            self.result_ready.emit('', str(error))


class PlotPanelDialog(QDialog):
    """Выбор общих параметров CTD, термоклина и слоёв SigmaT."""

    def __init__(self, repository, task_ids, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Панель графиков')
        self.setMinimumWidth(620)
        self.worker = None
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
        self.tasks, errors = self._collect_tasks(repository, task_ids)
        layout = QVBoxLayout(self)
        summary = QPlainTextEdit()
        summary.setReadOnly(True)
        summary.setPlainText('\n'.join(t['title'] for t in self.tasks))
        summary.setMaximumHeight(140)
        layout.addWidget(summary)
        info = QLabel('Одна задача — одна строка. CTD и KDE выровнены между задачами; глубина индивидуальная.')
        info.setWordWrap(True)
        layout.addWidget(info)
        self.form = QFormLayout()
        layout.addLayout(self.form)
        self.ctd_columns = QLineEdit('6,11,12,16')
        self.ctd_columns.setToolTip('0-based индексы колонок CTD, через запятую. Пустое поле отключает блок CTD.')
        self.form.addRow('Колонки CTD:', self.ctd_columns)
        self.depth_bin = QDoubleSpinBox()
        self.depth_bin.setRange(0.1, 10)
        self.depth_bin.setValue(2)
        self.form.addRow('Шаг по глубине, м:', self.depth_bin)
        self.mode = QComboBox()
        self.mode.addItem('Порог', 'threshold')
        self.mode.addItem('Отключить', 'off')
        self.form.addRow('Термоклин:', self.mode)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.01, 5)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(1)
        self.form.addRow('Порог, °C на 1 м:', self.threshold)
        self.mode.currentIndexChanged.connect(
            lambda: self.threshold.setEnabled(self.mode.currentData() == 'threshold'))
        self.temperature_column = QSpinBox()
        self.temperature_column.setRange(0, 999)
        self.temperature_column.setValue(6)
        self.form.addRow('Колонка температуры (0-based):', self.temperature_column)
        self.sigma_column = QSpinBox()
        self.sigma_column.setRange(0, 999)
        self.sigma_column.setValue(11)
        self.form.addRow('Колонка SigmaT (0-based):', self.sigma_column)
        self.levels = QLineEdit('14.5,15.5,16.2')
        self.levels.setToolTip('Пустое поле отключает слои плотности.')
        self.form.addRow('Уровни SigmaT:', self.levels)
        self.path = QLineEdit()
        self.form.addRow('Выходной HTML:', self.path)
        self.browse = QPushButton('Выбрать файл…')
        self.browse.clicked.connect(self._browse)
        layout.addWidget(self.browse)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.run_button = QPushButton('Построить и открыть')
        self.run_button.clicked.connect(self._run)
        layout.addWidget(self.run_button)
        close = QPushButton('Закрыть')
        close.clicked.connect(self.close)
        layout.addWidget(close)
        if errors:
            error_label = QPlainTextEdit()
            error_label.setReadOnly(True)
            error_label.setPlainText('\n'.join(errors))
            error_label.setMaximumHeight(130)
            error_label.setStyleSheet('color: #b00020;')
            layout.addWidget(error_label)
            self.run_button.setEnabled(False)
        elif self.tasks:
            self.path.setText(str(Path(self.tasks[0]['track_sizes_path']).parent / 'plot_panel.html'))

    @staticmethod
    def _collect_tasks(repo, task_ids):
        import pandas as pd
        tasks, errors = [], []
        for task_id in task_ids:
            task = repo.get_task(task_id)
            if not task:
                errors.append(f'Задача #{task_id} не найдена.')
                continue
            video = repo.get_video_file(task.video_id)
            dive = repo.get_dive(video.dive_id) if video else None
            title = f"{dive.name if dive else '?'} / {video.filename if video else '?'} (#{task_id})"
            outputs = repo.get_task_outputs(task_id)
            paths = {o.output_type: o.filepath for o in outputs if Path(o.filepath).is_file()}
            tracks = paths.get(OutputType.TRACK_SIZES_CSV)
            if not tracks:
                errors.append(f'{title}: отсутствует CSV размеров треков. Выполните расчёт размеров.')
                continue
            data = dict(title=title, track_sizes_path=tracks)
            if task.ctd_id:
                ctd = repo.get_ctd_file(task.ctd_id)
                if not ctd or not Path(ctd.filepath).is_file():
                    errors.append(f'{title}: привязанный CTD-файл не найден.')
                else:
                    data['ctd_path'] = ctd.filepath
            volume_path = paths.get(OutputType.VOLUME_CSV)
            if volume_path:
                try:
                    volume = pd.read_csv(volume_path)
                    for key in ('cross_section_area_m2', 'pleurobrachia_cross_section_area_m2'):
                        values = volume.loc[volume['parameter'] == key, 'value']
                        if not values.empty:
                            value = float(values.iloc[0])
                            if value > 0 and pd.notna(value) and value != float('inf'):
                                data[key] = value
                except Exception as error:
                    errors.append(f'{title}: не удалось прочитать параметры объёма: {error}')
            tasks.append(data)
        normalized = [bool(t.get('cross_section_area_m2')) for t in tasks]
        if any(normalized) and not all(normalized):
            errors.append('Несовместимые единицы KDE: площадь сечения есть только у части задач. '
                          'Выполните расчёт объёма для остальных задач.')
        if len(tasks) < 2 and not errors:
            errors.append('Для панели необходимы данные не менее двух задач.')
        return tasks, errors

    def _browse(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Сохранить панель', self.path.text(), 'HTML (*.html)')
        if path:
            self.path.setText(path)

    def _run(self):
        import math
        try:
            columns = [int(v.strip()) for v in self.ctd_columns.text().split(',') if v.strip()]
            levels = [float(v.strip()) for v in self.levels.text().split(',') if v.strip()]
            if any(v < 0 for v in columns) or any(not math.isfinite(v) for v in levels):
                raise ValueError('Проверьте индексы колонок CTD и уровни SigmaT.')
            if not self.path.text().strip():
                raise ValueError('Укажите путь сохранения HTML.')
            parameters = dict(ctd_columns=list(dict.fromkeys(columns)), depth_bin=self.depth_bin.value(),
                              thermocline_mode=self.mode.currentData(), thermocline_threshold=self.threshold.value(),
                              temperature_column=self.temperature_column.value(), sigma_t_column=self.sigma_column.value(),
                              density_levels=levels)
        except ValueError as error:
            QMessageBox.warning(self, 'Параметры панели', str(error))
            return
        self._set_running(True)
        self.progress.setRange(0, len(self.tasks))
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.worker = PlotPanelWorker(self.tasks, self.path.text().strip(), parameters, self)
        self.worker.progress.connect(lambda current, total: self.progress.setValue(current))
        self.worker.result_ready.connect(self._finished)
        self.worker.start()

    def _finished(self, path, error):
        self.worker.wait()
        self._set_running(False)
        self.progress.setVisible(False)
        if error:
            QMessageBox.warning(self, 'Ошибка построения панели', error)
        else:
            self.path.setText(path)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).resolve())))

    def _set_running(self, running):
        for widget in (self.ctd_columns, self.depth_bin, self.mode, self.threshold,
                       self.temperature_column, self.sigma_column, self.levels,
                       self.path, self.browse, self.run_button):
            widget.setEnabled(not running)
        self.threshold.setEnabled(not running and self.mode.currentData() == 'threshold')

    def reject(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, 'Панель графиков', 'Дождитесь завершения построения панели.')
            return
        super().reject()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            event.ignore()
            return
        super().closeEvent(event)
