"""
Главное окно приложения.
"""

import os
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QMenuBar,
    QMenu,
    QStatusBar,
    QToolBar,
    QMessageBox,
    QFileDialog,
    QToolButton,
    QLabel,
    QSpinBox,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QIcon, QKeySequence

from ..database import Repository
from ..core import TaskManager, get_config, save_config
from .widgets.dive_panel import DivePanel
from .widgets.model_panel import ModelPanel
from .widgets.task_table import TaskTable
from .widgets.status_bar import StatusBarWidget
from .dialogs import NewTaskDialog


class MainWindow(QMainWindow):
    """
    Главное окно приложения YOLO Jellyfish Batch Processor.
    """

    def __init__(self, repository: Repository):
        super().__init__()
        
        self.repo = repository
        self.task_manager = TaskManager(repository, self)
        
        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()
        self._connect_signals()
        self._restore_state()

    def _setup_ui(self):
        """Настройка интерфейса."""
        self.setWindowTitle("🦑 YOLO Jellyfish - Batch Processor")
        self.setMinimumSize(1100, 700)
        
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)
        
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.main_splitter)
        
        # Левая панель
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        
        self.left_splitter = QSplitter(Qt.Orientation.Vertical)
        left_layout.addWidget(self.left_splitter)
        
        self.dive_panel = DivePanel(self.repo)
        self.left_splitter.addWidget(self.dive_panel)
        
        self.model_panel = ModelPanel(self.repo)
        self.left_splitter.addWidget(self.model_panel)
        
        self.left_splitter.setSizes([400, 200])
        self.main_splitter.addWidget(left_widget)
        
        # Правая панель
        self.task_table = TaskTable(self.repo, self.task_manager)
        self.main_splitter.addWidget(self.task_table)
        
        self.main_splitter.setSizes([350, 750])

    def _setup_menu(self):
        """Настройка меню."""
        menubar = self.menuBar()
        
        # === Файл ===
        file_menu = menubar.addMenu("&Файл")
        
        self.action_add_dive = QAction("📁 Добавить погружение...", self)
        self.action_add_dive.setShortcut(QKeySequence("Ctrl+D"))
        self.action_add_dive.triggered.connect(self._on_add_dive)
        file_menu.addAction(self.action_add_dive)
        
        self.action_add_model = QAction("🧠 Добавить модель...", self)
        self.action_add_model.setShortcut(QKeySequence("Ctrl+M"))
        self.action_add_model.triggered.connect(self._on_add_model)
        file_menu.addAction(self.action_add_model)
        
        file_menu.addSeparator()
        
        self.action_exit = QAction("Выход", self)
        self.action_exit.setShortcut(QKeySequence("Ctrl+Q"))
        self.action_exit.triggered.connect(self.close)
        file_menu.addAction(self.action_exit)
        
        # === Очередь ===
        queue_menu = menubar.addMenu("&Очередь")
        
        self.action_start_queue = QAction("▶ Запустить", self)
        self.action_start_queue.setShortcut(QKeySequence("F5"))
        self.action_start_queue.triggered.connect(self._on_start_queue)
        queue_menu.addAction(self.action_start_queue)
        
        self.action_pause_queue = QAction("⏸ Пауза", self)
        self.action_pause_queue.setShortcut(QKeySequence("F6"))
        self.action_pause_queue.triggered.connect(self._on_pause_queue)
        self.action_pause_queue.setEnabled(False)
        queue_menu.addAction(self.action_pause_queue)
        
        self.action_stop_queue = QAction("⏹ Остановить", self)
        self.action_stop_queue.setShortcut(QKeySequence("F7"))
        self.action_stop_queue.triggered.connect(self._on_stop_queue)
        self.action_stop_queue.setEnabled(False)
        queue_menu.addAction(self.action_stop_queue)
        
        # === Справка ===
        help_menu = menubar.addMenu("&Справка")
        
        self.action_about = QAction("О программе", self)
        self.action_about.triggered.connect(self._on_about)
        help_menu.addAction(self.action_about)

    def _setup_toolbar(self):
        """Настройка панели инструментов."""
        toolbar = QToolBar("Основная")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        toolbar.addAction(self.action_add_dive)
        toolbar.addAction(self.action_add_model)
        toolbar.addSeparator()
        toolbar.addAction(self.action_start_queue)
        toolbar.addAction(self.action_pause_queue)
        toolbar.addAction(self.action_stop_queue)
        toolbar.addSeparator()

        # Количество параллельных воркеров
        toolbar.addWidget(QLabel(" Потоков: "))
        self._workers_spinbox = QSpinBox()
        self._workers_spinbox.setRange(1, 8)
        self._workers_spinbox.setValue(get_config().max_parallel_workers)
        self._workers_spinbox.setToolTip(
            "Количество параллельных задач.\n"
            "При device=cuda не ставьте больше, чем позволяет VRAM."
        )
        self._workers_spinbox.valueChanged.connect(self._on_workers_count_changed)
        toolbar.addWidget(self._workers_spinbox)
        toolbar.addSeparator()

        # Кнопка выбора глобального файла калибровки
        toolbar.addWidget(QLabel(" 📏 Калибровка: "))

        self.btn_calibration = QToolButton()
        self.btn_calibration.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.btn_calibration.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.btn_calibration.setToolTip(
            "Глобальный файл калибровки камеры.\n"
            "Используется при расчёте размеров во всех задачах."
        )
        self.btn_calibration.clicked.connect(self._on_calibration_browse)

        calib_menu = QMenu(self)
        self.action_calib_browse = QAction("Выбрать файл...", self)
        self.action_calib_browse.triggered.connect(self._on_calibration_browse)
        calib_menu.addAction(self.action_calib_browse)

        self.action_calib_clear = QAction("Очистить (использовать дефолтные)", self)
        self.action_calib_clear.triggered.connect(self._on_calibration_clear)
        calib_menu.addAction(self.action_calib_clear)

        self.btn_calibration.setMenu(calib_menu)
        toolbar.addWidget(self.btn_calibration)
        self._update_calibration_button()

    def _setup_statusbar(self):
        """Настройка статусной строки."""
        self.status_widget = StatusBarWidget(self.repo, self.task_manager)
        self.statusBar().addPermanentWidget(self.status_widget, 1)

    def _connect_signals(self):
        """Подключение сигналов."""
        self.task_manager.queue_changed.connect(self._on_queue_changed)
        self.task_manager.queue_state_changed.connect(self._on_queue_state_changed)
        self.task_manager.task_started.connect(self._on_task_started)
        self.task_manager.task_finished.connect(self._on_task_finished)
        self.task_manager.queue_finished.connect(self._on_queue_finished)
        
        self.dive_panel.dive_selected.connect(self._on_dive_selected)
        self.dive_panel.add_to_queue_requested.connect(self._on_add_to_queue)
        self.dive_panel.quick_add_to_queue_requested.connect(self._on_quick_add_to_queue)
        self.dive_panel.batch_add_to_queue_requested.connect(self._on_batch_add_to_queue)
        self.model_panel.model_selected.connect(self._on_model_selected)

    def _restore_state(self):
        """Восстанавливает состояние окна из конфига."""
        try:
            config = get_config()
            if config.ui.window_geometry:
                from PyQt6.QtCore import QByteArray
                import base64
                geometry = QByteArray(base64.b64decode(config.ui.window_geometry))
                self.restoreGeometry(geometry)
            
            if config.ui.splitter_sizes:
                self.main_splitter.setSizes(config.ui.splitter_sizes[:2])
        except Exception:
            pass

    def _save_state(self):
        """Сохраняет состояние окна в конфиг."""
        try:
            config = get_config()
            import base64
            config.ui.window_geometry = base64.b64encode(
                bytes(self.saveGeometry())
            ).decode('ascii')
            config.ui.splitter_sizes = self.main_splitter.sizes()
            save_config()
        except Exception:
            pass

    # ========== Обработчики меню ==========

    def _update_calibration_button(self):
        """Обновляет текст кнопки калибровки по текущему конфигу."""
        try:
            config = get_config()
            path = config.ui.calibration_json
            if path and os.path.exists(path):
                name = os.path.basename(path)
                self.btn_calibration.setText(name)
                self.btn_calibration.setStyleSheet("")
            else:
                self.btn_calibration.setText("не задана")
                self.btn_calibration.setStyleSheet("color: gray;")
        except Exception:
            self.btn_calibration.setText("не задана")

    def _on_calibration_browse(self):
        """Открывает диалог выбора файла калибровки."""
        try:
            config = get_config()
            start_dir = os.path.dirname(config.ui.calibration_json or "") or ""
        except Exception:
            start_dir = ""

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать файл калибровки",
            start_dir,
            "JSON файлы (*.json);;Все файлы (*)",
        )
        if path:
            try:
                config = get_config()
                config.ui.calibration_json = path
                save_config()
            except Exception:
                pass
            self._update_calibration_button()
            self.statusBar().showMessage(
                f"Калибровка: {os.path.basename(path)}", 3000
            )

    def _on_calibration_clear(self):
        """Сбрасывает выбранный файл калибровки."""
        try:
            config = get_config()
            config.ui.calibration_json = None
            save_config()
        except Exception:
            pass
        self._update_calibration_button()
        self.statusBar().showMessage("Калибровка сброшена — используются дефолтные коэффициенты", 3000)

    def _on_add_dive(self):
        self.dive_panel.add_dive()

    def _on_add_model(self):
        self.model_panel.add_model()

    def _on_start_queue(self):
        if self.task_manager.is_paused():
            self.task_manager.resume_queue()
        else:
            if not self.task_manager.start_queue():
                QMessageBox.information(
                    self, "Очередь пуста",
                    "Нет задач для выполнения.\nДобавьте видео в очередь."
                )

    def _on_pause_queue(self):
        if self.task_manager.is_paused():
            self.task_manager.resume_queue()
        else:
            self.task_manager.pause_queue()

    def _on_stop_queue(self):
        reply = QMessageBox.question(
            self, "Остановить очередь?",
            "Текущая задача будет завершена, остальные останутся в очереди.\nОстановить?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.task_manager.stop_queue()

    def _on_about(self):
        QMessageBox.about(
            self, "О программе",
            "<h2>🦑 YOLO Jellyfish Batch Processor</h2>"
            "<p>Пакетная обработка видео для детекции желетелого макрозоопланктона.</p>"
            "<p><b>Версия:</b> 0.3.0</p>"
            "<hr>"
            "<p>Использует YOLOv8 для детекции и трекинга.</p>"
            "<p><b>Постобработка задач (колонка «Пост.»):</b></p>"
            "<ul>"
            "<li><b>G</b> - Геометрия камеры (FOE)</li>"
            "<li><b>S</b> - Оценка размеров объектов</li>"
            "<li><b>V</b> - Расчёт осмотренного объёма</li>"
            "<li><b>A</b> - Анализ и визуализация</li>"
            "</ul>"
            "<p>Двойной клик на завершённой задаче → постобработка</p>"
            "<p>Или кнопка «📊 Постобработка» внизу таблицы</p>"
        )

    # ========== Обработчики сигналов ==========

    def _on_queue_changed(self):
        self.status_widget.update_stats()

    def _on_queue_state_changed(self, is_running: bool, is_paused: bool):
        self.action_start_queue.setEnabled(not is_running or is_paused)
        self.action_pause_queue.setEnabled(is_running)
        self.action_stop_queue.setEnabled(is_running)
        self._workers_spinbox.setEnabled(not is_running)

        if is_paused:
            self.action_start_queue.setText("▶ Продолжить")
            self.action_pause_queue.setText("⏸ На паузе")
        else:
            self.action_start_queue.setText("▶ Запустить")
            self.action_pause_queue.setText("⏸ Пауза")

    def _on_workers_count_changed(self, value: int) -> None:
        """Сохраняет количество параллельных воркеров в конфиг."""
        get_config().max_parallel_workers = value
        save_config()

    def _on_task_started(self, task_id: int):
        self.status_widget.set_current_task(task_id)

    def _on_task_finished(self, task_id: int, success: bool, error_message: str):
        if not success and error_message:
            self.statusBar().showMessage(f"Ошибка: {error_message}", 5000)

    def _on_queue_finished(self):
        self.status_widget.clear_current_task()
        QMessageBox.information(self, "Очередь завершена", "Все задачи в очереди выполнены.")

    def _on_dive_selected(self, dive_id: int):
        pass

    def _on_model_selected(self, model_id: int):
        pass

    def _on_add_to_queue(self, video_id: int, ctd_id: Optional[int]):
        model_id = self.model_panel.get_selected_model_id()
        
        models = self.repo.get_all_models()
        if not models:
            QMessageBox.warning(
                self, "Нет моделей",
                "Сначала добавьте модель для детекции.\n\nМеню: Файл → Добавить модель..."
            )
            return
        
        dialog = NewTaskDialog(
            repository=self.repo,
            video_id=video_id,
            ctd_id=ctd_id,
            model_id=model_id,
            parent=self,
        )
        
        if dialog.exec():
            selected_model_id = dialog.get_model_id()
            params = dialog.get_task_params()

            self.model_panel.select_model(selected_model_id)

            task = self.task_manager.add_task(
                video_id=video_id,
                model_id=selected_model_id,
                ctd_id=dialog.get_ctd_id(),
                **params,
            )

            if task:
                self.statusBar().showMessage(f"Задача #{task.id} добавлена в очередь", 3000)
            else:
                QMessageBox.warning(self, "Ошибка", "Не удалось создать задачу.")

    def _on_quick_add_to_queue(self, video_id: int, ctd_id: Optional[int]):
        model_id = self.model_panel.get_selected_model_id()
        if not model_id:
            models = self.repo.get_all_models()
            if not models:
                QMessageBox.warning(
                    self, "Нет моделей",
                    "Сначала добавьте модель для детекции.\n\nМеню: Файл → Добавить модель..."
                )
                return
            QMessageBox.warning(self, "Модель не выбрана", "Выберите модель в панели моделей.")
            return
        
        config = get_config()
        params = {
            "conf_threshold": config.default_detection_params.conf_threshold,
            "enable_tracking": config.default_detection_params.enable_tracking,
            "tracker_type": config.default_detection_params.tracker_type,
            "show_trails": config.default_detection_params.show_trails,
            "trail_length": config.default_detection_params.trail_length,
            "min_track_length": config.default_detection_params.min_track_length,
            "save_video": config.default_detection_params.save_video,
        }
        
        task = self.task_manager.add_task(video_id, model_id, ctd_id, **params)
        if task:
            self.statusBar().showMessage(f"Задача #{task.id} добавлена в очередь", 3000)
        else:
            QMessageBox.warning(self, "Ошибка", "Не удалось создать задачу.")

    def _on_batch_add_to_queue(self, video_ctd_pairs: list):
        """Добавляет несколько видео в очередь с одним диалогом настроек."""
        if not video_ctd_pairs:
            return

        model_id = self.model_panel.get_selected_model_id()
        models = self.repo.get_all_models()
        if not models:
            QMessageBox.warning(
                self, "Нет моделей",
                "Сначала добавьте модель для детекции.\n\nМеню: Файл → Добавить модель..."
            )
            return

        first_video_id, first_ctd_id = video_ctd_pairs[0]
        dialog = NewTaskDialog(
            repository=self.repo,
            video_id=first_video_id,
            ctd_id=first_ctd_id,
            model_id=model_id,
            parent=self,
        )
        if not dialog.exec():
            return

        selected_model_id = dialog.get_model_id()
        params = dialog.get_task_params()

        self.model_panel.select_model(selected_model_id)

        # Первое видео — CTD из диалога (пользователь мог изменить)
        self.task_manager.add_task(
            video_id=first_video_id,
            model_id=selected_model_id,
            ctd_id=dialog.get_ctd_id(),
            **params,
        )

        # Остальные видео — те же параметры, CTD из своего погружения
        for video_id, ctd_id in video_ctd_pairs[1:]:
            self.task_manager.add_task(
                video_id=video_id,
                model_id=selected_model_id,
                ctd_id=ctd_id,
                **params,
            )

        n = len(video_ctd_pairs)
        self.statusBar().showMessage(f"Добавлено задач: {n}", 3000)

    # ========== События окна ==========

    def closeEvent(self, event):
        if self.task_manager.is_running():
            reply = QMessageBox.question(
                self, "Выполняются задачи",
                "Сейчас выполняется обработка. Остановить и выйти?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            
            self.task_manager.stop_queue()
        
        self._save_state()
        event.accept()
