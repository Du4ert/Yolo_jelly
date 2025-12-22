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
        """
        Инициализация главного окна.
        
        Args:
            repository: Репозиторий для работы с БД.
        """
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
        self.setMinimumSize(1000, 700)
        
        # Центральный виджет
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)
        
        # Главный сплиттер (горизонтальный)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.main_splitter)
        
        # === Левая панель (погружения + модели) ===
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        
        # Вертикальный сплиттер для левой панели
        self.left_splitter = QSplitter(Qt.Orientation.Vertical)
        left_layout.addWidget(self.left_splitter)
        
        # Панель погружений
        self.dive_panel = DivePanel(self.repo)
        self.left_splitter.addWidget(self.dive_panel)
        
        # Панель моделей
        self.model_panel = ModelPanel(self.repo)
        self.left_splitter.addWidget(self.model_panel)
        
        # Пропорции левого сплиттера
        self.left_splitter.setSizes([400, 200])
        
        self.main_splitter.addWidget(left_widget)
        
        # === Правая панель (таблица задач) ===
        self.task_table = TaskTable(self.repo, self.task_manager)
        self.main_splitter.addWidget(self.task_table)
        
        # Пропорции главного сплиттера
        self.main_splitter.setSizes([350, 650])

    def _setup_menu(self):
        """Настройка меню."""
        menubar = self.menuBar()
        
        # === Файл ===
        file_menu = menubar.addMenu("&Файл")
        
        # Добавить погружение
        self.action_add_dive = QAction("Добавить погружение...", self)
        self.action_add_dive.setShortcut(QKeySequence("Ctrl+D"))
        self.action_add_dive.triggered.connect(self._on_add_dive)
        file_menu.addAction(self.action_add_dive)
        
        # Добавить модель
        self.action_add_model = QAction("Добавить модель...", self)
        self.action_add_model.setShortcut(QKeySequence("Ctrl+M"))
        self.action_add_model.triggered.connect(self._on_add_model)
        file_menu.addAction(self.action_add_model)
        
        file_menu.addSeparator()
        
        # Выход
        self.action_exit = QAction("Выход", self)
        self.action_exit.setShortcut(QKeySequence("Ctrl+Q"))
        self.action_exit.triggered.connect(self.close)
        file_menu.addAction(self.action_exit)
        
        # === Очередь ===
        queue_menu = menubar.addMenu("&Очередь")
        
        # Запустить
        self.action_start_queue = QAction("▶ Запустить", self)
        self.action_start_queue.setShortcut(QKeySequence("F5"))
        self.action_start_queue.triggered.connect(self._on_start_queue)
        queue_menu.addAction(self.action_start_queue)
        
        # Пауза
        self.action_pause_queue = QAction("⏸ Пауза", self)
        self.action_pause_queue.setShortcut(QKeySequence("F6"))
        self.action_pause_queue.triggered.connect(self._on_pause_queue)
        self.action_pause_queue.setEnabled(False)
        queue_menu.addAction(self.action_pause_queue)
        
        # Остановить
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
        
        # Добавить погружение
        toolbar.addAction(self.action_add_dive)
        
        # Добавить модель
        toolbar.addAction(self.action_add_model)
        
        toolbar.addSeparator()
        
        # Управление очередью
        toolbar.addAction(self.action_start_queue)
        toolbar.addAction(self.action_pause_queue)
        toolbar.addAction(self.action_stop_queue)

    def _setup_statusbar(self):
        """Настройка статусной строки."""
        self.status_widget = StatusBarWidget(self.repo, self.task_manager)
        self.statusBar().addPermanentWidget(self.status_widget, 1)

    def _connect_signals(self):
        """Подключение сигналов."""
        # Сигналы TaskManager
        self.task_manager.queue_changed.connect(self._on_queue_changed)
        self.task_manager.queue_state_changed.connect(self._on_queue_state_changed)
        self.task_manager.task_started.connect(self._on_task_started)
        self.task_manager.task_finished.connect(self._on_task_finished)
        self.task_manager.queue_finished.connect(self._on_queue_finished)
        
        # Сигналы панелей
        self.dive_panel.dive_selected.connect(self._on_dive_selected)
        self.dive_panel.add_to_queue_requested.connect(self._on_add_to_queue)
        self.dive_panel.quick_add_to_queue_requested.connect(self._on_quick_add_to_queue)
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

    def _on_add_dive(self):
        """Добавление нового погружения."""
        self.dive_panel.add_dive()

    def _on_add_model(self):
        """Добавление новой модели."""
        self.model_panel.add_model()

    def _on_start_queue(self):
        """Запуск очереди."""
        if self.task_manager.is_paused():
            self.task_manager.resume_queue()
        else:
            if not self.task_manager.start_queue():
                QMessageBox.information(
                    self,
                    "Очередь пуста",
                    "Нет задач для выполнения.\nДобавьте видео в очередь."
                )

    def _on_pause_queue(self):
        """Пауза/возобновление очереди."""
        if self.task_manager.is_paused():
            self.task_manager.resume_queue()
        else:
            self.task_manager.pause_queue()

    def _on_stop_queue(self):
        """Остановка очереди."""
        reply = QMessageBox.question(
            self,
            "Остановить очередь?",
            "Текущая задача будет завершена, остальные останутся в очереди.\nОстановить?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.task_manager.stop_queue()

    def _on_about(self):
        """Показывает информацию о программе."""
        QMessageBox.about(
            self,
            "О программе",
            "<h2>🦑 YOLO Jellyfish Batch Processor</h2>"
            "<p>Пакетная обработка видео для детекции желетелого макрозоопланктона.</p>"
            "<p><b>Версия:</b> 0.1.0</p>"
            "<hr>"
            "<p>Использует YOLOv8 для детекции и трекинга.</p>"
        )

    # ========== Обработчики сигналов ==========

    def _on_queue_changed(self):
        """Обновление при изменении очереди."""
        self.task_table.refresh()
        self.status_widget.update_stats()

    def _on_queue_state_changed(self, is_running: bool, is_paused: bool):
        """Обновление состояния кнопок."""
        self.action_start_queue.setEnabled(not is_running or is_paused)
        self.action_pause_queue.setEnabled(is_running)
        self.action_stop_queue.setEnabled(is_running)
        
        if is_paused:
            self.action_start_queue.setText("▶ Продолжить")
            self.action_pause_queue.setText("⏸ На паузе")
        else:
            self.action_start_queue.setText("▶ Запустить")
            self.action_pause_queue.setText("⏸ Пауза")

    def _on_task_started(self, task_id: int):
        """Задача начала выполняться."""
        self.status_widget.set_current_task(task_id)

    def _on_task_finished(self, task_id: int, success: bool, error_message: str):
        """Задача завершена."""
        if not success and error_message:
            self.statusBar().showMessage(f"Ошибка: {error_message}", 5000)

    def _on_queue_finished(self):
        """Все задачи выполнены."""
        self.status_widget.clear_current_task()
        QMessageBox.information(
            self,
            "Очередь завершена",
            "Все задачи в очереди выполнены."
        )

    def _on_dive_selected(self, dive_id: int):
        """Выбрано погружение."""
        pass  # Можно добавить логику

    def _on_model_selected(self, model_id: int):
        """Выбрана модель."""
        pass  # Можно добавить логику

    def _on_add_to_queue(self, video_id: int, ctd_id: Optional[int]):
        """Запрос на добавление в очередь."""
        # Получаем выбранную модель
        model_id = self.model_panel.get_selected_model_id()
        
        # Проверяем, есть ли модели вообще
        models = self.repo.get_all_models()
        if not models:
            QMessageBox.warning(
                self,
                "Нет моделей",
                "Сначала добавьте модель для детекции.\n\n"
                "Меню: Файл → Добавить модель..."
            )
            return
        
        # Показываем диалог создания задачи
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
        """Быстрое добавление в очередь с параметрами по умолчанию."""
        # Получаем выбранную модель
        model_id = self.model_panel.get_selected_model_id()
        if not model_id:
            # Проверяем, есть ли модели вообще
            models = self.repo.get_all_models()
            if not models:
                QMessageBox.warning(
                    self,
                    "Нет моделей",
                    "Сначала добавьте модель для детекции.\n\n"
                    "Меню: Файл → Добавить модель..."
                )
                return
            QMessageBox.warning(
                self,
                "Модель не выбрана",
                "Выберите модель в панели моделей."
            )
            return
        
        # Получаем параметры по умолчанию
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

    # ========== События окна ==========

    def closeEvent(self, event):
        """Обработка закрытия окна."""
        if self.task_manager.is_running():
            reply = QMessageBox.question(
                self,
                "Выполняются задачи",
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
