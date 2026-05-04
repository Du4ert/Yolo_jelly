"""
Диалог ручной проверки треков детекции.

Позволяет просмотреть треки, удалить неверные, изменить класс.
Использует готовые файлы _detected.mp4, _detections.csv, _tracks.csv.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QDialogButtonBox,
    QLabel,
    QSlider,
    QWidget,
    QComboBox,
    QSplitter,
    QHeaderView,
    QMessageBox,
    QApplication,
    QAbstractItemView,
    QFrame,
    QGraphicsView,
    QGraphicsScene,
)
from PyQt6.QtCore import Qt, QTimer, QUrl, QEvent, pyqtSignal, QPointF, QRectF
from PyQt6.QtGui import QPen, QColor, QBrush, QPalette, QMouseEvent

try:
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
    from PyQt6.QtMultimediaWidgets import (
        QVideoWidget,
        QGraphicsVideoItem,
    )

    HAS_MULTIMEDIA = True
except ImportError:
    HAS_MULTIMEDIA = False

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from verify_detections import (
    load_tracks,
    load_verified,
    load_track_detections,
    load_frame_detections,
    apply_changes,
    save_verified,
    verified_csv_path,
    TrackInfo,
    VerificationResult,
)
from constants import CLASS_NAMES

HIGHLIGHT_COLOR = QColor(255, 255, 0, 200)
HIGHLIGHT_FLASH_MS = 400


class _ClickableGraphicsView(QGraphicsView):
    clicked = pyqtSignal(QPointF)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            self.clicked.emit(scene_pos)
        super().mousePressEvent(event)


class VerifyDialog(QDialog):
    """Диалог ручной проверки треков."""

    def __init__(self, task, parent=None):
        super().__init__(parent)
        self._task = task
        self._tracks: List[TrackInfo] = []
        self._track_detections: Dict[int, pd.DataFrame] = {}
        self._deleted_detections: Dict[int, pd.DataFrame] = {}
        self._confirmed_detections: Dict[int, pd.DataFrame] = {}
        self._delete_crosses: Dict[int, tuple] = {}
        self._confirm_marks: Dict[int, tuple] = {}
        self._selected_track: Optional[TrackInfo] = None
        self._modified = False
        self._seeking = False
        self._player = None
        self._graphics_view = None
        self._scene = None
        self._video_item = None
        self._highlight_rect = None
        self._fitting = False
        self._track_play_active = False
        self._highlight_active = False
        self._sort_column = -1
        self._sort_reverse = False
        self._verified_backup = None
        self._closing = False

        self._resolve_paths()
        self._backup_verified()
        self._load_data()
        self._setup_ui()
        self._setup_video()
        self._setup_flash_timer()

        if self._tracks:
            self._table.selectRow(0)
            self._on_track_selected(0)

    # ── paths ────────────────────────────────────────────────────────

    def _resolve_paths(self):
        self._tracks_csv = ""
        self._detections_csv = ""
        self._video_path = ""
        self._verified_csv = ""

        for out in self._task.outputs:
            if not os.path.exists(out.filepath):
                continue
            if out.output_type.value == "csv":
                self._detections_csv = out.filepath
            elif out.output_type.value == "tracks_csv":
                self._tracks_csv = out.filepath
            elif out.output_type.value == "video":
                self._video_path = out.filepath

        if self._tracks_csv:
            self._verified_csv = verified_csv_path(self._tracks_csv)

    def _backup_verified(self):
        if self._verified_csv and os.path.exists(self._verified_csv):
            fd, self._verified_backup = tempfile.mkstemp(
                suffix=".csv", prefix="verify_backup_"
            )
            os.close(fd)
            shutil.copy2(self._verified_csv, self._verified_backup)

    def reject(self):
        if self._closing:
            return
        self._closing = True
        self._restore_backup()
        super().reject()

    def _restore_backup(self):
        try:
            if self._verified_backup and os.path.exists(self._verified_backup):
                if self._verified_csv:
                    shutil.copy2(self._verified_backup, self._verified_csv)
            elif (
                self._verified_csv
                and os.path.exists(self._verified_csv)
                and self._verified_backup is None
            ):
                os.remove(self._verified_csv)
        except Exception:
            pass
        self._cleanup_backup()

    def _cleanup_backup(self):
        if self._verified_backup and os.path.exists(self._verified_backup):
            try:
                os.unlink(self._verified_backup)
            except Exception:
                pass

    # ── data ─────────────────────────────────────────────────────────

    def _load_data(self):
        if not self._tracks_csv or not os.path.exists(self._tracks_csv):
            return

        self._tracks = load_tracks(self._tracks_csv)
        verified = load_verified(self._verified_csv)

        for t in self._tracks:
            if t.track_id in verified:
                t.deleted = verified[t.track_id]["deleted"]
                t.new_class_id = verified[t.track_id]["new_class_id"]
                t.confirmed = verified[t.track_id].get("confirmed", False)

        if self._detections_csv:
            for t in self._tracks:
                if t.deleted:
                    self._deleted_detections[t.track_id] = load_track_detections(
                        self._detections_csv, t.track_id
                    )
                if t.confirmed:
                    self._confirmed_detections[t.track_id] = load_track_detections(
                        self._detections_csv, t.track_id
                    )

    # ── UI ───────────────────────────────────────────────────────────

    def _setup_ui(self):
        self.setWindowTitle(
            f"Проверка треков — задача #{self._task.id}"
        )
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMinMaxButtonsHint
        )
        self.resize(1400, 800)

        layout = QVBoxLayout(self)

        # ── splitter: левая панель + правая панель ──
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Левая панель: таблица треков + кнопки
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self._task_status_label = QLabel()
        self._task_status_label.setStyleSheet(
            "padding: 4px 8px; font-size: 14px; font-weight: bold;"
        )
        left_layout.addWidget(self._task_status_label)

        self._table = QTableWidget()
        self._setup_table()
        left_layout.addWidget(self._table)

        # Кнопки действий над треками
        btn_layout = QHBoxLayout()
        self._btn_delete = QPushButton("🗑 Удалить трек")
        self._btn_delete.setEnabled(False)
        self._btn_delete.clicked.connect(self._toggle_delete)
        btn_layout.addWidget(self._btn_delete)

        self._btn_confirm = QPushButton("✔ Подтвердить")
        self._btn_confirm.setEnabled(False)
        self._btn_confirm.clicked.connect(self._on_confirm)
        btn_layout.addWidget(self._btn_confirm)

        self._class_combo = QComboBox()
        self._class_combo.setEnabled(False)
        for cid in sorted(CLASS_NAMES.keys()):
            self._class_combo.addItem(CLASS_NAMES[cid], cid)
        self._class_combo.currentIndexChanged.connect(self._on_class_changed)
        btn_layout.addWidget(QLabel("Сменить класс:"))
        btn_layout.addWidget(self._class_combo)

        left_layout.addLayout(btn_layout)
        splitter.addWidget(left)

        # Правая панель: видео + контролы
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        if HAS_MULTIMEDIA and self._video_path:
            self._graphics_view = _ClickableGraphicsView()
            self._graphics_view.setStyleSheet("background: black;")
            self._graphics_view.setFrameShape(QFrame.Shape.NoFrame)
            self._graphics_view.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self._graphics_view.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self._graphics_view.installEventFilter(self)
            self._graphics_view.clicked.connect(self._on_video_clicked)
            right_layout.addWidget(self._graphics_view)
        else:
            self._graphics_view = None
            self._video_item = None
            placeholder = QLabel(
                "Видеоплеер недоступен.\n"
                "Установите PyQt6.QtMultimedia или проверьте путь к файлу."
            )
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: #888;")
            right_layout.addWidget(placeholder)

        # Контролы воспроизведения
        controls = QHBoxLayout()

        self._btn_track_play = QPushButton("▶")
        self._btn_track_play.setFixedWidth(36)
        self._btn_track_play.setToolTip("Воспроизвести трек")
        self._btn_track_play.setEnabled(False)
        self._btn_track_play.setStyleSheet(
            "QPushButton { background-color: #E07000; color: white;"
            " border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background-color: #FF8C00; }"
            "QPushButton:pressed { background-color: #C06000; }"
            "QPushButton:disabled { background-color: #555; color: #888; }"
        )
        self._btn_track_play.clicked.connect(self._on_track_play_click)
        controls.addWidget(self._btn_track_play)

        self._btn_play = QPushButton("▶")
        self._btn_play.setFixedWidth(36)
        self._btn_play.setToolTip("Play / Pause — всё видео")
        self._btn_play.setEnabled(False)
        self._btn_play.clicked.connect(self._toggle_play)
        controls.addWidget(self._btn_play)

        self._time_slider = QSlider(Qt.Orientation.Horizontal)
        self._time_slider.setToolTip("Перемотка видео")
        self._time_slider.setEnabled(False)
        self._time_slider.sliderPressed.connect(self._on_slider_pressed)
        self._time_slider.sliderReleased.connect(self._on_slider_released)
        self._time_slider.sliderMoved.connect(self._on_slider_moved)
        controls.addWidget(self._time_slider)

        self._time_label = QLabel("00:00 / 00:00")
        self._time_label.setMinimumWidth(100)
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        controls.addWidget(self._time_label)

        self._btn_back = QPushButton("◀")
        self._btn_back.setFixedWidth(32)
        self._btn_back.setToolTip("Предыдущий кадр")
        self._btn_back.setEnabled(False)
        self._btn_back.clicked.connect(lambda: self._seek_frame(-1))
        controls.addWidget(self._btn_back)

        self._btn_fwd = QPushButton("▶")
        self._btn_fwd.setFixedWidth(32)
        self._btn_fwd.setToolTip("Следующий кадр")
        self._btn_fwd.setEnabled(False)
        self._btn_fwd.clicked.connect(lambda: self._seek_frame(1))
        controls.addWidget(self._btn_fwd)

        right_layout.addLayout(controls)

        # Информация о выбранном треке
        self._track_info_label = QLabel()
        self._track_info_label.setStyleSheet(
            "padding: 4px 8px; background: #333; color: #ddd; border-radius: 4px;"
        )
        self._track_info_label.setWordWrap(True)
        right_layout.addWidget(self._track_info_label)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        layout.addWidget(splitter)

        # ── нижние кнопки ──
        button_box = QDialogButtonBox()
        self._btn_cancel = button_box.addButton(
            "Отмена", QDialogButtonBox.ButtonRole.RejectRole
        )
        self._btn_apply = button_box.addButton(
            "Применить к данным", QDialogButtonBox.ButtonRole.ApplyRole
        )
        self._btn_save = button_box.addButton(
            "Сохранить и закрыть", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self._btn_cancel.clicked.connect(self.reject)
        self._btn_apply.clicked.connect(self._on_apply)
        self._btn_save.clicked.connect(self.accept)
        layout.addWidget(button_box)

    def _setup_table(self):
        columns = [
            "track_id",
            "Класс",
            "Статус",
            "Кадры",
            "Длит. (с)",
            "Глубина (м)",
            "Детекций",
            "Conf",
        ]
        self._table.setColumnCount(len(columns))
        self._table.setHorizontalHeaderLabels(columns)

        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(True)

        self._table.cellClicked.connect(self._on_track_selected)
        self._table.horizontalHeader().sectionClicked.connect(self._on_sort)

        palette = self._table.palette()
        palette.setColor(
            QPalette.ColorGroup.Inactive, QPalette.ColorRole.Highlight,
            palette.color(QPalette.ColorGroup.Active, QPalette.ColorRole.Highlight),
        )
        palette.setColor(
            QPalette.ColorGroup.Inactive, QPalette.ColorRole.HighlightedText,
            palette.color(QPalette.ColorGroup.Active, QPalette.ColorRole.HighlightedText),
        )
        self._table.setPalette(palette)

        self._table.setRowCount(len(self._tracks))
        for row, track in enumerate(self._tracks):
            self._set_track_row(row, track)

        self._table.resizeColumnsToContents()

        self._update_task_status_label()

    def _set_track_row(self, row: int, track: TrackInfo):
        depth_str = ""
        if track.first_depth_m is not None:
            depth_str = f"{track.first_depth_m:.1f}"
            if track.last_depth_m is not None:
                depth_str += f"–{track.last_depth_m:.1f}"

        frame_str = f"{track.first_frame}–{track.last_frame}"

        values = [
            str(track.track_id),
            track.effective_class_name,
            track.status_text,
            frame_str,
            str(track.duration_s),
            depth_str,
            str(track.detections_count),
            f"{track.avg_confidence:.2f}",
        ]

        for col, val in enumerate(values):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if track.deleted:
                item.setForeground(QBrush(QColor(180, 60, 60)))
                f = item.font()
                f.setStrikeOut(True)
                item.setFont(f)
            elif track.confirmed:
                item.setBackground(QBrush(QColor(60, 180, 60)))
            elif track.new_class_id is not None:
                item.setForeground(QBrush(QColor(200, 150, 0)))
            self._table.setItem(row, col, item)

    # ── video setup ──────────────────────────────────────────────────

    def _setup_video(self):
        if not HAS_MULTIMEDIA or not self._video_path or not self._graphics_view:
            return

        self._scene = QGraphicsScene(self._graphics_view)
        self._graphics_view.setScene(self._scene)

        self._video_item = QGraphicsVideoItem()
        self._scene.addItem(self._video_item)

        self._player = QMediaPlayer()
        self._audio = QAudioOutput()
        self._player.setAudioOutput(self._audio)
        self._audio.setVolume(0.5)
        self._player.setVideoOutput(self._video_item)

        self._player.setSource(QUrl.fromLocalFile(self._video_path))

        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.playbackStateChanged.connect(self._on_state_changed)
        self._player.errorOccurred.connect(self._on_video_error)

        # Прямоугольник подсветки (bbox)
        pen = QPen(HIGHLIGHT_COLOR)
        pen.setCosmetic(True)
        pen.setWidth(2)
        brush = QBrush(QColor(255, 255, 0, 30))
        self._highlight_rect = self._scene.addRect(0, 0, 0, 0, pen, brush)
        self._highlight_rect.setZValue(1)
        self._highlight_rect.setVisible(False)

        self._btn_play.setEnabled(True)
        self._btn_track_play.setEnabled(True)
        self._btn_back.setEnabled(True)
        self._btn_fwd.setEnabled(True)
        self._time_slider.setEnabled(True)

    def _setup_flash_timer(self):
        self._flash_timer = QTimer(self)
        self._flash_timer.timeout.connect(self._flash_tick)
        self._flash_timer.start(HIGHLIGHT_FLASH_MS)

    def _flash_tick(self):
        if self._highlight_active and self._highlight_rect:
            self._highlight_rect.setVisible(
                not self._highlight_rect.isVisible()
            )

    # ── event filter (auto-fit video on resize) ──────────────────────

    def eventFilter(self, obj, event):
        if (
            self._graphics_view
            and obj is self._graphics_view
            and event.type() == QEvent.Type.Resize
        ):
            self._fit_video()
        return super().eventFilter(obj, event)

    def _fit_video(self):
        if self._fitting or not self._video_item:
            return
        ns = self._video_item.nativeSize()
        if not ns.isValid():
            return
        self._fitting = True
        try:
            self._video_item.setSize(ns)
            self._scene.setSceneRect(self._video_item.boundingRect())
            self._graphics_view.fitInView(
                self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
            )
        finally:
            self._fitting = False

    # ── video slots ──────────────────────────────────────────────────

    def _on_duration_changed(self, ms: int):
        self._time_slider.setRange(0, ms)
        self._update_time_label()
        self._fit_video()

    def _on_position_changed(self, ms: int):
        if not self._seeking:
            self._time_slider.blockSignals(True)
            self._time_slider.setValue(ms)
            self._time_slider.blockSignals(False)
        self._update_time_label()
        self._update_highlight(ms)
        self._update_delete_crosses(ms)
        self._update_confirm_marks(ms)
        self._check_track_end(ms)
        self._update_track_button()

    def _check_track_end(self, ms: int):
        if not self._track_play_active or not self._selected_track or not self._player:
            return
        if self._player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            return
        end_ms = int(self._selected_track.last_timestamp_s * 1000)
        if ms >= end_ms:
            self._track_play_active = False
            self._player.pause()

    def _on_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self._btn_play.setText("⏸")
        else:
            self._btn_play.setText("▶")
        self._update_track_button()

    def _on_video_error(self, error, error_string):
        QMessageBox.warning(
            self, "Ошибка видео", f"Ошибка воспроизведения:\n{error_string}"
        )

    def _on_slider_pressed(self):
        self._seeking = True

    def _on_slider_released(self):
        self._seeking = False
        self._player.setPosition(self._time_slider.value())

    def _on_slider_moved(self, value: int):
        self._update_time_label()

    # ── highlight ────────────────────────────────────────────────────

    def _update_highlight(self, position_ms: int):
        if not self._highlight_rect or not self._selected_track or not self._video_item:
            self._highlight_active = False
            if self._highlight_rect:
                self._highlight_rect.setVisible(False)
            return

        native = self._video_item.nativeSize()
        if not native.isValid():
            self._highlight_active = False
            self._highlight_rect.setVisible(False)
            return

        scene_w = native.width()
        scene_h = native.height()

        current_sec = position_ms / 1000.0

        t = self._selected_track
        if current_sec < t.first_timestamp_s or current_sec > t.last_timestamp_s + 0.5:
            self._highlight_active = False
            self._highlight_rect.setVisible(False)
            return

        search_sec = max(t.first_timestamp_s, min(t.last_timestamp_s, current_sec))

        df = self._track_detections.get(self._selected_track.track_id)
        if df is None or df.empty:
            df = self._load_detections_for_track(self._selected_track.track_id)

        if df is None or df.empty:
            self._highlight_active = False
            self._highlight_rect.setVisible(False)
            return

        idx = (df["timestamp_s"] - search_sec).abs().idxmin()
        row = df.loc[idx]
        if search_sec == current_sec and abs(row["timestamp_s"] - current_sec) > 0.15:
            self._highlight_active = False
            self._highlight_rect.setVisible(False)
            return

        xc = float(row["x_center"])
        yc = float(row["y_center"])
        bw = float(row["width"])
        bh = float(row["height"])

        x = (xc - bw / 2) * scene_w
        y = (yc - bh / 2) * scene_h
        w = bw * scene_w
        h = bh * scene_h

        self._highlight_active = True
        self._highlight_rect.setRect(x, y, w, h)

    def _update_delete_crosses(self, position_ms: int):
        if not self._video_item:
            return

        native = self._video_item.nativeSize()
        if not native.isValid():
            return

        scene_w = native.width()
        scene_h = native.height()
        current_sec = position_ms / 1000.0

        deleted_ids = {t.track_id for t in self._tracks if t.deleted}

        for tid in list(self._delete_crosses.keys()):
            if tid not in deleted_ids:
                line1, line2 = self._delete_crosses.pop(tid)
                self._scene.removeItem(line1)
                self._scene.removeItem(line2)

        for tid in deleted_ids:
            df = self._deleted_detections.get(tid)
            if df is None or df.empty:
                continue

            idx = (df["timestamp_s"] - current_sec).abs().idxmin()
            row = df.loc[idx]
            if abs(row["timestamp_s"] - current_sec) > 0.15:
                if tid in self._delete_crosses:
                    line1, line2 = self._delete_crosses[tid]
                    line1.setVisible(False)
                    line2.setVisible(False)
                continue

            xc = float(row["x_center"])
            yc = float(row["y_center"])
            bw = float(row["width"])
            bh = float(row["height"])

            bx = (xc - bw / 2) * scene_w
            by = (yc - bh / 2) * scene_h
            bbox_w = bw * scene_w
            bbox_h = bh * scene_h

            if tid not in self._delete_crosses:
                pen = QPen(QColor(255, 60, 60, 200))
                pen.setCosmetic(True)
                pen.setWidth(2)
                line1 = self._scene.addLine(0, 0, 0, 0, pen)
                line2 = self._scene.addLine(0, 0, 0, 0, pen)
                line1.setZValue(2)
                line2.setZValue(2)
                self._delete_crosses[tid] = (line1, line2)
            else:
                line1, line2 = self._delete_crosses[tid]

            line1.setLine(bx, by, bx + bbox_w, by + bbox_h)
            line2.setLine(bx + bbox_w, by, bx, by + bbox_h)
            line1.setVisible(True)
            line2.setVisible(True)

    def _update_confirm_marks(self, position_ms: int):
        if not self._video_item:
            return

        native = self._video_item.nativeSize()
        if not native.isValid():
            return

        scene_w = native.width()
        scene_h = native.height()
        current_sec = position_ms / 1000.0

        confirmed_ids = {t.track_id for t in self._tracks if t.confirmed}

        for tid in list(self._confirm_marks.keys()):
            if tid not in confirmed_ids:
                line1, line2 = self._confirm_marks.pop(tid)
                self._scene.removeItem(line1)
                self._scene.removeItem(line2)

        for tid in confirmed_ids:
            df = self._confirmed_detections.get(tid)
            if df is None or df.empty:
                continue

            idx = (df["timestamp_s"] - current_sec).abs().idxmin()
            row = df.loc[idx]
            if abs(row["timestamp_s"] - current_sec) > 0.15:
                if tid in self._confirm_marks:
                    line1, line2 = self._confirm_marks[tid]
                    line1.setVisible(False)
                    line2.setVisible(False)
                continue

            xc = float(row["x_center"])
            yc = float(row["y_center"])
            bw = float(row["width"])
            bh = float(row["height"])

            bx = (xc - bw / 2) * scene_w
            by = (yc - bh / 2) * scene_h
            bbox_w = bw * scene_w
            bbox_h = bh * scene_h

            if tid not in self._confirm_marks:
                pen = QPen(QColor(60, 255, 60, 200))
                pen.setCosmetic(True)
                pen.setWidth(2)
                line1 = self._scene.addLine(0, 0, 0, 0, pen)
                line2 = self._scene.addLine(0, 0, 0, 0, pen)
                line1.setZValue(2)
                line2.setZValue(2)
                self._confirm_marks[tid] = (line1, line2)
            else:
                line1, line2 = self._confirm_marks[tid]

            # checkmark ✓ shape
            cx1 = bx + bbox_w * 0.2
            cy1 = by + bbox_h * 0.5
            cx2 = bx + bbox_w * 0.45
            cy2 = by + bbox_h * 0.8
            cx3 = bx + bbox_w * 0.8
            cy3 = by + bbox_h * 0.25

            line1.setLine(cx1, cy1, cx2, cy2)
            line2.setLine(cx2, cy2, cx3, cy3)
            line1.setVisible(True)
            line2.setVisible(True)

    def _load_detections_for_track(self, track_id: int):
        if not self._detections_csv:
            return None
        df = load_track_detections(self._detections_csv, track_id)
        self._track_detections[track_id] = df
        return df

    # ── play controls ────────────────────────────────────────────────

    def _toggle_play(self):
        if not HAS_MULTIMEDIA or not self._player:
            return
        self._track_play_active = False
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _seek_frame(self, direction: int):
        if not HAS_MULTIMEDIA or not self._player:
            return

        if self._selected_track:
            t = self._selected_track
            fps = t.frame_span / t.duration_s if t.duration_s > 0 else 30.0
        else:
            fps = 30.0
        frame_ms = int(1000.0 / fps)

        new_pos = self._player.position() + direction * frame_ms
        new_pos = max(0, min(self._player.duration(), new_pos))
        self._player.setPosition(new_pos)

    def _restart_track(self):
        if not self._selected_track:
            return
        first_ms = int(self._selected_track.first_timestamp_s * 1000)
        self._seek_to(first_ms)

    def _on_track_play_click(self):
        if not self._selected_track or not self._player:
            return

        t = self._selected_track
        end_ms = int(t.last_timestamp_s * 1000)
        pos = self._player.position()

        if self._track_play_active:
            self._track_play_active = False
            self._player.pause()
        elif self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._track_play_active = True
            self._player.pause()
        elif pos >= end_ms:
            self._track_play_active = False
            self._btn_track_play.setText("▶")
            self._seek_to(int(t.first_timestamp_s * 1000))
        else:
            self._track_play_active = True
            self._player.play()

    def _update_track_button(self):
        if not self._btn_track_play:
            return

        t = self._selected_track
        if not t:
            self._btn_track_play.setText("▶")
            return

        end_ms = int(t.last_timestamp_s * 1000)

        if self._track_play_active:
            self._btn_track_play.setText("⏸")
        elif (
            self._player
            and self._player.playbackState() == QMediaPlayer.PlaybackState.PausedState
            and self._player.position() >= end_ms
        ):
            self._btn_track_play.setText("↻")
        else:
            self._btn_track_play.setText("▶")

    def _seek_to(self, ms: int):
        if not HAS_MULTIMEDIA or not self._player:
            return
        ms = max(0, min(ms, self._player.duration()))
        self._player.setPosition(ms)
        self._player.pause()

    def _update_time_label(self):
        pos = self._time_slider.value()
        dur = self._time_slider.maximum()
        self._time_label.setText(
            f"{_format_ms(pos)} / {_format_ms(dur)}"
        )

    # ── track actions ────────────────────────────────────────────────

    def _on_track_selected(self, row: int, seek_to_first: bool = True):
        if row < 0 or row >= len(self._tracks):
            return

        self._selected_track = self._tracks[row]

        self._btn_delete.setEnabled(True)
        self._btn_delete.setText(
            "↩ Восстановить трек"
            if self._selected_track.deleted
            else "🗑 Удалить трек"
        )
        self._btn_confirm.setEnabled(not self._selected_track.deleted)
        self._btn_confirm.setText(
            "✘ Отменить подтверждение"
            if self._selected_track.confirmed
            else "✔ Подтвердить"
        )
        self._class_combo.setEnabled(True)
        self._class_combo.blockSignals(True)
        cid = self._selected_track.effective_class_id
        idx = self._class_combo.findData(cid)
        if idx >= 0:
            self._class_combo.setCurrentIndex(idx)
        self._class_combo.blockSignals(False)

        # Информация о треке
        t = self._selected_track
        self._track_info_label.setText(
            f"Трек #{t.track_id}  |  {t.effective_class_name}  |  "
            f"Кадры {t.first_frame}–{t.last_frame} ({t.frame_span})  |  "
            f"{t.duration_s} с  |  "
            f"{t.detections_count} детекций  |  "
            f"Conf {t.avg_confidence:.2f}"
        )

        if seek_to_first:
            self._track_play_active = False
            first_ms = int(t.first_timestamp_s * 1000)
            self._seek_to(first_ms)
        self._update_track_button()

    def _on_video_clicked(self, scene_pos: QPointF):
        if not self._player or not self._video_item or not self._detections_csv:
            return

        native = self._video_item.nativeSize()
        if not native.isValid():
            return

        scene_w = native.width()
        scene_h = native.height()
        current_sec = self._player.position() / 1000.0

        frame_df = load_frame_detections(
            self._detections_csv, current_sec, tolerance_s=0.15
        )
        if frame_df.empty:
            return

        best_track_id = None
        best_dist = float("inf")

        for _, row in frame_df.iterrows():
            xc = float(row["x_center"])
            yc = float(row["y_center"])
            bw = float(row["width"])
            bh = float(row["height"])

            bx = (xc - bw / 2) * scene_w
            by = (yc - bh / 2) * scene_h
            bbox_w = bw * scene_w
            bbox_h = bh * scene_h
            bbox = QRectF(bx, by, bbox_w, bbox_h)

            if bbox.contains(scene_pos):
                tid = int(row["track_id"])
                for row_idx, t in enumerate(self._tracks):
                    if t.track_id == tid:
                        self._table.selectRow(row_idx)
                        self._on_track_selected(row_idx, seek_to_first=False)
                        self._update_highlight(self._player.position())
                        self._update_delete_crosses(self._player.position())
                        self._update_confirm_marks(self._player.position())
                        return
            else:
                cx = bx + bbox_w / 2
                cy = by + bbox_h / 2
                dx = scene_pos.x() - cx
                dy = scene_pos.y() - cy
                dist = dx * dx + dy * dy
                if dist < best_dist:
                    best_dist = dist
                    best_track_id = int(row["track_id"])

        if best_track_id is not None:
            for row_idx, t in enumerate(self._tracks):
                if t.track_id == best_track_id:
                    self._table.selectRow(row_idx)
                    self._on_track_selected(row_idx, seek_to_first=False)
                    self._update_highlight(self._player.position())
                    self._update_delete_crosses(self._player.position())
                    self._update_confirm_marks(self._player.position())
                    return

    def _toggle_delete(self):
        if not self._selected_track:
            return

        self._selected_track.deleted = not self._selected_track.deleted
        self._modified = True

        self._btn_delete.setText(
            "↩ Восстановить трек"
            if self._selected_track.deleted
            else "🗑 Удалить трек"
        )
        self._btn_confirm.setEnabled(not self._selected_track.deleted)
        if self._selected_track.deleted:
            self._selected_track.confirmed = False

        tid = self._selected_track.track_id
        if self._selected_track.deleted:
            self._confirmed_detections.pop(tid, None)
            mark = self._confirm_marks.pop(tid, None)
            if mark:
                self._scene.removeItem(mark[0])
                self._scene.removeItem(mark[1])
            if tid not in self._deleted_detections:
                self._deleted_detections[tid] = self._load_detections_for_track(tid)
        else:
            self._deleted_detections.pop(tid, None)
            cross = self._delete_crosses.pop(tid, None)
            if cross:
                self._scene.removeItem(cross[0])
                self._scene.removeItem(cross[1])

        row = self._table.currentRow()
        self._set_track_row(row, self._selected_track)
        self._table.resizeColumnsToContents()
        self._update_task_status_label()

        if self._tracks_csv:
            try:
                save_verified(self._tracks, self._tracks_csv)
            except Exception:
                pass

        if self._player:
            self._update_delete_crosses(self._player.position())

        if self._selected_track.deleted:
            self._select_next_unconfirmed()

    def _on_class_changed(self, index: int):
        if not self._selected_track:
            return

        new_cid = self._class_combo.currentData()
        if new_cid == self._selected_track.class_id:
            self._selected_track.new_class_id = None
        else:
            self._selected_track.new_class_id = new_cid
        self._modified = True

        row = self._table.currentRow()
        self._set_track_row(row, self._selected_track)
        self._table.resizeColumnsToContents()
        self._update_task_status_label()

        if self._tracks_csv:
            try:
                save_verified(self._tracks, self._tracks_csv)
            except Exception:
                pass

    # ── confirm / apply / sort ────────────────────────────────────────

    def _on_confirm(self):
        if not self._selected_track or self._selected_track.deleted:
            return

        self._selected_track.confirmed = not self._selected_track.confirmed
        self._modified = True

        tid = self._selected_track.track_id
        if self._selected_track.confirmed:
            if tid not in self._confirmed_detections:
                self._confirmed_detections[tid] = self._load_detections_for_track(tid)
        else:
            self._confirmed_detections.pop(tid, None)
            mark = self._confirm_marks.pop(tid, None)
            if mark:
                self._scene.removeItem(mark[0])
                self._scene.removeItem(mark[1])

        self._btn_confirm.setText(
            "✘ Отменить подтверждение"
            if self._selected_track.confirmed
            else "✔ Подтвердить"
        )

        row = self._table.currentRow()
        self._set_track_row(row, self._selected_track)
        self._table.resizeColumnsToContents()
        self._update_task_status_label()

        if self._tracks_csv:
            try:
                save_verified(self._tracks, self._tracks_csv)
            except Exception:
                pass

        if self._player:
            self._update_confirm_marks(self._player.position())

        if self._selected_track.confirmed:
            self._select_next_unconfirmed()

    def _on_apply(self):
        modified_tracks = [t for t in self._tracks if t.is_modified]
        if not modified_tracks:
            QMessageBox.information(self, "Применить", "Нет изменений для применения.")
            return

        destructive = [t for t in self._tracks if t.deleted or t.new_class_id is not None]
        if destructive:
            reply = QMessageBox.question(
                self,
                "Применить изменения",
                "Изменения будут применены к файлам _tracks.csv и _detections.csv.\n"
                "Удалённые треки будут безвозвратно удалены из этих файлов.\n\n"
                "Продолжить?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        try:
            result = apply_changes(
                self._tracks,
                self._tracks_csv,
                self._detections_csv,
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Ошибка сохранения",
                f"Не удалось применить изменения:\n{e}",
            )
            return

        self._load_data()
        self._rebuild_table()
        self._modified = False
        self._cleanup_backup()

        QMessageBox.information(
            self,
            "Применено",
            f"Изменения применены.\n"
            f"Всего треков: {result.total_tracks}\n"
            f"Удалено: {result.deleted_count}\n"
            f"Изменён класс: {result.changed_class_count}",
        )

        if self._tracks:
            self._table.selectRow(0)
            self._on_track_selected(0)

    def _on_sort(self, column: int):
        if column == self._sort_column:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = column
            self._sort_reverse = False

        key_map = {
            0: lambda t: t.track_id,
            1: lambda t: t.effective_class_name.lower(),
            2: lambda t: t.status_text,
            3: lambda t: t.first_frame,
            4: lambda t: t.duration_s,
            5: lambda t: t.first_depth_m or 0.0,
            6: lambda t: t.detections_count,
            7: lambda t: t.avg_confidence,
        }
        key = key_map.get(column, lambda t: t.track_id)

        self._tracks.sort(key=key, reverse=self._sort_reverse)

        self._table.horizontalHeader().setSortIndicator(
            self._sort_column,
            Qt.SortOrder.DescendingOrder if self._sort_reverse else Qt.SortOrder.AscendingOrder,
        )

        self._rebuild_table()

        if self._tracks:
            self._table.selectRow(0)
            self._on_track_selected(0)

    def _rebuild_table(self):
        selected_track_id = self._selected_track.track_id if self._selected_track else None

        self._table.setRowCount(len(self._tracks))
        for row, track in enumerate(self._tracks):
            self._set_track_row(row, track)

        self._table.resizeColumnsToContents()

        if selected_track_id is not None:
            for row, track in enumerate(self._tracks):
                if track.track_id == selected_track_id:
                    self._table.selectRow(row)
                    self._on_track_selected(row)
                    break

        self._update_task_status_label()

    def _select_next_unconfirmed(self):
        current_row = self._table.currentRow()
        n = len(self._tracks)

        for offset in range(1, n):
            idx = (current_row + offset) % n
            t = self._tracks[idx]
            if not t.confirmed and not t.deleted:
                self._table.selectRow(idx)
                self._on_track_selected(idx)
                return

    def _compute_task_status(self) -> str:
        if not self._tracks:
            return ""

        resolved = sum(1 for t in self._tracks if t.confirmed or t.deleted)
        if resolved == 0:
            return ""
        if resolved == len(self._tracks):
            return "Подтверждено"
        return "В работе"

    def _update_task_status_label(self):
        status = self._compute_task_status()
        if status == "Подтверждено":
            self._task_status_label.setText("✓ Подтверждено")
            self._task_status_label.setStyleSheet(
                "padding: 4px 8px; font-size: 14px; font-weight: bold;"
                " background: #2D5A2D; color: #A0FFA0; border-radius: 4px;"
            )
        elif status == "В работе":
            self._task_status_label.setText("⏳ В работе")
            self._task_status_label.setStyleSheet(
                "padding: 4px 8px; font-size: 14px; font-weight: bold;"
                " background: #5A5A20; color: #FFE080; border-radius: 4px;"
            )
        else:
            self._task_status_label.setText("")
            self._task_status_label.setStyleSheet("")

        self.setWindowTitle(
            f"Проверка треков — задача #{self._task.id}"
            + (f"  [{status}]" if status else "")
        )

    # ── close ─────────────────────────────────────────────────────────

    def accept(self):
        if self._closing:
            return
        if self._modified and self._tracks_csv:
            try:
                save_verified(self._tracks, self._tracks_csv)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Ошибка сохранения",
                    f"Не удалось сохранить состояние проверки:\n{e}",
                )
                return
        self._closing = True
        self._cleanup_backup()
        super().accept()


def _format_ms(ms: int) -> str:
    total_s = ms // 1000
    minutes = total_s // 60
    seconds = total_s % 60
    return f"{minutes:02d}:{seconds:02d}"
