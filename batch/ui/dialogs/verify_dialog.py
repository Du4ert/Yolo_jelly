"""
Диалог ручной проверки треков детекции.

Позволяет просмотреть треки, удалить неверные, изменить класс.
Использует готовые файлы _detected.mp4, _detections.csv, _tracks.csv.
"""

import os
import sys
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
from PyQt6.QtCore import Qt, QTimer, QUrl, QEvent
from PyQt6.QtGui import QPen, QColor, QBrush

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
    apply_changes,
    verified_csv_path,
    TrackInfo,
    VerificationResult,
)
from constants import CLASS_NAMES

HIGHLIGHT_COLOR = QColor(255, 255, 0, 200)
HIGHLIGHT_FLASH_MS = 400


class VerifyDialog(QDialog):
    """Диалог ручной проверки треков."""

    def __init__(self, task, parent=None):
        super().__init__(parent)
        self._task = task
        self._tracks: List[TrackInfo] = []
        self._track_detections: Dict[int, pd.DataFrame] = {}
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

        self._resolve_paths()
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

        self._table = QTableWidget()
        self._setup_table()
        left_layout.addWidget(self._table)

        # Кнопки действий над треками
        btn_layout = QHBoxLayout()
        self._btn_delete = QPushButton("🗑 Удалить трек")
        self._btn_delete.setEnabled(False)
        self._btn_delete.clicked.connect(self._toggle_delete)
        btn_layout.addWidget(self._btn_delete)

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
            self._graphics_view = QGraphicsView()
            self._graphics_view.setStyleSheet("background: black;")
            self._graphics_view.setFrameShape(QFrame.Shape.NoFrame)
            self._graphics_view.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self._graphics_view.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self._graphics_view.installEventFilter(self)
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
        self._btn_play.setEnabled(False)
        self._btn_play.clicked.connect(self._toggle_play)
        controls.addWidget(self._btn_play)

        self._time_slider = QSlider(Qt.Orientation.Horizontal)
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
        self._btn_save = button_box.addButton(
            "Сохранить и закрыть", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self._btn_cancel = button_box.addButton(
            "Отмена", QDialogButtonBox.ButtonRole.RejectRole
        )
        self._btn_save.clicked.connect(self._save_and_close)
        self._btn_cancel.clicked.connect(self.reject)
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

        self._table.setRowCount(len(self._tracks))
        for row, track in enumerate(self._tracks):
            self._set_track_row(row, track)

        self._table.resizeColumnsToContents()

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
        if not HAS_MULTIMEDIA or not self._player or not self._selected_track:
            return

        t = self._selected_track
        fps = t.frame_span / t.duration_s if t.duration_s > 0 else 30.0
        frame_ms = int(1000.0 / fps)

        new_pos = self._player.position() + direction * frame_ms
        first_ms = int(t.first_timestamp_s * 1000)
        last_ms = int(t.last_timestamp_s * 1000)
        new_pos = max(first_ms, min(last_ms, new_pos))
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

    def _on_track_selected(self, row: int):
        if row < 0 or row >= len(self._tracks):
            return

        self._selected_track = self._tracks[row]

        # Обновить кнопки
        self._btn_delete.setEnabled(True)
        self._btn_delete.setText(
            "↩ Восстановить трек"
            if self._selected_track.deleted
            else "🗑 Удалить трек"
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

        # Перемотка видео на начало трека
        self._track_play_active = False
        first_ms = int(t.first_timestamp_s * 1000)
        self._seek_to(first_ms)
        self._update_track_button()

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

        row = self._table.currentRow()
        self._set_track_row(row, self._selected_track)
        self._table.resizeColumnsToContents()

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

    # ── save ─────────────────────────────────────────────────────────

    def _save_and_close(self):
        if not self._modified:
            self.accept()
            return

        modified_tracks = [t for t in self._tracks if t.is_modified]
        if not modified_tracks:
            self.accept()
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
                f"Не удалось сохранить изменения:\n{e}",
            )
            return

        QMessageBox.information(
            self,
            "Сохранено",
            f"Изменения применены.\n"
            f"Всего треков: {result.total_tracks}\n"
            f"Удалено: {result.deleted_count}\n"
            f"Изменён класс: {result.changed_class_count}\n\n"
            f"Файл верификации: {result.verified_csv_path}",
        )
        self.accept()


def _format_ms(ms: int) -> str:
    total_s = ms // 1000
    minutes = total_s // 60
    seconds = total_s % 60
    return f"{minutes:02d}:{seconds:02d}"
