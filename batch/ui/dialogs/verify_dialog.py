"""
Диалог ручной проверки треков детекции.

Позволяет просмотреть треки, удалить неверные, изменить класс.
Использует готовые файлы _detected.mp4, _detections.csv, _tracks.csv.
"""

import os
import sys
import shutil
import tempfile
import math
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
from PyQt6.QtCore import Qt, QTimer, QUrl, QEvent, pyqtSignal, QPointF, QRectF, QItemSelectionModel
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
    load_frame_detections,
    apply_changes,
    save_verified,
    verified_csv_path,
    TrackInfo,
    VerificationResult,
    load_untracked_fragments,
    load_track_proximity,
    load_verification_track_detections,
    refresh_track_statistics,
    detection_file_signature,
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
        self._visible_tracks: List[TrackInfo] = []
        self._expanded_tracks = set()
        self._group_children = {}
        self._group_depth = {}
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
        self._fit_pending = False
        self._last_fit_state = None
        self._track_play_active = False
        self._highlight_active = False
        self._sort_column = 8
        self._sort_reverse = False
        self._verified_backup = None
        self._baseline_state: Dict[int, tuple] = {}
        self._closing = False
        self._fragment_data = pd.DataFrame()
        self._fragment_by_row = {}
        self._detached_by_row = {}
        self._proximity_scores = {}
        self._proximity_cache_key = None
        self._proximity_reference_id = None
        self._proximity_timer = QTimer(self)
        self._proximity_timer.setSingleShot(True)
        self._proximity_timer.setInterval(150)
        self._proximity_timer.timeout.connect(self._refresh_proximity)

        self._resolve_paths()
        self._backup_verified()
        self._load_data()
        self._baseline_state = self._snapshot_state()
        self._setup_ui()
        self._setup_video()
        self._setup_flash_timer()

        if self._visible_tracks:
            self._table.selectRow(0)
            self._on_track_selected(0)
        else:
            self._clear_track_selection()

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
        self._restore_backup(cleanup=True)
        super().reject()

    def _restore_backup(self, cleanup: bool):
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

        if cleanup:
            self._cleanup_backup()

    def _cleanup_backup(self):
        if self._verified_backup and os.path.exists(self._verified_backup):
            try:
                os.unlink(self._verified_backup)
            except Exception:
                pass
        self._verified_backup = None

    def _update_verified_backup(self):
        self._cleanup_backup()
        if self._verified_csv and os.path.exists(self._verified_csv):
            fd, self._verified_backup = tempfile.mkstemp(
                suffix=".csv", prefix="verify_backup_"
            )
            os.close(fd)
            shutil.copy2(self._verified_csv, self._verified_backup)

    # ── data ─────────────────────────────────────────────────────────

    def _load_data(self):
        self._tracks = []
        self._track_detections.clear()
        self._deleted_detections.clear()
        self._confirmed_detections.clear()
        self._selected_track = None
        self._fragment_data = pd.DataFrame()
        self._fragment_by_row = {}
        self._detached_by_row = {}
        self._proximity_scores = {}
        self._proximity_cache_key = None
        self._proximity_reference_id = None

        if not self._tracks_csv or not os.path.exists(self._tracks_csv):
            return

        self._tracks = load_tracks(self._tracks_csv)
        verified = load_verified(self._verified_csv)
        loaded_ids = {t.track_id for t in self._tracks}
        pending_detaches = {tid: state["track_snapshot"] for tid, state in verified.items()
                            if state.get("track_snapshot", {}) and state["track_snapshot"].get("detached_rows")}
        signature = detection_file_signature(self._detections_csv) if pending_detaches else None
        valid_detaches = {tid for tid, snapshot in pending_detaches.items()
                          if snapshot.get("detached_signature") == signature
                          and set(snapshot.get("detached_owner_ids") or (
                              snapshot.get("detached_from_track_id"),)).issubset(loaded_ids)}
        planned_ids = valid_detaches | {tid for tid, state in verified.items()
                                       if tid > 0 and state.get("fragment_assigned")}
        for tid, state in verified.items():
            snapshot = state.get("track_snapshot")
            if (snapshot and tid not in loaded_ids and (
                    tid in valid_detaches or snapshot.get("is_archive")
                    and not snapshot.get("detached_rows")
                    and state.get("merged_into_track_id") in loaded_ids | planned_ids)):
                snapshot = dict(snapshot)
                snapshot["fragment_rows"] = tuple(snapshot.get("fragment_rows", ()))
                snapshot["merged_member_ids"] = tuple(snapshot.get("merged_member_ids", ()))
                snapshot["detached_rows"] = tuple(snapshot.get("detached_rows", ()))
                snapshot["detached_owner_ids"] = tuple(snapshot.get("detached_owner_ids", ()))
                self._tracks.append(TrackInfo(**snapshot))
                loaded_ids.add(tid)
        if set(pending_detaches) - valid_detaches:
            QMessageBox.warning(self, "Отсоединение частей",
                                "CSV изменился: сохранённые отсоединения, не соответствующие "
                                "исходному файлу, не восстановлены. Повторите их по текущим данным.")

        preserved_groups = {state["fragment_rows"]: state["fragment_signature"]
                            for state in verified.values() if state.get("fragment_rows")
                            and not state.get("fragment_assigned")}
        fragments, self._fragment_data = load_untracked_fragments(self._detections_csv, preserved_groups)
        saved_fragments = {
            state["fragment_rows"]: (tid, state)
            for tid, state in verified.items() if state.get("fragment_rows") and tid not in pending_detaches
            and not (state.get("track_snapshot") or {}).get("is_archive")
        }
        restored = set()
        occupied = {t.track_id for t in self._tracks}
        for fragment in fragments:
            saved = saved_fragments.get(fragment.fragment_rows)
            if saved and saved[1]["fragment_signature"] == fragment.fragment_signature:
                tid, state = saved
                if tid not in occupied:
                    fragment.track_id = tid
                    fragment.fragment_assigned = state["fragment_assigned"]
                    restored.add(fragment.fragment_rows)
            if fragment.track_id in occupied:
                fragment.track_id = min(-1, min(occupied, default=0) - 1)
            occupied.add(fragment.track_id)
        self._tracks.extend(fragments)
        self._index_assigned_rows()
        if set(saved_fragments) - restored:
            QMessageBox.warning(
                self, "Назначения фрагментов",
                "Исходные данные изменились. Часть сохранённых назначений фрагментов "
                "не восстановлена. Проверьте фрагменты без ID и назначьте их заново.",
            )

        for t in self._tracks:
            if t.track_id in verified and (
                not t.fragment_rows or t.fragment_rows in restored or t.detached_rows
            ):
                t.deleted = verified[t.track_id]["deleted"]
                t.new_class_id = verified[t.track_id]["new_class_id"]
                t.confirmed = verified[t.track_id].get("confirmed", False)
                t.merged_into_track_id = verified[t.track_id].get("merged_into_track_id")
                snapshot = verified[t.track_id].get("track_snapshot")
                if snapshot:
                    t.merged_member_ids = tuple(snapshot.get("merged_member_ids", ()))
                    t.group_parent_id = snapshot.get("group_parent_id")

        self._refresh_merged_track_stats()
        if valid_detaches:
            affected = set(valid_detaches)
            for track in self._tracks:
                if track.detached_rows:
                    affected.update((track.detached_from_track_id, track.detached_parent_id,
                                     track.merged_into_track_id, track.group_parent_id))
                    affected.update(track.detached_owner_ids)
            self._refresh_statistics(affected)

        if self._detections_csv:
            for t in self._tracks:
                if t.deleted:
                    self._deleted_detections[t.track_id] = self._load_detections_for_track(t.track_id)
                if t.confirmed:
                    self._confirmed_detections[t.track_id] = self._load_detections_for_track(t.track_id)

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

        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Показывать:"))
        self._track_filter = QComboBox()
        self._track_filter.addItem("Только треки с ID", "tracks")
        self._track_filter.addItem("Треки и фрагменты без ID", "all")
        self._track_filter.addItem("Только фрагменты без ID", "fragments")
        self._track_filter.setToolTip(
            "Неназначенные фрагменты не препятствуют завершению проверки треков. "
            "По умолчанию список упорядочен по времени появления."
        )
        filter_layout.addWidget(self._track_filter)
        left_layout.addLayout(filter_layout)

        self._table = QTableWidget()
        self._setup_table()
        left_layout.addWidget(self._table)
        self._track_filter.currentIndexChanged.connect(self._on_filter_changed)

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

        self._btn_merge = QPushButton("Объединить")
        self._btn_merge.setEnabled(False)
        self._btn_merge.clicked.connect(self._on_merge_tracks)
        btn_layout.addWidget(self._btn_merge)

        self._btn_detach = QPushButton("Отсоединить")
        self._btn_detach.setEnabled(False)
        self._btn_detach.setToolTip("Выберите один вложенный трек или фрагмент. "
                                    "Подгруппа отсоединяется вместе со своим составом.")
        self._btn_detach.clicked.connect(self._on_detach_track)
        btn_layout.addWidget(self._btn_detach)

        self._class_combo = QComboBox()
        self._class_combo.setEnabled(False)
        for cid in sorted(CLASS_NAMES.keys()):
            self._class_combo.addItem(CLASS_NAMES[cid], cid)
        self._class_combo.currentIndexChanged.connect(self._on_class_changed)
        btn_layout.addWidget(QLabel("Сменить класс:"))
        btn_layout.addWidget(self._class_combo)

        left_layout.addLayout(btn_layout)
        fragment_buttons = QHBoxLayout()
        self._btn_create_track = QPushButton("Создать трек из фрагмента")
        self._btn_attach_fragment = QPushButton("Присоединить выбранные фрагменты")
        self._btn_create_track.setEnabled(False)
        self._btn_attach_fragment.setEnabled(False)
        self._btn_create_track.clicked.connect(self._on_create_fragment_track)
        self._btn_attach_fragment.clicked.connect(self._on_attach_fragment)
        self._btn_merge.setToolTip(
            "Выделите через Ctrl трек с ID и один или несколько фрагментов без ID "
            "одного класса. Сохранится ID выбранного трека."
        )
        self._btn_attach_fragment.setToolTip(self._btn_merge.toolTip())
        fragment_buttons.addWidget(self._btn_create_track)
        fragment_buttons.addWidget(self._btn_attach_fragment)
        left_layout.addLayout(fragment_buttons)
        hint = QLabel("Для присоединения включите общий список и выделите через Ctrl "
                      "фрагмент и трек, затем нажмите «Объединить». Колонка «Близость» "
                      "учитывает время, положение и перекрытие рамок; меньше — ближе.")
        hint.setWordWrap(True)
        left_layout.addWidget(hint)
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
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        self._btn_close = QPushButton("Закрыть")
        self._btn_cancel = QPushButton("Отменить")
        self._btn_save = QPushButton("Сохранить")
        self._btn_apply = QPushButton("Применить к данным")
        button_layout.addWidget(self._btn_close)
        button_layout.addWidget(self._btn_cancel)
        button_layout.addWidget(self._btn_save)
        button_layout.addWidget(self._btn_apply)
        self._btn_close.clicked.connect(self.reject)
        self._btn_cancel.clicked.connect(self._on_cancel_changes)
        self._btn_save.clicked.connect(self._on_save)
        self._btn_apply.clicked.connect(self._on_apply)
        self._set_modified(False)
        layout.addLayout(button_layout)

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
            "Начало (с)",
            "Близость",
        ]
        self._table.setColumnCount(len(columns))
        self._table.setHorizontalHeaderLabels(columns)
        self._table.horizontalHeaderItem(9).setToolTip(
            "К выбранному фрагменту (или треку): Δt + расстояние центров в диагоналях "
            "средней рамки + (1 − IoU). Меньше — ближе. Подробности — в подсказке ячейки."
        )

        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(True)

        self._table.cellClicked.connect(self._on_table_cell_clicked)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
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

        self._update_visible_tracks()
        self._table.horizontalHeader().setSortIndicator(8, Qt.SortOrder.AscendingOrder)
        self._table.setRowCount(len(self._visible_tracks))
        for row, track in enumerate(self._visible_tracks):
            self._set_track_row(row, track)

        self._table.resizeColumnsToContents()

        self._update_task_status_label()

    def _set_modified(self, modified: bool):
        self._modified = modified
        for button in (self._btn_cancel, self._btn_save):
            button.setEnabled(modified)
        self._update_apply_button()

    def _has_pending_data_changes(self) -> bool:
        return any(
            t.detached_rows or not t.is_candidate and not t.is_archive and (
                t.deleted
                or t.merged_into_track_id is not None
                or (t.new_class_id is not None and t.new_class_id != t.class_id)
                or t.fragment_assigned
            )
            for t in self._tracks
        )

    def _update_apply_button(self):
        self._btn_apply.setEnabled(self._has_pending_data_changes())

    def _snapshot_state(self) -> Dict[int, tuple]:
        return {
            t.track_id: (
                bool(t.deleted),
                t.new_class_id,
                bool(t.confirmed),
                t.merged_into_track_id,
                t.fragment_assigned,
                t.is_archive, t.group_parent_id, t.merged_member_ids,
                t.detached_from_track_id, t.detached_rows,
                t.detached_parent_id, t.detached_signature, t.detached_owner_ids,
            )
            for t in self._tracks
        }

    def _update_modified_from_baseline(self):
        self._set_modified(self._snapshot_state() != self._baseline_state)
        self._update_merge_button()
        self._proximity_timer.start()

    def _selected_tracks(self) -> List[TrackInfo]:
        rows = sorted({idx.row() for idx in self._table.selectionModel().selectedRows()})
        return [self._visible_tracks[row] for row in rows if 0 <= row < len(self._visible_tracks)]

    def _can_merge_selected_tracks(self) -> bool:
        selected = self._selected_tracks()
        if len(selected) < 2:
            return False
        if any(t.deleted or t.merged_into_track_id is not None for t in selected):
            return False
        regular = [t for t in selected if not t.is_candidate]
        if not regular or (any(t.is_candidate for t in selected) and len(regular) != 1):
            return False
        class_ids = {t.effective_class_id for t in selected}
        return len(class_ids) == 1

    def _update_merge_button(self):
        if not hasattr(self, "_btn_merge"):
            return
        self._btn_merge.setEnabled(self._can_merge_selected_tracks())
        if hasattr(self, "_btn_attach_fragment"):
            self._btn_attach_fragment.setEnabled(
                self._can_merge_selected_tracks() and any(t.is_candidate for t in self._selected_tracks())
            )
        if hasattr(self, "_btn_detach"):
            selected = self._selected_tracks()
            self._btn_detach.setEnabled(len(selected) == 1 and selected[0].merged_into_track_id is not None)

    def _on_selection_changed(self):
        self._update_merge_button()
        self._proximity_timer.start()

    def _refresh_proximity(self, force: bool = False):
        selected = [t for t in self._selected_tracks()
                    if not t.deleted and t.merged_into_track_id is None]
        fragments = [t for t in selected if t.is_candidate]
        reference = min(fragments, key=self._chronological_key) if fragments else next(
            (t for t in selected if t is self._selected_track), selected[0] if selected else None,
        )
        # Обычная проверка с отключёнными фрагментами не требует дополнительного
        # прохода по CSV на каждом выборе. Для обычных треков оценка включается
        # нажатием на заголовок «Близость».
        if self._track_filter.currentData() == "tracks" and self._sort_column != 9 and not force:
            reference = None
        key = (reference.track_id if reference else None, tuple(self._snapshot_state().items()))
        if key == self._proximity_cache_key:
            return
        self._proximity_cache_key = key
        self._proximity_reference_id = reference.track_id if reference else None
        self._proximity_scores = {}
        if reference is not None:
            by_id = {t.track_id: t for t in self._tracks}
            targets = {t.track_id for t in self._tracks
                       if t is not reference and not t.deleted and t.merged_into_track_id is None
                       and t.effective_class_id == reference.effective_class_id}
            track_ids = {}
            orphan_ids = {}
            row_targets = {}
            for track in self._tracks:
                root_id = track.merged_into_track_id if track.merged_into_track_id is not None else track.track_id
                row_targets.update((idx, root_id if root_id in targets else None) for idx in track.detached_rows)
                if root_id not in targets or root_id not in by_id:
                    continue
                if track.fragment_rows:
                    orphan_ids.update((idx, root_id) for idx in track.fragment_rows)
                else:
                    track_ids[track.track_id] = root_id
            if track_ids or orphan_ids:
                try:
                    reference_df = self._track_detections.get(reference.track_id)
                    if reference_df is None:
                        reference_df = self._load_detections_for_track(reference.track_id)
                    self._proximity_scores = load_track_proximity(
                        self._detections_csv, reference_df, track_ids, orphan_ids, row_targets,
                    )
                except Exception as e:
                    self._table.horizontalHeaderItem(9).setToolTip(f"Не удалось оценить близость: {e}")
        blocked = self._table.blockSignals(True)
        for row, track in enumerate(self._visible_tracks):
            self._set_proximity_cell(row, track)
        self._table.blockSignals(blocked)
        if self._sort_column == 9:
            self._rebuild_table(seek_to_first=False)

    def _set_proximity_cell(self, row: int, track: TrackInfo):
        result = self._proximity_scores.get(track.track_id)
        if track.track_id == self._proximity_reference_id:
            text, tooltip = "Опорный", "Положение этого элемента используется для сравнения."
        elif result:
            text = f"{result['score']:.2f}"
            tooltip = (
                f"К элементу {self._proximity_reference_id}:\n"
                f"Разрыв по времени: {result['time_gap_s']:.3f} с\n"
                f"Расстояние центров (нормированные координаты): {result['center_distance']:.4f}\n"
                f"Перекрытие рамок IoU: {result['iou']:.3f}\n"
                "Меньшая оценка означает большую пространственно-временную близость."
            )
        else:
            text, tooltip = "—", "Нет подходящих данных для сравнения с выбранным элементом."
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        item.setToolTip(tooltip)
        self._table.setItem(row, 9, item)

    def _refresh_merged_track_stats(self):
        children = {}
        for track in self._tracks:
            if track.merged_into_track_id is not None:
                parent = track.group_parent_id if track.group_parent_id is not None else track.merged_into_track_id
                children.setdefault(parent, []).append(track)
        visited = set()

        def refresh(target):
            if target.track_id in visited or target.is_archive:
                return
            visited.add(target.track_id)
            pending = [t for t in children.get(target.track_id, []) if not t.is_archive]
            for child in pending:
                refresh(child)
            if pending:
                self._apply_merged_stats(target, [target] + pending)

        for track in self._tracks:
            refresh(track)

    def _apply_merged_stats(self, target: TrackInfo, tracks: List[TrackInfo]):
        first = min(tracks, key=lambda t: t.first_frame)
        last = max(tracks, key=lambda t: t.last_frame)
        detections_count = sum(t.detections_count for t in tracks)
        if detections_count:
            target.avg_confidence = round(
                sum(t.avg_confidence * t.detections_count for t in tracks) / detections_count,
                3,
            )
        target.first_frame = min(t.first_frame for t in tracks)
        target.last_frame = max(t.last_frame for t in tracks)
        target.frame_span = target.last_frame - target.first_frame + 1
        target.first_timestamp_s = min(t.first_timestamp_s for t in tracks)
        target.last_timestamp_s = max(t.last_timestamp_s for t in tracks)
        target.duration_s = round(target.last_timestamp_s - target.first_timestamp_s, 2)
        target.first_depth_m = first.first_depth_m
        target.last_depth_m = last.last_depth_m
        if target.first_depth_m is not None and target.last_depth_m is not None:
            target.depth_change_m = round(target.last_depth_m - target.first_depth_m, 2)
        target.detections_count = detections_count

    def _clear_visual_marks(self):
        if not self._scene:
            self._delete_crosses.clear()
            self._confirm_marks.clear()
            return

        for marks in (self._delete_crosses, self._confirm_marks):
            for line1, line2 in marks.values():
                self._scene.removeItem(line1)
                self._scene.removeItem(line2)
            marks.clear()

        if self._highlight_rect:
            self._highlight_rect.setVisible(False)
        self._highlight_active = False

    def _reload_verified_state(self):
        self._clear_visual_marks()
        self._load_data()
        self._rebuild_table()
        self._baseline_state = self._snapshot_state()
        self._set_modified(False)

        if self._player:
            self._update_delete_crosses(self._player.position())
            self._update_confirm_marks(self._player.position())

    def _set_track_row(self, row: int, track: TrackInfo):
        depth_str = ""
        if track.first_depth_m is not None:
            depth_str = f"{track.first_depth_m:.1f}"
            if track.last_depth_m is not None:
                depth_str += f"–{track.last_depth_m:.1f}"

        frame_str = f"{track.first_frame}–{track.last_frame}"

        values = [
            f"Фрагмент {abs(track.track_id)}" if track.track_id < 0 else str(track.track_id),
            track.effective_class_name,
            track.status_text,
            frame_str,
            str(track.duration_s),
            depth_str,
            str(track.detections_count),
            f"{track.avg_confidence:.2f}",
            f"{track.first_timestamp_s:.3f}",
        ]

        for col, val in enumerate(values):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if track.deleted:
                item.setForeground(QBrush(QColor(180, 60, 60)))
                f = item.font()
                f.setStrikeOut(True)
                item.setFont(f)
            elif track.merged_into_track_id is not None:
                item.setForeground(QBrush(QColor(120, 120, 200)))
            elif track.confirmed:
                item.setBackground(QBrush(QColor(60, 180, 60)))
            elif track.new_class_id is not None:
                item.setForeground(QBrush(QColor(200, 150, 0)))
            elif track.is_candidate:
                item.setForeground(QBrush(QColor(220, 160, 70)))
            self._table.setItem(row, col, item)
        name = self._table.item(row, 0)
        indent = "    " * self._group_depth.get(track.track_id, 0)
        if track.track_id in self._group_children:
            arrow = "▼" if track.track_id in self._expanded_tracks else "▶"
            name.setText(f"{indent}{arrow} {values[0]} ({len(self._group_children[track.track_id])})")
            name.setToolTip("Нажмите на ID, чтобы развернуть или свернуть состав объединённого трека.")
            font = name.font()
            font.setBold(True)
            name.setFont(font)
        elif track.merged_into_track_id is not None:
            name.setText(indent + "↳ " + values[0])
            name.setToolTip(f"Часть объединённого трека #{track.merged_into_track_id}; не отдельный трек.")
        self._set_proximity_cell(row, track)

    # ── video setup ──────────────────────────────────────────────────

    def _setup_video(self):
        if not HAS_MULTIMEDIA or not self._video_path or not self._graphics_view:
            return

        self._scene = QGraphicsScene(self._graphics_view)
        self._graphics_view.setScene(self._scene)

        self._video_item = QGraphicsVideoItem()
        self._video_item.nativeSizeChanged.connect(
            lambda *_: self._schedule_fit_video()
        )
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

        self._schedule_fit_video()

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
            self._schedule_fit_video()
        return super().eventFilter(obj, event)

    def showEvent(self, event):
        super().showEvent(event)
        self._schedule_fit_video()

    def _schedule_fit_video(self):
        if self._fit_pending:
            return
        self._fit_pending = True
        QTimer.singleShot(0, self._run_scheduled_fit_video)

    def _run_scheduled_fit_video(self):
        self._fit_pending = False
        self._fit_video()

    def _fit_video(self):
        if self._fitting or not self._video_item:
            return
        ns = self._video_item.nativeSize()
        if not ns.isValid():
            return
        viewport_size = self._graphics_view.viewport().size()
        fit_state = (
            int(ns.width()),
            int(ns.height()),
            viewport_size.width(),
            viewport_size.height(),
        )
        if fit_state == self._last_fit_state:
            return
        self._fitting = True
        try:
            self._video_item.setSize(ns)
            self._scene.setSceneRect(self._video_item.boundingRect())
            self._graphics_view.fitInView(
                self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
            )
            self._last_fit_state = fit_state
        finally:
            self._fitting = False

    # ── video slots ──────────────────────────────────────────────────

    def _on_duration_changed(self, ms: int):
        self._time_slider.setRange(0, ms)
        self._update_time_label()
        self._schedule_fit_video()

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
        df = load_verification_track_detections(self._detections_csv, self._tracks, track_id)
        self._track_detections[track_id] = df
        return df

    def _refresh_statistics(self, track_ids):
        track_ids = set(track_ids)
        by_id = {t.track_id: t for t in self._tracks}
        for tid in list(track_ids):
            visited = set()
            while tid in by_id and tid not in visited:
                visited.add(tid)
                track_ids.add(tid)
                track = by_id[tid]
                tid = track.group_parent_id if track.group_parent_id is not None else track.merged_into_track_id
        self._track_detections.clear()
        self._deleted_detections.clear()
        self._confirmed_detections.clear()
        for track in self._tracks:
            if track.track_id in track_ids:
                refresh_track_statistics(track, self._load_detections_for_track(track.track_id))
        for track in self._tracks:
            if track.deleted or track.confirmed:
                data = self._track_detections.get(track.track_id)
                if data is None:
                    data = self._load_detections_for_track(track.track_id)
                if track.deleted:
                    self._deleted_detections[track.track_id] = data
                if track.confirmed:
                    self._confirmed_detections[track.track_id] = data

    def _index_assigned_rows(self):
        self._fragment_by_row = {idx: t for t in self._tracks for idx in t.fragment_rows}
        self._detached_by_row = {idx: t for t in self._tracks for idx in t.detached_rows}

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

    def _on_table_cell_clicked(self, row: int, column: int):
        self._on_track_selected(row)
        if row < 0 or row >= len(self._visible_tracks):
            return
        track = self._visible_tracks[row]
        modifiers = QApplication.keyboardModifiers()
        if (column == 0 and track.track_id in self._group_children
                and not modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)):
            if track.track_id in self._expanded_tracks:
                self._expanded_tracks.remove(track.track_id)
            else:
                self._expanded_tracks.add(track.track_id)
            self._rebuild_table(seek_to_first=False)

    def _on_track_selected(self, row: int, seek_to_first: bool = True):
        if row < 0 or row >= len(self._visible_tracks):
            return

        self._selected_track = self._visible_tracks[row]

        self._btn_delete.setEnabled(True)
        self._btn_delete.setText(
            "↩ Восстановить трек"
            if self._selected_track.deleted
            else "🗑 Удалить трек"
        )
        is_merged_source = self._selected_track.merged_into_track_id is not None
        is_candidate = self._selected_track.is_candidate
        self._btn_create_track.setEnabled(is_candidate)
        self._update_merge_button()
        self._btn_delete.setEnabled(not is_merged_source and not is_candidate)
        self._btn_confirm.setEnabled(not self._selected_track.deleted and not is_merged_source and not is_candidate)
        self._btn_confirm.setText(
            "✘ Отменить подтверждение"
            if self._selected_track.confirmed
            else "✔ Подтвердить"
        )
        self._class_combo.setEnabled(not is_merged_source)
        self._class_combo.blockSignals(True)
        cid = self._selected_track.effective_class_id
        idx = self._class_combo.findData(cid)
        if idx >= 0:
            self._class_combo.setCurrentIndex(idx)
        self._class_combo.blockSignals(False)

        # Информация о треке
        t = self._selected_track
        if t.track_id < 0:
            title = f"Фрагмент {abs(t.track_id)}" + (" без ID" if t.is_candidate else "")
        else:
            title = f"Трек #{t.track_id}"
        self._track_info_label.setText(
            f"{title}  |  {t.effective_class_name}  |  "
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
        self._proximity_timer.start()

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

        candidates = []
        known_tracks = {t.track_id: t for t in self._tracks}
        visible_ids = {t.track_id for t in self._visible_tracks}

        for _, row in frame_df.iterrows():
            if int(row["_detection_row"]) in self._detached_by_row:
                track = self._detached_by_row[int(row["_detection_row"])]
            elif pd.isna(row["track_id"]):
                track = self._fragment_by_row.get(int(row["_detection_row"]))
            else:
                track = known_tracks.get(int(row["track_id"]))
            if track is None:
                continue
            if track.merged_into_track_id is not None:
                track = known_tracks.get(track.merged_into_track_id, track)
            if track.track_id not in visible_ids:
                continue
            xc = float(row["x_center"])
            yc = float(row["y_center"])
            bw = float(row["width"])
            bh = float(row["height"])
            if not all(math.isfinite(v) for v in (xc, yc, bw, bh)) or bw <= 0 or bh <= 0:
                continue

            bx = (xc - bw / 2) * scene_w
            by = (yc - bh / 2) * scene_h
            bbox_w = bw * scene_w
            bbox_h = bh * scene_h
            bbox = QRectF(bx, by, bbox_w, bbox_h)

            dt = abs(float(row["timestamp_s"]) - current_sec)
            dist = (scene_pos.x() - xc * scene_w) ** 2 + (scene_pos.y() - yc * scene_h) ** 2
            # Внутри рамок приоритет у ближайшего по времени кадра.
            key = (0, dt, bbox_w * bbox_h) if bbox.contains(scene_pos) else (1, dist, dt)
            candidates.append((key, track.track_id))

        if candidates:
            best_track_id = min(candidates)[1]
            for row_idx, t in enumerate(self._visible_tracks):
                if t.track_id == best_track_id:
                    self._table.selectRow(row_idx)
                    self._on_track_selected(row_idx, seek_to_first=False)
                    self._update_highlight(self._player.position())
                    self._update_delete_crosses(self._player.position())
                    self._update_confirm_marks(self._player.position())
                    return

    def _on_create_fragment_track(self):
        track = self._selected_track
        if not track or not track.is_candidate:
            return
        # Учитываем и ID детекций, отсутствующих в таблице треков.
        max_id = max((t.track_id for t in self._tracks), default=0)
        try:
            for chunk in pd.read_csv(self._detections_csv, usecols=["track_id"], chunksize=50000):
                ids = chunk["track_id"].dropna()
                if not ids.empty:
                    max_id = max(max_id, int(ids.max()))
        except Exception as e:
            QMessageBox.critical(self, "Создать трек", f"Не удалось прочитать ID детекций:\n{e}")
            return
        old_id = track.track_id
        track.track_id = max(0, max_id) + 1
        if track.detached_rows:
            track.merged_member_ids = (track.track_id,)
        track.fragment_assigned = True
        self._track_detections.pop(old_id, None)
        self._update_modified_from_baseline()
        self._rebuild_table()

    def _on_attach_fragment(self):
        self._on_merge_tracks()

    def _on_detach_track(self):
        selected = self._selected_tracks()
        if len(selected) != 1 or selected[0].merged_into_track_id is None:
            return
        source = selected[0]
        by_id = {t.track_id: t for t in self._tracks}
        class_id = by_id[source.merged_into_track_id].effective_class_id
        ancestors = []
        parent = source.group_parent_id if source.group_parent_id is not None else source.merged_into_track_id
        seen = set()
        while parent in by_id and parent not in seen:
            seen.add(parent)
            ancestors.append(by_id[parent])
            current = by_id[parent]
            parent = current.group_parent_id if current.group_parent_id is not None else current.merged_into_track_id
        descendants = []
        for track in self._tracks:
            if track is source:
                continue
            parent = track.group_parent_id if track.group_parent_id is not None else track.merged_into_track_id
            visited = set()
            while parent in by_id and parent not in visited:
                if parent == source.track_id:
                    descendants.append(track)
                    break
                visited.add(parent)
                current = by_id[parent]
                parent = current.group_parent_id if current.group_parent_id is not None else current.merged_into_track_id
        try:
            signature = detection_file_signature(self._detections_csv)
            if any((t.detached_rows and t.detached_signature != signature)
                   or (t.fragment_rows and t.fragment_signature != signature) for t in self._tracks):
                raise ValueError("CSV изменился. Откройте окно проверки заново перед отсоединением.")
            data = load_verification_track_detections(self._detections_csv, self._tracks, source.track_id)
            if data.empty:
                raise ValueError("Для выбранной части не найдены детекции в исходном CSV.")
            ancestor_data = {
                t.track_id: load_verification_track_detections(self._detections_csv, self._tracks, t.track_id)
                for t in ancestors
            }
            if detection_file_signature(self._detections_csv) != signature:
                raise ValueError("CSV изменился во время чтения. Повторно откройте окно проверки.")
        except Exception as e:
            QMessageBox.warning(self, "Отсоединить", f"Не удалось отсоединить часть:\n{e}")
            return

        members = set(source.merged_member_ids or (source.track_id,))
        for track in descendants:
            members.update(track.merged_member_ids or (track.track_id,))
        if source.is_archive:
            rows = tuple(int(idx) for idx in data.index)
            row_set = set(rows)
            for track in self._tracks:
                if track is not source and track.detached_rows:
                    track.detached_rows = tuple(idx for idx in track.detached_rows if idx not in row_set)
                    if not track.detached_rows:
                        track.detached_from_track_id = None
                        track.detached_parent_id = None
                        track.detached_signature = ""
                        track.detached_owner_ids = ()
            if source.detached_from_track_id is None:
                source.detached_from_track_id = int(data["track_id"].dropna().iloc[0])
                source.detached_parent_id = (source.group_parent_id if source.group_parent_id is not None
                                             else source.merged_into_track_id)
            source.detached_rows = rows
            source.detached_signature = signature
            source.detached_owner_ids = tuple(sorted(int(tid) for tid in data["track_id"].dropna().unique()))
            if source.track_id < 0:
                source.fragment_rows = rows
                source.fragment_signature = signature
        source.merged_into_track_id = None
        source.group_parent_id = None
        source.deleted = False
        source.confirmed = False
        source.new_class_id = class_id if class_id != source.class_id else None
        if source.track_id < 0:
            source.fragment_assigned = False
        for child in descendants:
            child.merged_into_track_id = source.track_id
        for ancestor in ancestors:
            ancestor.confirmed = False
            if ancestor.merged_member_ids:
                ancestor.merged_member_ids = tuple(sorted(set(ancestor.merged_member_ids) - members))
        self._index_assigned_rows()
        self._selected_track = source
        self._table.clearSelection()
        self._clear_visual_marks()
        self._track_detections.clear()
        self._track_detections[source.track_id] = data
        refresh_track_statistics(source, data)
        self._deleted_detections.pop(source.track_id, None)
        self._confirmed_detections.pop(source.track_id, None)
        for ancestor in ancestors:
            remaining = ancestor_data[ancestor.track_id].drop(index=data.index, errors="ignore")
            refresh_track_statistics(ancestor, remaining)
            self._track_detections[ancestor.track_id] = remaining
            self._confirmed_detections.pop(ancestor.track_id, None)
            if ancestor.deleted:
                self._deleted_detections[ancestor.track_id] = remaining
        self._update_modified_from_baseline()
        if source.is_candidate and self._track_filter.currentData() == "tracks":
            blocked = self._track_filter.blockSignals(True)
            self._track_filter.setCurrentIndex(self._track_filter.findData("all"))
            self._track_filter.blockSignals(blocked)
        self._rebuild_table()
        if self._player:
            self._update_delete_crosses(self._player.position())
            self._update_confirm_marks(self._player.position())

    def _toggle_delete(self):
        if not self._selected_track or self._selected_track.is_candidate or self._selected_track.merged_into_track_id is not None:
            return

        self._selected_track.deleted = not self._selected_track.deleted
        self._update_modified_from_baseline()

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

        if self._player:
            self._update_delete_crosses(self._player.position())

        if self._selected_track.deleted:
            self._select_next_unconfirmed()

    def _on_class_changed(self, index: int):
        if not self._selected_track or self._selected_track.merged_into_track_id is not None:
            return

        new_cid = self._class_combo.currentData()
        if new_cid == self._selected_track.class_id:
            self._selected_track.new_class_id = None
        else:
            self._selected_track.new_class_id = new_cid
        self._update_modified_from_baseline()

        row = self._table.currentRow()
        self._set_track_row(row, self._selected_track)
        self._table.resizeColumnsToContents()
        self._update_task_status_label()

    def _on_merge_tracks(self):
        selected = self._selected_tracks()
        if len(selected) < 2:
            return
        if not self._can_merge_selected_tracks():
            QMessageBox.warning(
                self,
                "Объединить",
                "Выберите трек с ID и один или несколько фрагментов без ID одного класса.\n"
                "Для объединения обычных треков выберите два или более трека одного класса.",
            )
            return

        # Фрагмент может появиться раньше трека, но ID цели всегда сохраняется.
        target = min((t for t in selected if not t.is_candidate), key=self._chronological_key)
        for source in selected:
            if source is target:
                continue
            members = set(source.merged_member_ids or (source.track_id,))
            for child in self._tracks:
                if child.merged_into_track_id == source.track_id:
                    members.update(child.merged_member_ids or (child.track_id,))
            source.merged_member_ids = tuple(sorted(members))
        source_ids = {t.track_id for t in selected if t is not target}
        # Переносим и ранее присоединённые фрагменты, не создавая цепочки слияний.
        for track in self._tracks:
            if track.merged_into_track_id in source_ids:
                if track.group_parent_id is None:
                    track.group_parent_id = track.merged_into_track_id
                track.merged_into_track_id = target.track_id
        for track in selected:
            if track.track_id == target.track_id:
                continue
            if track.is_candidate:
                track.fragment_assigned = True
            track.group_parent_id = target.track_id
            track.merged_into_track_id = target.track_id
            track.new_class_id = None
            track.confirmed = False
            track.deleted = False
            self._deleted_detections.pop(track.track_id, None)
            self._confirmed_detections.pop(track.track_id, None)

        target.confirmed = False
        self._apply_merged_stats(target, selected)
        self._track_detections.clear()
        self._confirmed_detections.pop(target.track_id, None)
        self._update_modified_from_baseline()
        self._selected_track = target
        self._expanded_tracks.discard(target.track_id)
        self._table.clearSelection()
        self._rebuild_table()

        if self._player:
            self._update_delete_crosses(self._player.position())
            self._update_confirm_marks(self._player.position())

    # ── confirm / apply / sort ────────────────────────────────────────

    def _on_confirm(self):
        if (
            not self._selected_track
            or self._selected_track.is_candidate
            or self._selected_track.deleted
            or self._selected_track.merged_into_track_id is not None
        ):
            return

        self._selected_track.confirmed = not self._selected_track.confirmed
        self._update_modified_from_baseline()

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

        if self._player:
            self._update_confirm_marks(self._player.position())

        if self._selected_track.confirmed:
            self._select_next_unconfirmed()

    def _on_cancel_changes(self):
        self._restore_backup(cleanup=False)
        self._reload_verified_state()

    def _on_save(self):
        if not self._tracks_csv:
            return

        try:
            save_verified(self._tracks, self._tracks_csv)
            self._update_verified_backup()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Ошибка сохранения",
                f"Не удалось сохранить состояние проверки:\n{e}",
            )
            return

        self._baseline_state = self._snapshot_state()
        self._set_modified(False)

    def _on_apply(self):
        if not self._has_pending_data_changes():
            QMessageBox.information(self, "Применить", "Нет изменений для применения.")
            return

        destructive = [
            t for t in self._tracks
            if t.deleted or t.new_class_id is not None or t.merged_into_track_id is not None
        ]
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

        self._update_verified_backup()
        self._reload_verified_state()

        QMessageBox.information(
            self,
            "Применено",
            f"Изменения применены.\n"
            f"Всего треков: {result.total_tracks}\n"
            f"Удалено: {result.deleted_count}\n"
            f"Изменён класс: {result.changed_class_count}\n"
            f"Объединено: {result.merged_count}\n"
            f"Отсоединено от применённых групп: {result.detached_count}",
        )

    def _on_sort(self, column: int):
        if column == 9:
            self._proximity_timer.stop()
            self._refresh_proximity(force=True)
        if column == self._sort_column:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = column
            self._sort_reverse = False

        self._table.horizontalHeader().setSortIndicator(
            self._sort_column,
            Qt.SortOrder.DescendingOrder if self._sort_reverse else Qt.SortOrder.AscendingOrder,
        )
        self._rebuild_table()

    @staticmethod
    def _chronological_key(track: TrackInfo):
        return (
            track.first_frame, track.first_timestamp_s,
            track.fragment_rows[0] if track.fragment_rows else -1, track.track_id,
        )

    def _update_visible_tracks(self):
        mode = self._track_filter.currentData()
        self._group_children = {}
        self._group_depth = {}
        known_ids = {t.track_id for t in self._tracks}
        for track in self._tracks:
            if track.merged_into_track_id is not None:
                parent = track.group_parent_id if track.group_parent_id in known_ids else track.merged_into_track_id
                self._group_children.setdefault(parent, []).append(track)
        self._visible_tracks = [
            t for t in self._tracks
            if t.merged_into_track_id is None and (
                mode == "all" or (t.is_candidate if mode == "fragments" else not t.is_candidate)
            )
        ]
        # Стабильный порядок появления используется и при совпадении значений
        # выбранной пользователем колонки сортировки.
        self._visible_tracks.sort(key=self._chronological_key)
        key_map = {
            0: lambda t: t.track_id,
            1: lambda t: t.effective_class_name.lower(),
            2: lambda t: t.status_text,
            3: lambda t: t.first_frame,
            4: lambda t: t.duration_s,
            5: lambda t: t.first_depth_m or 0.0,
            6: lambda t: t.detections_count,
            7: lambda t: t.avg_confidence,
            8: lambda t: t.first_timestamp_s,
            9: lambda t: -1 if t.track_id == self._proximity_reference_id else
                self._proximity_scores.get(t.track_id, {}).get("score", float("inf")),
        }
        key = key_map.get(self._sort_column)
        if key is not None:
            self._visible_tracks.sort(key=key, reverse=self._sort_reverse)
        rows = []
        visited = set()

        def append_group(track, depth):
            if track.track_id in visited:
                return
            visited.add(track.track_id)
            self._group_depth[track.track_id] = depth
            rows.append(track)
            if track.track_id in self._expanded_tracks:
                for child in sorted(self._group_children.get(track.track_id, []), key=self._chronological_key):
                    append_group(child, depth + 1)

        for track in self._visible_tracks:
            append_group(track, 0)
        self._visible_tracks = rows

    def _on_filter_changed(self, index: int):
        self._rebuild_table(seek_to_first=False)

    def _clear_track_selection(self):
        self._proximity_timer.stop()
        self._selected_track = None
        self._table.clearSelection()
        self._table.setCurrentCell(-1, -1)
        self._track_play_active = False
        self._highlight_active = False
        if self._highlight_rect:
            self._highlight_rect.setVisible(False)
        for button in (self._btn_delete, self._btn_confirm, self._btn_create_track,
                       self._btn_attach_fragment, self._btn_merge, self._btn_detach):
            button.setEnabled(False)
        self._class_combo.setEnabled(False)
        self._track_info_label.setText("Нет элементов для выбранного фильтра.")
        self._update_track_button()

    def _rebuild_table(self, seek_to_first: bool = True):
        selected_track_id = self._selected_track.track_id if self._selected_track else None
        selected_ids = {t.track_id for t in self._selected_tracks()}
        if selected_track_id is not None:
            selected_ids.add(selected_track_id)

        blocked = self._table.blockSignals(True)
        self._table.clearSelection()
        self._table.setCurrentCell(-1, -1)
        self._update_visible_tracks()
        self._table.setRowCount(len(self._visible_tracks))
        for row, track in enumerate(self._visible_tracks):
            self._set_track_row(row, track)

        self._table.resizeColumnsToContents()
        selected_row = next((row for row, track in enumerate(self._visible_tracks)
                             if track.track_id == selected_track_id), None)
        row = selected_row if selected_row is not None else next(
            (i for i, t in enumerate(self._visible_tracks) if t.track_id in selected_ids), 0,
        )
        if self._visible_tracks:
            selection = self._table.selectionModel()
            flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
            selected_rows = [i for i, t in enumerate(self._visible_tracks) if t.track_id in selected_ids]
            for selected in selected_rows or [row]:
                selection.select(self._table.model().index(selected, 0), flags)
            selection.setCurrentIndex(self._table.model().index(row, 0),
                                      QItemSelectionModel.SelectionFlag.NoUpdate)
        self._table.blockSignals(blocked)
        if self._visible_tracks:
            self._on_track_selected(
                row, seek_to_first=seek_to_first or selected_row is None,
            )
            self._update_merge_button()
        else:
            self._clear_track_selection()
        if self._player:
            self._update_highlight(self._player.position())

        self._update_task_status_label()

    def _select_next_unconfirmed(self):
        current_row = self._table.currentRow()
        n = len(self._visible_tracks)

        for offset in range(1, n):
            idx = (current_row + offset) % n
            t = self._visible_tracks[idx]
            if not t.is_candidate and not t.confirmed and not t.deleted and t.merged_into_track_id is None:
                self._table.selectRow(idx)
                self._on_track_selected(idx)
                return

    def _compute_task_status(self) -> str:
        tracks = [t for t in self._tracks if not t.is_candidate]
        if not tracks:
            return ""
        if any(t.fragment_assigned or t.detached_rows for t in self._tracks):
            return "В работе"

        resolved = sum(
            1 for t in tracks
            if t.confirmed or t.deleted or t.merged_into_track_id is not None
        )
        if resolved == 0:
            return ""
        if resolved == len(tracks):
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

        candidates = sum(t.is_candidate for t in self._tracks)
        if candidates:
            text = self._task_status_label.text()
            self._task_status_label.setText(
                (text + "  |  " if text else "") + f"Фрагментов без ID: {candidates}"
            )

        self.setWindowTitle(
            f"Проверка треков — задача #{self._task.id}"
            + (f"  [{status}]" if status else "")
        )

    # ── close ─────────────────────────────────────────────────────────

    def accept(self):
        self._on_save()


def _format_ms(ms: int) -> str:
    total_s = ms // 1000
    minutes = total_s // 60
    seconds = total_s % 60
    return f"{minutes:02d}:{seconds:02d}"
