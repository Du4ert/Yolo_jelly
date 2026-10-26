"""Чтение кадров для верификации без блокировки интерфейса Qt."""

import math
from collections import OrderedDict
from threading import Condition

import cv2
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage


class VideoFrameReader(QThread):
    """Декодер с абсолютными номерами кадров и отменой устаревших запросов.

    VideoCapture принадлежит исключительно рабочему потоку. При близком
    переходе кадры дочитываются последовательно, назад используется небольшой
    кеш. Произвольная перемотка проверяется по позиции и временной метке;
    если backend не подтверждает позицию, чтение начинается с начала файла.
    """

    metadataReady = pyqtSignal(float, int, int, int)
    frameReady = pyqtSignal(int, int, QImage)
    failed = pyqtSignal(int, str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self._path = path
        self._condition = Condition()
        self._pending = None
        self._revision = 0
        self._stopping = False
        self._cache = OrderedDict()
        self._cache_bytes = 0

    def request_frame(self, revision: int, frame_index: int):
        with self._condition:
            self._revision = revision
            self._pending = (revision, frame_index)
            self._condition.notify()

    def cancel(self, revision: int):
        with self._condition:
            self._revision = revision
            self._pending = None
            self._condition.notify()

    def stop(self):
        with self._condition:
            self._stopping = True
            self._pending = None
            self._condition.notify()

    def _obsolete(self, revision):
        with self._condition:
            return self._stopping or revision != self._revision

    def _remember(self, index, image):
        previous = self._cache.pop(index, None)
        if previous is not None:
            self._cache_bytes -= previous.sizeInBytes()
        self._cache[index] = image
        self._cache_bytes += image.sizeInBytes()
        while len(self._cache) > 1 and (len(self._cache) > 8 or self._cache_bytes > 64 * 1024 * 1024):
            _, removed = self._cache.popitem(last=False)
            self._cache_bytes -= removed.sizeInBytes()

    def run(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self._path)
            if not cap.isOpened():
                raise ValueError("Не удалось открыть видео для покадрового чтения.")
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count < 1:
                raise ValueError("Не удалось определить FPS и число кадров видео.")
            frame_count = int(count)
            self.metadataReady.emit(fps, frame_count, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            next_index = 0
            verified_seek = True
            while True:
                with self._condition:
                    while self._pending is None and not self._stopping:
                        self._condition.wait()
                    if self._stopping:
                        break
                    revision, target = self._pending
                    self._pending = None
                target = max(0, min(frame_count - 1, target))
                cached = self._cache.get(target)
                if cached is not None:
                    self._cache.move_to_end(target)
                    if not self._obsolete(revision):
                        self.frameReady.emit(revision, target, cached)
                    continue
                try:
                    if target < next_index or target - next_index > 120:
                        anchor = max(0, target - 16)
                        positioned = cap.set(cv2.CAP_PROP_POS_FRAMES, anchor)
                        position = cap.get(cv2.CAP_PROP_POS_FRAMES)
                        if positioned and math.isfinite(position) and abs(position - anchor) < 0.5:
                            next_index = anchor
                            verified_seek = False
                        else:
                            cap.release()
                            cap = cv2.VideoCapture(self._path)
                            if not cap.isOpened():
                                raise ValueError("Не удалось повторно открыть видео.")
                            next_index, verified_seek = 0, True
                    while next_index <= target and not self._obsolete(revision):
                        index = next_index
                        ok, frame = cap.read()
                        if not ok:
                            raise ValueError(f"Не удалось прочитать кадр {index}.")
                        next_index += 1
                        if not verified_seek:
                            position = cap.get(cv2.CAP_PROP_POS_FRAMES)
                            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                            if (not math.isfinite(position) or abs(position - next_index) >= 0.5
                                    or not math.isfinite(timestamp)
                                    or abs(timestamp - index * 1000.0 / fps) > 500.0 / fps):
                                cap.release()
                                cap = cv2.VideoCapture(self._path)
                                if not cap.isOpened():
                                    raise ValueError("Не удалось открыть видео для точного дочитывания.")
                                next_index, verified_seek = 0, True
                                continue
                            verified_seek = True
                        if index >= target - 2:
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                                           QImage.Format.Format_RGB888).copy()
                            if image.isNull():
                                raise ValueError(f"Не удалось подготовить изображение кадра {index}.")
                            self._remember(index, image)
                        if index == target and not self._obsolete(revision):
                            self.frameReady.emit(revision, target, self._cache[target])
                except Exception as e:
                    # После ошибки позиция декодера неизвестна: следующий запрос
                    # должен выполнить проверенную перемотку или открыть файл заново.
                    next_index = frame_count + 1
                    if not self._obsolete(revision):
                        self.failed.emit(revision, str(e))
        except Exception as e:
            self.failed.emit(0, str(e))
        finally:
            if cap is not None:
                cap.release()
            self._cache.clear()
            self._cache_bytes = 0
