"""
Утилиты для ускоренного чтения и записи видео.

- ThreadedVideoCapture: фоновое декодирование кадров в отдельном потоке
- NvencVideoWriter: аппаратное кодирование через FFmpeg NVENC (с fallback на cv2)
"""

import cv2
import threading
import queue
import subprocess
import shutil
from typing import Optional, Tuple


class ThreadedVideoCapture:
    """
    Обёртка над cv2.VideoCapture с фоновым чтением кадров.

    Пока основной поток обрабатывает кадр N (GPU-инференс, optical flow),
    фоновый поток декодирует кадр N+1 из видеофайла.
    Это позволяет перекрыть I/O-задержку декодирования.

    Использование:
        cap = ThreadedVideoCapture(video_path).start()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ...
        cap.release()
    """

    def __init__(self, path: str, queue_size: int = 32):
        self.cap = cv2.VideoCapture(path)
        self._queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)

    def start(self) -> "ThreadedVideoCapture":
        """Запускает фоновый поток чтения."""
        self._thread.start()
        return self

    def _reader(self):
        """Фоновый цикл чтения кадров."""
        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self._queue.put((False, None))
                break
            # Блокируется если очередь полна — обеспечивает back-pressure
            try:
                self._queue.put((ret, frame), timeout=1.0)
            except queue.Full:
                if self._stop_event.is_set():
                    break
                continue

    def read(self) -> Tuple[bool, Optional["cv2.Mat"]]:
        """Возвращает следующий кадр из очереди."""
        try:
            return self._queue.get(timeout=5.0)
        except queue.Empty:
            return False, None

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def get(self, prop_id: int) -> float:
        """Делегирует свойства VideoCapture."""
        return self.cap.get(prop_id)

    def release(self):
        """Останавливает фоновый поток и освобождает ресурсы."""
        self._stop_event.set()
        # Очищаем очередь чтобы разблокировать поток-писатель
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self.cap.release()


class NvencVideoWriter:
    """
    Видеозаписыватель через FFmpeg NVENC (GPU-кодирование H.264).
    Если FFmpeg или NVENC недоступен — fallback на cv2.VideoWriter.

    Использование:
        writer = NvencVideoWriter(output_path, fps, width, height)
        writer.write(frame)
        ...
        writer.release()
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        bitrate: str = "20M",
    ):
        self.output_path = output_path
        self._proc: Optional[subprocess.Popen] = None
        self._fallback: Optional[cv2.VideoWriter] = None

        if self._ffmpeg_nvenc_available():
            try:
                self._proc = subprocess.Popen(
                    [
                        "ffmpeg", "-y",
                        "-f", "rawvideo",
                        "-pix_fmt", "bgr24",
                        "-s", f"{width}x{height}",
                        "-r", str(fps),
                        "-i", "-",
                        "-c:v", "h264_nvenc",
                        "-preset", "p4",
                        "-b:v", bitrate,
                        "-pix_fmt", "yuv420p",
                        output_path,
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
            except (OSError, subprocess.SubprocessError):
                pass

        # Fallback на cv2.VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._fallback = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    @staticmethod
    def _ffmpeg_nvenc_available() -> bool:
        """Проверяет доступность FFmpeg с поддержкой h264_nvenc."""
        if not shutil.which("ffmpeg"):
            return False
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=5,
            )
            return "h264_nvenc" in result.stdout
        except (subprocess.SubprocessError, OSError):
            return False

    def write(self, frame: "cv2.Mat"):
        """Записывает кадр."""
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                pass
        elif self._fallback:
            self._fallback.write(frame)

    def release(self):
        """Завершает запись и освобождает ресурсы."""
        if self._proc:
            if self._proc.stdin:
                try:
                    self._proc.stdin.close()
                except OSError:
                    pass
            self._proc.wait(timeout=30)
            self._proc = None
        if self._fallback:
            self._fallback.release()
            self._fallback = None
