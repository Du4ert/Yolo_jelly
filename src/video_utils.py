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
    Если NVENC недоступен — fallback на libx264 через FFmpeg.
    Если FFmpeg недоступен — аварийный fallback на cv2.VideoWriter.

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
        self._fps = fps
        self._width = width
        self._height = height
        self._bitrate = bitrate
        self._encoder: Optional[str] = None
        self._proc: Optional[subprocess.Popen] = None
        self._fallback: Optional[cv2.VideoWriter] = None

        nvenc_available, nvenc_reason = self._ffmpeg_nvenc_available()
        if nvenc_available and self._start_ffmpeg_writer("h264_nvenc"):
            print(f"  Видео: используется h264_nvenc, bitrate {bitrate}")
            return

        if shutil.which("ffmpeg") and self._start_ffmpeg_writer("libx264"):
            reason = f": {nvenc_reason}" if nvenc_reason else ""
            print(f"  Видео: h264_nvenc недоступен{reason}; используется libx264, bitrate {bitrate}")
            return

        reason = "ffmpeg недоступен" if not shutil.which("ffmpeg") else "ffmpeg writer не запустился"
        print(f"  Видео: {reason}; используется OpenCV mp4v, bitrate не гарантируется")
        self._start_cv2_fallback(width, height)

    def _start_ffmpeg_writer(self, encoder: str) -> bool:
        """Запускает FFmpeg writer с выбранным H.264 энкодером."""
        if not shutil.which("ffmpeg"):
            return False

        codec_args = ["-c:v", encoder, "-b:v", self._bitrate]
        if encoder == "h264_nvenc":
            codec_args = ["-c:v", encoder, "-preset", "p4", "-b:v", self._bitrate]
        elif encoder == "libx264":
            codec_args = ["-c:v", encoder, "-preset", "veryfast", "-b:v", self._bitrate]

        try:
            self._proc = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "rawvideo",
                    "-pix_fmt", "bgr24",
                    "-s", f"{self._width}x{self._height}",
                    "-r", str(self._fps),
                    "-i", "-",
                    *codec_args,
                    "-pix_fmt", "yuv420p",
                    self.output_path,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self._encoder = encoder
            return True
        except (OSError, subprocess.SubprocessError):
            self._proc = None
            self._encoder = None
            return False

    def _start_cv2_fallback(self, width: int, height: int):
        """Аварийный fallback на OpenCV, если FFmpeg недоступен."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._fallback = cv2.VideoWriter(self.output_path, fourcc, self._fps, (width, height))

    @staticmethod
    def _ffmpeg_nvenc_available() -> Tuple[bool, str]:
        """Проверяет реальную доступность h264_nvenc (пробный encode)."""
        if not shutil.which("ffmpeg"):
            return False, "ffmpeg недоступен"
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1",
                    "-c:v", "h264_nvenc", "-f", "null", "-",
                ],
                capture_output=True, timeout=10,
            )
            if result.returncode == 0:
                return True, ""
            reason = result.stderr.decode("utf-8", errors="replace").strip()
            return False, reason.splitlines()[0] if reason else f"код возврата {result.returncode}"
        except subprocess.TimeoutExpired:
            return False, "таймаут проверки NVENC"
        except (subprocess.SubprocessError, OSError) as exc:
            return False, str(exc)

    def write(self, frame: "cv2.Mat"):
        """Записывает кадр."""
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                failed_encoder = self._encoder
                # ffmpeg упал — читаем stderr и переключаемся на следующий fallback
                stderr_out = ""
                if self._proc.stderr:
                    try:
                        stderr_out = self._proc.stderr.read().decode("utf-8", errors="replace").strip()
                    except OSError:
                        pass
                print(f"  Предупреждение: ffmpeg {self._encoder or 'writer'} завершился с ошибкой")
                if stderr_out:
                    print(f"  ffmpeg stderr: {stderr_out}")
                self._proc = None
                self._encoder = None
                if failed_encoder != "libx264" and self._start_ffmpeg_writer("libx264"):
                    print(f"  Видео: переключение на libx264, bitrate {self._bitrate}")
                    self._proc.stdin.write(frame.tobytes())
                else:
                    print("  Видео: переключение на OpenCV mp4v, bitrate не гарантируется")
                    self._start_cv2_fallback(frame.shape[1], frame.shape[0])
                    self._fallback.write(frame)
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
            try:
                self._proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                print("  Предупреждение: ffmpeg не завершился за 60 сек, принудительное завершение")
                self._proc.kill()
                self._proc.wait()
            if self._proc.returncode != 0:
                stderr_out = ""
                if self._proc.stderr:
                    try:
                        stderr_out = self._proc.stderr.read().decode("utf-8", errors="replace").strip()
                    except OSError:
                        pass
                print(f"  Предупреждение: ffmpeg завершился с кодом {self._proc.returncode} — видео может быть повреждено")
                if stderr_out:
                    print(f"  ffmpeg stderr: {stderr_out}")
            self._proc = None
        if self._fallback:
            self._fallback.release()
            self._fallback = None
