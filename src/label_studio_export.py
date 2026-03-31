"""
Экспорт кадров и предразметки в формате Label Studio.

Сохраняет кадры с детекциями в папки по классам (по наивысшему приоритету)
и генерирует JSON-файл предразметки для импорта в Label Studio.
"""

import json
import uuid
from datetime import date
from pathlib import Path
from typing import Optional, Set

import cv2
import numpy as np

from constants import CLASS_NAMES, CLASS_PRIORITY


def _class_folder_name(class_name: str) -> str:
    """Aurelia aurita -> Aurelia_aurita"""
    return class_name.replace(' ', '_')


def _yolo_to_ls(x_center: float, y_center: float,
                bbox_width: float, bbox_height: float) -> dict:
    """Конвертация нормализованных YOLO-координат (center, 0-1) в Label Studio (top-left, 0-100)."""
    x = max(0.0, (x_center - bbox_width / 2)) * 100
    y = max(0.0, (y_center - bbox_height / 2)) * 100
    w = min(bbox_width * 100, 100.0 - x)
    h = min(bbox_height * 100, 100.0 - y)
    return {'x': round(x, 2), 'y': round(y, 2),
            'width': round(w, 2), 'height': round(h, 2)}


class LabelStudioExporter:
    """Накапливает кадры и детекции, в конце записывает JSON для Label Studio."""

    def __init__(
        self,
        output_dir: str,
        video_name: str = "",
        model_version: str = "yolo_v8",
        image_quality: int = 95,
        frame_interval: int = 1,
        export_classes: Optional[Set[str]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.video_name = video_name
        self.model_version = model_version
        self.image_quality = image_quality
        self.frame_interval = max(1, frame_interval)
        self.export_classes = export_classes  # None = все классы
        self.date_str = date.today().isoformat()  # 2026-03-31

        # Создаём папки для экспортируемых классов
        for class_name in CLASS_NAMES.values():
            if self.export_classes is None or class_name in self.export_classes:
                (self.output_dir / _class_folder_name(class_name)).mkdir(parents=True, exist_ok=True)

        # Накопитель для JSON
        self._annotations: list[dict] = []

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
        detections: list[dict],
    ) -> None:
        """
        Обработка одного кадра с детекциями.

        Args:
            frame: BGR-кадр из OpenCV
            frame_number: номер кадра в видео
            timestamp: временная метка кадра (секунды)
            detections: список детекций текущего кадра (dict с ключами
                        class_name, confidence, x_center, y_center, width, height, track_id)
        """
        if not detections:
            return

        # Фильтрация по выбранным классам
        if self.export_classes is not None:
            detections = [d for d in detections if d['class_name'] in self.export_classes]
            if not detections:
                return

        # Прореживание по интервалу кадров
        if self.frame_interval > 1 and frame_number % self.frame_interval != 0:
            return

        # Определяем класс с наивысшим приоритетом
        best_det = max(detections, key=lambda d: CLASS_PRIORITY.get(d['class_name'], 0))
        folder_name = _class_folder_name(best_det['class_name'])

        # track_id наиболее приоритетной детекции
        track_id = best_det.get('track_id') or 0

        # Формируем имя файла: T005-F000042-GH012345-2026-03-31.jpg
        filename = f"T{track_id:03d}-F{frame_number:06d}-{self.video_name}-{self.date_str}.jpg"
        rel_path = f"{folder_name}/{filename}"
        abs_path = self.output_dir / rel_path

        # Сохраняем кадр
        cv2.imwrite(str(abs_path), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.image_quality])

        # Собираем Label Studio результаты для всех детекций на кадре
        results = []
        for det in detections:
            coords = _yolo_to_ls(
                det['x_center'], det['y_center'],
                det['width'], det['height'],
            )
            results.append({
                'id': uuid.uuid4().hex[:10],
                'from_name': 'label',
                'to_name': 'image',
                'type': 'rectanglelabels',
                'value': {
                    **coords,
                    'rotation': 0,
                    'rectanglelabels': [det['class_name']],
                },
                'score': det['confidence'],
            })

        self._annotations.append({
            'data': {'image': rel_path},
            'predictions': [{
                'model_version': self.model_version,
                'result': results,
            }],
        })

    def finalize(self) -> str:
        """Записывает preannotations.json и возвращает путь к нему."""
        json_path = self.output_dir / 'preannotations.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self._annotations, f, ensure_ascii=False, indent=2)
        return str(json_path)
