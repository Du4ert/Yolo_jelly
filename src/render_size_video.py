"""
Рендеринг видео с информацией о размерах объектов.

Использует:
- detected.mp4 - видео с рамками детекций
- detections.csv - базовые детекции (координаты рамок)
- detections_with_size.csv - детекции с рассчитанными размерами
- geometry.csv - данные по углам наклона камеры

Выводит:
- Под рамками объектов: distance_to_object_m, estimated_size_cm
- В левом нижнем углу: углы наклона камеры (tilt_horizontal_deg, tilt_vertical_deg)
"""

import cv2
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
import time

from constants import CLASS_COLORS_BGR


def load_size_data(detections_csv: str, size_csv: str) -> pd.DataFrame:
    """
    Загружает и объединяет данные детекций и размеров.
    
    Args:
        detections_csv: путь к CSV с базовыми детекциями
        size_csv: путь к CSV с детекциями и размерами
    
    Returns:
        DataFrame с объединёнными данными
    """
    # Загружаем детекции с размерами как основу
    if Path(size_csv).exists():
        df = pd.read_csv(size_csv)
    else:
        # Если файла с размерами нет, используем базовые детекции
        df = pd.read_csv(detections_csv)
        df['distance_to_object_m'] = None
        df['estimated_size_cm'] = None
    
    return df


def load_geometry_data(geometry_csv: str) -> Optional[pd.DataFrame]:
    """
    Загружает данные геометрии камеры.
    
    Args:
        geometry_csv: путь к CSV с данными геометрии
    
    Returns:
        DataFrame с данными геометрии или None
    """
    if geometry_csv and Path(geometry_csv).exists():
        return pd.read_csv(geometry_csv)
    return None


def get_tilt_for_frame(frame: int, geometry_df: Optional[pd.DataFrame]) -> Tuple[float, float]:
    """
    Получает углы наклона камеры для заданного кадра.
    
    Args:
        frame: номер кадра
        geometry_df: DataFrame с данными геометрии
    
    Returns:
        (tilt_horizontal_deg, tilt_vertical_deg)
    """
    if geometry_df is None or len(geometry_df) == 0:
        return 0.0, 0.0
    
    # Находим интервал, содержащий кадр
    mask = (geometry_df['frame_start'] <= frame) & (geometry_df['frame_end'] >= frame)
    matching = geometry_df[mask]
    
    if len(matching) == 0:
        # Кадр вне диапазона - берём ближайший интервал
        geometry_df = geometry_df.copy()
        geometry_df['dist_to_frame'] = geometry_df.apply(
            lambda r: min(abs(r['frame_start'] - frame), abs(r['frame_end'] - frame)), axis=1
        )
        matching = geometry_df.nsmallest(1, 'dist_to_frame')
    
    if len(matching) == 0:
        return 0.0, 0.0
    
    row = matching.iloc[0]
    
    # Проверяем уверенность (выбросы фильтруются)
    confidence = row.get('confidence', 1.0)
    if confidence < 0.5:
        return 0.0, 0.0
    
    tilt_h = row.get('tilt_horizontal_deg', 0.0)
    tilt_v = row.get('tilt_vertical_deg', 0.0)
    
    return float(tilt_h), float(tilt_v)


def draw_size_info(
    frame: np.ndarray,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    distance_m: Optional[float],
    size_cm: Optional[float],
    class_name: str,
    frame_width: int,
    frame_height: int
) -> np.ndarray:
    """
    Рисует информацию о размере под рамкой объекта.
    
    Args:
        frame: кадр изображения
        x_center, y_center: нормализованные координаты центра
        width, height: нормализованные размеры рамки
        distance_m: дистанция до объекта (м)
        size_cm: размер объекта (см)
        class_name: название класса
        frame_width, frame_height: размеры кадра в пикселях
    
    Returns:
        Модифицированный кадр
    """
    # Конвертируем нормализованные координаты в пиксели
    cx = int(x_center * frame_width)
    cy = int(y_center * frame_height)
    w = int(width * frame_width)
    h = int(height * frame_height)
    
    # Нижняя граница рамки
    bottom_y = cy + h // 2
    
    # Формируем текст
    text_parts = []
    if distance_m is not None and not pd.isna(distance_m):
        text_parts.append(f"{distance_m:.2f}m")
    if size_cm is not None and not pd.isna(size_cm):
        text_parts.append(f"{size_cm:.1f}cm")
    
    if not text_parts:
        return frame
    
    text = " | ".join(text_parts)
    
    # Параметры текста
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    # Получаем размер текста для центрирования
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Позиция текста (центрированно под рамкой)
    text_x = cx - text_width // 2
    text_y = bottom_y + text_height + 8  # Небольшой отступ под рамкой
    
    # Убедимся, что текст не выходит за границы кадра
    text_x = max(5, min(text_x, frame_width - text_width - 5))
    text_y = min(text_y, frame_height - 5)
    
    # Цвет текста по классу
    color = CLASS_COLORS_BGR.get(class_name, (255, 255, 255))
    
    # Фон для текста (полупрозрачный чёрный)
    bg_x1 = text_x - 3
    bg_y1 = text_y - text_height - 3
    bg_x2 = text_x + text_width + 3
    bg_y2 = text_y + baseline + 3
    
    # Рисуем фон
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Рисуем текст
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    return frame


def draw_tilt_info(
    frame: np.ndarray,
    tilt_h: float,
    tilt_v: float,
    frame_width: int,
    frame_height: int
) -> np.ndarray:
    """
    Рисует информацию об углах наклона камеры в левом нижнем углу.
    
    Args:
        frame: кадр изображения
        tilt_h: горизонтальный наклон (градусы)
        tilt_v: вертикальный наклон (градусы)
        frame_width, frame_height: размеры кадра
    
    Returns:
        Модифицированный кадр
    """
    # Вычисляем полный наклон
    total_tilt = np.sqrt(tilt_h**2 + tilt_v**2)
    
    # Формируем текст
    text_lines = [
        f"Tilt H: {tilt_h:+.1f}°",
        f"Tilt V: {tilt_v:+.1f}°",
        f"Total: {total_tilt:.1f}°"
    ]
    
    # Параметры текста
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    line_height = 25
    
    # Позиция в левом нижнем углу
    margin = 15
    start_y = frame_height - margin - (len(text_lines) - 1) * line_height
    
    # Вычисляем максимальную ширину текста для фона
    max_width = 0
    for text in text_lines:
        (text_width, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
        max_width = max(max_width, text_width)
    
    # Рисуем полупрозрачный фон
    bg_x1 = margin - 5
    bg_y1 = start_y - 20
    bg_x2 = margin + max_width + 10
    bg_y2 = frame_height - margin + 10
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Цвет в зависимости от величины наклона
    if total_tilt < 15:
        color = (0, 255, 0)  # Зелёный - хороший наклон
    elif total_tilt < 30:
        color = (0, 255, 255)  # Жёлтый - умеренный наклон
    else:
        color = (0, 0, 255)  # Красный - большой наклон
    
    # Рисуем текст
    for i, text in enumerate(text_lines):
        y = start_y + i * line_height
        cv2.putText(frame, text, (margin, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    return frame


def render_size_video(
    input_video: str,
    detections_csv: str,
    size_csv: str,
    geometry_csv: Optional[str] = None,
    output_video: Optional[str] = None,
    verbose: bool = True,
    progress_callback=None,
) -> str:
    """
    Рендерит видео с информацией о размерах объектов.
    
    Args:
        input_video: путь к входному видео (detected.mp4)
        detections_csv: путь к CSV с базовыми детекциями
        size_csv: путь к CSV с детекциями и размерами (detections_with_size.csv)
        geometry_csv: путь к CSV с данными геометрии (geometry.csv)
        output_video: путь к выходному видео (по умолчанию: добавляется суффикс _sized)
        verbose: выводить ли информацию о прогрессе
    
    Returns:
        Путь к выходному видео
    """
    # Определяем выходной файл
    if output_video is None:
        input_path = Path(input_video)
        output_video = str(input_path.parent / f"{input_path.stem}_sized{input_path.suffix}")
    
    # Загружаем данные
    if verbose:
        print("Загрузка данных...")
    
    df = load_size_data(detections_csv, size_csv)
    geometry_df = load_geometry_data(geometry_csv)
    
    if verbose:
        print(f"  Детекций: {len(df)}")
        print(f"  С размерами: {df['estimated_size_cm'].notna().sum()}")
        if geometry_df is not None:
            print(f"  Интервалов геометрии: {len(geometry_df)}")
    
    # Открываем видео
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {input_video}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if verbose:
        print(f"\nВидео: {input_video}")
        print(f"  Разрешение: {frame_width}x{frame_height}")
        print(f"  FPS: {fps:.1f}")
        print(f"  Кадров: {total_frames}")
    
    # Создаём выходное видео
    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
    
    # Группируем детекции по кадрам для быстрого доступа
    frame_detections = df.groupby('frame')
    
    # Обработка кадров
    frame_idx = 0
    start_time = time.time()
    
    if verbose:
        print(f"\nРендеринг...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Получаем углы наклона для текущего кадра
        tilt_h, tilt_v = get_tilt_for_frame(frame_idx, geometry_df)
        
        # Рисуем информацию о наклоне
        frame = draw_tilt_info(frame, tilt_h, tilt_v, frame_width, frame_height)
        
        # Получаем детекции для текущего кадра
        if frame_idx in frame_detections.groups:
            detections = frame_detections.get_group(frame_idx)
            
            for _, det in detections.iterrows():
                frame = draw_size_info(
                    frame,
                    x_center=det['x_center'],
                    y_center=det['y_center'],
                    width=det['width'],
                    height=det['height'],
                    distance_m=det.get('distance_to_object_m'),
                    size_cm=det.get('estimated_size_cm'),
                    class_name=det['class_name'],
                    frame_width=frame_width,
                    frame_height=frame_height
                )
        
        out.write(frame)
        frame_idx += 1

        # Прогресс
        if progress_callback is not None and frame_idx % 100 == 0:
            progress_callback(frame_idx, total_frames)
        if verbose and frame_idx % 500 == 0:
            progress = frame_idx / total_frames * 100
            elapsed = time.time() - start_time
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0
            remaining = (total_frames - frame_idx) / fps_proc if fps_proc > 0 else 0
            print(f"  {progress:.1f}% ({frame_idx}/{total_frames}), "
                  f"{fps_proc:.1f} FPS, осталось ~{remaining:.0f}с")
    
    # Завершение
    cap.release()
    out.release()
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\nГотово!")
        print(f"  Время: {elapsed:.1f} сек")
        print(f"  Выходной файл: {output_video}")
    
    return output_video


def main():
    """CLI интерфейс."""
    parser = argparse.ArgumentParser(
        description="Рендеринг видео с информацией о размерах объектов"
    )
    parser.add_argument(
        "--video", "-v",
        required=True,
        help="Путь к входному видео (detected.mp4)"
    )
    parser.add_argument(
        "--detections", "-d",
        required=True,
        help="Путь к CSV с базовыми детекциями (detections.csv)"
    )
    parser.add_argument(
        "--size", "-s",
        required=True,
        help="Путь к CSV с размерами (detections_with_size.csv)"
    )
    parser.add_argument(
        "--geometry", "-g",
        default=None,
        help="Путь к CSV с геометрией камеры (geometry.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Путь к выходному видео (по умолчанию: input_sized.mp4)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Тихий режим (без вывода прогресса)"
    )
    
    args = parser.parse_args()
    
    try:
        render_size_video(
            input_video=args.video,
            detections_csv=args.detections,
            size_csv=args.size,
            geometry_csv=args.geometry,
            output_video=args.output,
            verbose=not args.quiet
        )
        return 0
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
