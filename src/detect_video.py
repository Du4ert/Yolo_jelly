"""
Детекция желетелых на видео с экспортом результатов.
"""

import cv2
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from ultralytics import YOLO


# Соответствие классов
CLASS_NAMES = {
    0: 'Aurelia aurita',
    1: 'Beroe ovata',
    2: 'Mnemiopsis leidyi',
    3: 'Pleurobrachia pileus',
    4: 'Rhizostoma pulmo'
}


def load_ctd_data(ctd_path: str) -> pd.DataFrame:
    """
    Загружает данные CTD из CSV файла.
    
    Ожидаемые колонки: time, depth, temperature, salinity (опционально)
    
    Args:
        ctd_path: путь к CSV файлу
    
    Returns:
        DataFrame с данными CTD
    """
    df = pd.read_csv(ctd_path)
    required_cols = ['time', 'depth']
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Отсутствует обязательная колонка: {col}")
    
    return df


def interpolate_ctd(ctd_data: pd.DataFrame, timestamp: float, param: str) -> Optional[float]:
    """
    Интерполирует параметр CTD по времени.
    
    Args:
        ctd_data: DataFrame с данными CTD
        timestamp: временная метка (секунды)
        param: имя параметра (depth, temperature, salinity)
    
    Returns:
        Интерполированное значение или None
    """
    if param not in ctd_data.columns:
        return None
    
    return float(np.interp(timestamp, ctd_data['time'], ctd_data[param]))


def detect_on_video(
    video_path: str,
    model_path: str,
    output_video: Optional[str] = None,
    output_csv: Optional[str] = None,
    conf_threshold: float = 0.25,
    depth_rate: Optional[float] = None,
    ctd_path: Optional[str] = None,
    save_video: bool = True
) -> pd.DataFrame:
    """
    Запускает детекцию на видео и экспортирует результаты.
    
    Args:
        video_path: путь к входному видео
        model_path: путь к модели YOLO
        output_video: путь к выходному видео (или None)
        output_csv: путь к CSV с детекциями (или None)
        conf_threshold: порог уверенности
        depth_rate: скорость погружения м/с (если нет CTD)
        ctd_path: путь к CSV с данными CTD
        save_video: сохранять ли видео с разметкой
    
    Returns:
        DataFrame с детекциями
    """
    # Загрузка модели
    print(f"Загрузка модели: {model_path}")
    model = YOLO(model_path)
    
    # Загрузка CTD данных
    ctd_data = None
    if ctd_path:
        print(f"Загрузка CTD данных: {ctd_path}")
        ctd_data = load_ctd_data(ctd_path)
    
    # Открытие видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print()
    print("="*60)
    print("ДЕТЕКЦИЯ НА ВИДЕО")
    print("="*60)
    print(f"Видео: {video_path}")
    print(f"  Разрешение: {width}x{height}")
    print(f"  FPS: {fps:.1f}")
    print(f"  Длительность: {duration:.1f} сек ({total_frames} кадров)")
    print(f"Порог уверенности: {conf_threshold}")
    if depth_rate:
        print(f"Скорость погружения: {depth_rate} м/с")
    if ctd_data is not None:
        print(f"CTD данные загружены: {len(ctd_data)} записей")
    print("="*60)
    print()
    
    # Подготовка выходного видео
    out = None
    if save_video and output_video:
        output_path = Path(output_video)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Список детекций
    detections = []
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = frame_count / fps if fps > 0 else 0
        
        # Определение глубины
        depth = None
        temperature = None
        salinity = None
        
        if ctd_data is not None:
            depth = interpolate_ctd(ctd_data, timestamp, 'depth')
            temperature = interpolate_ctd(ctd_data, timestamp, 'temperature')
            salinity = interpolate_ctd(ctd_data, timestamp, 'salinity')
        elif depth_rate:
            depth = timestamp * depth_rate
        
        # Детекция
        results = model(frame, conf=conf_threshold, verbose=False)[0]
        
        # Обработка результатов
        for box in results.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            
            # Нормализованные координаты
            xywhn = box.xywhn[0]
            x_center = float(xywhn[0])
            y_center = float(xywhn[1])
            bbox_width = float(xywhn[2])
            bbox_height = float(xywhn[3])
            
            det = {
                'frame': frame_count,
                'timestamp_s': round(timestamp, 3),
                'depth_m': round(depth, 2) if depth is not None else None,
                'temperature_c': round(temperature, 2) if temperature is not None else None,
                'salinity_psu': round(salinity, 2) if salinity is not None else None,
                'class_id': class_id,
                'class_name': CLASS_NAMES.get(class_id, f'unknown_{class_id}'),
                'confidence': round(confidence, 3),
                'x_center': round(x_center, 4),
                'y_center': round(y_center, 4),
                'width': round(bbox_width, 4),
                'height': round(bbox_height, 4),
                'bbox_area_norm': round(bbox_width * bbox_height, 6)
            }
            detections.append(det)
        
        # Отрисовка на кадре
        if out is not None:
            annotated_frame = results.plot(
                line_width=2,
                font_size=0.6,
                labels=True,
                conf=True
            )
            
            # Добавляем информацию
            info_text = f"Time: {timestamp:.2f}s | Frame: {frame_count}"
            if depth is not None:
                info_text += f" | Depth: {depth:.1f}m"
            
            cv2.putText(
                annotated_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Количество детекций на кадре
            det_count = len(results.boxes)
            if det_count > 0:
                cv2.putText(
                    annotated_frame,
                    f"Detections: {det_count}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            
            out.write(annotated_frame)
        
        frame_count += 1
        
        # Прогресс
        if frame_count % 100 == 0:
            progress = frame_count / total_frames * 100
            print(f"  Прогресс: {progress:.1f}% ({frame_count}/{total_frames}), детекций: {len(detections)}")
    
    # Освобождение ресурсов
    cap.release()
    if out is not None:
        out.release()
    
    # Создание DataFrame
    df = pd.DataFrame(detections)
    
    # Сохранение CSV
    if output_csv:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"\nCSV сохранён: {csv_path}")
    
    if output_video and save_video:
        print(f"Видео сохранено: {output_video}")
    
    # Вывод статистики
    print()
    print("="*60)
    print("РЕЗУЛЬТАТЫ")
    print("="*60)
    print(f"Всего детекций: {len(df)}")
    print(f"Кадров с детекциями: {df['frame'].nunique()}")
    
    if len(df) > 0:
        print()
        print("Детекции по видам:")
        for class_name in df['class_name'].unique():
            class_df = df[df['class_name'] == class_name]
            print(f"  {class_name}:")
            print(f"    Количество: {len(class_df)}")
            print(f"    Средняя уверенность: {class_df['confidence'].mean():.3f}")
            if class_df['depth_m'].notna().any():
                print(f"    Глубины: {class_df['depth_m'].min():.1f} - {class_df['depth_m'].max():.1f} м")
    
    print("="*60)
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Детекция желетелых на видео"
    )
    parser.add_argument(
        "--video", "-v",
        required=True,
        help="Путь к входному видео"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Путь к модели YOLO (.pt)"
    )
    parser.add_argument(
        "--output", "-o",
        default="output/detected.mp4",
        help="Путь к выходному видео (по умолчанию: output/detected.mp4)"
    )
    parser.add_argument(
        "--csv",
        default="output/detections.csv",
        help="Путь к CSV с детекциями (по умолчанию: output/detections.csv)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Порог уверенности (по умолчанию: 0.25)"
    )
    parser.add_argument(
        "--depth-rate",
        type=float,
        default=None,
        help="Скорость погружения м/с (если нет CTD данных)"
    )
    parser.add_argument(
        "--ctd",
        default=None,
        help="Путь к CSV с данными CTD"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Не сохранять видео (только CSV)"
    )
    
    args = parser.parse_args()
    
    try:
        detect_on_video(
            video_path=args.video,
            model_path=args.model,
            output_video=args.output,
            output_csv=args.csv,
            conf_threshold=args.conf,
            depth_rate=args.depth_rate,
            ctd_path=args.ctd,
            save_video=not args.no_video
        )
        return 0
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
