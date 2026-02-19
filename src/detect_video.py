"""
Детекция желетелых на видео с экспортом результатов и трекингом объектов.
"""

import cv2
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
import time
from ultralytics import YOLO

from constants import CLASS_NAMES


def load_ctd_data(ctd_path: str) -> pd.DataFrame:
    """
    Загружает данные CTD из CSV файла.
    
    Ожидаемые колонки: time, depth, temperature, salinity (опционально)
    Поддерживаемые разделители: запятая, точка с запятой, табуляция, pipe (|)
    Регистр колонок не имеет значения (Time, TIME, time все принимаются)
    
    Args:
        ctd_path: путь к CSV файлу
    
    Returns:
        DataFrame с данными CTD
    """
    # Определяем разделитель автоматически
    with open(ctd_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
    
    # Проверяем возможные разделители
    for sep in ['|', ';', '\t', ',']:
        if sep in first_line:
            delimiter = sep
            break
    else:
        delimiter = ','  # по умолчанию
    
    df = pd.read_csv(ctd_path, sep=delimiter)
    
    # Нормализуем имена колонок к нижнему регистру
    df.columns = df.columns.str.lower()
    
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


def generate_color_for_id(track_id: int) -> Tuple[int, int, int]:
    """
    Генерирует уникальный цвет для ID трека.
    
    Args:
        track_id: ID трека
    
    Returns:
        Кортеж (B, G, R) для OpenCV
    """
    # Используем golden ratio для равномерного распределения цветов
    golden_ratio = 0.618033988749895
    hue = (track_id * golden_ratio) % 1.0
    
    # Конвертация HSV в RGB
    h = hue * 6
    c = 1.0
    x = c * (1 - abs(h % 2 - 1))
    
    if h < 1:
        r, g, b = c, x, 0
    elif h < 2:
        r, g, b = x, c, 0
    elif h < 3:
        r, g, b = 0, c, x
    elif h < 4:
        r, g, b = 0, x, c
    elif h < 5:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    return (int(b * 255), int(g * 255), int(r * 255))


def detect_on_video(
    video_path: str,
    model_path: str,
    output_video: Optional[str] = None,
    output_csv: Optional[str] = None,
    conf_threshold: float = 0.25,
    depth_rate: Optional[float] = None,
    ctd_path: Optional[str] = None,
    save_video: bool = True,
    enable_tracking: bool = False,
    tracker_type: str = "bytetrack.yaml",
    show_trails: bool = False,
    trail_length: int = 30,
    min_track_length: int = 3,
    use_dominant_class: bool = True
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
        enable_tracking: включить трекинг объектов
        tracker_type: тип трекера (bytetrack.yaml или botsort.yaml)
        show_trails: показывать траектории движения объектов
        trail_length: длина траектории в кадрах
        min_track_length: минимальная длина трека в кадрах (короче игнорируются)
        use_dominant_class: присваивать всем детекциям трека доминирующий класс
    
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
    if enable_tracking:
        print(f"Трекинг: включен (тип: {tracker_type})")
        if show_trails:
            print(f"  Отображение траекторий: да (длина: {trail_length} кадров)")
    else:
        print(f"Трекинг: выключен")
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
    
    # Данные для трекинга
    track_history = defaultdict(list)  # track_id -> [(x, y), ...]
    track_info = {}  # track_id -> {'class_id', 'class_name', 'first_frame', 'last_frame'}
    track_class_votes = defaultdict(list)  # track_id -> [class_id, class_id, ...] для определения доминирующего класса
    
    frame_count = 0
    start_time = time.time()
    
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
        
        # Детекция или трекинг
        if enable_tracking:
            results = model.track(
                frame, 
                conf=conf_threshold, 
                persist=True,
                tracker=tracker_type,
                verbose=False
            )[0]
        else:
            results = model(frame, conf=conf_threshold, verbose=False)[0]
        
        # Обработка результатов
        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                
                # Нормализованные координаты
                xywhn = box.xywhn[0]
                x_center = float(xywhn[0])
                y_center = float(xywhn[1])
                bbox_width = float(xywhn[2])
                bbox_height = float(xywhn[3])
                
                # ID трека (если трекинг включен)
                track_id = None
                if enable_tracking and box.id is not None:
                    track_id = int(box.id)
                    
                    # Обновляем историю трека (для визуализации траектории)
                    # Используем абсолютные координаты для рисования
                    center_x = int(x_center * width)
                    center_y = int(y_center * height)
                    track_history[track_id].append((center_x, center_y))
                    
                    # Ограничиваем длину истории
                    if len(track_history[track_id]) > trail_length:
                        track_history[track_id] = track_history[track_id][-trail_length:]
                    
                    # Сохраняем информацию о треке
                    track_class_votes[track_id].append(class_id)  # Для голосования за класс
                    
                    if track_id not in track_info:
                        track_info[track_id] = {
                            'class_id': class_id,
                            'class_name': CLASS_NAMES.get(class_id, f'unknown_{class_id}'),
                            'first_frame': frame_count,
                            'last_frame': frame_count,
                            'first_timestamp': timestamp,
                            'last_timestamp': timestamp,
                            'first_depth': depth,
                            'last_depth': depth
                        }
                    else:
                        track_info[track_id]['last_frame'] = frame_count
                        track_info[track_id]['last_timestamp'] = timestamp
                        track_info[track_id]['last_depth'] = depth
                
                det = {
                    'frame': frame_count,
                    'timestamp_s': round(timestamp, 3),
                    'depth_m': round(depth, 2) if depth is not None else None,
                    'temperature_c': round(temperature, 2) if temperature is not None else None,
                    'salinity_psu': round(salinity, 2) if salinity is not None else None,
                    'track_id': track_id,
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
            
            # Рисуем траектории, если включено
            if enable_tracking and show_trails:
                for track_id, trail in track_history.items():
                    if len(trail) > 1:
                        # Цвет для трека
                        color = generate_color_for_id(track_id)
                        
                        # Рисуем линию траектории
                        points = np.array(trail, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(
                            annotated_frame,
                            [points],
                            isClosed=False,
                            color=color,
                            thickness=2
                        )
                        
                        # Рисуем точки на траектории
                        for i, point in enumerate(trail):
                            # Размер точки зависит от "возраста" (новые больше)
                            alpha = (i + 1) / len(trail)
                            radius = int(3 * alpha) + 1
                            cv2.circle(
                                annotated_frame,
                                point,
                                radius,
                                color,
                                -1
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
            det_count = len(boxes) if boxes is not None else 0
            if det_count > 0:
                if enable_tracking:
                    active_tracks = len([tid for tid, trail in track_history.items() 
                                       if len(trail) > 0 and trail[-1] == (int(boxes[0].xywhn[0][0] * width), 
                                                                           int(boxes[0].xywhn[0][1] * height))])
                    cv2.putText(
                        annotated_frame,
                        f"Detections: {det_count} | Active tracks: {len(track_info)}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                else:
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
            if enable_tracking:
                print(f"  Прогресс: {progress:.1f}% ({frame_count}/{total_frames}), "
                      f"детекций: {len(detections)}, треков: {len(track_info)}")
            else:
                print(f"  Прогресс: {progress:.1f}% ({frame_count}/{total_frames}), "
                      f"детекций: {len(detections)}")
    
    # Замер времени
    end_time = time.time()
    processing_time = end_time - start_time
    time_per_frame = processing_time / frame_count if frame_count > 0 else 0
    fps_processing = frame_count / processing_time if processing_time > 0 else 0
    
    # Освобождение ресурсов
    cap.release()
    if out is not None:
        out.release()
    
    # Создание DataFrame
    df = pd.DataFrame(detections)
    
    # Постобработка треков
    if enable_tracking and len(df) > 0:
        # 1. Определяем доминирующий класс для каждого трека
        track_dominant_class = {}
        for track_id, class_votes in track_class_votes.items():
            # Находим класс, который встречается чаще всего
            class_counts = Counter(class_votes)
            dominant_class_id = class_counts.most_common(1)[0][0]
            track_dominant_class[track_id] = dominant_class_id
            
            # Обновляем track_info
            track_info[track_id]['class_id'] = dominant_class_id
            track_info[track_id]['class_name'] = CLASS_NAMES.get(dominant_class_id, f'unknown_{dominant_class_id}')
        
        # 2. Фильтруем короткие треки и обновляем классы
        valid_tracks = set()
        short_tracks = set()
        
        for track_id, info in track_info.items():
            track_length = len(track_class_votes[track_id])
            if track_length >= min_track_length:
                valid_tracks.add(track_id)
            else:
                short_tracks.add(track_id)
        
        if short_tracks:
            print(f"\nОтфильтровано коротких треков (< {min_track_length} кадров): {len(short_tracks)}")
        
        # 3. Применяем фильтры к DataFrame
        # Удаляем детекции из коротких треков
        df = df[df['track_id'].isin(valid_tracks) | df['track_id'].isna()].copy()
        
        # 4. Обновляем классы на доминирующие (если включено)
        if use_dominant_class:
            def update_class(row):
                if pd.notna(row['track_id']) and row['track_id'] in track_dominant_class:
                    dominant_id = track_dominant_class[row['track_id']]
                    row['class_id'] = dominant_id
                    row['class_name'] = CLASS_NAMES.get(dominant_id, f'unknown_{dominant_id}')
                return row
            
            df = df.apply(update_class, axis=1)
            
            # Подсчитываем треки с изменёнными классами
            changed_tracks = 0
            for track_id in valid_tracks:
                class_votes = track_class_votes[track_id]
                if len(set(class_votes)) > 1:  # Было более одного класса
                    changed_tracks += 1
            
            if changed_tracks > 0:
                print(f"Треков с исправленным классом (доминирующий): {changed_tracks}")
        
        # 5. Удаляем короткие треки из track_info
        for track_id in short_tracks:
            del track_info[track_id]
    
    # Сохранение CSV
    if output_csv:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"\nCSV сохранён: {csv_path}")
        
        # Сохранение статистики по трекам
        if enable_tracking and len(track_info) > 0:
            track_stats = []
            for track_id, info in track_info.items():
                duration = info['last_timestamp'] - info['first_timestamp']
                frame_span = info['last_frame'] - info['first_frame'] + 1
                
                # Подсчет детекций для этого трека
                track_detections = df[df['track_id'] == track_id]
                
                stat = {
                    'track_id': track_id,
                    'class_id': info['class_id'],
                    'class_name': info['class_name'],
                    'first_frame': info['first_frame'],
                    'last_frame': info['last_frame'],
                    'frame_span': frame_span,
                    'duration_s': round(duration, 2),
                    'first_timestamp_s': round(info['first_timestamp'], 2),
                    'last_timestamp_s': round(info['last_timestamp'], 2),
                    'first_depth_m': round(info['first_depth'], 2) if info['first_depth'] is not None else None,
                    'last_depth_m': round(info['last_depth'], 2) if info['last_depth'] is not None else None,
                    'detections_count': len(track_detections),
                    'avg_confidence': round(track_detections['confidence'].mean(), 3)
                }
                
                # Вычисляем перемещение по глубине
                if info['first_depth'] is not None and info['last_depth'] is not None:
                    stat['depth_change_m'] = round(info['last_depth'] - info['first_depth'], 2)
                else:
                    stat['depth_change_m'] = None
                
                track_stats.append(stat)
            
            track_df = pd.DataFrame(track_stats)
            track_csv_path = csv_path.parent / (csv_path.stem + '_tracks.csv')
            track_df.to_csv(track_csv_path, index=False)
            print(f"Статистика треков сохранена: {track_csv_path}")
    
    if output_video and save_video:
        print(f"Видео сохранено: {output_video}")
    
    # Вывод статистики
    print()
    print("="*60)
    print("РЕЗУЛЬТАТЫ")
    print("="*60)
    print(f"Время обработки: {processing_time:.1f} сек ({processing_time/60:.1f} мин)")
    print(f"Скорость: {fps_processing:.1f} FPS ({time_per_frame*1000:.1f} мс/кадр)")
    print()
    print(f"Всего детекций: {len(df)}")
    print(f"Кадров с детекциями: {df['frame'].nunique()}")
    print(f"Обработано кадров: {frame_count}")
    
    if enable_tracking:
        print(f"Всего треков: {len(track_info)}")
        print()
        print("Статистика треков:")
        for class_name in sorted(set(info['class_name'] for info in track_info.values())):
            class_tracks = [tid for tid, info in track_info.items() if info['class_name'] == class_name]
            print(f"  {class_name}: {len(class_tracks)} треков")
            
            if class_tracks:
                durations = [track_info[tid]['last_timestamp'] - track_info[tid]['first_timestamp'] 
                           for tid in class_tracks]
                avg_duration = sum(durations) / len(durations)
                print(f"    Средняя длительность: {avg_duration:.2f} сек")
    
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
            if enable_tracking:
                unique_tracks = class_df['track_id'].nunique()
                print(f"    Уникальных треков: {unique_tracks}")
    
    print("="*60)
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Детекция желетелых на видео с поддержкой трекинга"
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
    parser.add_argument(
        "--track",
        action="store_true",
        help="Включить трекинг объектов"
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        choices=["bytetrack.yaml", "botsort.yaml"],
        help="Тип трекера (по умолчанию: bytetrack.yaml)"
    )
    parser.add_argument(
        "--show-trails",
        action="store_true",
        help="Показывать траектории движения объектов"
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=30,
        help="Длина траектории в кадрах (по умолчанию: 30)"
    )
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=3,
        help="Минимальная длина трека в кадрах (короче игнорируются, по умолчанию: 3)"
    )
    parser.add_argument(
        "--no-dominant-class",
        action="store_true",
        help="Не присваивать доминирующий класс детекциям трека"
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
            save_video=not args.no_video,
            enable_tracking=args.track,
            tracker_type=args.tracker,
            show_trails=args.show_trails,
            trail_length=args.trail_length,
            min_track_length=args.min_track_length,
            use_dominant_class=not args.no_dominant_class
        )
        return 0
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
