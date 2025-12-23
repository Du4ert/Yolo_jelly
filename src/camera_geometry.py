"""
Модуль оценки геометрии камеры и размеров объектов.

Функции:
- Оценка наклона камеры по морскому снегу (Focus of Expansion)
- Расчёт реального размера объектов по калибровочным данным
- Коррекция измерений с учётом положения в кадре

Калибровочные данные получены для GoPro 12 Wide 4K (3840x2160).
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from scipy.optimize import minimize
import argparse


@dataclass
class CameraCalibration:
    """Калибровочные параметры камеры."""
    # Константа k = размер_пикс × дистанция для эталонного объекта
    k: float = 156.1  # пиксель·м (средний Feret для теннисного мяча)
    
    # Размер кадра
    frame_width: int = 1920
    frame_height: int = 1080
    
    # Угол обзора камеры (градусы)
    fov_horizontal: float = 100.0  # GoPro Wide
    
    # Оптимальный диапазон дистанций для измерений
    min_reliable_distance: float = 1.0  # м
    max_reliable_distance: float = 2.5  # м
    
    @property
    def pixels_per_degree(self) -> float:
        """Пикселей на градус угла обзора."""
        return self.frame_width / self.fov_horizontal
    
    @property
    def frame_center(self) -> Tuple[float, float]:
        """Центр кадра (x, y)."""
        return self.frame_width / 2, self.frame_height / 2


@dataclass
class FOEResult:
    """Результат оценки Focus of Expansion."""
    foe_x: float  # X координата FOE
    foe_y: float  # Y координата FOE
    tilt_horizontal: float  # Наклон по горизонтали (градусы, + вправо)
    tilt_vertical: float  # Наклон по вертикали (градусы, + вниз)
    confidence: float  # Уверенность оценки (0-1)
    n_vectors: int  # Количество использованных векторов


@dataclass 
class SizeEstimate:
    """Оценка реального размера объекта."""
    size_m: float  # Размер в метрах
    distance_m: float  # Дистанция до объекта
    confidence: float  # Уверенность (зависит от дистанции)
    in_reliable_range: bool  # Находится ли в оптимальном диапазоне


def estimate_foe(
    points: np.ndarray,
    vectors: np.ndarray,
    frame_size: Tuple[int, int] = (1920, 1080),
    min_vector_length: float = 0.5
) -> FOEResult:
    """
    Оценивает Focus of Expansion по набору точек и их векторов движения.
    
    FOE — точка, из которой "разлетаются" все неподвижные объекты при 
    движении камеры вперёд. Смещение FOE от центра кадра указывает 
    на наклон камеры.
    
    Args:
        points: массив координат точек (N, 2)
        vectors: массив векторов движения (N, 2)
        frame_size: размер кадра (width, height)
        min_vector_length: минимальная длина вектора для учёта
    
    Returns:
        FOEResult с координатами FOE и оценкой наклона
    """
    width, height = frame_size
    
    # Фильтруем короткие векторы (шум)
    vec_lengths = np.linalg.norm(vectors, axis=1)
    mask = vec_lengths > min_vector_length
    points_filt = points[mask]
    vectors_filt = vectors[mask]
    
    if len(points_filt) < 10:
        # Недостаточно данных
        return FOEResult(
            foe_x=width / 2,
            foe_y=height / 2,
            tilt_horizontal=0.0,
            tilt_vertical=0.0,
            confidence=0.0,
            n_vectors=len(points_filt)
        )
    
    def foe_error(foe):
        """Ошибка: насколько векторы отклоняются от радиального направления."""
        fx, fy = foe
        radial = points_filt - np.array([fx, fy])
        radial_norm = radial / (np.linalg.norm(radial, axis=1, keepdims=True) + 1e-6)
        vec_norm = vectors_filt / (np.linalg.norm(vectors_filt, axis=1, keepdims=True) + 1e-6)
        dot = np.sum(radial_norm * vec_norm, axis=1)
        return np.mean(1 - np.abs(dot))
    
    # Оптимизация — ищем FOE
    initial_foe = [width / 2, height / 2]
    result = minimize(foe_error, initial_foe, method='Nelder-Mead')
    foe_x, foe_y = result.x
    
    # Оценка качества (1 - ошибка)
    confidence = 1 - result.fun
    
    # Расчёт наклона камеры
    pixels_per_degree = width / 100.0  # примерно для GoPro Wide
    tilt_horizontal = (foe_x - width / 2) / pixels_per_degree
    tilt_vertical = (foe_y - height / 2) / pixels_per_degree
    
    return FOEResult(
        foe_x=foe_x,
        foe_y=foe_y,
        tilt_horizontal=tilt_horizontal,
        tilt_vertical=tilt_vertical,
        confidence=confidence,
        n_vectors=len(points_filt)
    )


def estimate_foe_from_frames(
    gray1: np.ndarray,
    gray2: np.ndarray,
    max_corners: int = 1000,
    min_vector_length: float = 0.5
) -> Tuple[FOEResult, np.ndarray, np.ndarray]:
    """
    Оценивает FOE по двум последовательным кадрам.
    
    Args:
        gray1: первый кадр (grayscale)
        gray2: второй кадр (grayscale)
        max_corners: максимум точек для отслеживания
        min_vector_length: минимальная длина вектора
    
    Returns:
        (FOEResult, points, vectors)
    """
    height, width = gray1.shape
    
    # Параметры детектора точек
    feature_params = dict(
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=7
    )
    
    # Параметры Lucas-Kanade
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    
    # Находим точки
    points1 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
    
    if points1 is None or len(points1) < 10:
        return FOEResult(
            foe_x=width / 2, foe_y=height / 2,
            tilt_horizontal=0, tilt_vertical=0,
            confidence=0, n_vectors=0
        ), np.array([]), np.array([])
    
    # Отслеживаем
    points2, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, points1, None, **lk_params)
    
    good_old = points1[status == 1].reshape(-1, 2)
    good_new = points2[status == 1].reshape(-1, 2)
    vectors = good_new - good_old
    
    foe_result = estimate_foe(good_old, vectors, (width, height), min_vector_length)
    
    return foe_result, good_old, vectors


def estimate_object_size(
    bbox_size_pixels: float,
    distance_m: float,
    calibration: CameraCalibration = None
) -> SizeEstimate:
    """
    Оценивает реальный размер объекта по его размеру в пикселях и дистанции.
    
    Формула: размер_реальный = размер_пикс × дистанция / K
    где K = k / эталонный_размер ≈ 2365 для GoPro 12 Wide 4K
    
    Args:
        bbox_size_pixels: размер bbox в пикселях (ширина или высота)
        distance_m: дистанция до объекта в метрах
        calibration: калибровочные параметры (или None для значений по умолчанию)
    
    Returns:
        SizeEstimate с оценкой размера
    """
    if calibration is None:
        calibration = CameraCalibration()
    
    # K = k / эталонный_размер_м
    # k = 156.1 для теннисного мяча (d = 0.066 м)
    K = calibration.k / 0.066  # ≈ 2365
    
    size_m = bbox_size_pixels * distance_m / K
    
    # Оценка уверенности на основе дистанции
    in_range = calibration.min_reliable_distance <= distance_m <= calibration.max_reliable_distance
    
    if in_range:
        confidence = 0.95
    elif distance_m < calibration.min_reliable_distance:
        # Близко — небольшое снижение точности
        confidence = 0.8
    else:
        # Далеко — точность падает
        confidence = max(0.5, 0.95 - (distance_m - calibration.max_reliable_distance) * 0.1)
    
    return SizeEstimate(
        size_m=size_m,
        distance_m=distance_m,
        confidence=confidence,
        in_reliable_range=in_range
    )


def estimate_distance_from_size(
    bbox_size_pixels: float,
    known_size_m: float,
    calibration: CameraCalibration = None
) -> float:
    """
    Оценивает дистанцию до объекта по его известному размеру.
    
    Формула: дистанция = k / размер_пикс × (известный_размер / эталон)
    
    Args:
        bbox_size_pixels: размер bbox в пикселях
        known_size_m: известный реальный размер объекта в метрах
        calibration: калибровочные параметры
    
    Returns:
        Дистанция в метрах
    """
    if calibration is None:
        calibration = CameraCalibration()
    
    # k для эталона (мяч 0.066 м) = 156.1
    # k для объекта размера S: k_obj = 156.1 × (S / 0.066)
    k_object = calibration.k * (known_size_m / 0.066)
    
    distance_m = k_object / bbox_size_pixels
    
    return distance_m


def process_video_geometry(
    video_path: str,
    output_csv: Optional[str] = None,
    frame_interval: int = 30,
    calibration: CameraCalibration = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Обрабатывает видео и оценивает геометрию камеры для каждого интервала.
    
    Args:
        video_path: путь к видео
        output_csv: путь для сохранения результатов (опционально)
        frame_interval: интервал между оценками FOE (кадры)
        calibration: калибровочные параметры
        verbose: выводить прогресс
    
    Returns:
        DataFrame с результатами по кадрам
    """
    if calibration is None:
        calibration = CameraCalibration()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Обновляем калибровку под размер видео
    calibration.frame_width = width
    calibration.frame_height = height
    
    if verbose:
        print(f"Видео: {video_path}")
        print(f"  Разрешение: {width}x{height}, FPS: {fps:.1f}")
        print(f"  Кадров: {total_frames}, интервал оценки: {frame_interval}")
    
    results = []
    
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return pd.DataFrame()
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_idx = 0
    
    # Накапливаем точки и векторы за интервал
    interval_points = []
    interval_vectors = []
    interval_start = 0
    
    # Параметры трекинга
    feature_params = dict(maxCorners=500, qualityLevel=0.01, minDistance=10, blockSize=7)
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Трекинг точек
        points1 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        if points1 is not None and len(points1) > 10:
            points2, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points1, None, **lk_params)
            
            good_old = points1[status == 1].reshape(-1, 2)
            good_new = points2[status == 1].reshape(-1, 2)
            
            interval_points.extend(good_old)
            interval_vectors.extend(good_new - good_old)
        
        # Оценка FOE на каждом интервале
        if frame_idx % frame_interval == 0 and len(interval_points) > 50:
            points_arr = np.array(interval_points)
            vectors_arr = np.array(interval_vectors)
            
            foe_result = estimate_foe(points_arr, vectors_arr, (width, height))
            
            timestamp = frame_idx / fps
            
            results.append({
                'frame_start': interval_start,
                'frame_end': frame_idx,
                'timestamp_s': round(timestamp, 2),
                'foe_x': round(foe_result.foe_x, 1),
                'foe_y': round(foe_result.foe_y, 1),
                'tilt_horizontal_deg': round(foe_result.tilt_horizontal, 1),
                'tilt_vertical_deg': round(foe_result.tilt_vertical, 1),
                'confidence': round(foe_result.confidence, 3),
                'n_vectors': foe_result.n_vectors
            })
            
            # Сброс накопителя
            interval_points = []
            interval_vectors = []
            interval_start = frame_idx
            
            if verbose and len(results) % 10 == 0:
                print(f"  Обработано: {frame_idx}/{total_frames} кадров")
        
        prev_gray = gray
    
    cap.release()
    
    df = pd.DataFrame(results)
    
    if output_csv and len(df) > 0:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        if verbose:
            print(f"\nРезультаты сохранены: {output_csv}")
    
    if verbose and len(df) > 0:
        print(f"\n=== СТАТИСТИКА ===")
        print(f"Интервалов обработано: {len(df)}")
        print(f"Средний наклон по горизонтали: {df['tilt_horizontal_deg'].mean():.1f}°")
        print(f"Средний наклон по вертикали: {df['tilt_vertical_deg'].mean():.1f}°")
        print(f"Средняя уверенность: {df['confidence'].mean():.2f}")
    
    return df


def add_size_estimates_to_detections(
    detections_csv: str,
    output_csv: Optional[str] = None,
    geometry_csv: Optional[str] = None,
    calibration: CameraCalibration = None,
    frame_width: int = 1920,
    frame_height: int = 1080
) -> pd.DataFrame:
    """
    Добавляет оценки реального размера к таблице детекций.
    
    Args:
        detections_csv: путь к CSV с детекциями (из detect_video.py)
        output_csv: путь для сохранения результатов
        geometry_csv: путь к CSV с геометрией камеры (опционально)
        calibration: калибровочные параметры
        frame_width: ширина кадра в пикселях
        frame_height: высота кадра в пикселях
    
    Returns:
        DataFrame с добавленными оценками размера
    """
    if calibration is None:
        calibration = CameraCalibration()
        calibration.frame_width = frame_width
        calibration.frame_height = frame_height
    
    df = pd.read_csv(detections_csv)
    
    # Загружаем геометрию если есть
    geometry_df = None
    if geometry_csv and Path(geometry_csv).exists():
        geometry_df = pd.read_csv(geometry_csv)
    
    # Рассчитываем размеры
    size_estimates = []
    distance_estimates = []
    confidence_estimates = []
    
    for _, row in df.iterrows():
        # Размер bbox в пикселях (берём максимум из ширины и высоты)
        bbox_w_pix = row['width'] * frame_width
        bbox_h_pix = row['height'] * frame_height
        bbox_size_pix = max(bbox_w_pix, bbox_h_pix)
        
        # Дистанция из глубины (если есть)
        depth = row.get('depth_m')
        
        if pd.notna(depth) and depth > 0:
            # Используем глубину как приближение дистанции
            # (это упрощение — реальная дистанция зависит от наклона камеры)
            distance = depth
            
            # Коррекция на наклон камеры (если есть данные геометрии)
            if geometry_df is not None:
                # Находим ближайший интервал
                frame = row['frame']
                closest = geometry_df.iloc[(geometry_df['frame_end'] - frame).abs().argmin()]
                tilt = closest['tilt_horizontal_deg']
                # Корректируем дистанцию
                distance = depth / np.cos(np.radians(abs(tilt)))
            
            estimate = estimate_object_size(bbox_size_pix, distance, calibration)
            size_estimates.append(round(estimate.size_m, 3))
            distance_estimates.append(round(estimate.distance_m, 2))
            confidence_estimates.append(round(estimate.confidence, 2))
        else:
            size_estimates.append(None)
            distance_estimates.append(None)
            confidence_estimates.append(None)
    
    df['estimated_size_m'] = size_estimates
    df['estimated_distance_m'] = distance_estimates
    df['size_confidence'] = confidence_estimates
    
    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Результаты с размерами сохранены: {output_csv}")
    
    # Статистика
    valid_sizes = df[df['estimated_size_m'].notna()]
    if len(valid_sizes) > 0:
        print(f"\n=== СТАТИСТИКА РАЗМЕРОВ ===")
        print(f"Детекций с оценкой размера: {len(valid_sizes)}/{len(df)}")
        
        for class_name in valid_sizes['class_name'].unique():
            class_df = valid_sizes[valid_sizes['class_name'] == class_name]
            sizes = class_df['estimated_size_m']
            print(f"\n{class_name}:")
            print(f"  Средний размер: {sizes.mean()*100:.1f} см")
            print(f"  Диапазон: {sizes.min()*100:.1f} - {sizes.max()*100:.1f} см")
            print(f"  Медиана: {sizes.median()*100:.1f} см")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Оценка геометрии камеры и размеров объектов"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Команды')
    
    # Команда: geometry — оценка наклона камеры
    geom_parser = subparsers.add_parser('geometry', help='Оценка наклона камеры по видео')
    geom_parser.add_argument('--video', '-v', required=True, help='Путь к видео')
    geom_parser.add_argument('--output', '-o', default='output/geometry.csv', help='Выходной CSV')
    geom_parser.add_argument('--interval', '-i', type=int, default=30, help='Интервал оценки (кадры)')
    
    # Команда: size — добавление размеров к детекциям
    size_parser = subparsers.add_parser('size', help='Добавить оценки размеров к детекциям')
    size_parser.add_argument('--detections', '-d', required=True, help='CSV с детекциями')
    size_parser.add_argument('--output', '-o', help='Выходной CSV (по умолчанию перезаписывает)')
    size_parser.add_argument('--geometry', '-g', help='CSV с геометрией камеры')
    size_parser.add_argument('--width', type=int, default=1920, help='Ширина кадра')
    size_parser.add_argument('--height', type=int, default=1080, help='Высота кадра')
    
    args = parser.parse_args()
    
    if args.command == 'geometry':
        process_video_geometry(
            video_path=args.video,
            output_csv=args.output,
            frame_interval=args.interval
        )
    
    elif args.command == 'size':
        output = args.output or args.detections
        add_size_estimates_to_detections(
            detections_csv=args.detections,
            output_csv=output,
            geometry_csv=args.geometry,
            frame_width=args.width,
            frame_height=args.height
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
