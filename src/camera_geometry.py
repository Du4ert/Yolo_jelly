"""
Модуль оценки геометрии камеры и размеров объектов.

Функции:
- Оценка наклона камеры по морскому снегу (Focus of Expansion)
- Расчёт реального размера объектов по динамике изменения размера в треке
- Определение абсолютной глубины объекта в толще воды

Калибровочные данные получены для GoPro 12 Wide 4K (3840x2160).
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
from scipy.optimize import minimize
from scipy.stats import linregress
import argparse
import warnings


# Классы, для которых не считаем размер (слишком мелкие)
SKIP_SIZE_CLASSES = {'Pleurobrachia pileus'}


@dataclass
class CameraCalibration:
    """Калибровочные параметры камеры."""
    k: float = 156.1  # пиксель·м (средний Feret для теннисного мяча 6.6 см)
    frame_width: int = 1920
    frame_height: int = 1080
    fov_horizontal: float = 100.0  # GoPro Wide
    min_reliable_distance: float = 0.5  # м
    max_reliable_distance: float = 3.0  # м
    
    @property
    def K(self) -> float:
        """Коэффициент K = k / эталонный_размер ≈ 2365 пикс/м на 1м."""
        return self.k / 0.066
    
    @property
    def pixels_per_degree(self) -> float:
        return self.frame_width / self.fov_horizontal
    
    @property
    def frame_center(self) -> Tuple[float, float]:
        return self.frame_width / 2, self.frame_height / 2


@dataclass
class FOEResult:
    """Результат оценки Focus of Expansion."""
    foe_x: float
    foe_y: float
    tilt_horizontal: float  # градусы, + вправо
    tilt_vertical: float    # градусы, + вниз
    confidence: float
    n_vectors: int


@dataclass
class TrackSizeEstimate:
    """Результат оценки размера объекта по треку."""
    track_id: int
    class_name: str
    real_size_m: float           # реальный размер объекта (м)
    real_size_cm: float          # реальный размер объекта (см)
    distance_at_max_size: float  # дистанция в момент макс. размера (м)
    object_depth_m: float        # абсолютная глубина объекта (м)
    camera_depth_at_max: float   # глубина камеры в момент макс. размера (м)
    max_size_frame: int          # кадр с максимальным размером
    max_size_pixels: float       # максимальный размер в пикселях
    confidence: float            # уверенность оценки (0-1)
    method: str                  # метод оценки
    n_points_used: int           # точек использовано для оценки
    fit_r_squared: float         # R² фита
    warnings: List[str] = field(default_factory=list)


def estimate_foe(
    points: np.ndarray,
    vectors: np.ndarray,
    frame_size: Tuple[int, int] = (1920, 1080),
    min_vector_length: float = 0.5
) -> FOEResult:
    """
    Оценивает Focus of Expansion по набору точек и их векторов движения.
    """
    width, height = frame_size
    
    vec_lengths = np.linalg.norm(vectors, axis=1)
    mask = vec_lengths > min_vector_length
    points_filt = points[mask]
    vectors_filt = vectors[mask]
    
    if len(points_filt) < 10:
        return FOEResult(width/2, height/2, 0, 0, 0, len(points_filt))
    
    def foe_error(foe):
        fx, fy = foe
        radial = points_filt - np.array([fx, fy])
        radial_norm = radial / (np.linalg.norm(radial, axis=1, keepdims=True) + 1e-6)
        vec_norm = vectors_filt / (np.linalg.norm(vectors_filt, axis=1, keepdims=True) + 1e-6)
        dot = np.sum(radial_norm * vec_norm, axis=1)
        return np.mean(1 - np.abs(dot))
    
    result = minimize(foe_error, [width/2, height/2], method='Nelder-Mead')
    foe_x, foe_y = result.x
    confidence = 1 - result.fun
    pixels_per_degree = width / 100.0
    tilt_h = (foe_x - width/2) / pixels_per_degree
    tilt_v = (foe_y - height/2) / pixels_per_degree
    
    return FOEResult(foe_x, foe_y, tilt_h, tilt_v, confidence, len(points_filt))


def estimate_foe_from_frames(
    gray1: np.ndarray,
    gray2: np.ndarray,
    max_corners: int = 1000,
    min_vector_length: float = 0.5
) -> Tuple[FOEResult, np.ndarray, np.ndarray]:
    """Оценивает FOE по двум последовательным кадрам."""
    height, width = gray1.shape
    
    feature_params = dict(maxCorners=max_corners, qualityLevel=0.01, minDistance=10, blockSize=7)
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    points1 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
    
    if points1 is None or len(points1) < 10:
        return FOEResult(width/2, height/2, 0, 0, 0, 0), np.array([]), np.array([])
    
    points2, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, points1, None, **lk_params)
    
    good_old = points1[status == 1].reshape(-1, 2)
    good_new = points2[status == 1].reshape(-1, 2)
    vectors = good_new - good_old
    
    foe_result = estimate_foe(good_old, vectors, (width, height), min_vector_length)
    return foe_result, good_old, vectors


def _get_bbox_size_pixels(row: pd.Series, frame_width: int, frame_height: int) -> float:
    """Получает размер bbox в пикселях (максимум из ширины и высоты)."""
    w_pix = row['width'] * frame_width
    h_pix = row['height'] * frame_height
    return max(w_pix, h_pix)


def _find_max_size_frame(track_df: pd.DataFrame, frame_width: int, frame_height: int) -> Tuple[int, pd.Series]:
    """
    Находит кадр с максимальным размером объекта.
    
    Исключает последние кадры, где размер может падать из-за выхода за границы.
    """
    track_df = track_df.copy()
    track_df['size_pix'] = track_df.apply(
        lambda r: _get_bbox_size_pixels(r, frame_width, frame_height), axis=1
    )
    
    # Находим точку максимума
    max_idx = track_df['size_pix'].idxmax()
    max_pos = track_df.index.get_loc(max_idx)
    
    # Проверяем, не в самом ли конце максимум
    n_points = len(track_df)
    if n_points > 3 and max_pos >= n_points - 2:
        # Берём точку раньше
        cutoff = int(n_points * 0.8)
        if cutoff > 0:
            track_subset = track_df.iloc[:cutoff]
            if len(track_subset) > 0:
                max_idx = track_subset['size_pix'].idxmax()
    
    return max_idx, track_df.loc[max_idx]


def estimate_size_from_track(
    track_df: pd.DataFrame,
    calibration: CameraCalibration,
    camera_tilt_deg: float = 0.0,
    min_depth_change: float = 0.3,
    min_points: int = 5,
    min_r_squared: float = 0.5,
    min_size_change_ratio: float = 0.3
) -> Optional[TrackSizeEstimate]:
    """
    Оценивает реальный размер объекта по динамике изменения размера в треке.
    
    Алгоритм:
    1. Строит зависимость 1/size от depth_camera
    2. Линейная регрессия даёт наклон и пересечение
    3. Из них вычисляется реальный размер и дистанция
    
    Args:
        track_df: DataFrame с детекциями одного трека
        calibration: калибровочные параметры
        camera_tilt_deg: наклон камеры от вертикали (градусы)
        min_depth_change: минимальное изменение глубины для оценки (м)
        min_points: минимум точек в треке
        min_r_squared: минимальный R² для принятия результата
        min_size_change_ratio: мин. относительное изменение размера (max-min)/max
    
    Returns:
        TrackSizeEstimate или None если оценка невозможна
    """
    track_id = track_df['track_id'].iloc[0]
    class_name = track_df['class_name'].iloc[0]
    
    # Пропускаем мелкие виды
    if class_name in SKIP_SIZE_CLASSES:
        return None
    
    # Проверяем наличие данных глубины
    if 'depth_m' not in track_df.columns or track_df['depth_m'].isna().all():
        return None
    
    # Фильтруем точки с валидной глубиной
    valid_df = track_df[track_df['depth_m'].notna()].copy()
    
    if len(valid_df) < min_points:
        return None
    
    # Проверяем достаточное изменение глубины
    depth_change = valid_df['depth_m'].max() - valid_df['depth_m'].min()
    if depth_change < min_depth_change:
        return None
    
    # Вычисляем размеры в пикселях
    frame_width = calibration.frame_width
    frame_height = calibration.frame_height
    
    valid_df['size_pix'] = valid_df.apply(
        lambda r: _get_bbox_size_pixels(r, frame_width, frame_height), axis=1
    )
    
    # Проверяем достаточное изменение размера
    size_min = valid_df['size_pix'].min()
    size_max = valid_df['size_pix'].max()
    size_change_ratio = (size_max - size_min) / size_max
    
    if size_change_ratio < min_size_change_ratio:
        return None
    
    # Фильтруем выбросы по размеру (больше 3 сигм)
    size_mean = valid_df['size_pix'].mean()
    size_std = valid_df['size_pix'].std()
    if size_std > 0:
        valid_df = valid_df[
            (valid_df['size_pix'] > size_mean - 3*size_std) &
            (valid_df['size_pix'] < size_mean + 3*size_std)
        ]
    
    if len(valid_df) < min_points:
        return None
    
    # Данные для регрессии
    camera_depths = valid_df['depth_m'].values
    sizes_pix = valid_df['size_pix'].values
    
    # Защита от нулевых размеров
    sizes_pix = np.maximum(sizes_pix, 1.0)
    inv_sizes = 1.0 / sizes_pix
    
    # Линейная регрессия: 1/size = slope × camera_depth + intercept
    slope, intercept, r_value, p_value, std_err = linregress(camera_depths, inv_sizes)
    r_squared = r_value ** 2
    
    warnings_list = []
    
    # Проверка качества фита
    if r_squared < min_r_squared:
        return None
    
    if abs(slope) < 1e-10:
        return None
    
    K = calibration.K
    
    # Коррекция на наклон камеры
    tilt_correction = np.cos(np.radians(abs(camera_tilt_deg))) if camera_tilt_deg != 0 else 1.0
    
    # Вычисляем реальный размер
    real_size = 1.0 / (K * abs(slope))
    
    # Вычисляем глубину объекта
    if slope < 0:
        # Объект выше камеры
        object_depth = intercept * K * real_size
    else:
        # Объект ниже камеры
        object_depth = -intercept * K * real_size
    
    # Находим кадр с максимальным размером
    max_idx, max_row = _find_max_size_frame(valid_df, frame_width, frame_height)
    max_size_pix = _get_bbox_size_pixels(max_row, frame_width, frame_height)
    camera_depth_at_max = max_row['depth_m']
    max_frame = max_row['frame']
    
    # Дистанция в момент максимального размера (с учётом наклона)
    vertical_distance = abs(object_depth - camera_depth_at_max)
    distance_at_max = vertical_distance / tilt_correction if tilt_correction > 0 else vertical_distance
    
    # Проверка разумности результата
    if real_size < 0.005:  # меньше 5 мм
        warnings_list.append("size_too_small")
    
    if real_size > 1.0:  # больше 1 м
        warnings_list.append("size_too_large")
    
    if distance_at_max < calibration.min_reliable_distance:
        warnings_list.append("distance_below_min")
    
    if distance_at_max > calibration.max_reliable_distance:
        warnings_list.append("distance_above_max")
    
    # Оценка уверенности на основе R²
    confidence = r_squared
    
    # Снижаем при предупреждениях
    if "size_too_large" in warnings_list or "size_too_small" in warnings_list:
        confidence *= 0.5
    
    return TrackSizeEstimate(
        track_id=track_id,
        class_name=class_name,
        real_size_m=round(real_size, 4),
        real_size_cm=round(real_size * 100, 2),
        distance_at_max_size=round(distance_at_max, 3),
        object_depth_m=round(object_depth, 2),
        camera_depth_at_max=round(camera_depth_at_max, 2),
        max_size_frame=int(max_frame),
        max_size_pixels=round(max_size_pix, 1),
        confidence=round(confidence, 3),
        method="linear_regression",
        n_points_used=len(valid_df),
        fit_r_squared=round(r_squared, 4),
        warnings=warnings_list
    )


def load_geometry_data(geometry_csv: str) -> Optional[pd.DataFrame]:
    """Загружает данные о наклоне камеры."""
    if geometry_csv and Path(geometry_csv).exists():
        return pd.read_csv(geometry_csv)
    return None


def get_camera_tilt_for_frame(geometry_df: Optional[pd.DataFrame], frame: int) -> float:
    """Получает наклон камеры для заданного кадра."""
    if geometry_df is None or len(geometry_df) == 0:
        return 0.0
    
    # Находим ближайший интервал
    idx = (geometry_df['frame_end'] - frame).abs().argmin()
    row = geometry_df.iloc[idx]
    
    # Общий наклон от вертикали
    tilt_h = row.get('tilt_horizontal_deg', 0)
    tilt_v = row.get('tilt_vertical_deg', 0)
    
    # Суммарный угол
    total_tilt = np.sqrt(tilt_h**2 + tilt_v**2)
    return total_tilt


def process_detections_with_size(
    detections_csv: str,
    output_csv: Optional[str] = None,
    tracks_output_csv: Optional[str] = None,
    geometry_csv: Optional[str] = None,
    calibration: CameraCalibration = None,
    frame_width: int = 1920,
    frame_height: int = 1080,
    min_depth_change: float = 0.3,
    min_track_points: int = 5,
    min_r_squared: float = 0.5,
    min_size_change_ratio: float = 0.3,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Обрабатывает детекции и добавляет оценки размеров на основе анализа треков.
    
    Args:
        detections_csv: путь к CSV с детекциями (из detect_video.py)
        output_csv: путь для сохранения детекций с размерами
        tracks_output_csv: путь для сохранения статистики по трекам
        geometry_csv: путь к CSV с геометрией камеры (наклон)
        calibration: калибровочные параметры
        frame_width: ширина кадра
        frame_height: высота кадра
        min_depth_change: мин. изменение глубины для оценки (м)
        min_track_points: мин. точек в треке
        min_r_squared: мин. R² для принятия результата
        min_size_change_ratio: мин. относительное изменение размера
        verbose: выводить прогресс
    
    Returns:
        (DataFrame детекций с размерами, DataFrame статистики треков)
    """
    if calibration is None:
        calibration = CameraCalibration()
    calibration.frame_width = frame_width
    calibration.frame_height = frame_height
    
    df = pd.read_csv(detections_csv)
    
    # Загружаем данные о наклоне камеры
    geometry_df = load_geometry_data(geometry_csv)
    
    if verbose:
        print(f"Загружено детекций: {len(df)}")
        print(f"Уникальных треков: {df['track_id'].nunique()}")
        if geometry_df is not None:
            print(f"Загружены данные о наклоне камеры: {len(geometry_df)} интервалов")
            mean_tilt = np.sqrt(
                geometry_df['tilt_horizontal_deg'].mean()**2 + 
                geometry_df['tilt_vertical_deg'].mean()**2
            )
            print(f"  Средний наклон от вертикали: {mean_tilt:.1f}°")
    
    # Проверяем наличие необходимых колонок
    required_cols = ['track_id', 'frame', 'depth_m', 'width', 'height', 'class_name']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют колонки: {missing}")
    
    # Обрабатываем каждый трек
    track_estimates = []
    tracks_processed = 0
    tracks_skipped_r2 = 0
    tracks_skipped_size_change = 0
    tracks_skipped_depth = 0
    tracks_with_size = 0
    
    for track_id, track_df in df.groupby('track_id'):
        if pd.isna(track_id):
            continue
        
        tracks_processed += 1
        
        # Получаем средний наклон камеры для трека
        mean_frame = track_df['frame'].mean()
        camera_tilt = get_camera_tilt_for_frame(geometry_df, mean_frame)
        
        estimate = estimate_size_from_track(
            track_df, 
            calibration,
            camera_tilt_deg=camera_tilt,
            min_depth_change=min_depth_change,
            min_points=min_track_points,
            min_r_squared=min_r_squared,
            min_size_change_ratio=min_size_change_ratio
        )
        
        if estimate is not None:
            track_estimates.append(estimate)
            tracks_with_size += 1
    
    if verbose:
        print(f"\nТреков обработано: {tracks_processed}")
        print(f"Треков с оценкой размера: {tracks_with_size}")
        print(f"Отфильтровано (не прошли критерии): {tracks_processed - tracks_with_size}")
    
    # Создаём DataFrame с результатами по трекам
    if track_estimates:
        tracks_df = pd.DataFrame([
            {
                'track_id': e.track_id,
                'class_name': e.class_name,
                'real_size_cm': e.real_size_cm,
                'real_size_m': e.real_size_m,
                'object_depth_m': e.object_depth_m,
                'distance_at_max_m': e.distance_at_max_size,
                'camera_depth_at_max_m': e.camera_depth_at_max,
                'max_size_frame': e.max_size_frame,
                'max_size_pixels': e.max_size_pixels,
                'confidence': e.confidence,
                'method': e.method,
                'n_points': e.n_points_used,
                'fit_r_squared': e.fit_r_squared,
                'warnings': ';'.join(e.warnings) if e.warnings else ''
            }
            for e in track_estimates
        ])
    else:
        tracks_df = pd.DataFrame()
    
    # Добавляем размеры к детекциям
    size_map = {e.track_id: e for e in track_estimates}
    
    df['estimated_size_cm'] = None
    df['estimated_size_m'] = None
    df['object_depth_m'] = None
    df['distance_to_object_m'] = None
    df['size_confidence'] = None
    
    for idx, row in df.iterrows():
        track_id = row['track_id']
        if pd.notna(track_id) and track_id in size_map:
            est = size_map[track_id]
            df.at[idx, 'estimated_size_cm'] = est.real_size_cm
            df.at[idx, 'estimated_size_m'] = est.real_size_m
            df.at[idx, 'object_depth_m'] = est.object_depth_m
            df.at[idx, 'size_confidence'] = est.confidence
            
            # Дистанция для текущего кадра
            camera_depth = row['depth_m']
            if pd.notna(camera_depth):
                df.at[idx, 'distance_to_object_m'] = round(abs(est.object_depth_m - camera_depth), 3)
    
    # Сохранение результатов
    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        if verbose:
            print(f"\nДетекции с размерами сохранены: {output_csv}")
    
    if tracks_output_csv and len(tracks_df) > 0:
        tracks_path = Path(tracks_output_csv)
        tracks_path.parent.mkdir(parents=True, exist_ok=True)
        tracks_df.to_csv(tracks_path, index=False)
        if verbose:
            print(f"Статистика треков сохранена: {tracks_output_csv}")
    
    # Вывод статистики по видам
    if verbose and len(tracks_df) > 0:
        print("\n" + "="*60)
        print("СТАТИСТИКА РАЗМЕРОВ ПО ВИДАМ")
        print("="*60)
        
        for class_name in sorted(tracks_df['class_name'].unique()):
            class_df = tracks_df[tracks_df['class_name'] == class_name]
            sizes = class_df['real_size_cm']
            depths = class_df['object_depth_m']
            
            print(f"\n{class_name}:")
            print(f"  Треков: {len(class_df)}")
            print(f"  Размер: {sizes.mean():.1f} ± {sizes.std():.1f} см")
            print(f"    мин: {sizes.min():.1f} см, макс: {sizes.max():.1f} см")
            print(f"  Глубина объектов: {depths.mean():.1f} ± {depths.std():.1f} м")
            print(f"    мин: {depths.min():.1f} м, макс: {depths.max():.1f} м")
            print(f"  Средний R²: {class_df['fit_r_squared'].mean():.2f}")
            print(f"  Средняя уверенность: {class_df['confidence'].mean():.2f}")
    
    return df, tracks_df


def process_video_geometry(
    video_path: str,
    output_csv: Optional[str] = None,
    frame_interval: int = 30,
    calibration: CameraCalibration = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Обрабатывает видео и оценивает геометрию камеры для каждого интервала.
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
    
    interval_points = []
    interval_vectors = []
    interval_start = 0
    
    feature_params = dict(maxCorners=500, qualityLevel=0.01, minDistance=10, blockSize=7)
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        points1 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        if points1 is not None and len(points1) > 10:
            points2, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points1, None, **lk_params)
            
            good_old = points1[status == 1].reshape(-1, 2)
            good_new = points2[status == 1].reshape(-1, 2)
            
            interval_points.extend(good_old)
            interval_vectors.extend(good_new - good_old)
        
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
        total_tilt = np.sqrt(df['tilt_horizontal_deg'].mean()**2 + df['tilt_vertical_deg'].mean()**2)
        print(f"Средний наклон от вертикали: {total_tilt:.1f}°")
        print(f"Средняя уверенность: {df['confidence'].mean():.2f}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Оценка геометрии камеры и размеров объектов"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Команды')
    
    # Команда: geometry
    geom_parser = subparsers.add_parser('geometry', help='Оценка наклона камеры по видео')
    geom_parser.add_argument('--video', '-v', required=True, help='Путь к видео')
    geom_parser.add_argument('--output', '-o', default='output/geometry.csv', help='Выходной CSV')
    geom_parser.add_argument('--interval', '-i', type=int, default=30, help='Интервал оценки (кадры)')
    
    # Команда: size
    size_parser = subparsers.add_parser('size', help='Оценка размеров объектов по трекам')
    size_parser.add_argument('--detections', '-d', required=True, help='CSV с детекциями')
    size_parser.add_argument('--output', '-o', help='Выходной CSV с детекциями')
    size_parser.add_argument('--tracks', '-t', help='Выходной CSV со статистикой треков')
    size_parser.add_argument('--geometry', '-g', help='CSV с геометрией камеры (наклон)')
    size_parser.add_argument('--width', type=int, default=1920, help='Ширина кадра')
    size_parser.add_argument('--height', type=int, default=1080, help='Высота кадра')
    size_parser.add_argument('--min-depth-change', type=float, default=0.3, 
                            help='Мин. изменение глубины для оценки (м)')
    size_parser.add_argument('--min-track-points', type=int, default=5,
                            help='Мин. точек в треке')
    size_parser.add_argument('--min-r-squared', type=float, default=0.5,
                            help='Мин. R² регрессии для принятия результата')
    size_parser.add_argument('--min-size-change', type=float, default=0.3,
                            help='Мин. относительное изменение размера (0-1)')
    
    args = parser.parse_args()
    
    if args.command == 'geometry':
        process_video_geometry(
            video_path=args.video,
            output_csv=args.output,
            frame_interval=args.interval
        )
    
    elif args.command == 'size':
        output = args.output or args.detections.replace('.csv', '_with_size.csv')
        tracks = args.tracks or args.detections.replace('.csv', '_track_sizes.csv')
        
        process_detections_with_size(
            detections_csv=args.detections,
            output_csv=output,
            tracks_output_csv=tracks,
            geometry_csv=args.geometry,
            frame_width=args.width,
            frame_height=args.height,
            min_depth_change=args.min_depth_change,
            min_track_points=args.min_track_points,
            min_r_squared=args.min_r_squared,
            min_size_change_ratio=args.min_size_change
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
