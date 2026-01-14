"""
Модуль оценки геометрии камеры и размеров объектов.

Функции:
- Оценка наклона камеры по морскому снегу (Focus of Expansion)
- Расчёт реального размера объектов по динамике изменения размера в треке
- Определение абсолютной глубины объекта в толще воды

Калибровочные данные получены для GoPro 12 Wide 4K (3840x2160).

Формулы расчёта размеров (по методике научрука):
1. k = (Δpixels/pixels₁) / Δd  - удельный прирост размера (%/м)
2. d = 24.68 * |k|^(-0.644)    - дистанция до объекта (м)
3. p = 2.432 * d^(-1.0334)     - калибровка (px/мм)
4. size = pixels / p           - размер объекта (мм)
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
from scipy.optimize import minimize
import argparse


# Классы, для которых не считаем размер (слишком мелкие)
SKIP_SIZE_CLASSES = {'Pleurobrachia pileus'}

# Типичные размеры видов (см) для референсной оценки
TYPICAL_SIZES_CM = {
    'Aurelia aurita': {'mean': 10.0, 'std': 5.0, 'min': 3.0, 'max': 40.0},
    'Mnemiopsis leidyi': {'mean': 5.0, 'std': 2.5, 'min': 2.0, 'max': 12.0},
    'Beroe ovata': {'mean': 8.0, 'std': 3.0, 'min': 3.0, 'max': 15.0},
    'Rhizostoma pulmo': {'mean': 30.0, 'std': 15.0, 'min': 10.0, 'max': 60.0},
}


@dataclass
class CameraCalibration:
    """Калибровочные параметры камеры GoPro 12 Wide 4K.
    
    Калибровка выполнена для разрешения 3840x2160 (4K).
    Коэффициенты получены эмпирически по формулам научрука.
    """
    # Размер кадра (4K)
    frame_width: int = 3840
    frame_height: int = 2160
    
    # Угол обзора
    fov_horizontal: float = 100.0
    
    # === Калибровочные коэффициенты (эмпирические, GoPro 12 Wide 4K) ===
    # Формула: d = A * k^B, где k - удельный прирост размера (%/м)
    distance_coef_A: float = 24.68
    distance_coef_B: float = -0.644
    
    # Формула: p = C * d^D, где p - px/мм, d - дистанция (м)
    pixel_calib_C: float = 2.432
    pixel_calib_D: float = -1.0334
    
    # Диапазон надёжных измерений (по SNR анализу)
    min_reliable_distance: float = 0.3   # ближе - слишком крупно
    max_reliable_distance: float = 3.0   # дальше - шум > сигнал
    
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
    tilt_horizontal: float
    tilt_vertical: float
    confidence: float
    n_vectors: int


@dataclass
class TrackSizeEstimate:
    """
    Результат оценки размера объекта по треку.
    
    Метод расчёта по формулам научрука:
    1. k = (Δpixels/pixels₁) / Δdepth_camera  - удельный прирост размера для каждой пары
    2. d = 24.68 * |k|^(-0.644)               - дистанция до объекта (м)
    3. p = 2.432 * d^(-1.0334)                - калибровка px/мм
    4. size = pixels_max / p                  - размер объекта по макс. кадру (мм)
    5. object_depth = camera_depth + distance - глубина объекта (камера смотрит вниз)
    """
    track_id: int
    class_name: str
    real_size_mm: float            # Размер в миллиметрах
    real_size_cm: float            # Размер в сантиметрах
    distance_m: float              # Дистанция до объекта на момент max размера (м)
    object_depth_m: float          # Глубина объекта в воде (м)
    first_frame: int               # Кадр с макс. размером (переименовать в max_frame)
    first_size_pixels: float       # Макс. размер в пикселях (переименовать в max_size_pixels)
    camera_depth_first: float      # Глубина камеры на кадре с макс. размером (м)
    k_mean: float                  # Средний k по всем парам (%/м)
    k_std: float                   # Стд k (%/м)
    pixel_calibration: float       # Калибровка px/мм на момент max размера
    confidence: float              # Уверенность оценки (0-1)
    method: str                    # 'k_method', 'typical'
    n_points_used: int             # Количество пар для расчёта k
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# FOE (Focus of Expansion) - оценка наклона камеры
# =============================================================================

def estimate_foe(
    points: np.ndarray,
    vectors: np.ndarray,
    frame_size: Tuple[int, int] = (1920, 1080),
    min_vector_length: float = 0.5,
    max_tilt_deg: float = 60.0
) -> FOEResult:
    """
    Оценивает Focus of Expansion по точкам и векторам движения.
    
    При малом движении камеры FOE уходит в бесконечность, что даёт
    нефизичные значения наклона. Фильтруем такие случаи.
    """
    width, height = frame_size
    cx, cy = width / 2, height / 2
    
    vec_lengths = np.linalg.norm(vectors, axis=1)
    mask = vec_lengths > min_vector_length
    points_filt = points[mask]
    vectors_filt = vectors[mask]
    
    if len(points_filt) < 10:
        return FOEResult(cx, cy, 0, 0, 0, len(points_filt))
    
    def foe_error(foe):
        fx, fy = foe
        radial = points_filt - np.array([fx, fy])
        radial_norm = radial / (np.linalg.norm(radial, axis=1, keepdims=True) + 1e-6)
        vec_norm = vectors_filt / (np.linalg.norm(vectors_filt, axis=1, keepdims=True) + 1e-6)
        dot = np.sum(radial_norm * vec_norm, axis=1)
        return np.mean(1 - np.abs(dot))
    
    result = minimize(foe_error, [cx, cy], method='Nelder-Mead')
    foe_x, foe_y = result.x
    confidence = 1 - result.fun
    
    pixels_per_degree = width / 100.0
    tilt_h = (foe_x - cx) / pixels_per_degree
    tilt_v = (foe_y - cy) / pixels_per_degree
    
    diagonal = np.sqrt(width**2 + height**2)
    foe_distance = np.sqrt((foe_x - cx)**2 + (foe_y - cy)**2)
    max_foe_distance = 3 * diagonal
    
    is_outlier = (
        foe_distance > max_foe_distance or
        abs(tilt_h) > max_tilt_deg or
        abs(tilt_v) > max_tilt_deg or
        not np.isfinite(foe_x) or
        not np.isfinite(foe_y)
    )
    
    if is_outlier:
        return FOEResult(cx, cy, 0.0, 0.0, 0.0, len(points_filt))
    
    return FOEResult(foe_x, foe_y, tilt_h, tilt_v, confidence, len(points_filt))


# =============================================================================
# Вспомогательные функции для расчёта размеров
# =============================================================================

def _get_bbox_size_pixels(row: pd.Series, frame_width: int, frame_height: int) -> float:
    """
    Получает размер bbox в пикселях.
    Берёт максимум из ширины и высоты.
    Калибровка выполнена без коррекции дисторсии.
    """
    w_pix = row['width'] * frame_width
    h_pix = row['height'] * frame_height
    return max(w_pix, h_pix)


def _calculate_k_for_pair(
    pixels1: float, pixels2: float, 
    depth1: float, depth2: float
) -> Optional[float]:
    """
    Вычисляет удельный прирост k для пары точек.
    
    k = (Δpixels / pixels₁) / Δd
    
    Возвращает k в долях/м. Для %/м умножить на 100.
    """
    delta_d = depth2 - depth1
    
    if abs(delta_d) < 0.05:  # меньше 5 см - шум
        return None
    
    if pixels1 <= 0:
        return None
    
    delta_pixels = pixels2 - pixels1
    relative_change = delta_pixels / pixels1
    k = relative_change / delta_d
    
    return k


def _calculate_distance_from_k(k_abs: float, calibration: CameraCalibration) -> float:
    """
    Вычисляет дистанцию до объекта по удельному приросту.
    d = A * k^B
    
    Args:
        k_abs: модуль k в %/м (NOT долях!)
    """
    A = calibration.distance_coef_A
    B = calibration.distance_coef_B
    k_abs = max(k_abs, 1.0)  # минимум 1%/м
    return A * (k_abs ** B)


def _calculate_pixel_calibration(distance: float, calibration: CameraCalibration) -> float:
    """
    Вычисляет калибровку px/мм для данной дистанции.
    p = C * d^D
    """
    C = calibration.pixel_calib_C
    D = calibration.pixel_calib_D
    distance = max(distance, 0.1)
    return C * (distance ** D)


def _calculate_size_mm(pixels: float, pixel_calibration: float) -> float:
    """Вычисляет размер объекта в миллиметрах: size_mm = pixels / p"""
    if pixel_calibration <= 0:
        return 0.0
    return pixels / pixel_calibration


def _get_first_frame_data(
    track_df: pd.DataFrame, 
    frame_width: int, 
    frame_height: int
) -> Tuple[int, pd.Series, float]:
    """Получает данные первого кадра трека."""
    track_df = track_df.copy()
    track_df['size_pix'] = track_df.apply(
        lambda r: _get_bbox_size_pixels(r, frame_width, frame_height), axis=1
    )
    first_idx = track_df['frame'].idxmin()
    first_row = track_df.loc[first_idx]
    return first_idx, first_row, first_row['size_pix']


# =============================================================================
# Основные функции оценки размеров
# =============================================================================

def _find_max_size_frame(
    track_df: pd.DataFrame,
    frame_width: int,
    frame_height: int
) -> Tuple[int, pd.Series, float]:
    """
    Находит кадр с максимальным размером объекта.
    
    Логика: из последних 20% трека берём последний кадр,
    где объект ещё увеличивается (до начала выхода за границы кадра).
    """
    track_df = track_df.copy()
    track_df['size_pix'] = track_df.apply(
        lambda r: _get_bbox_size_pixels(r, frame_width, frame_height), axis=1
    )
    track_df = track_df.sort_values('frame').reset_index(drop=True)
    
    n_points = len(track_df)
    
    if n_points < 3:
        # Слишком короткий трек — просто берём максимум
        max_idx = track_df['size_pix'].idxmax()
        max_row = track_df.loc[max_idx]
        return max_idx, max_row, max_row['size_pix']
    
    # Последние 20% трека
    start_idx = int(n_points * 0.8)
    last_20_pct = track_df.iloc[start_idx:]
    
    # Ищем последний кадр, где размер ещё растёт
    # (следующий кадр больше или равен текущему)
    best_idx = None
    
    for i in range(len(last_20_pct) - 1):
        current_size = last_20_pct.iloc[i]['size_pix']
        next_size = last_20_pct.iloc[i + 1]['size_pix']
        
        if next_size >= current_size:
            # Размер ещё растёт — запоминаем следующий кадр
            best_idx = last_20_pct.index[i + 1]
    
    if best_idx is None:
        # Размер везде уменьшается — берём первый кадр из последних 20%
        best_idx = last_20_pct.index[0]
    
    max_row = track_df.loc[best_idx]
    return best_idx, max_row, max_row['size_pix']


def estimate_size_by_k_method(
    track_df: pd.DataFrame,
    calibration: CameraCalibration,
    min_depth_change: float = 0.1,
    min_points: int = 3
) -> Optional[TrackSizeEstimate]:
    """
    Оценивает размер объекта методом удельного прироста k.
    
    Алгоритм:
    1. Для каждой пары соседних точек вычисляем k = (Δpx/px₁) / Δdepth_camera
    2. По k вычисляем дистанцию: d = 24.68 × |k|^(-0.644)
    3. Находим кадр с максимальным размером (объект целиком в кадре)
    4. Берём дистанцию от ближайшей пары к этому кадру
    5. Калибровка пикселя: p = 2.432 × d^(-1.0334)
    6. Размер: size_mm = pixels_max / p
    7. Глубина объекта = глубина камеры + дистанция (камера смотрит вниз)
    """
    track_id = track_df['track_id'].iloc[0]
    class_name = track_df['class_name'].iloc[0]
    
    if class_name in SKIP_SIZE_CLASSES:
        return None
    
    if 'depth_m' not in track_df.columns or track_df['depth_m'].isna().all():
        return None
    
    valid_df = track_df[track_df['depth_m'].notna()].copy()
    
    if len(valid_df) < min_points:
        return None
    
    frame_width = calibration.frame_width
    frame_height = calibration.frame_height
    
    valid_df['size_pix'] = valid_df.apply(
        lambda r: _get_bbox_size_pixels(r, frame_width, frame_height), axis=1
    )
    valid_df = valid_df.sort_values('frame').reset_index(drop=True)
    
    # Проверяем достаточное изменение глубины
    depth_change = valid_df['depth_m'].max() - valid_df['depth_m'].min()
    if depth_change < min_depth_change:
        return None
    
    # Вычисляем k и дистанцию для каждой пары соседних точек
    pair_data = []  # список (frame_mid, k, distance)
    
    for i in range(len(valid_df) - 1):
        row1 = valid_df.iloc[i]
        row2 = valid_df.iloc[i + 1]
        
        pixels1 = row1['size_pix']
        pixels2 = row2['size_pix']
        depth1 = row1['depth_m']
        depth2 = row2['depth_m']
        
        delta_depth = depth2 - depth1
        
        # Пропускаем пары с малым изменением глубины
        if abs(delta_depth) < 0.05:
            continue
        
        if pixels1 <= 0:
            continue
        
        delta_pixels = pixels2 - pixels1
        k = (delta_pixels / pixels1) / delta_depth  # доли/м
        
        # k должен быть положительным (размер растёт при погружении)
        # Если k <= 0, значит объект удаляется или данные шумные
        if k <= 0:
            continue
        
        k_percent = k * 100  # в %/м
        distance = _calculate_distance_from_k(k_percent, calibration)
        
        # Средний кадр пары
        frame_mid = (row1['frame'] + row2['frame']) / 2
        
        pair_data.append({
            'frame_mid': frame_mid,
            'k': k,
            'k_percent': k_percent,
            'distance': distance,
            'depth_camera': (depth1 + depth2) / 2
        })
    
    if len(pair_data) == 0:
        return None
    
    # Находим кадр с максимальным размером
    max_idx, max_row, max_size_pix = _find_max_size_frame(valid_df, frame_width, frame_height)
    max_frame = int(max_row['frame'])
    camera_depth_max = max_row['depth_m']
    
    # Находим ближайшую пару к кадру с максимальным размером
    closest_pair = min(pair_data, key=lambda p: abs(p['frame_mid'] - max_frame))
    
    # Дистанция на момент максимального размера
    # Корректируем по разнице глубин камеры
    depth_diff = camera_depth_max - closest_pair['depth_camera']
    distance_at_max = closest_pair['distance'] - depth_diff  # камера приблизилась
    distance_at_max = max(distance_at_max, 0.1)  # защита от отрицательных
    
    # Калибровка пикселя для этой дистанции
    pixel_calib = _calculate_pixel_calibration(distance_at_max, calibration)
    
    # Размер объекта
    size_mm = _calculate_size_mm(max_size_pix, pixel_calib)
    size_cm = size_mm / 10.0
    
    # Глубина объекта = глубина камеры + дистанция (камера смотрит вниз)
    object_depth = camera_depth_max + distance_at_max
    
    # Статистика k
    k_values = [p['k_percent'] for p in pair_data]
    k_mean = np.mean(k_values)
    k_std = np.std(k_values) if len(k_values) > 1 else 0.0
    
    # Оценка уверенности
    warnings_list = []
    confidence = 1.0
    
    if distance_at_max > calibration.max_reliable_distance:
        warnings_list.append("distance_above_reliable")
        confidence = 0.3
    elif distance_at_max > 2.0:
        warnings_list.append("distance_marginal")
        confidence = 0.6
    
    if distance_at_max < calibration.min_reliable_distance:
        warnings_list.append("distance_too_close")
        confidence *= 0.8
    
    # Проверка стабильности k
    if k_std > 0 and k_mean > 0:
        cv = k_std / k_mean  # коэффициент вариации
        if cv > 0.5:
            warnings_list.append("k_unstable")
            confidence *= 0.7
    
    if class_name in TYPICAL_SIZES_CM:
        typical = TYPICAL_SIZES_CM[class_name]
        if size_cm < typical['min'] * 0.3 or size_cm > typical['max'] * 3:
            warnings_list.append("size_outside_typical")
            confidence *= 0.5
    
    if size_mm < 5:
        warnings_list.append("size_too_small")
    if size_mm > 1000:
        warnings_list.append("size_too_large")
    
    return TrackSizeEstimate(
        track_id=track_id,
        class_name=class_name,
        real_size_mm=round(size_mm, 1),
        real_size_cm=round(size_cm, 2),
        distance_m=round(distance_at_max, 3),
        object_depth_m=round(object_depth, 2),
        first_frame=max_frame,
        first_size_pixels=round(max_size_pix, 1),
        camera_depth_first=round(camera_depth_max, 2),
        k_mean=round(k_mean, 2),
        k_std=round(k_std, 2),
        pixel_calibration=round(pixel_calib, 4),
        confidence=round(confidence, 3),
        method="k_method",
        n_points_used=len(pair_data),
        warnings=warnings_list
    )


def estimate_size_from_typical(
    track_df: pd.DataFrame,
    calibration: CameraCalibration
) -> Optional[TrackSizeEstimate]:
    """
    Оценивает размер на основе типичных размеров вида.
    Используется как fallback когда k-метод не работает.
    """
    track_id = track_df['track_id'].iloc[0]
    class_name = track_df['class_name'].iloc[0]
    
    if class_name in SKIP_SIZE_CLASSES:
        return None
    
    if class_name not in TYPICAL_SIZES_CM:
        return None
    
    typical = TYPICAL_SIZES_CM[class_name]
    typical_size_mm = typical['mean'] * 10  # см -> мм
    
    frame_width = calibration.frame_width
    frame_height = calibration.frame_height
    
    valid_df = track_df[track_df['depth_m'].notna()].copy() if 'depth_m' in track_df.columns else track_df.copy()
    if len(valid_df) == 0:
        valid_df = track_df.copy()
    
    _, first_row, first_size_pix = _get_first_frame_data(valid_df, frame_width, frame_height)
    
    camera_depth_first = first_row.get('depth_m', np.nan)
    first_frame = int(first_row['frame'])
    
    # Обратная калибровка: из типичного размера и пикселей находим p, затем d
    pixel_calib = first_size_pix / typical_size_mm
    
    C = calibration.pixel_calib_C
    D = calibration.pixel_calib_D
    
    if pixel_calib > 0 and C > 0:
        distance = (pixel_calib / C) ** (1.0 / D)
    else:
        distance = 1.5
    
    distance = np.clip(distance, 0.2, 5.0)
    
    if pd.notna(camera_depth_first):
        object_depth = camera_depth_first - distance
    else:
        object_depth = np.nan
    
    return TrackSizeEstimate(
        track_id=track_id,
        class_name=class_name,
        real_size_mm=round(typical_size_mm, 1),
        real_size_cm=round(typical['mean'], 2),
        distance_m=round(distance, 3),
        object_depth_m=round(object_depth, 2) if pd.notna(object_depth) else None,
        first_frame=first_frame,
        first_size_pixels=round(first_size_pix, 1),
        camera_depth_first=round(camera_depth_first, 2) if pd.notna(camera_depth_first) else None,
        k_mean=0.0,
        k_std=0.0,
        pixel_calibration=round(pixel_calib, 4),
        confidence=0.2,
        method="typical",
        n_points_used=len(valid_df),
        warnings=["estimated_from_typical_size"]
    )


# =============================================================================
# Загрузка данных геометрии
# =============================================================================

def load_geometry_data(geometry_csv: str) -> Optional[pd.DataFrame]:
    """Загружает данные о наклоне камеры."""
    if geometry_csv and Path(geometry_csv).exists():
        return pd.read_csv(geometry_csv)
    return None


# =============================================================================
# Основная функция обработки детекций
# =============================================================================

def process_detections_with_size(
    detections_csv: str,
    output_csv: Optional[str] = None,
    tracks_output_csv: Optional[str] = None,
    geometry_csv: Optional[str] = None,
    calibration: CameraCalibration = None,
    frame_width: int = 3840,
    frame_height: int = 2160,
    min_depth_change: float = 0.1,
    min_track_points: int = 3,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Обрабатывает детекции и добавляет оценки размеров.
    
    Стратегия:
    1. Сначала пытаемся k-метод для треков с хорошими данными
    2. Fallback — оценка по типичным размерам вида
    """
    if calibration is None:
        calibration = CameraCalibration()
    calibration.frame_width = frame_width
    calibration.frame_height = frame_height
    
    df = pd.read_csv(detections_csv)
    
    if verbose:
        print(f"Загружено детекций: {len(df)}")
        print(f"Уникальных треков: {df['track_id'].nunique()}")
        print(f"Разрешение: {frame_width}x{frame_height}")
    
    required_cols = ['track_id', 'frame', 'width', 'height', 'class_name', 'x_center', 'y_center']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют колонки: {missing}")
    
    # Этап 1: k-метод для хороших треков
    k_method_estimates = []
    tracks_for_typical = []
    
    for track_id, track_df in df.groupby('track_id'):
        if pd.isna(track_id):
            continue
        
        class_name = track_df['class_name'].iloc[0]
        if class_name in SKIP_SIZE_CLASSES:
            continue
        
        estimate = estimate_size_by_k_method(
            track_df, calibration,
            min_depth_change=min_depth_change,
            min_points=min_track_points
        )
        
        if estimate is not None:
            k_method_estimates.append(estimate)
        else:
            tracks_for_typical.append((track_id, track_df))
    
    if verbose:
        print(f"\nТреков с k-методом: {len(k_method_estimates)}")
        print(f"Треков для типичной оценки: {len(tracks_for_typical)}")
    
    # Этап 2: оценка по типичным размерам
    typical_estimates = []
    
    for track_id, track_df in tracks_for_typical:
        estimate = estimate_size_from_typical(track_df, calibration)
        if estimate is not None:
            typical_estimates.append(estimate)
    
    if verbose:
        print(f"Треков с типичной оценкой: {len(typical_estimates)}")
    
    # Объединяем все оценки
    all_estimates = k_method_estimates + typical_estimates
    
    if verbose:
        print(f"\nВсего треков с оценкой: {len(all_estimates)}")
    
    # Создаём DataFrame треков
    if all_estimates:
        tracks_df = pd.DataFrame([
            {
                'track_id': e.track_id,
                'class_name': e.class_name,
                'real_size_mm': e.real_size_mm,
                'real_size_cm': e.real_size_cm,
                'distance_m': e.distance_m,
                'object_depth_m': e.object_depth_m,
                'first_frame': e.first_frame,
                'first_size_pixels': e.first_size_pixels,
                'camera_depth_first_m': e.camera_depth_first,
                'k_mean_pct_per_m': e.k_mean,
                'k_std_pct_per_m': e.k_std,
                'pixel_calibration': e.pixel_calibration,
                'confidence': e.confidence,
                'method': e.method,
                'n_points': e.n_points_used,
                'warnings': ';'.join(e.warnings) if e.warnings else ''
            }
            for e in all_estimates
        ])
    else:
        tracks_df = pd.DataFrame()
    
    # Добавляем к детекциям
    size_map = {e.track_id: e for e in all_estimates}
    
    df['estimated_size_mm'] = None
    df['estimated_size_cm'] = None
    df['object_depth_m'] = None
    df['distance_to_object_m'] = None
    df['size_confidence'] = None
    df['size_method'] = None
    
    for idx, row in df.iterrows():
        track_id = row['track_id']
        if pd.notna(track_id) and track_id in size_map:
            est = size_map[track_id]
            df.at[idx, 'estimated_size_mm'] = est.real_size_mm
            df.at[idx, 'estimated_size_cm'] = est.real_size_cm
            df.at[idx, 'object_depth_m'] = est.object_depth_m
            df.at[idx, 'size_confidence'] = est.confidence
            df.at[idx, 'size_method'] = est.method
            
            camera_depth = row.get('depth_m')
            if pd.notna(camera_depth) and est.object_depth_m is not None:
                df.at[idx, 'distance_to_object_m'] = round(abs(est.object_depth_m - camera_depth), 3)
    
    # Сохранение
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        if verbose:
            print(f"\nДетекции сохранены: {output_csv}")
    
    if tracks_output_csv and len(tracks_df) > 0:
        Path(tracks_output_csv).parent.mkdir(parents=True, exist_ok=True)
        tracks_df.to_csv(tracks_output_csv, index=False)
        if verbose:
            print(f"Статистика треков: {tracks_output_csv}")
    
    # Статистика
    if verbose and len(tracks_df) > 0:
        print("\n" + "="*60)
        print("СТАТИСТИКА РАЗМЕРОВ ПО ВИДАМ")
        print("="*60)
        
        for class_name in sorted(tracks_df['class_name'].unique()):
            class_df = tracks_df[tracks_df['class_name'] == class_name]
            sizes = class_df['real_size_cm']
            
            print(f"\n{class_name}:")
            print(f"  Треков: {len(class_df)}")
            print(f"  Размер: {sizes.mean():.1f} ± {sizes.std():.1f} см")
            print(f"    диапазон: {sizes.min():.1f} - {sizes.max():.1f} см")
            
            methods = class_df['method'].value_counts()
            print(f"  Методы: {dict(methods)}")
    
    return df, tracks_df


# =============================================================================
# Обработка геометрии видео (FOE)
# =============================================================================

def process_video_geometry(
    video_path: str,
    output_csv: Optional[str] = None,
    frame_interval: int = 30,
    calibration: CameraCalibration = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Обрабатывает видео и оценивает наклон камеры."""
    if calibration is None:
        calibration = CameraCalibration()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if verbose:
        print(f"Видео: {video_path}")
        print(f"  {width}x{height}, {fps:.1f} fps, {total_frames} кадров")
    
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
            foe = estimate_foe(points_arr, vectors_arr, (width, height))
            
            results.append({
                'frame_start': interval_start,
                'frame_end': frame_idx,
                'timestamp_s': round(frame_idx / fps, 2),
                'foe_x': round(foe.foe_x, 1),
                'foe_y': round(foe.foe_y, 1),
                'tilt_horizontal_deg': round(foe.tilt_horizontal, 1),
                'tilt_vertical_deg': round(foe.tilt_vertical, 1),
                'confidence': round(foe.confidence, 3),
                'n_vectors': foe.n_vectors
            })
            
            interval_points = []
            interval_vectors = []
            interval_start = frame_idx
        
        prev_gray = gray
    
    cap.release()
    
    df = pd.DataFrame(results)
    
    if output_csv and len(df) > 0:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        if verbose:
            print(f"Сохранено: {output_csv}")
    
    if verbose and len(df) > 0:
        if 'confidence' in df.columns:
            valid_df = df[df['confidence'] >= 0.5]
        else:
            valid_df = df
        
        if len(valid_df) > 0:
            total_tilt = np.sqrt(
                valid_df['tilt_horizontal_deg'].mean()**2 + 
                valid_df['tilt_vertical_deg'].mean()**2
            )
            n_outliers = len(df) - len(valid_df)
            outlier_info = f" (отфильтровано выбросов: {n_outliers})" if n_outliers > 0 else ""
            print(f"Средний наклон: {total_tilt:.1f}°{outlier_info}")
    
    return df


# =============================================================================
# Расчёт объёма (будет в следующей части)
# =============================================================================

@dataclass
class VolumeEstimate:
    """Результат оценки осмотренного объёма воды."""
    total_volume_m3: float
    frustum_volume_m3: float
    swept_volume_m3: float
    depth_range_m: Tuple[float, float]
    depth_traversed_m: float
    detection_distance_m: float
    near_distance_m: float
    cross_section_area_m2: float
    fov_horizontal_deg: float
    fov_vertical_deg: float
    duration_s: float
    descent_rate_m_s: float
    density_by_class: Dict[str, float]
    counts_by_class: Dict[str, int]


DEFAULT_DETECTION_DISTANCES = {
    'Aurelia aurita': 2.0,
    'Rhizostoma pulmo': 2.5,
    'Mnemiopsis leidyi': 1.0,
    'Beroe ovata': 1.2,
    'Pleurobrachia pileus': 0.5
}


def calculate_frustum_volume(d_near: float, d_far: float, fov_h_rad: float, fov_v_rad: float) -> float:
    """Вычисляет объём усечённой пирамиды (frustum)."""
    w_near = 2 * d_near * np.tan(fov_h_rad / 2)
    h_near = 2 * d_near * np.tan(fov_v_rad / 2)
    A_near = w_near * h_near
    
    w_far = 2 * d_far * np.tan(fov_h_rad / 2)
    h_far = 2 * d_far * np.tan(fov_v_rad / 2)
    A_far = w_far * h_far
    
    depth = d_far - d_near
    V = (depth / 3) * (A_near + A_far + np.sqrt(A_near * A_far))
    return V


def calculate_cross_section_area(distance: float, fov_h_rad: float, fov_v_rad: float) -> float:
    """Вычисляет площадь поперечного сечения на заданной дистанции."""
    w = 2 * distance * np.tan(fov_h_rad / 2)
    h = 2 * distance * np.tan(fov_v_rad / 2)
    return w * h


def estimate_detection_distance(
    tracks_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    calibration: CameraCalibration,
    reference_class: str = 'Aurelia aurita'
) -> float:
    """Оценивает эффективную дистанцию обнаружения по данным треков."""
    if tracks_df is None or len(tracks_df) == 0:
        return DEFAULT_DETECTION_DISTANCES.get(reference_class, 1.5)
    
    if 'method' not in tracks_df.columns:
        return DEFAULT_DETECTION_DISTANCES.get(reference_class, 1.5)
    
    ref_tracks = tracks_df[
        (tracks_df['class_name'] == reference_class) & 
        (tracks_df['method'] == 'k_method')
    ]
    
    if len(ref_tracks) == 0:
        ref_tracks = tracks_df[tracks_df['method'] == 'k_method']
    
    if len(ref_tracks) == 0:
        return DEFAULT_DETECTION_DISTANCES.get(reference_class, 1.5)
    
    max_distances = []
    
    for _, track in ref_tracks.iterrows():
        track_id = track['track_id']
        object_depth = track['object_depth_m']
        
        if pd.isna(object_depth):
            continue
        
        track_detections = detections_df[detections_df['track_id'] == track_id]
        if len(track_detections) == 0:
            continue
        
        min_camera_depth = track_detections['depth_m'].min()
        max_dist = abs(object_depth - min_camera_depth)
        
        if 0.5 < max_dist < 5.0:
            max_distances.append(max_dist)
    
    if max_distances:
        return np.mean(max_distances) + np.std(max_distances) * 0.5
    
    return DEFAULT_DETECTION_DISTANCES.get(reference_class, 1.5)


def calculate_surveyed_volume(
    detections_df: pd.DataFrame,
    tracks_df: Optional[pd.DataFrame] = None,
    ctd_df: Optional[pd.DataFrame] = None,
    calibration: CameraCalibration = None,
    fov_horizontal_deg: float = 100.0,
    near_distance_m: float = 0.3,
    detection_distance_m: Optional[float] = None,
    depth_range: Optional[Tuple[float, float]] = None,
    total_duration_s: Optional[float] = None,
    fps: float = 60.0,
    verbose: bool = True
) -> VolumeEstimate:
    """Вычисляет осмотренный объём воды на основе данных погружения."""
    if calibration is None:
        calibration = CameraCalibration()
    
    fov_h_deg = fov_horizontal_deg
    aspect_ratio = calibration.frame_width / calibration.frame_height
    fov_v_deg = fov_h_deg / aspect_ratio
    
    fov_h_rad = np.radians(fov_h_deg)
    fov_v_rad = np.radians(fov_v_deg)
    
    if depth_range is not None:
        depth_min, depth_max = depth_range
        source = "явно задан"
    elif ctd_df is not None and 'depth_m' in ctd_df.columns:
        depths = ctd_df['depth_m'].dropna()
        if len(depths) > 0:
            depth_min = depths.min()
            depth_max = depths.max()
            source = "CTD"
        else:
            raise ValueError("Нет данных о глубине в CTD")
    else:
        depths = detections_df['depth_m'].dropna()
        if len(depths) == 0:
            raise ValueError("Нет данных о глубине")
        depth_min = depths.min()
        depth_max = depths.max()
        source = "детекции"
    
    depth_traversed = depth_max - depth_min
    
    if total_duration_s is not None:
        duration = total_duration_s
    elif ctd_df is not None and 'timestamp_s' in ctd_df.columns:
        timestamps = ctd_df['timestamp_s'].dropna()
        duration = timestamps.max() - timestamps.min() if len(timestamps) > 1 else 0
    else:
        frames = detections_df['frame'].dropna()
        if len(frames) > 1:
            duration = (frames.max() - frames.min()) / fps
        else:
            timestamps = detections_df['timestamp_s'].dropna()
            duration = timestamps.max() - timestamps.min() if len(timestamps) > 1 else 0
    
    descent_rate = depth_traversed / duration if duration > 0 else 0
    
    if detection_distance_m is None:
        d_far = estimate_detection_distance(tracks_df, detections_df, calibration)
    else:
        d_far = detection_distance_m
    
    d_near = near_distance_m
    
    if verbose:
        print(f"=== РАСЧЁТ ОСМОТРЕННОГО ОБЪЁМА ===")
        print(f"Источник: {source}")
        print(f"Диапазон глубин: {depth_min:.1f} - {depth_max:.1f} м")
        print(f"Дистанция обнаружения: {d_near:.1f} - {d_far:.2f} м")
    
    V_frustum = calculate_frustum_volume(d_near, d_far, fov_h_rad, fov_v_rad)
    A_far = calculate_cross_section_area(d_far, fov_h_rad, fov_v_rad)
    V_swept = A_far * depth_traversed
    V_total = V_frustum + V_swept
    
    if verbose:
        print(f"ИТОГО объём: {V_total:.1f} м³")
    
    counts_by_class = {}
    density_by_class = {}
    
    for class_name in detections_df['class_name'].unique():
        if class_name in SKIP_SIZE_CLASSES:
            continue
        
        class_df = detections_df[detections_df['class_name'] == class_name]
        n_tracks = class_df['track_id'].nunique()
        
        if n_tracks == 0 or class_df['track_id'].isna().all():
            n_tracks = len(class_df)
        
        counts_by_class[class_name] = n_tracks
        density_by_class[class_name] = n_tracks / V_total if V_total > 0 else 0
    
    return VolumeEstimate(
        total_volume_m3=round(V_total, 2),
        frustum_volume_m3=round(V_frustum, 2),
        swept_volume_m3=round(V_swept, 2),
        depth_range_m=(round(depth_min, 2), round(depth_max, 2)),
        depth_traversed_m=round(depth_traversed, 2),
        detection_distance_m=round(d_far, 2),
        near_distance_m=round(d_near, 2),
        cross_section_area_m2=round(A_far, 2),
        fov_horizontal_deg=fov_h_deg,
        fov_vertical_deg=round(fov_v_deg, 1),
        duration_s=round(duration, 1),
        descent_rate_m_s=round(descent_rate, 3),
        density_by_class=density_by_class,
        counts_by_class=counts_by_class
    )


def process_volume_estimation(
    detections_csv: str,
    tracks_csv: Optional[str] = None,
    ctd_csv: Optional[str] = None,
    output_csv: Optional[str] = None,
    fov_horizontal: float = 100.0,
    near_distance: float = 0.3,
    detection_distance: Optional[float] = None,
    depth_min: Optional[float] = None,
    depth_max: Optional[float] = None,
    total_duration: Optional[float] = None,
    fps: float = 60.0,
    frame_width: int = 3840,
    frame_height: int = 2160,
    verbose: bool = True
) -> VolumeEstimate:
    """Обрабатывает файлы и вычисляет осмотренный объём."""
    calibration = CameraCalibration()
    calibration.frame_width = frame_width
    calibration.frame_height = frame_height
    
    detections_df = pd.read_csv(detections_csv)
    
    tracks_df = None
    if tracks_csv and Path(tracks_csv).exists():
        tracks_df = pd.read_csv(tracks_csv)
    
    ctd_df = None
    if ctd_csv and Path(ctd_csv).exists():
        ctd_df = pd.read_csv(ctd_csv)
        column_mapping = {}
        for col in ctd_df.columns:
            col_lower = col.lower().strip()
            if col_lower in ('depth', 'depth_m', 'depth (m)', 'глубина'):
                column_mapping[col] = 'depth_m'
            elif col_lower in ('time', 'time_s', 'timestamp', 'timestamp_s', 'время'):
                column_mapping[col] = 'timestamp_s'
        if column_mapping:
            ctd_df = ctd_df.rename(columns=column_mapping)
    
    depth_range = None
    if depth_min is not None and depth_max is not None:
        depth_range = (depth_min, depth_max)
    
    result = calculate_surveyed_volume(
        detections_df=detections_df,
        tracks_df=tracks_df,
        ctd_df=ctd_df,
        calibration=calibration,
        fov_horizontal_deg=fov_horizontal,
        near_distance_m=near_distance,
        detection_distance_m=detection_distance,
        depth_range=depth_range,
        total_duration_s=total_duration,
        fps=fps,
        verbose=verbose
    )
    
    if output_csv:
        output_data = {
            'parameter': [
                'total_volume_m3', 'frustum_volume_m3', 'swept_volume_m3',
                'depth_min_m', 'depth_max_m', 'depth_traversed_m',
                'detection_distance_m', 'near_distance_m', 'cross_section_area_m2',
                'fov_horizontal_deg', 'fov_vertical_deg', 'duration_s', 'descent_rate_m_s'
            ],
            'value': [
                result.total_volume_m3, result.frustum_volume_m3, result.swept_volume_m3,
                result.depth_range_m[0], result.depth_range_m[1], result.depth_traversed_m,
                result.detection_distance_m, result.near_distance_m, result.cross_section_area_m2,
                result.fov_horizontal_deg, result.fov_vertical_deg, result.duration_s, result.descent_rate_m_s
            ]
        }
        
        for class_name, count in result.counts_by_class.items():
            output_data['parameter'].append(f'count_{class_name.replace(" ", "_")}')
            output_data['value'].append(count)
            output_data['parameter'].append(f'density_{class_name.replace(" ", "_")}_per_m3')
            output_data['value'].append(result.density_by_class[class_name])
        
        output_df = pd.DataFrame(output_data)
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_csv, index=False)
        
        if verbose:
            print(f"\nРезультаты сохранены: {output_csv}")
    
    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Оценка геометрии и размеров")
    subparsers = parser.add_subparsers(dest='command')
    
    # geometry
    geom = subparsers.add_parser('geometry', help='Оценка наклона камеры')
    geom.add_argument('--video', '-v', required=True)
    geom.add_argument('--output', '-o', default='output/geometry.csv')
    geom.add_argument('--interval', '-i', type=int, default=30)
    
    # size
    size = subparsers.add_parser('size', help='Оценка размеров по трекам')
    size.add_argument('--detections', '-d', required=True)
    size.add_argument('--output', '-o')
    size.add_argument('--tracks', '-t')
    size.add_argument('--geometry', '-g')
    size.add_argument('--width', type=int, default=3840)
    size.add_argument('--height', type=int, default=2160)
    size.add_argument('--min-depth-change', type=float, default=0.1)
    size.add_argument('--min-track-points', type=int, default=3)
    
    # volume
    vol = subparsers.add_parser('volume', help='Расчёт осмотренного объёма воды')
    vol.add_argument('--detections', '-d', required=True)
    vol.add_argument('--tracks', '-t')
    vol.add_argument('--ctd', '-c')
    vol.add_argument('--output', '-o')
    vol.add_argument('--fov', type=float, default=100.0)
    vol.add_argument('--near-distance', type=float, default=0.3)
    vol.add_argument('--detection-distance', type=float)
    vol.add_argument('--depth-min', type=float)
    vol.add_argument('--depth-max', type=float)
    vol.add_argument('--duration', type=float)
    vol.add_argument('--fps', type=float, default=60.0)
    vol.add_argument('--width', type=int, default=3840)
    vol.add_argument('--height', type=int, default=2160)
    
    args = parser.parse_args()
    
    if args.command == 'geometry':
        process_video_geometry(args.video, args.output, args.interval)
    
    elif args.command == 'size':
        output = args.output or args.detections.replace('.csv', '_with_size.csv')
        tracks = args.tracks or args.detections.replace('.csv', '_track_sizes.csv')
        process_detections_with_size(
            args.detections, output, tracks, args.geometry,
            frame_width=args.width, frame_height=args.height,
            min_depth_change=args.min_depth_change,
            min_track_points=args.min_track_points
        )
    
    elif args.command == 'volume':
        output = args.output or args.detections.replace('.csv', '_volume.csv')
        process_volume_estimation(
            detections_csv=args.detections,
            tracks_csv=args.tracks,
            ctd_csv=args.ctd,
            output_csv=output,
            fov_horizontal=args.fov,
            near_distance=args.near_distance,
            detection_distance=args.detection_distance,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            total_duration=args.duration,
            fps=args.fps,
            frame_width=args.width,
            frame_height=args.height
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
