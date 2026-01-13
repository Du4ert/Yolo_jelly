"""
Модуль оценки геометрии камеры и размеров объектов.

Функции:
- Оценка наклона камеры по морскому снегу (Focus of Expansion)
- Расчёт реального размера объектов по динамике изменения размера в треке
- Коррекция дисторсии GoPro fisheye
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
    """Калибровочные параметры камеры GoPro 12 Wide."""
    # Калибровочная константа k = размер_пикс × дистанция
    k: float = 154.8  # после коррекции дисторсии
    
    # Размер кадра
    frame_width: int = 1920
    frame_height: int = 1080
    
    # Параметры fisheye дисторсии: correction = 1 + a*r² + b*r⁴
    distortion_a: float = 0.3146
    distortion_b: float = 0.0382
    
    # Нормализующий радиус (диагональ/2 для 4K)
    distortion_r_max: float = 2203.0
    
    # Угол обзора
    fov_horizontal: float = 100.0
    
    # Диапазон надёжных измерений
    min_reliable_distance: float = 0.5
    max_reliable_distance: float = 3.0
    
    @property
    def K(self) -> float:
        """Коэффициент K = k / эталонный_размер."""
        return self.k / 0.066  # ≈ 2345
    
    @property
    def pixels_per_degree(self) -> float:
        return self.frame_width / self.fov_horizontal
    
    @property
    def frame_center(self) -> Tuple[float, float]:
        return self.frame_width / 2, self.frame_height / 2
    
    def get_distortion_correction(self, x: float, y: float) -> float:
        """
        Вычисляет коэффициент коррекции дисторсии для точки (x, y).
        
        Fisheye растягивает объекты радиально от центра.
        Возвращает коэффициент, на который нужно ДЕЛИТЬ измеренный размер.
        """
        cx, cy = self.frame_center
        
        # Масштабируем для текущего разрешения относительно 4K
        scale = self.frame_width / 1920.0
        r_max_scaled = self.distortion_r_max / 2.0 * scale  # для HD
        
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        r = dist / r_max_scaled
        r = min(r, 1.5)  # ограничиваем для экстраполяции
        
        correction = 1 + self.distortion_a * r**2 + self.distortion_b * r**4
        return correction


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
    """Результат оценки размера объекта по треку."""
    track_id: int
    class_name: str
    real_size_m: float
    real_size_cm: float
    distance_at_max_size: float
    object_depth_m: float
    camera_depth_at_max: float
    max_size_frame: int
    max_size_pixels: float
    confidence: float
    method: str  # 'regression', 'reference', 'typical'
    n_points_used: int
    fit_r_squared: Optional[float]
    warnings: List[str] = field(default_factory=list)


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
    
    Args:
        points: координаты точек (N, 2)
        vectors: векторы смещения (N, 2)
        frame_size: размер кадра (width, height)
        min_vector_length: минимальная длина вектора для учёта
        max_tilt_deg: максимальный допустимый наклон (градусы)
    
    Returns:
        FOEResult с оценкой FOE и наклона
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
    
    # Вычисляем наклон
    pixels_per_degree = width / 100.0
    tilt_h = (foe_x - cx) / pixels_per_degree
    tilt_v = (foe_y - cy) / pixels_per_degree
    
    # Проверяем на выбросы: если FOE слишком далеко от кадра или наклон нереален
    # FOE дальше 3 диагоналей от центра — признак малого движения
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
        # Возвращаем центр кадра с нулевым наклоном и низкой уверенностью
        # Это означает "камера не двигалась достаточно для определения наклона"
        return FOEResult(
            foe_x=cx,
            foe_y=cy,
            tilt_horizontal=0.0,
            tilt_vertical=0.0,
            confidence=0.0,  # Нулевая уверенность — признак невалидного результата
            n_vectors=len(points_filt)
        )
    
    return FOEResult(foe_x, foe_y, tilt_h, tilt_v, confidence, len(points_filt))


def _get_bbox_size_pixels(
    row: pd.Series, 
    frame_width: int, 
    frame_height: int,
    calibration: CameraCalibration,
    apply_distortion_correction: bool = True
) -> float:
    """
    Получает размер bbox в пикселях с коррекцией дисторсии.
    
    Берёт максимум из ширины и высоты (для объектов которые могут поворачиваться).
    Применяет коррекцию fisheye дисторсии по позиции в кадре.
    """
    w_pix = row['width'] * frame_width
    h_pix = row['height'] * frame_height
    size_pix = max(w_pix, h_pix)
    
    if apply_distortion_correction:
        # Позиция центра bbox
        x = row['x_center'] * frame_width
        y = row['y_center'] * frame_height
        
        correction = calibration.get_distortion_correction(x, y)
        size_pix = size_pix / correction
    
    return size_pix


def _find_max_size_frame(
    track_df: pd.DataFrame, 
    frame_width: int, 
    frame_height: int,
    calibration: CameraCalibration
) -> Tuple[int, pd.Series, float]:
    """
    Находит кадр с максимальным размером объекта.
    
    Исключает последние кадры, где размер может падать из-за выхода за границы.
    Возвращает (индекс, строку, размер_в_пикселях).
    """
    track_df = track_df.copy()
    track_df['size_pix'] = track_df.apply(
        lambda r: _get_bbox_size_pixels(r, frame_width, frame_height, calibration), 
        axis=1
    )
    
    max_idx = track_df['size_pix'].idxmax()
    max_pos = track_df.index.get_loc(max_idx)
    
    n_points = len(track_df)
    if n_points > 3 and max_pos >= n_points - 2:
        cutoff = int(n_points * 0.8)
        if cutoff > 0:
            track_subset = track_df.iloc[:cutoff]
            if len(track_subset) > 0:
                max_idx = track_subset['size_pix'].idxmax()
    
    max_row = track_df.loc[max_idx]
    max_size = max_row['size_pix']
    
    return max_idx, max_row, max_size


def estimate_size_from_track_regression(
    track_df: pd.DataFrame,
    calibration: CameraCalibration,
    camera_tilt_deg: float = 0.0,
    min_depth_change: float = 0.3,
    min_points: int = 5,
    min_r_squared: float = 0.5,
    min_size_change_ratio: float = 0.3
) -> Optional[TrackSizeEstimate]:
    """
    Оценивает размер объекта методом линейной регрессии 1/size vs depth.
    
    Возвращает None если данных недостаточно или фит плохой.
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
    
    depth_change = valid_df['depth_m'].max() - valid_df['depth_m'].min()
    if depth_change < min_depth_change:
        return None
    
    frame_width = calibration.frame_width
    frame_height = calibration.frame_height
    
    valid_df['size_pix'] = valid_df.apply(
        lambda r: _get_bbox_size_pixels(r, frame_width, frame_height, calibration),
        axis=1
    )
    
    # Проверка изменения размера
    size_min = valid_df['size_pix'].min()
    size_max = valid_df['size_pix'].max()
    if size_max <= 0:
        return None
    size_change_ratio = (size_max - size_min) / size_max
    
    if size_change_ratio < min_size_change_ratio:
        return None
    
    # Фильтр выбросов
    size_mean = valid_df['size_pix'].mean()
    size_std = valid_df['size_pix'].std()
    if size_std > 0:
        valid_df = valid_df[
            (valid_df['size_pix'] > size_mean - 3*size_std) &
            (valid_df['size_pix'] < size_mean + 3*size_std)
        ]
    
    if len(valid_df) < min_points:
        return None
    
    camera_depths = valid_df['depth_m'].values
    sizes_pix = np.maximum(valid_df['size_pix'].values, 1.0)
    inv_sizes = 1.0 / sizes_pix
    
    slope, intercept, r_value, _, _ = linregress(camera_depths, inv_sizes)
    r_squared = r_value ** 2
    
    if r_squared < min_r_squared:
        return None
    
    if abs(slope) < 1e-10:
        return None
    
    K = calibration.K
    
    # Коррекция на наклон камеры
    tilt_correction = np.cos(np.radians(abs(camera_tilt_deg))) if camera_tilt_deg != 0 else 1.0
    
    real_size = 1.0 / (K * abs(slope))
    
    if slope < 0:
        object_depth = intercept * K * real_size
    else:
        object_depth = -intercept * K * real_size
    
    _, max_row, max_size_pix = _find_max_size_frame(valid_df, frame_width, frame_height, calibration)
    camera_depth_at_max = max_row['depth_m']
    max_frame = max_row['frame']
    
    vertical_distance = abs(object_depth - camera_depth_at_max)
    distance_at_max = vertical_distance / tilt_correction if tilt_correction > 0 else vertical_distance
    
    warnings_list = []
    if real_size < 0.005:
        warnings_list.append("size_too_small")
    if real_size > 1.0:
        warnings_list.append("size_too_large")
    if distance_at_max > calibration.max_reliable_distance:
        warnings_list.append("distance_above_max")
    
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
        confidence=round(r_squared, 3),
        method="regression",
        n_points_used=len(valid_df),
        fit_r_squared=round(r_squared, 4),
        warnings=warnings_list
    )


def estimate_size_from_reference(
    track_df: pd.DataFrame,
    calibration: CameraCalibration,
    reference_estimates: List[TrackSizeEstimate],
    camera_tilt_deg: float = 0.0
) -> Optional[TrackSizeEstimate]:
    """
    Оценивает размер объекта по референсным данным от хороших треков того же класса.
    
    Находит ближайшие по глубине хорошие треки и интерполирует дистанцию.
    """
    track_id = track_df['track_id'].iloc[0]
    class_name = track_df['class_name'].iloc[0]
    
    if class_name in SKIP_SIZE_CLASSES:
        return None
    
    # Находим референсные треки того же класса
    same_class_refs = [e for e in reference_estimates if e.class_name == class_name]
    
    if not same_class_refs:
        return None
    
    frame_width = calibration.frame_width
    frame_height = calibration.frame_height
    
    # Находим максимальный размер и глубину для текущего трека
    valid_df = track_df[track_df['depth_m'].notna()].copy()
    if len(valid_df) == 0:
        return None
    
    _, max_row, max_size_pix = _find_max_size_frame(valid_df, frame_width, frame_height, calibration)
    camera_depth_at_max = max_row['depth_m']
    max_frame = max_row['frame']
    
    # Находим ближайший референсный трек по глубине камеры
    closest_ref = min(same_class_refs, key=lambda e: abs(e.camera_depth_at_max - camera_depth_at_max))
    
    # Оцениваем дистанцию как среднюю от референсных
    ref_distances = [e.distance_at_max_size for e in same_class_refs]
    estimated_distance = np.mean(ref_distances)
    
    # Коррекция на наклон
    tilt_correction = np.cos(np.radians(abs(camera_tilt_deg))) if camera_tilt_deg != 0 else 1.0
    
    # Вычисляем размер
    K = calibration.K
    real_size = max_size_pix * estimated_distance / K
    
    # Глубина объекта
    object_depth = camera_depth_at_max - estimated_distance * tilt_correction
    
    # Уверенность ниже чем у регрессии
    confidence = 0.5
    
    warnings_list = ["estimated_from_reference"]
    
    # Проверка разумности
    if class_name in TYPICAL_SIZES_CM:
        typical = TYPICAL_SIZES_CM[class_name]
        if real_size * 100 < typical['min'] * 0.5 or real_size * 100 > typical['max'] * 2:
            warnings_list.append("size_outside_typical_range")
            confidence *= 0.5
    
    return TrackSizeEstimate(
        track_id=track_id,
        class_name=class_name,
        real_size_m=round(real_size, 4),
        real_size_cm=round(real_size * 100, 2),
        distance_at_max_size=round(estimated_distance, 3),
        object_depth_m=round(object_depth, 2),
        camera_depth_at_max=round(camera_depth_at_max, 2),
        max_size_frame=int(max_frame),
        max_size_pixels=round(max_size_pix, 1),
        confidence=round(confidence, 3),
        method="reference",
        n_points_used=len(valid_df),
        fit_r_squared=None,
        warnings=warnings_list
    )


def estimate_size_from_typical(
    track_df: pd.DataFrame,
    calibration: CameraCalibration,
    camera_tilt_deg: float = 0.0
) -> Optional[TrackSizeEstimate]:
    """
    Оценивает размер на основе типичных размеров вида.
    
    Используется как fallback когда нет ни регрессии, ни референсов.
    """
    track_id = track_df['track_id'].iloc[0]
    class_name = track_df['class_name'].iloc[0]
    
    if class_name in SKIP_SIZE_CLASSES:
        return None
    
    if class_name not in TYPICAL_SIZES_CM:
        return None
    
    typical = TYPICAL_SIZES_CM[class_name]
    
    frame_width = calibration.frame_width
    frame_height = calibration.frame_height
    
    valid_df = track_df[track_df['depth_m'].notna()].copy()
    if len(valid_df) == 0:
        valid_df = track_df.copy()
    
    _, max_row, max_size_pix = _find_max_size_frame(
        valid_df if len(valid_df) > 0 else track_df, 
        frame_width, frame_height, calibration
    )
    
    camera_depth_at_max = max_row.get('depth_m', np.nan)
    max_frame = max_row['frame']
    
    # Используем средний типичный размер
    typical_size_m = typical['mean'] / 100.0
    
    # Вычисляем дистанцию исходя из типичного размера
    K = calibration.K
    estimated_distance = typical_size_m * K / max_size_pix
    
    # Ограничиваем разумными пределами
    estimated_distance = np.clip(estimated_distance, 0.2, 5.0)
    
    # Глубина объекта
    tilt_correction = np.cos(np.radians(abs(camera_tilt_deg))) if camera_tilt_deg != 0 else 1.0
    
    if pd.notna(camera_depth_at_max):
        object_depth = camera_depth_at_max - estimated_distance * tilt_correction
    else:
        object_depth = np.nan
    
    # Пересчитываем размер для консистентности
    real_size = max_size_pix * estimated_distance / K
    
    confidence = 0.3  # низкая уверенность
    warnings_list = ["estimated_from_typical_size"]
    
    return TrackSizeEstimate(
        track_id=track_id,
        class_name=class_name,
        real_size_m=round(real_size, 4),
        real_size_cm=round(real_size * 100, 2),
        distance_at_max_size=round(estimated_distance, 3),
        object_depth_m=round(object_depth, 2) if pd.notna(object_depth) else None,
        camera_depth_at_max=round(camera_depth_at_max, 2) if pd.notna(camera_depth_at_max) else None,
        max_size_frame=int(max_frame),
        max_size_pixels=round(max_size_pix, 1),
        confidence=round(confidence, 3),
        method="typical",
        n_points_used=len(valid_df),
        fit_r_squared=None,
        warnings=warnings_list
    )


def load_geometry_data(geometry_csv: str) -> Optional[pd.DataFrame]:
    """Загружает данные о наклоне камеры."""
    if geometry_csv and Path(geometry_csv).exists():
        return pd.read_csv(geometry_csv)
    return None


def get_camera_tilt_for_frame(
    geometry_df: Optional[pd.DataFrame], 
    frame: int,
    min_confidence: float = 0.5
) -> float:
    """
    Получает наклон камеры для заданного кадра.
    
    Игнорирует записи с низкой уверенностью (выбросы).
    
    Args:
        geometry_df: DataFrame с данными геометрии
        frame: номер кадра
        min_confidence: минимальная уверенность для учёта записи
    
    Returns:
        Общий наклон камеры в градусах
    """
    if geometry_df is None or len(geometry_df) == 0:
        return 0.0
    
    # Фильтруем невалидные записи (выбросы)
    if 'confidence' in geometry_df.columns:
        valid_df = geometry_df[geometry_df['confidence'] >= min_confidence]
    else:
        valid_df = geometry_df
    
    if len(valid_df) == 0:
        return 0.0
    
    # Находим ближайший валидный кадр
    idx = (valid_df['frame_end'] - frame).abs().argmin()
    row = valid_df.iloc[idx]
    
    tilt_h = row.get('tilt_horizontal_deg', 0)
    tilt_v = row.get('tilt_vertical_deg', 0)
    
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
    min_track_points: int = 3,
    min_r_squared: float = 0.5,
    min_size_change_ratio: float = 0.3,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Обрабатывает детекции и добавляет оценки размеров.
    
    Стратегия:
    1. Сначала пытаемся регрессию для треков с хорошими данными
    2. Для остальных — оценка по референсам от хороших треков
    3. Fallback — оценка по типичным размерам вида
    """
    if calibration is None:
        calibration = CameraCalibration()
    calibration.frame_width = frame_width
    calibration.frame_height = frame_height
    
    df = pd.read_csv(detections_csv)
    
    geometry_df = load_geometry_data(geometry_csv)
    
    if verbose:
        print(f"Загружено детекций: {len(df)}")
        print(f"Уникальных треков: {df['track_id'].nunique()}")
        if geometry_df is not None:
            # Фильтруем выбросы (confidence >= 0.5)
            if 'confidence' in geometry_df.columns:
                valid_geom = geometry_df[geometry_df['confidence'] >= 0.5]
            else:
                valid_geom = geometry_df
            
            if len(valid_geom) > 0:
                mean_tilt = np.sqrt(
                    valid_geom['tilt_horizontal_deg'].mean()**2 + 
                    valid_geom['tilt_vertical_deg'].mean()**2
                )
                n_outliers = len(geometry_df) - len(valid_geom)
                outlier_info = f" (отфильтровано выбросов: {n_outliers})" if n_outliers > 0 else ""
                print(f"Средний наклон камеры: {mean_tilt:.1f}°{outlier_info}")
            else:
                print("Средний наклон камеры: нет валидных данных")
    
    required_cols = ['track_id', 'frame', 'width', 'height', 'class_name', 'x_center', 'y_center']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют колонки: {missing}")
    
    # Этап 1: регрессия для хороших треков
    regression_estimates = []
    tracks_for_reference = []
    
    for track_id, track_df in df.groupby('track_id'):
        if pd.isna(track_id):
            continue
        
        class_name = track_df['class_name'].iloc[0]
        if class_name in SKIP_SIZE_CLASSES:
            continue
        
        mean_frame = track_df['frame'].mean()
        camera_tilt = get_camera_tilt_for_frame(geometry_df, mean_frame)
        
        estimate = estimate_size_from_track_regression(
            track_df, calibration,
            camera_tilt_deg=camera_tilt,
            min_depth_change=min_depth_change,
            min_points=min_track_points,
            min_r_squared=min_r_squared,
            min_size_change_ratio=min_size_change_ratio
        )
        
        if estimate is not None:
            regression_estimates.append(estimate)
        else:
            tracks_for_reference.append((track_id, track_df, camera_tilt))
    
    if verbose:
        print(f"\nТреков с регрессией: {len(regression_estimates)}")
        print(f"Треков для референсной оценки: {len(tracks_for_reference)}")
    
    # Этап 2: референсная оценка
    reference_estimates = []
    tracks_for_typical = []
    
    for track_id, track_df, camera_tilt in tracks_for_reference:
        estimate = estimate_size_from_reference(
            track_df, calibration, regression_estimates, camera_tilt
        )
        
        if estimate is not None:
            reference_estimates.append(estimate)
        else:
            tracks_for_typical.append((track_id, track_df, camera_tilt))
    
    if verbose:
        print(f"Треков с референсной оценкой: {len(reference_estimates)}")
    
    # Этап 3: оценка по типичным размерам
    typical_estimates = []
    
    for track_id, track_df, camera_tilt in tracks_for_typical:
        estimate = estimate_size_from_typical(track_df, calibration, camera_tilt)
        
        if estimate is not None:
            typical_estimates.append(estimate)
    
    if verbose:
        print(f"Треков с типичной оценкой: {len(typical_estimates)}")
    
    # Объединяем все оценки
    all_estimates = regression_estimates + reference_estimates + typical_estimates
    
    if verbose:
        print(f"\nВсего треков с оценкой: {len(all_estimates)}")
    
    # Создаём DataFrame треков
    if all_estimates:
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
            for e in all_estimates
        ])
    else:
        tracks_df = pd.DataFrame()
    
    # Добавляем к детекциям
    size_map = {e.track_id: e for e in all_estimates}
    
    df['estimated_size_cm'] = None
    df['estimated_size_m'] = None
    df['object_depth_m'] = None
    df['distance_to_object_m'] = None
    df['size_confidence'] = None
    df['size_method'] = None
    
    for idx, row in df.iterrows():
        track_id = row['track_id']
        if pd.notna(track_id) and track_id in size_map:
            est = size_map[track_id]
            df.at[idx, 'estimated_size_cm'] = est.real_size_cm
            df.at[idx, 'estimated_size_m'] = est.real_size_m
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
        # Фильтруем выбросы
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
        else:
            print("Средний наклон: нет валидных данных")
    
    return df


@dataclass
class VolumeEstimate:
    """Результат оценки осмотренного объёма воды."""
    total_volume_m3: float           # Общий осмотренный объём (м³)
    frustum_volume_m3: float         # Объём начального frustum (м³)
    swept_volume_m3: float           # Объём пройденных слоёв (м³)
    depth_range_m: Tuple[float, float]  # Диапазон глубин (мин, макс)
    depth_traversed_m: float         # Пройденная глубина (м)
    detection_distance_m: float      # Эффективная дистанция обнаружения (м)
    near_distance_m: float           # Ближняя граница (м)
    cross_section_area_m2: float     # Площадь сечения на дальней границе (м²)
    fov_horizontal_deg: float        # Горизонтальный угол обзора
    fov_vertical_deg: float          # Вертикальный угол обзора
    duration_s: float                # Длительность записи (с)
    descent_rate_m_s: float          # Скорость погружения (м/с)
    density_by_class: Dict[str, float]  # Плотность по классам (особей/м³)
    counts_by_class: Dict[str, int]  # Количество особей по классам


# Дистанции обнаружения по умолчанию (из эмпирических данных)
DEFAULT_DETECTION_DISTANCES = {
    'Aurelia aurita': 2.0,      # крупная, хорошо заметная
    'Rhizostoma pulmo': 2.5,    # очень крупная
    'Mnemiopsis leidyi': 1.0,   # средняя, прозрачная
    'Beroe ovata': 1.2,         # средняя
    'Pleurobrachia pileus': 0.5  # мелкая
}


def calculate_frustum_volume(
    d_near: float,
    d_far: float,
    fov_h_rad: float,
    fov_v_rad: float
) -> float:
    """
    Вычисляет объём усечённой пирамиды (frustum).
    
    Args:
        d_near: ближняя граница (м)
        d_far: дальняя граница (м)
        fov_h_rad: горизонтальный FOV (радианы)
        fov_v_rad: вертикальный FOV (радианы)
    
    Returns:
        Объём в м³
    """
    # Размеры на ближней границе
    w_near = 2 * d_near * np.tan(fov_h_rad / 2)
    h_near = 2 * d_near * np.tan(fov_v_rad / 2)
    A_near = w_near * h_near
    
    # Размеры на дальней границе
    w_far = 2 * d_far * np.tan(fov_h_rad / 2)
    h_far = 2 * d_far * np.tan(fov_v_rad / 2)
    A_far = w_far * h_far
    
    # Объём усечённой пирамиды
    depth = d_far - d_near
    V = (depth / 3) * (A_near + A_far + np.sqrt(A_near * A_far))
    
    return V


def calculate_cross_section_area(
    distance: float,
    fov_h_rad: float,
    fov_v_rad: float
) -> float:
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
    """
    Оценивает эффективную дистанцию обнаружения по данным треков.
    
    Использует максимальную дистанцию, на которой объекты были обнаружены.
    
    Args:
        tracks_df: DataFrame со статистикой треков (из process_detections_with_size)
        detections_df: DataFrame с детекциями
        calibration: параметры калибровки
        reference_class: референсный класс для оценки (по умолчанию Aurelia)
    
    Returns:
        Эффективная дистанция обнаружения (м)
    """
    if tracks_df is None or len(tracks_df) == 0:
        return DEFAULT_DETECTION_DISTANCES.get(reference_class, 1.5)
    
    # Проверяем что это правильный DataFrame с колонкой method
    if 'method' not in tracks_df.columns:
        return DEFAULT_DETECTION_DISTANCES.get(reference_class, 1.5)
    
    # Фильтруем по референсному классу и методу регрессии (надёжные данные)
    ref_tracks = tracks_df[
        (tracks_df['class_name'] == reference_class) & 
        (tracks_df['method'] == 'regression')
    ]
    
    if len(ref_tracks) == 0:
        # Пробуем любой класс с регрессией
        ref_tracks = tracks_df[tracks_df['method'] == 'regression']
    
    if len(ref_tracks) == 0:
        return DEFAULT_DETECTION_DISTANCES.get(reference_class, 1.5)
    
    # Для каждого трека вычисляем максимальную дистанцию
    max_distances = []
    
    for _, track in ref_tracks.iterrows():
        track_id = track['track_id']
        object_depth = track['object_depth_m']
        
        if pd.isna(object_depth):
            continue
        
        # Находим детекции этого трека
        track_detections = detections_df[detections_df['track_id'] == track_id]
        if len(track_detections) == 0:
            continue
        
        # Минимальная глубина камеры = момент первого обнаружения (самый далёкий)
        min_camera_depth = track_detections['depth_m'].min()
        max_dist = abs(object_depth - min_camera_depth)
        
        if 0.5 < max_dist < 5.0:  # разумные пределы
            max_distances.append(max_dist)
    
    if max_distances:
        # Берём среднее + 0.5 std для консервативной оценки
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
    """
    Вычисляет осмотренный объём воды на основе данных погружения.
    
    ВАЖНО: Объём считается по ВСЕМУ диапазону погружения, а не только там,
    где есть детекции. Пустая вода тоже учитывается!
    
    Приоритет источников данных о глубине:
    1. depth_range — если задан явно
    2. ctd_df — если передан файл CTD
    3. detections_df — fallback на данные из детекций
    
    Args:
        detections_df: DataFrame с детекциями
        tracks_df: DataFrame со статистикой треков (опционально)
        ctd_df: DataFrame с данными CTD (опционально, приоритетный источник)
        calibration: параметры калибровки
        fov_horizontal_deg: горизонтальный угол обзора камеры
        near_distance_m: ближняя граница обнаружения
        detection_distance_m: дистанция обнаружения (если None — оценивается автоматически)
        depth_range: (min_depth, max_depth) — явно заданный диапазон глубин
        total_duration_s: общая длительность записи (если известна)
        fps: частота кадров видео
        verbose: выводить детали
    
    Returns:
        VolumeEstimate с результатами
    """
    if calibration is None:
        calibration = CameraCalibration()
    
    # Параметры поля зрения
    fov_h_deg = fov_horizontal_deg
    aspect_ratio = calibration.frame_width / calibration.frame_height
    fov_v_deg = fov_h_deg / aspect_ratio
    
    fov_h_rad = np.radians(fov_h_deg)
    fov_v_rad = np.radians(fov_v_deg)
    
    # Определяем диапазон глубин (приоритет: явный > CTD > детекции)
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
        source = "детекции (ВНИМАНИЕ: может быть неполный диапазон!)"
    
    depth_traversed = depth_max - depth_min
    
    # Определяем длительность записи
    if total_duration_s is not None:
        duration = total_duration_s
    elif ctd_df is not None and 'timestamp_s' in ctd_df.columns:
        timestamps = ctd_df['timestamp_s'].dropna()
        duration = timestamps.max() - timestamps.min() if len(timestamps) > 1 else 0
    else:
        # Оцениваем по номерам кадров
        frames = detections_df['frame'].dropna()
        if len(frames) > 1:
            duration = (frames.max() - frames.min()) / fps
        else:
            timestamps = detections_df['timestamp_s'].dropna()
            duration = timestamps.max() - timestamps.min() if len(timestamps) > 1 else 0
    
    descent_rate = depth_traversed / duration if duration > 0 else 0
    
    # Эффективная дистанция обнаружения
    if detection_distance_m is None:
        d_far = estimate_detection_distance(tracks_df, detections_df, calibration)
    else:
        d_far = detection_distance_m
    
    d_near = near_distance_m
    
    if verbose:
        print(f"=== РАСЧЁТ ОСМОТРЕННОГО ОБЪЁМА ===")
        print(f"Источник данных о глубине: {source}")
        print(f"Диапазон глубин: {depth_min:.1f} - {depth_max:.1f} м")
        print(f"Пройденная глубина: {depth_traversed:.2f} м")
        print(f"Длительность: {duration:.1f} с")
        print(f"Скорость погружения: {descent_rate:.3f} м/с")
        print(f"FOV: {fov_h_deg:.0f}° × {fov_v_deg:.0f}°")
        print(f"Дистанция обнаружения: {d_near:.1f} - {d_far:.2f} м")
    
    # Объём начального frustum
    V_frustum = calculate_frustum_volume(d_near, d_far, fov_h_rad, fov_v_rad)
    
    # Площадь сечения на дальней границе
    A_far = calculate_cross_section_area(d_far, fov_h_rad, fov_v_rad)
    
    # Объём пройденных слоёв
    V_swept = A_far * depth_traversed
    
    # Общий объём
    V_total = V_frustum + V_swept
    
    if verbose:
        print(f"\nОбъём frustum: {V_frustum:.2f} м³")
        print(f"Площадь сечения: {A_far:.2f} м²")
        print(f"Объём слоёв: {V_swept:.2f} м³")
        print(f"ИТОГО: {V_total:.1f} м³")
    
    # Подсчёт особей по классам
    counts_by_class = {}
    density_by_class = {}
    
    for class_name in detections_df['class_name'].unique():
        if class_name in SKIP_SIZE_CLASSES:
            continue
        
        # Считаем уникальные треки
        class_df = detections_df[detections_df['class_name'] == class_name]
        n_tracks = class_df['track_id'].nunique()
        
        # Если track_id пустой, считаем детекции
        if n_tracks == 0 or class_df['track_id'].isna().all():
            n_tracks = len(class_df)
        
        counts_by_class[class_name] = n_tracks
        density_by_class[class_name] = n_tracks / V_total if V_total > 0 else 0
    
    if verbose:
        print(f"\n=== ПЛОТНОСТЬ ПО КЛАССАМ ===")
        for class_name, count in counts_by_class.items():
            density = density_by_class[class_name]
            print(f"{class_name}: {count} особей, {density:.4f} ос./м³ ({density*1000:.1f} ос./1000м³)")
    
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
    frame_width: int = 1920,
    frame_height: int = 1080,
    verbose: bool = True
) -> VolumeEstimate:
    """
    Обрабатывает файлы и вычисляет осмотренный объём.
    
    Args:
        detections_csv: путь к CSV с детекциями
        tracks_csv: путь к CSV со статистикой треков (опционально)
        ctd_csv: путь к CSV с данными CTD (приоритетный источник глубины)
        output_csv: путь для сохранения результатов
        fov_horizontal: горизонтальный FOV камеры (градусы)
        near_distance: ближняя граница обнаружения (м)
        detection_distance: дистанция обнаружения (м), None для автоматической
        depth_min: минимальная глубина (если задана явно)
        depth_max: максимальная глубина (если задана явно)
        total_duration: общая длительность записи (с)
        fps: частота кадров видео
        frame_width: ширина кадра
        frame_height: высота кадра
        verbose: выводить детали
    
    Returns:
        VolumeEstimate
    """
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
        # Нормализуем названия колонок CTD (могут быть разные варианты)
        column_mapping = {}
        for col in ctd_df.columns:
            col_lower = col.lower().strip()
            if col_lower in ('depth', 'depth_m', 'depth (m)', 'глубина'):
                column_mapping[col] = 'depth_m'
            elif col_lower in ('time', 'time_s', 'timestamp', 'timestamp_s', 'время'):
                column_mapping[col] = 'timestamp_s'
            elif col_lower in ('temperature', 'temp', 'temp_c', 'температура'):
                column_mapping[col] = 'temperature_c'
        if column_mapping:
            ctd_df = ctd_df.rename(columns=column_mapping)
    
    # Диапазон глубин
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
    
    # Сохранение результатов
    if output_csv:
        output_data = {
            'parameter': [
                'total_volume_m3',
                'frustum_volume_m3',
                'swept_volume_m3',
                'depth_min_m',
                'depth_max_m',
                'depth_traversed_m',
                'detection_distance_m',
                'near_distance_m',
                'cross_section_area_m2',
                'fov_horizontal_deg',
                'fov_vertical_deg',
                'duration_s',
                'descent_rate_m_s'
            ],
            'value': [
                result.total_volume_m3,
                result.frustum_volume_m3,
                result.swept_volume_m3,
                result.depth_range_m[0],
                result.depth_range_m[1],
                result.depth_traversed_m,
                result.detection_distance_m,
                result.near_distance_m,
                result.cross_section_area_m2,
                result.fov_horizontal_deg,
                result.fov_vertical_deg,
                result.duration_s,
                result.descent_rate_m_s
            ]
        }
        
        # Добавляем данные по классам
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
    size.add_argument('--width', type=int, default=1920)
    size.add_argument('--height', type=int, default=1080)
    size.add_argument('--min-depth-change', type=float, default=0.3)
    size.add_argument('--min-track-points', type=int, default=3)
    size.add_argument('--min-r-squared', type=float, default=0.5)
    size.add_argument('--min-size-change', type=float, default=0.3)
    
    # volume
    vol = subparsers.add_parser('volume', help='Расчёт осмотренного объёма воды')
    vol.add_argument('--detections', '-d', required=True, help='CSV с детекциями')
    vol.add_argument('--tracks', '-t', help='CSV со статистикой треков')
    vol.add_argument('--ctd', '-c', help='CSV с данными CTD (полный диапазон глубин)')
    vol.add_argument('--output', '-o', help='Выходной CSV')
    vol.add_argument('--fov', type=float, default=100.0, help='Горизонтальный FOV камеры (градусы)')
    vol.add_argument('--near-distance', type=float, default=0.3, help='Ближняя граница (м)')
    vol.add_argument('--detection-distance', type=float, help='Дистанция обнаружения (м)')
    vol.add_argument('--depth-min', type=float, help='Минимальная глубина (м)')
    vol.add_argument('--depth-max', type=float, help='Максимальная глубина (м)')
    vol.add_argument('--duration', type=float, help='Длительность записи (с)')
    vol.add_argument('--fps', type=float, default=60.0, help='Частота кадров')
    vol.add_argument('--width', type=int, default=1920, help='Ширина кадра')
    vol.add_argument('--height', type=int, default=1080, help='Высота кадра')
    
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
            min_track_points=args.min_track_points,
            min_r_squared=args.min_r_squared,
            min_size_change_ratio=args.min_size_change
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
