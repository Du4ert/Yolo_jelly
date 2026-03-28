"""
Модуль оценки геометрии камеры и размеров объектов.

Функции:
- Оценка наклона камеры по морскому снегу (Focus of Expansion)
- Расчёт реального размера объектов по динамике изменения размера в треке
- Определение абсолютной глубины объекта в толще воды

Калибровочные данные получены для GoPro 12 Wide 4K (3840x2160).

Формулы расчёта размеров:
1. k = (Δpixels/pixels₁) / Δd  - удельный прирост размера (%/м)
2. d = 80.00 * |k|^(-0.9)     - дистанция до объекта (м)
3. p = 4.35 * d^(-1.25)       - калибровка (px/мм)
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
from video_utils import ThreadedVideoCapture


# Ширина кадра, для которой получены калибровочные коэффициенты (GoPro 12 Wide 4K)
REFERENCE_FRAME_WIDTH = 3840

# Классы с фиксированным размером (слишком мелкие для k-метода)
FIXED_SIZE_CLASSES = {
    'Pleurobrachia pileus': {
        'size_mm': 10.0,      # фиксированный размер
        'distance_m': 0.2,    # фиксированная дистанция (200 мм)
    }
}

# Типичные размеры видов (см) для референсной оценки
TYPICAL_SIZES_CM = {
    'Aurelia aurita': {'mean': 5.0, 'std': 5.0, 'min': 3.0, 'max': 40.0},
    'Mnemiopsis leidyi': {'mean': 3.0, 'std': 2.5, 'min': 2.0, 'max': 12.0},
    'Beroe ovata': {'mean': 5.0, 'std': 3.0, 'min': 3.0, 'max': 15.0},
    'Rhizostoma pulmo': {'mean': 30.0, 'std': 15.0, 'min': 10.0, 'max': 60.0},
}


@dataclass
class CameraCalibration:
    """Калибровочные параметры камеры GoPro 12 Wide 4K.
    
    Калибровка выполнена для разрешения 3840x2160 (4K).
    Коэффициенты получены эмпирически по формулам.
    """
    # Размер кадра (4K)
    frame_width: int = 3840
    frame_height: int = 2160
    
    # Угол обзора
    fov_horizontal: float = 156.0
    
    # === Калибровочные коэффициенты (эмпирические, GoPro 12 Wide 4K) ===
    # Формула: d = A * k^B, где k - удельный прирост размера (%/м)
    distance_coef_A: float = 80.00
    distance_coef_B: float = -0.9
    
    # Формула: p = C * d^D, где p - px/мм, d - дистанция (м)
    pixel_calib_C: float = 4.35
    pixel_calib_D: float = -1.25
    
    # Диапазон надёжных измерений (по SNR анализу)
    min_reliable_distance: float = 0.1   # ближе - слишком крупно
    max_reliable_distance: float = 3.0   # дальше - шум > сигнал

    # Радиальная дисторсия (коррекция fisheye GoPro 156°)
    # Коррекция: size_corrected = size_raw * (1 + k1*r² + k2*r⁴)
    # r — нормализованное расстояние от оптического центра (0 = центр, 1 = угол кадра)
    # При k1 < 0 — уменьшает размер на периферии (компенсация раздутия fisheye)
    distortion_k1: float = 0.0
    distortion_k2: float = 0.0
    optical_center_x: float = 0.5  # нормализован (0..1 от ширины)
    optical_center_y: float = 0.5  # нормализован (0..1 от высоты)

    # Параллакс-метод: перцентиль скорости оптического потока для референса.
    # Объекты на P95 скорости считаются на дистанции min_reliable_distance.
    parallax_ref_percentile: float = 95.0

    @property
    def pixels_per_degree(self) -> float:
        return self.frame_width / self.fov_horizontal

    @property
    def frame_center(self) -> Tuple[float, float]:
        return self.frame_width / 2, self.frame_height / 2

    @property
    def resolution_scale(self) -> float:
        """Масштаб разрешения относительно референсного (3840x2160).

        Используется для нормализации пикселей перед применением калибровочных
        коэффициентов, которые получены для разрешения REFERENCE_FRAME_WIDTH.
        """
        return self.frame_width / REFERENCE_FRAME_WIDTH

    @classmethod
    def from_json(cls, json_path: str) -> 'CameraCalibration':
        """Загружает калибровку из JSON файла."""
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(
            distance_coef_A=data.get('distance_coef_A', 80.0),
            distance_coef_B=data.get('distance_coef_B', -0.9),
            pixel_calib_C=data.get('pixel_calib_C', 4.35),
            pixel_calib_D=data.get('pixel_calib_D', -1.25),
            distortion_k1=data.get('distortion_k1', 0.0),
            distortion_k2=data.get('distortion_k2', 0.0),
            optical_center_x=data.get('optical_center_x', 0.5),
            optical_center_y=data.get('optical_center_y', 0.5),
            frame_width=data.get('frame_width', 3840),
            frame_height=data.get('frame_height', 2160),
            parallax_ref_percentile=data.get('parallax_ref_percentile', 95.0),
        )

    def to_json(self, json_path: str):
        """Сохраняет калибровку в JSON файл."""
        import json
        data = {
            'distance_coef_A': self.distance_coef_A,
            'distance_coef_B': self.distance_coef_B,
            'pixel_calib_C': self.pixel_calib_C,
            'pixel_calib_D': self.pixel_calib_D,
            'distortion_k1': self.distortion_k1,
            'distortion_k2': self.distortion_k2,
            'optical_center_x': self.optical_center_x,
            'optical_center_y': self.optical_center_y,
            'parallax_ref_percentile': self.parallax_ref_percentile,
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


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
    
    Метод расчёта:
    1. k = (Δpixels/pixels₁) / Δdepth_camera  - удельный прирост размера для каждой пары
    2. d = 80.00 * |k|^(-0.9)                - дистанция до объекта (м)
    3. p = 4.35 * d^(-1.25)                  - калибровка px/мм
    4. size = pixels_max / p                  - размер объекта по макс. кадру (мм)
    5. object_depth = camera_depth + distance - глубина объекта (камера смотрит вниз)
    """
    track_id: int
    class_name: str
    real_size_mm: float            # Финальный размер в миллиметрах
    real_size_cm: float            # Финальный размер в сантиметрах
    distance_m: float              # Дистанция до объекта на финальный момент (м)
    object_depth_m: float          # Глубина объекта в воде (м)
    first_frame: int               # Кадр финального измерения
    first_size_pixels: float       # Размер в пикселях на финальном кадре
    camera_depth_first: float      # Глубина камеры на финальном кадре (м)
    k_mean: float                  # Средний k по всем парам (%/м)
    k_std: float                   # Стд k (%/м)
    pixel_calibration: float       # Калибровка px/мм на финальный момент
    confidence: float              # Уверенность оценки (0-1)
    method: str                    # 'k_method', 'typical', 'fixed'
    n_points_used: int             # Количество пар для расчёта k
    warnings: List[str] = field(default_factory=list)
    # Покадровые размеры для k-method: {frame: size_mm}
    # Расстояние вычисляется динамически: object_depth_m - camera_depth
    frame_data: Dict[int, float] = field(default_factory=dict)


# =============================================================================
# FOE (Focus of Expansion) - оценка наклона камеры
# =============================================================================

def _foe_error_directed(points, vectors, foe):
    """Ошибка FOE: 1 - cos(угол) между радиальным и реальным вектором.

    Без abs() — вектора должны быть направлены ОТ FOE (расхождение при спуске).
    """
    fx, fy = foe
    radial = points - np.array([fx, fy])
    radial_norm = radial / (np.linalg.norm(radial, axis=1, keepdims=True) + 1e-6)
    vec_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-6)
    dot = np.sum(radial_norm * vec_norm, axis=1)
    return dot


def _fit_foe_on_subset(points, vectors, cx, cy):
    """Оптимизация FOE на подмножестве точек (Nelder-Mead)."""
    def error(foe):
        dot = _foe_error_directed(points, vectors, foe)
        return np.mean(1 - dot)

    result = minimize(error, [cx, cy], method='Nelder-Mead',
                      options={'maxiter': 300, 'xatol': 1.0, 'fatol': 1e-4})
    return result.x, 1 - result.fun


def estimate_foe(
    points: np.ndarray,
    vectors: np.ndarray,
    frame_size: Tuple[int, int] = (3840, 2160),
    min_vector_length: float = 1.0,
    max_tilt_deg: float = 90.0,
    calibration: 'CameraCalibration' = None,
    n_ransac: int = 150,
    ransac_sample_size: int = 30,
    inlier_threshold: float = 0.7,
) -> FOEResult:
    """
    Оценивает Focus of Expansion по точкам и векторам движения (RANSAC).

    Вращение камеры (покачивание от течений) создаёт тангенциальный поток,
    который не совпадает с радиальным паттерном FOE. RANSAC отсекает такие
    точки как выбросы, оставляя только трансляционный компонент.

    Args:
        n_ransac: число итераций RANSAC
        ransac_sample_size: размер случайной выборки для оптимизации
        inlier_threshold: порог dot product для inlier (cos(45°) ≈ 0.7)
    """
    width, height = frame_size
    cx, cy = width / 2, height / 2

    if calibration is not None:
        pixels_per_degree = calibration.pixels_per_degree
    else:
        pixels_per_degree = width / 156.0

    vec_lengths = np.linalg.norm(vectors, axis=1)
    mask = vec_lengths > min_vector_length
    points_filt = points[mask]
    vectors_filt = vectors[mask]

    if len(points_filt) < 30:
        return FOEResult(cx, cy, 0, 0, 0, len(points_filt))

    # --- RANSAC: найти FOE, устойчивый к вращательным выбросам ---
    best_n_inliers = 0
    best_foe = np.array([cx, cy])
    best_inlier_mask = np.zeros(len(points_filt), dtype=bool)
    rng = np.random.default_rng(42)

    sample_size = min(ransac_sample_size, len(points_filt))

    for _ in range(n_ransac):
        idx = rng.choice(len(points_filt), size=sample_size, replace=False)
        foe_candidate, _ = _fit_foe_on_subset(
            points_filt[idx], vectors_filt[idx], cx, cy
        )

        # Подсчёт inliers: dot > threshold (вектор направлен от FOE)
        dot = _foe_error_directed(points_filt, vectors_filt, foe_candidate)
        inlier_mask = dot > inlier_threshold
        n_inliers = np.sum(inlier_mask)

        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_foe = foe_candidate
            best_inlier_mask = inlier_mask

    # Финальная оценка на всех inliers лучшей модели
    if best_n_inliers >= 30:
        foe_final, confidence = _fit_foe_on_subset(
            points_filt[best_inlier_mask], vectors_filt[best_inlier_mask],
            best_foe[0], best_foe[1]
        )
        foe_x, foe_y = foe_final
    else:
        # Мало inliers — используем лучший RANSAC-результат напрямую
        foe_x, foe_y = best_foe
        dot = _foe_error_directed(points_filt, vectors_filt, best_foe)
        confidence = float(np.mean(np.clip(dot, 0, 1)))

    tilt_h = (foe_x - cx) / pixels_per_degree
    tilt_v = (foe_y - cy) / pixels_per_degree

    max_foe_distance = 1.5 * max(width, height)
    foe_distance = np.sqrt((foe_x - cx)**2 + (foe_y - cy)**2)

    # Ветка 1: NaN/бесконечность или FOE улетел за пределы кадра — невалидно
    if not np.isfinite(foe_x) or not np.isfinite(foe_y) or foe_distance > max_foe_distance:
        return FOEResult(cx, cy, 0.0, 0.0, 0.0, len(points_filt))

    # Ветка 2: угол превышает физический предел — клэмп до ±max_tilt_deg,
    # но confidence=0: строка видна в CSV, но исключена из сглаживания и size-коррекции
    if abs(tilt_h) > max_tilt_deg or abs(tilt_v) > max_tilt_deg:
        tilt_h = float(np.clip(tilt_h, -max_tilt_deg, max_tilt_deg))
        tilt_v = float(np.clip(tilt_v, -max_tilt_deg, max_tilt_deg))
        foe_x = cx + tilt_h * pixels_per_degree
        foe_y = cy + tilt_v * pixels_per_degree
        return FOEResult(foe_x, foe_y, tilt_h, tilt_v, 0.0, len(points_filt))

    # Ветка 3: валидный результат
    return FOEResult(foe_x, foe_y, tilt_h, tilt_v, confidence, len(points_filt))


def _classify_flow_regime(
    foe_confidence: float,
    flow_direction_std_deg: float,
    flow_median_speed: float,
    min_parallel_speed: float = 2.0,
    max_parallel_direction_std: float = 45.0,
    min_radial_confidence: float = 0.5,
) -> str:
    """Классифицирует режим оптического потока.

    Returns:
        'radial'    — FOE надёжный, k-метод применим
        'parallel'  — параллельный поток (снос/наклон), параллакс-метод применим
        'ambiguous' — неопределённый (ни один метод не надёжен)
    """
    if foe_confidence >= min_radial_confidence:
        return 'radial'
    if (flow_direction_std_deg < max_parallel_direction_std
            and flow_median_speed > min_parallel_speed):
        return 'parallel'
    return 'ambiguous'


# =============================================================================
# Вспомогательные функции для расчёта размеров
# =============================================================================

def _get_bbox_size_pixels(
    row: pd.Series, frame_width: int, frame_height: int,
    calibration: 'CameraCalibration' = None
) -> float:
    """
    Получает размер bbox в пикселях (сырой, без коррекции дисторсии).
    Берёт максимум из ширины и высоты.

    Коррекция дисторсии применяется позже — к финальному size_mm в _find_size_pairs,
    чтобы не искажать k-значения (которые зависят от отношения пикселей).
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
    frame_height: int,
    calibration: 'CameraCalibration' = None
) -> Tuple[int, pd.Series, float]:
    """Получает данные первого кадра трека."""
    track_df = track_df.copy()
    track_df['size_pix'] = track_df.apply(
        lambda r: _get_bbox_size_pixels(r, frame_width, frame_height, calibration), axis=1
    )
    first_idx = track_df['frame'].idxmin()
    first_row = track_df.loc[first_idx]
    return first_idx, first_row, first_row['size_pix']


# =============================================================================
# Основные функции оценки размеров
# =============================================================================

def _find_size_pairs(
    valid_df: pd.DataFrame,
    min_change_ratio: float,
    apply_tilt_correction: bool,
    geometry_df: Optional[pd.DataFrame],
    calibration: 'CameraCalibration',
) -> Tuple[List[dict], bool, float]:
    """
    Жадно ищет пары кадров с достаточным приростом размера.
    Для каждой пары вычисляет k, distance, size_mm и object_depth.

    Возвращает:
        pair_data: список словарей с данными каждой пары
        tilt_correction_applied: была ли применена коррекция наклона
        avg_tilt_deg: максимальный угол наклона среди пар
    """
    pair_data = []
    tilt_correction_applied = False
    avg_tilt_deg = 0.0

    # Извлекаем массивы для быстрого поиска (numpy вместо iloc)
    sizes_arr = valid_df['size_pix'].values
    depths_arr = valid_df['depth_m'].values
    frames_arr = valid_df['frame'].values
    xcenter_arr = valid_df['x_center'].values if 'x_center' in valid_df.columns else None
    ycenter_arr = valid_df['y_center'].values if 'y_center' in valid_df.columns else None
    n = len(valid_df)

    # Параметры дисторсии
    frame_width = calibration.frame_width
    frame_height = calibration.frame_height
    k1 = calibration.distortion_k1
    k2 = calibration.distortion_k2
    has_distortion = (k1 != 0 or k2 != 0) and xcenter_arr is not None
    if has_distortion:
        diag_half = np.sqrt(frame_width**2 + frame_height**2) / 2
        ocx = calibration.optical_center_x * frame_width
        ocy = calibration.optical_center_y * frame_height

    i = 0
    while i < n:
        pixels1 = sizes_arr[i]
        depth1 = depths_arr[i]
        frame1 = frames_arr[i]

        if pixels1 <= 0:
            i += 1
            continue

        target_size = pixels1 * min_change_ratio

        # Векторизованный поиск первого элемента >= target_size
        remaining = sizes_arr[i + 1:]
        candidates = np.where(remaining >= target_size)[0]
        if len(candidates) == 0:
            break
        found_j = candidates[0] + i + 1

        pixels2 = sizes_arr[found_j]
        depth2 = depths_arr[found_j]
        frame2 = frames_arr[found_j]

        delta_depth = depth2 - depth1

        if abs(delta_depth) < 0.01:
            i = found_j
            continue

        delta_pixels = pixels2 - pixels1
        k_raw = (delta_pixels / pixels1) / delta_depth
        k = k_raw

        cos_tilt = 1.0
        tilt_deg_pair = 0.0
        if apply_tilt_correction and geometry_df is not None:
            cos_tilt, tilt_deg_pair = get_average_tilt_for_range(
                int(frame1), int(frame2), geometry_df
            )
            if cos_tilt < 1.0:
                tilt_correction_applied = True
                avg_tilt_deg = max(avg_tilt_deg, tilt_deg_pair)
            cos_tilt = max(cos_tilt, 0.17)
            k = k_raw / cos_tilt

        if k > 0:
            k_percent = k * 100
            distance = _calculate_distance_from_k(k_percent, calibration)
            pixel_calib = _calculate_pixel_calibration(distance, calibration)

            # Коррекция дисторсии: применяем к пикселям конечного кадра
            # перед вычислением size_mm (как в _compute_sizes_from_pairs)
            corrected_pixels2 = pixels2
            if has_distortion:
                cx = xcenter_arr[found_j] * frame_width
                cy = ycenter_arr[found_j] * frame_height
                r = np.sqrt((cx - ocx)**2 + (cy - ocy)**2) / diag_half
                distortion_factor = 1.0 + k1 * r**2 + k2 * r**4
                if distortion_factor > 0:
                    corrected_pixels2 = pixels2 * distortion_factor

            # Нормализуем пиксели к референсному разрешению (3840px),
            # т.к. pixel_calib откалиброван для REFERENCE_FRAME_WIDTH
            pixels2_ref = corrected_pixels2 / calibration.resolution_scale
            size_mm = _calculate_size_mm(pixels2_ref, pixel_calib)
            camera_depth_end = depth2
            object_depth = camera_depth_end + distance

            pair_data.append({
                'frame_start': frame1,
                'frame_end': frame2,
                'frame_mid': (frame1 + frame2) / 2,
                'k': k,
                'k_percent': k_percent,
                'k_raw_percent': k_raw * 100,
                'cos_tilt': cos_tilt,
                'tilt_deg': tilt_deg_pair,
                'distance': distance,
                'pixel_calib': pixel_calib,
                'size_mm': size_mm,
                'size_pixels': pixels2,
                'depth_camera': camera_depth_end,
                'object_depth': object_depth,
                'size_change_pct': (pixels2 / pixels1 - 1) * 100
            })

        i = found_j

    return pair_data, tilt_correction_applied, avg_tilt_deg


def _filter_pairs_by_mad(pair_data: List[dict]) -> List[dict]:
    """
    Фильтрует выбросы по k с помощью MAD (Median Absolute Deviation).
    Порог: 3 * MAD * 1.4826.
    Если всё отфильтровалось — возвращает исходный список без изменений.
    """
    k_values_raw = [p['k_percent'] for p in pair_data]
    k_median = np.median(k_values_raw)
    k_mad = np.median(np.abs(np.array(k_values_raw) - k_median))

    if k_mad > 0:
        max_k_threshold = k_median + 3 * k_mad * 1.4826
        min_k_threshold = max(k_median - 3 * k_mad * 1.4826, 1.0)
    else:
        max_k_threshold = k_median * 3 if k_median > 0 else 500
        min_k_threshold = 1.0

    filtered = [
        p for p in pair_data
        if min_k_threshold <= p['k_percent'] <= max_k_threshold
    ]

    return filtered if filtered else pair_data


def _apply_moving_median(pair_data: List[dict], smoothing_window: int) -> None:
    """
    Применяет скользящую медиану к size_mm в pair_data.
    Добавляет ключ 'size_mm_smoothed' в каждый элемент (изменение на месте).
    """
    sizes_raw = [p['size_mm'] for p in pair_data]

    if len(sizes_raw) >= smoothing_window:
        half_window = smoothing_window // 2
        for i in range(len(sizes_raw)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(sizes_raw), i + half_window + 1)
            window_values = sizes_raw[start_idx:end_idx]
            pair_data[i]['size_mm_smoothed'] = np.median(window_values)
    else:
        for p in pair_data:
            p['size_mm_smoothed'] = p['size_mm']


def _select_final_size(
    smoothed_sizes: List[float],
    pair_data_filtered: List[dict],
) -> Tuple[float, int]:
    """
    Выбирает финальный размер как последнюю стабильную точку
    до начала устойчивого уменьшения (≥2 подряд снижений > 2%).

    Если выбранный размер отличается от медианы > 50% — fallback на медиану.

    Возвращает: (final_size_mm, final_idx)
    """
    final_idx = len(smoothed_sizes) - 1

    consecutive_decreases = 0
    for i in range(1, len(smoothed_sizes)):
        if smoothed_sizes[i] < smoothed_sizes[i - 1] * 0.98:
            consecutive_decreases += 1
            if consecutive_decreases >= 2:
                final_idx = max(0, i - consecutive_decreases)
                break
        else:
            consecutive_decreases = 0

    final_size_mm = smoothed_sizes[final_idx]
    median_size_mm = np.median(smoothed_sizes)

    if abs(final_size_mm - median_size_mm) > median_size_mm * 0.5:
        for i in range(len(smoothed_sizes) - 1, -1, -1):
            if abs(smoothed_sizes[i] - median_size_mm) <= median_size_mm * 0.5:
                final_idx = i
                final_size_mm = smoothed_sizes[i]
                break
        else:
            final_size_mm = median_size_mm
            final_idx = len(smoothed_sizes) // 2

    return final_size_mm, final_idx


def _find_max_size_frame(
    track_df: pd.DataFrame,
    frame_width: int,
    frame_height: int,
    calibration: 'CameraCalibration' = None
) -> Tuple[int, pd.Series, float]:
    """
    Находит кадр с максимальным размером объекта.

    Логика: из последних 20% трека берём последний кадр,
    где объект ещё увеличивается (до начала выхода за границы кадра).
    """
    track_df = track_df.copy()
    track_df['size_pix'] = track_df.apply(
        lambda r: _get_bbox_size_pixels(r, frame_width, frame_height, calibration), axis=1
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
    min_points: int = 3,
    min_size_change_pct: float = 10.0,
    geometry_df: Optional[pd.DataFrame] = None,
    apply_tilt_correction: bool = True,
    smoothing_window: int = 3
) -> Optional[TrackSizeEstimate]:
    """
    Оценивает размер объекта методом удельного прироста k.
    
    Алгоритм:
    1. Берём пары точек, где размер изменился минимум на 10%
    2. Для каждой пары вычисляем:
       - k = (Δpx/px₁) / Δdepth_camera
       - k_corr = k / cos(θ) (коррекция наклона)
       - distance = 80.00 × |k|^(-0.9)
       - pixel_calib = 4.35 × d^(-1.25)
       - size_mm = pixels_end / pixel_calib
    3. Сглаживаем ряд размеров скользящей медианой
    4. Выбираем последний стабильный размер до начала уменьшения
    5. Глубина объекта = глубина камеры + дистанция
    
    Коррекция наклона:
    При наклоне камеры на угол θ от вертикали, реальное изменение дистанции
    до объекта = Δdepth × cos(θ). Измеренный k занижен, поэтому:
    k_corrected = k_measured / cos(θ)
    
    Args:
        track_df: DataFrame с детекциями трека
        calibration: калибровочные параметры камеры
        min_depth_change: минимальное изменение глубины (м)
        min_points: минимальное количество точек
        min_size_change_pct: минимальное изменение размера между точками пары (%)
        geometry_df: DataFrame с данными геометрии (наклон камеры)
        apply_tilt_correction: применять ли коррекцию наклона
        smoothing_window: размер окна для сглаживания медианой
    """
    track_id = track_df['track_id'].iloc[0]
    class_name = track_df['class_name'].iloc[0]
    
    # Пропускаем классы с фиксированным размером
    if class_name in FIXED_SIZE_CLASSES:
        return None
    
    if 'depth_m' not in track_df.columns or track_df['depth_m'].isna().all():
        return None
    
    valid_df = track_df[track_df['depth_m'].notna()].copy()
    
    if len(valid_df) < min_points:
        return None
    
    frame_width = calibration.frame_width
    frame_height = calibration.frame_height
    
    valid_df['size_pix'] = valid_df.apply(
        lambda r: _get_bbox_size_pixels(r, frame_width, frame_height, calibration), axis=1
    )
    valid_df = valid_df.sort_values('frame').reset_index(drop=True)

    # Проверяем достаточное изменение глубины
    depth_change = valid_df['depth_m'].max() - valid_df['depth_m'].min()
    if depth_change < min_depth_change:
        return None
    
    min_change_ratio = 1.0 + min_size_change_pct / 100.0
    pair_data, tilt_correction_applied, avg_tilt_deg = _find_size_pairs(
        valid_df, min_change_ratio, apply_tilt_correction, geometry_df, calibration
    )

    if len(pair_data) == 0:
        return None
    
    # ===== Фильтрация выбросов и сглаживание =====
    pair_data_filtered = _filter_pairs_by_mad(pair_data)
    _apply_moving_median(pair_data_filtered, smoothing_window)

    smoothed_sizes = [p['size_mm_smoothed'] for p in pair_data_filtered]
    final_size_mm, final_idx = _select_final_size(smoothed_sizes, pair_data_filtered)
    
    # Берём данные финальной пары
    final_pair = pair_data_filtered[final_idx]
    
    size_mm = final_size_mm
    size_cm = size_mm / 10.0
    distance_final = final_pair['distance']
    camera_depth_final = final_pair['depth_camera']
    pixel_calib = final_pair['pixel_calib']
    final_frame = int(final_pair['frame_end'])
    final_size_pixels = final_pair['size_pixels']
    
    # Статистика k (по отфильтрованным данным)
    k_values = [p['k_percent'] for p in pair_data_filtered]
    k_mean = np.mean(k_values)
    k_std = np.std(k_values) if len(k_values) > 1 else 0.0
    
    # Оценка уверенности
    warnings_list = []
    confidence = 1.0
    
    if distance_final > calibration.max_reliable_distance:
        warnings_list.append("distance_above_reliable")
        confidence = 0.3
    elif distance_final > 2.0:
        warnings_list.append("distance_marginal")
        confidence = 0.6
    
    if distance_final < calibration.min_reliable_distance:
        warnings_list.append("distance_too_close")
        confidence *= 0.8
    
    # Проверка стабильности k
    if k_std > 0 and k_mean > 0:
        cv = k_std / k_mean  # коэффициент вариации
        if cv > 0.5:
            warnings_list.append("k_unstable")
            confidence *= 0.7
    
    # Проверка стабильности размеров
    size_cv = np.std(smoothed_sizes) / np.mean(smoothed_sizes) if np.mean(smoothed_sizes) > 0 else 0
    if size_cv > 0.3:
        warnings_list.append("size_unstable")
        confidence *= 0.8
    
    if class_name in TYPICAL_SIZES_CM:
        typical = TYPICAL_SIZES_CM[class_name]
        if size_cm < typical['min'] * 0.3 or size_cm > typical['max'] * 3:
            warnings_list.append("size_outside_typical")
            confidence *= 0.5
    
    if size_mm < 5:
        warnings_list.append("size_too_small")
    if size_mm > 1000:
        warnings_list.append("size_too_large")
    
    # Добавляем информацию о коррекции наклона
    if tilt_correction_applied:
        warnings_list.append(f"tilt_corrected_{avg_tilt_deg:.0f}deg")
    
    # Информация о сглаживании
    n_filtered = len(pair_data) - len(pair_data_filtered)
    if n_filtered > 0:
        warnings_list.append(f"outliers_filtered_{n_filtered}")
    
    # ===== Определяем единую object_depth с фильтрацией выбросов =====
    # Объект находится на фиксированной глубине, она не должна "прыгать"
    object_depths_raw = [p['object_depth'] for p in pair_data_filtered]
    obj_depth_median = np.median(object_depths_raw)
    obj_depth_mad = np.median(np.abs(np.array(object_depths_raw) - obj_depth_median))
    
    # Фильтруем выбросы по object_depth (3*MAD)
    if obj_depth_mad > 0:
        obj_depth_threshold = 3 * obj_depth_mad * 1.4826
    else:
        obj_depth_threshold = obj_depth_median * 0.2  # 20% от медианы
    
    valid_depths = [d for d in object_depths_raw if abs(d - obj_depth_median) <= obj_depth_threshold]
    
    if valid_depths:
        # Берём медиану отфильтрованных значений как финальную глубину объекта
        object_depth_final = np.median(valid_depths)
    else:
        object_depth_final = obj_depth_median
    
    # Собираем покадровые данные: {frame: size_mm}
    # Расстояние будем вычислять динамически в process_detections_with_size
    frame_sizes = {}
    for p in pair_data_filtered:
        frame_end = int(p['frame_end'])
        frame_sizes[frame_end] = round(p['size_mm_smoothed'], 1)
    
    return TrackSizeEstimate(
        track_id=track_id,
        class_name=class_name,
        real_size_mm=round(size_mm, 1),
        real_size_cm=round(size_cm, 2),
        distance_m=round(distance_final, 3),
        object_depth_m=round(object_depth_final, 2),  # единая фиксированная глубина
        first_frame=final_frame,
        first_size_pixels=round(final_size_pixels, 1),
        camera_depth_first=round(camera_depth_final, 2),
        k_mean=round(k_mean, 2),
        k_std=round(k_std, 2),
        pixel_calibration=round(pixel_calib, 4),
        confidence=round(confidence, 3),
        method="k_method",
        n_points_used=len(pair_data_filtered),
        warnings=warnings_list,
        frame_data=frame_sizes  # теперь только размеры, расстояние вычисляется динамически
    )


def estimate_size_fixed(
    track_df: pd.DataFrame,
    calibration: CameraCalibration
) -> Optional[TrackSizeEstimate]:
    """
    Оценивает размер для классов с фиксированным размером.
    Глубина объекта вычисляется по последнему кадру трека.
    """
    track_id = track_df['track_id'].iloc[0]
    class_name = track_df['class_name'].iloc[0]
    
    if class_name not in FIXED_SIZE_CLASSES:
        return None
    
    fixed = FIXED_SIZE_CLASSES[class_name]
    size_mm = fixed['size_mm']
    distance = fixed['distance_m']
    
    frame_width = calibration.frame_width
    frame_height = calibration.frame_height
    
    # Берём последний кадр трека
    valid_df = track_df[track_df['depth_m'].notna()].copy() if 'depth_m' in track_df.columns else track_df.copy()
    if len(valid_df) == 0:
        valid_df = track_df.copy()
    
    valid_df = valid_df.sort_values('frame')
    last_row = valid_df.iloc[-1]
    
    last_frame = int(last_row['frame'])
    camera_depth_last = last_row.get('depth_m', np.nan)
    
    # Размер в пикселях на последнем кадре
    last_size_pix = _get_bbox_size_pixels(last_row, frame_width, frame_height, calibration)
    
    # Глубина объекта = глубина камеры + дистанция
    if pd.notna(camera_depth_last):
        object_depth = camera_depth_last + distance
    else:
        object_depth = np.nan
    
    # Калибровка пикселя для фиксированной дистанции
    pixel_calib = _calculate_pixel_calibration(distance, calibration)
    
    return TrackSizeEstimate(
        track_id=track_id,
        class_name=class_name,
        real_size_mm=size_mm,
        real_size_cm=round(size_mm / 10.0, 2),
        distance_m=distance,
        object_depth_m=round(object_depth, 2) if pd.notna(object_depth) else None,
        first_frame=last_frame,
        first_size_pixels=round(last_size_pix, 1),
        camera_depth_first=round(camera_depth_last, 2) if pd.notna(camera_depth_last) else None,
        k_mean=0.0,
        k_std=0.0,
        pixel_calibration=round(pixel_calib, 4),
        confidence=0.5,  # средняя уверенность для фиксированных
        method="fixed",
        n_points_used=len(valid_df),
        warnings=["fixed_size_estimate"]
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
    
    # Пропускаем классы с фиксированным размером
    if class_name in FIXED_SIZE_CLASSES:
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
    
    # Берём кадр с максимальным размером (ближайший к камере)
    _, max_row, max_size_pix = _find_max_size_frame(valid_df, frame_width, frame_height, calibration)
    
    camera_depth_max = max_row.get('depth_m', np.nan)
    max_frame = int(max_row['frame'])
    
    # Обратная калибровка: из типичного размера и пикселей находим p, затем d.
    # Нормализуем пиксели к референсному разрешению (3840px),
    # т.к. коэффициент C в формуле p = C * d^D получен для REFERENCE_FRAME_WIDTH.
    max_size_pix_ref = max_size_pix / calibration.resolution_scale
    pixel_calib = max_size_pix_ref / typical_size_mm
    
    C = calibration.pixel_calib_C
    D = calibration.pixel_calib_D
    
    if pixel_calib > 0 and C > 0:
        distance = (pixel_calib / C) ** (1.0 / D)
    else:
        distance = 1.5
    
    distance = np.clip(distance, calibration.min_reliable_distance,
                       calibration.max_reliable_distance + 2.0)
    
    if pd.notna(camera_depth_max):
        object_depth = camera_depth_max + distance  # камера смотрит вниз, объект глубже
    else:
        object_depth = np.nan
    
    return TrackSizeEstimate(
        track_id=track_id,
        class_name=class_name,
        real_size_mm=round(typical_size_mm, 1),
        real_size_cm=round(typical['mean'], 2),
        distance_m=round(distance, 3),
        object_depth_m=round(object_depth, 2) if pd.notna(object_depth) else None,
        first_frame=max_frame,
        first_size_pixels=round(max_size_pix, 1),
        camera_depth_first=round(camera_depth_max, 2) if pd.notna(camera_depth_max) else None,
        k_mean=0.0,
        k_std=0.0,
        pixel_calibration=round(pixel_calib, 4),
        confidence=0.2,
        method="typical",
        n_points_used=len(valid_df),
        warnings=["estimated_from_typical_size"]
    )


def estimate_size_by_parallax(
    track_df: pd.DataFrame,
    calibration: CameraCalibration,
    geometry_df: pd.DataFrame,
    frame_width: int = 3840,
    frame_height: int = 2160,
    min_track_points: int = 3,
) -> Optional[TrackSizeEstimate]:
    """Оценивает размер объекта по параллаксу движения.

    При боковом потоке (параллельном режиме) ближние объекты движутся быстрее.
    Скорость ближнего снега (P95) на известной дистанции даёт масштаб:
        d_obj = d_ref × (v_ref / v_obj)
    """
    track_id = track_df['track_id'].iloc[0]
    class_name = track_df['class_name'].iloc[0]

    if class_name in FIXED_SIZE_CLASSES:
        return None

    valid_df = track_df.sort_values('frame').reset_index(drop=True)
    if len(valid_df) < min_track_points:
        return None

    # Найти параллельные интервалы, пересекающиеся с треком
    track_start = int(valid_df['frame'].iloc[0])
    track_end = int(valid_df['frame'].iloc[-1])

    parallel_intervals = geometry_df[
        (geometry_df['flow_regime'] == 'parallel')
        & (geometry_df['frame_start'] < track_end)
        & (geometry_df['frame_end'] > track_start)
    ]

    if len(parallel_intervals) == 0:
        return None

    # Средняя скорость фона (P95) по параллельным интервалам
    v_ref = float(parallel_intervals['flow_p95_speed'].median())
    if v_ref < 1.0:
        return None

    # Скорость объекта: медиана покадровых смещений центра bbox
    x_px = valid_df['x_center'].values * frame_width
    y_px = valid_df['y_center'].values * frame_height
    frames = valid_df['frame'].values

    dx = np.diff(x_px)
    dy = np.diff(y_px)
    d_frames = np.diff(frames).astype(float)
    d_frames[d_frames == 0] = 1.0

    speeds = np.sqrt(dx**2 + dy**2) / d_frames  # px/кадр
    v_obj = float(np.median(speeds))

    if v_obj < 0.5:
        return None  # объект почти неподвижен — дистанция неопределима

    # Дистанция по параллаксу
    d_ref = calibration.min_reliable_distance
    d_obj = d_ref * (v_ref / v_obj)
    d_obj = float(np.clip(d_obj, calibration.min_reliable_distance,
                          calibration.max_reliable_distance))

    # Размер через стандартную калибровку
    _, max_row, max_size_pix = _find_max_size_frame(
        valid_df, frame_width, frame_height, calibration)
    max_size_pix_ref = max_size_pix / calibration.resolution_scale
    pixel_calib = _calculate_pixel_calibration(d_obj, calibration)
    size_mm = _calculate_size_mm(max_size_pix_ref, pixel_calib)

    camera_depth = max_row.get('depth_m', np.nan)
    if pd.notna(camera_depth):
        object_depth = camera_depth + d_obj
    else:
        object_depth = np.nan

    max_frame = int(max_row['frame'])

    # Confidence: базовый 0.5, штрафы
    confidence = 0.5
    warnings = []

    # Плохая дискриминация: v_obj слишком близко к v_ref
    velocity_ratio = v_obj / v_ref
    if velocity_ratio > 0.7:
        confidence *= 0.7
        warnings.append("low_parallax_discrimination")

    # Мало параллельных интервалов
    if len(parallel_intervals) == 1:
        confidence *= 0.9
        warnings.append("single_parallel_interval")

    # Дистанция на границе надёжного диапазона
    if d_obj >= calibration.max_reliable_distance:
        confidence *= 0.5
        warnings.append("distance_above_reliable")
    elif d_obj <= calibration.min_reliable_distance:
        confidence *= 0.8
        warnings.append("distance_too_close")

    # Проверка по типичным размерам вида
    if class_name in TYPICAL_SIZES_CM:
        typical = TYPICAL_SIZES_CM[class_name]
        size_cm = size_mm / 10.0
        if size_cm < typical['min'] / 3 or size_cm > typical['max'] * 3:
            confidence *= 0.5
            warnings.append("size_outside_typical")

    return TrackSizeEstimate(
        track_id=track_id,
        class_name=class_name,
        real_size_mm=round(size_mm, 1),
        real_size_cm=round(size_mm / 10.0, 2),
        distance_m=round(d_obj, 3),
        object_depth_m=round(object_depth, 2) if pd.notna(object_depth) else None,
        first_frame=max_frame,
        first_size_pixels=round(max_size_pix, 1),
        camera_depth_first=round(camera_depth, 2) if pd.notna(camera_depth) else None,
        k_mean=0.0,
        k_std=0.0,
        pixel_calibration=round(pixel_calib, 4),
        confidence=round(confidence, 3),
        method="parallax",
        n_points_used=len(valid_df),
        warnings=warnings,
    )


# =============================================================================
# Загрузка данных геометрии
# =============================================================================

def load_geometry_data(geometry_csv: str) -> Optional[pd.DataFrame]:
    """Загружает данные о наклоне камеры."""
    if geometry_csv and Path(geometry_csv).exists():
        return pd.read_csv(geometry_csv)
    return None


def get_tilt_correction_for_frame(
    frame: int,
    geometry_df: Optional[pd.DataFrame],
    min_confidence: float = 0.5
) -> float:
    """
    Получает коэффициент коррекции наклона для заданного кадра.
    
    При наклоне камеры на угол θ от вертикали, изменение дистанции до объекта
    при погружении камеры на Δd составляет Δd × cos(θ).
    
    Возвращает cos(θ) для коррекции. Если геометрия недоступна, возвращает 1.0.
    
    Args:
        frame: номер кадра
        geometry_df: DataFrame с данными геометрии (из process_video_geometry)
        min_confidence: минимальная уверенность для использования данных
    
    Returns:
        cos(θ) - коэффициент коррекции (0..1), где 1.0 = камера смотрит вертикально
    """
    if geometry_df is None or len(geometry_df) == 0:
        return 1.0
    
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
        return 1.0
    
    row = matching.iloc[0]
    
    # Проверяем уверенность
    if 'confidence' in row and row['confidence'] < min_confidence:
        return 1.0
    
    # Вычисляем полный угол наклона
    tilt_h = row.get('tilt_horizontal_deg', 0.0)
    tilt_v = row.get('tilt_vertical_deg', 0.0)
    
    # Полный наклон = sqrt(h² + v²)
    total_tilt_deg = np.sqrt(tilt_h**2 + tilt_v**2)
    
    # Ограничиваем максимальный наклон (при >80° cos близок к 0)
    total_tilt_deg = min(total_tilt_deg, 80.0)
    
    # cos(θ) - коэффициент коррекции
    cos_tilt = np.cos(np.radians(total_tilt_deg))
    
    return cos_tilt


def get_average_tilt_for_range(
    frame_start: int,
    frame_end: int,
    geometry_df: Optional[pd.DataFrame],
    min_confidence: float = 0.5
) -> Tuple[float, float]:
    """
    Вычисляет средний наклон камеры для диапазона кадров.
    
    Returns:
        (cos_tilt, tilt_deg) - коэффициент коррекции и угол в градусах
    """
    if geometry_df is None or len(geometry_df) == 0:
        return 1.0, 0.0
    
    # Фильтруем по уверенности
    if 'confidence' in geometry_df.columns:
        valid_df = geometry_df[geometry_df['confidence'] >= min_confidence].copy()
    else:
        valid_df = geometry_df.copy()
    
    if len(valid_df) == 0:
        return 1.0, 0.0
    
    # Находим интервалы, пересекающиеся с диапазоном
    mask = (valid_df['frame_end'] >= frame_start) & (valid_df['frame_start'] <= frame_end)
    overlapping = valid_df[mask]
    
    if len(overlapping) == 0:
        # Берём все данные если нет пересечений
        overlapping = valid_df
    
    # Вычисляем средний наклон
    tilt_h = overlapping['tilt_horizontal_deg'].mean()
    tilt_v = overlapping['tilt_vertical_deg'].mean()
    
    total_tilt_deg = np.sqrt(tilt_h**2 + tilt_v**2)
    total_tilt_deg = min(total_tilt_deg, 80.0)
    
    cos_tilt = np.cos(np.radians(total_tilt_deg))
    
    return cos_tilt, total_tilt_deg


# =============================================================================
# Основная функция обработки детекций
# =============================================================================

def _build_tracks_dataframe(all_estimates: List['TrackSizeEstimate']) -> pd.DataFrame:
    """Преобразует список оценок треков в DataFrame для сохранения."""
    if not all_estimates:
        return pd.DataFrame()
    return pd.DataFrame([
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


def _assign_size_columns_to_detections(
    df: pd.DataFrame,
    size_map: dict,
) -> pd.DataFrame:
    """
    Добавляет к DataFrame детекций колонки размеров:
    estimated_size_mm, estimated_size_cm, object_depth_m,
    distance_to_object_m, size_confidence, size_method.

    Для k-method: размер интерполируется между кадрами трека.
    Расстояние вычисляется динамически: object_depth - camera_depth.
    """
    df['estimated_size_mm'] = None
    df['estimated_size_cm'] = None
    df['object_depth_m'] = None
    df['distance_to_object_m'] = None
    df['size_confidence'] = None
    df['size_method'] = None

    if not size_map:
        return df

    # Маска строк с валидным track_id, присутствующим в size_map
    has_track = df['track_id'].notna()
    if not has_track.any():
        return df

    valid_track_ids = set(size_map.keys())
    mask = has_track & df['track_id'].isin(valid_track_ids)
    if not mask.any():
        return df

    # Построить массивы скалярных атрибутов из size_map по track_id
    track_ids_col = df.loc[mask, 'track_id']
    object_depths = track_ids_col.map(lambda tid: size_map[tid].object_depth_m)
    confidences = track_ids_col.map(lambda tid: size_map[tid].confidence)
    methods = track_ids_col.map(lambda tid: size_map[tid].method)

    df.loc[mask, 'object_depth_m'] = object_depths.values
    df.loc[mask, 'size_confidence'] = confidences.values
    df.loc[mask, 'size_method'] = methods.values

    # distance_to_object_m = max(object_depth - camera_depth, 0)
    dist_mask = mask & df['depth_m'].notna() & df['object_depth_m'].notna()
    if dist_mask.any():
        obj_d = df.loc[dist_mask, 'object_depth_m'].astype(float)
        cam_d = df.loc[dist_mask, 'depth_m'].astype(float)
        distances = (obj_d - cam_d).clip(lower=0.0).round(3)
        df.loc[dist_mask, 'distance_to_object_m'] = distances.values

    # Интерполяция размеров по кадрам для каждого трека
    # Предварительно подготовим кеш отсортированных frame_data
    _interp_cache = {}
    for tid, est in size_map.items():
        if est.frame_data:
            frames_sorted = sorted(est.frame_data.keys())
            sizes_sorted = [est.frame_data[f] for f in frames_sorted]
            _interp_cache[tid] = (np.array(frames_sorted, dtype=float),
                                  np.array(sizes_sorted, dtype=float))
        else:
            _interp_cache[tid] = None

    # Группируем по track_id для эффективной интерполяции
    for tid, group_idx in df.loc[mask].groupby('track_id').groups.items():
        est = size_map[tid]
        cache = _interp_cache[tid]

        if cache is not None:
            xp, fp = cache
            frames = df.loc[group_idx, 'frame'].astype(float).values
            # np.interp делает линейную интерполяцию с clamp на краях —
            # точно повторяет логику оригинала
            sizes = np.round(np.interp(frames, xp, fp), 1)
            df.loc[group_idx, 'estimated_size_mm'] = sizes
        else:
            df.loc[group_idx, 'estimated_size_mm'] = est.real_size_mm

    # estimated_size_cm из estimated_size_mm
    size_assigned = mask & df['estimated_size_mm'].notna()
    if size_assigned.any():
        sizes_mm = df.loc[size_assigned, 'estimated_size_mm'].astype(float)
        df.loc[size_assigned, 'estimated_size_cm'] = (sizes_mm / 10.0).round(2).values

    return df


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
    apply_tilt_correction: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Обрабатывает детекции и добавляет оценки размеров.
    
    Стратегия:
    1. k-метод для треков с хорошими данными (вертикальный спуск)
    2. Параллакс-метод для треков в зонах бокового потока
    3. Fallback — оценка по типичным размерам вида
    
    Args:
        detections_csv: CSV с детекциями
        output_csv: выходной CSV с детекциями + размеры
        tracks_output_csv: CSV со статистикой треков
        geometry_csv: CSV с данными геометрии (наклон камеры)
        calibration: калибровочные параметры
        frame_width: ширина кадра
        frame_height: высота кадра
        min_depth_change: мин. изменение глубины для k-метода
        min_track_points: мин. точек в треке
        apply_tilt_correction: применять ли коррекцию наклона камеры
        verbose: выводить ли информацию о прогрессе
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
    
    # Загружаем данные геометрии (наклон камеры)
    geometry_df = load_geometry_data(geometry_csv) if geometry_csv else None
    if verbose and geometry_df is not None:
        print(f"Загружены данные геометрии: {len(geometry_df)} интервалов")
    
    # Этап 0: классы с фиксированным размером (P. pileus)
    fixed_estimates = []

    # Разделяем треки по типу обработки
    tracks_for_k_method = []
    tracks_after_k = []  # треки, для которых k-метод не сработал

    for track_id, track_df in df.groupby('track_id'):
        if pd.isna(track_id):
            continue

        class_name = track_df['class_name'].iloc[0]

        # Сначала проверяем фиксированные классы
        if class_name in FIXED_SIZE_CLASSES:
            estimate = estimate_size_fixed(track_df, calibration)
            if estimate is not None:
                fixed_estimates.append(estimate)
            continue

        tracks_for_k_method.append(track_df)

    # Этап 1: k-метод с параллельной обработкой треков
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os

    def _process_track_k_method(track_df):
        return estimate_size_by_k_method(
            track_df, calibration,
            min_depth_change=min_depth_change,
            min_points=min_track_points,
            geometry_df=geometry_df,
            apply_tilt_correction=apply_tilt_correction
        )

    k_method_estimates = []
    max_workers = min(os.cpu_count() or 4, len(tracks_for_k_method) or 1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_track_k_method, tdf): tdf
                   for tdf in tracks_for_k_method}
        for future in as_completed(futures):
            estimate = future.result()
            if estimate is not None:
                k_method_estimates.append(estimate)
            else:
                tdf = futures[future]
                track_id = tdf['track_id'].iloc[0]
                tracks_after_k.append((track_id, tdf))

    if verbose:
        print(f"\nТреков с фиксированным размером: {len(fixed_estimates)}")
        print(f"Треков с k-методом: {len(k_method_estimates)}")

    # Этап 1.5: параллакс-метод для треков с параллельным потоком
    parallax_estimates = []
    tracks_for_typical = []

    has_parallax_data = (geometry_df is not None
                         and 'flow_regime' in geometry_df.columns)

    if has_parallax_data:
        for track_id, track_df in tracks_after_k:
            estimate = estimate_size_by_parallax(
                track_df, calibration, geometry_df,
                frame_width=frame_width, frame_height=frame_height,
                min_track_points=min_track_points,
            )
            if estimate is not None:
                parallax_estimates.append(estimate)
            else:
                tracks_for_typical.append((track_id, track_df))
    else:
        tracks_for_typical = tracks_after_k

    if verbose:
        print(f"Треков с параллаксом: {len(parallax_estimates)}")
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
    all_estimates = (fixed_estimates + k_method_estimates
                     + parallax_estimates + typical_estimates)
    
    if verbose:
        print(f"\nВсего треков с оценкой: {len(all_estimates)}")
    
    tracks_df = _build_tracks_dataframe(all_estimates)

    size_map = {e.track_id: e for e in all_estimates}
    df = _assign_size_columns_to_detections(df, size_map)
    
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
    verbose: bool = True,
    frame_step: int = 1,
) -> pd.DataFrame:
    """Обрабатывает видео и оценивает наклон камеры.

    Args:
        frame_step: шаг обработки кадров для optical flow (1 = каждый кадр,
                    2 = через кадр и т.д.). Увеличение ускоряет обработку,
                    но может незначительно повлиять на точность FOE.
    """
    if calibration is None:
        calibration = CameraCalibration()
    
    cap = ThreadedVideoCapture(video_path).start()
    if not cap.isOpened():
        cap.release()
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

        # Пропуск кадров для ускорения (frame_step > 1)
        if frame_step > 1 and frame_idx % frame_step != 0:
            continue

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
            foe = estimate_foe(points_arr, vectors_arr, (width, height),
                               calibration=calibration)

            # Статистика оптического потока для параллакс-метода
            vec_magnitudes = np.linalg.norm(vectors_arr, axis=1)
            valid_flow = vec_magnitudes > 1.0
            if valid_flow.sum() > 10:
                valid_mags = vec_magnitudes[valid_flow]
                valid_vecs = vectors_arr[valid_flow]
                flow_median_speed = float(np.median(valid_mags))
                flow_p95_speed = float(np.percentile(
                    valid_mags, calibration.parallax_ref_percentile))
                # Круговое стд направлений: R = |mean(exp(iθ))|, σ = √(-2·ln(R))
                angles = np.arctan2(valid_vecs[:, 1], valid_vecs[:, 0])
                R = float(np.abs(np.mean(np.exp(1j * angles))))
                flow_direction_std_deg = float(np.degrees(
                    np.sqrt(-2.0 * np.log(max(R, 1e-6)))))
            else:
                flow_median_speed = 0.0
                flow_p95_speed = 0.0
                flow_direction_std_deg = 180.0

            flow_regime = _classify_flow_regime(
                foe.confidence, flow_direction_std_deg, flow_median_speed)

            results.append({
                'frame_start': interval_start,
                'frame_end': frame_idx,
                'timestamp_s': round(frame_idx / fps, 2),
                'foe_x': round(foe.foe_x, 1),
                'foe_y': round(foe.foe_y, 1),
                'tilt_horizontal_deg': round(foe.tilt_horizontal, 1),
                'tilt_vertical_deg': round(foe.tilt_vertical, 1),
                'confidence': round(foe.confidence, 3),
                'n_vectors': foe.n_vectors,
                'flow_median_speed': round(flow_median_speed, 2),
                'flow_p95_speed': round(flow_p95_speed, 2),
                'flow_direction_std_deg': round(flow_direction_std_deg, 1),
                'flow_regime': flow_regime,
            })

            interval_points = []
            interval_vectors = []
            interval_start = frame_idx

        prev_gray = gray

    cap.release()

    df = pd.DataFrame(results)

    # Временное сглаживание: скользящая медиана по tilt (окно=3)
    # Наклон камеры меняется плавно, резкие скачки = шум/вращение
    if len(df) >= 3:
        for col in ('tilt_horizontal_deg', 'tilt_vertical_deg'):
            if col in df.columns:
                valid_mask = df['confidence'] > 0
                if valid_mask.sum() >= 3:
                    smoothed = df.loc[valid_mask, col].rolling(
                        window=3, min_periods=1, center=True
                    ).median()
                    df.loc[valid_mask, col] = smoothed.round(1)

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
    fov_horizontal_deg: float = 156.0,
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
    fov_horizontal: float = 156.0,
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
# Калибровка коэффициентов
# =============================================================================

def _compute_sizes_from_pairs(
    pairs: List[dict],
    A: float, B: float, C: float, D: float,
    k1: float, k2: float,
    resolution_scale: float
) -> np.ndarray:
    """
    Быстрое вычисление размеров по предвычисленным парам для заданных коэффициентов.
    Используется в цикле оптимизации — не перечитывает CSV.
    """
    sizes = np.empty(len(pairs))
    for i, p in enumerate(pairs):
        k_abs = max(p['k_percent'], 1.0)
        distance = A * (k_abs ** B)
        pixel_calib = C * (max(distance, 0.1) ** D)

        # Коррекция дисторсии
        r = p.get('r_norm', 0.0)
        distortion_factor = 1.0 + k1 * r**2 + k2 * r**4
        corrected_pixels = p['size_pixels'] * distortion_factor if distortion_factor > 0 else p['size_pixels']

        pixels_ref = corrected_pixels / resolution_scale
        sizes[i] = pixels_ref / pixel_calib if pixel_calib > 0 else 0.0
    return sizes


def _extract_calibration_pairs(
    detections_csv: str,
    geometry_csv: Optional[str],
    frame_width: int,
    frame_height: int,
    apply_tilt_correction: bool = True,
    min_depth_change: float = 0.1,
    min_track_points: int = 3,
    min_size_change_pct: float = 10.0,
) -> Dict[int, List[dict]]:
    """
    Извлекает сырые пары из detection CSV для калибровки.

    Возвращает dict {track_id: list_of_pairs}, где каждая пара содержит:
      k_percent, size_pixels, r_norm, frame_end, x_center, y_center
    """
    df = pd.read_csv(detections_csv)
    geometry_df = pd.read_csv(geometry_csv) if geometry_csv else None

    calibration = CameraCalibration(frame_width=frame_width, frame_height=frame_height)
    diag_half = np.sqrt(frame_width**2 + frame_height**2) / 2

    result = {}
    min_change_ratio = 1.0 + min_size_change_pct / 100.0

    for track_id, track_df in df.groupby('track_id'):
        track_df = track_df.copy()
        if 'depth_m' not in track_df.columns or track_df['depth_m'].isna().all():
            continue

        valid_df = track_df[track_df['depth_m'].notna()].copy()
        if len(valid_df) < min_track_points:
            continue

        valid_df['size_pix'] = valid_df.apply(
            lambda r: _get_bbox_size_pixels(r, frame_width, frame_height), axis=1
        )
        valid_df = valid_df.sort_values('frame').reset_index(drop=True)

        depth_change = valid_df['depth_m'].max() - valid_df['depth_m'].min()
        if depth_change < min_depth_change:
            continue

        pair_data, _, _ = _find_size_pairs(
            valid_df, min_change_ratio, apply_tilt_correction, geometry_df, calibration
        )
        if not pair_data:
            continue

        pair_data_filtered = _filter_pairs_by_mad(pair_data)

        # Добавляем r_norm и координаты конечного кадра
        for p in pair_data_filtered:
            frame_end = int(p['frame_end'])
            end_rows = valid_df[valid_df['frame'] == frame_end]
            if len(end_rows) > 0:
                end_row = end_rows.iloc[0]
                cx = end_row['x_center'] * frame_width
                cy = end_row['y_center'] * frame_height
                ocx = frame_width / 2
                ocy = frame_height / 2
                p['r_norm'] = np.sqrt((cx - ocx)**2 + (cy - ocy)**2) / diag_half
                p['x_center'] = end_row['x_center']
                p['y_center'] = end_row['y_center']
            else:
                p['r_norm'] = 0.0
                p['x_center'] = 0.5
                p['y_center'] = 0.5

        result[int(track_id)] = pair_data_filtered

    return result


def _calibration_loss(
    params: np.ndarray,
    all_track_pairs: Dict[int, List[dict]],
    known_sizes_mm: Dict[int, float],
    resolution_scale: float,
    known_depths: Optional[Dict[int, float]] = None,
) -> float:
    """
    Loss function для оптимизации калибровочных коэффициентов.

    Компоненты:
    1. Size loss: Huber loss по относительной ошибке размера
    2. Distance loss: если known_depths задан — штраф за расхождение
       расчётной дистанции (A*k^B) и реальной (track_depth - camera_depth)
    3. Регуляризация дисторсии
    """
    A, B, C, D, k1, k2 = params
    huber_delta = 0.3
    size_loss = 0.0
    dist_loss = 0.0
    n_tracks = 0

    for track_id, known_size in known_sizes_mm.items():
        pairs = all_track_pairs.get(track_id)
        if not pairs:
            continue

        estimated = _compute_sizes_from_pairs(pairs, A, B, C, D, k1, k2, resolution_scale)
        rel_errors = (estimated - known_size) / known_size
        abs_err = np.abs(rel_errors)
        huber = np.where(
            abs_err <= huber_delta,
            0.5 * rel_errors**2,
            huber_delta * abs_err - 0.5 * huber_delta**2
        )
        size_loss += np.mean(huber)

        # Distance loss: привязка к известной глубине объекта
        if known_depths and track_id in known_depths:
            track_depth = known_depths[track_id]
            for p in pairs:
                k_abs = max(p['k_percent'], 1.0)
                computed_dist = A * (k_abs ** B)
                camera_depth = p['depth_camera']
                true_dist = track_depth - camera_depth
                if true_dist > 0.05:
                    dist_err = (computed_dist - true_dist) / true_dist
                    dist_loss += min(dist_err**2, 1.0)
            dist_loss /= len(pairs)

        n_tracks += 1

    if n_tracks == 0:
        return 1e6

    size_loss /= n_tracks
    dist_loss /= max(n_tracks, 1)

    # Регуляризация дисторсии (мягкая — не мешает оптимизатору)
    reg_distortion = 0.001 * (k1**2 + k2**2)

    return size_loss + dist_loss + reg_distortion


def _fit_power_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Фитирует степенной закон y = A * x^B по данным в log-log пространстве.
    Использует робастную медианную регрессию.
    Возвращает (A, B).
    """
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return 1.0, -1.0
    log_x = np.log(x)
    log_y = np.log(y)
    # OLS в log-пространстве
    B, log_A = np.polyfit(log_x, log_y, 1)
    A = np.exp(log_A)
    return A, B


def _parse_size_depth_file(path: str) -> Dict[int, tuple]:
    """
    Парсит size-depth.txt.
    Формат строки: track_idN:размер_мм:глубина_м
    Возвращает {track_id: (size_mm, depth_m)}
    """
    import re
    result = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r'track_id(\d+):([0-9.]+):([0-9.]+)', line)
            if m:
                tid = int(m.group(1))
                size_mm = float(m.group(2))
                depth_m = float(m.group(3))
                result[tid] = (size_mm, depth_m)
    return result


def _discover_test_dirs(parent_dir: str,
                        detections_name: str = 'ball_detections.csv',
                        geometry_name: str = 'ball_geometry.csv') -> List[dict]:
    """
    Автоматически находит подпапки с калибровочными данными.

    Ищет в каждой подпапке parent_dir:
    - size-depth.txt (в корне подпапки или в output/)
    - output/<detections_name>
    - output/<geometry_name>

    Возвращает список video_spec словарей для calibrate_coefficients().
    """
    parent = Path(parent_dir)
    specs = []
    for subdir in sorted(parent.iterdir()):
        if not subdir.is_dir():
            continue
        # Ищем size-depth.txt
        sd_path = subdir / 'size-depth.txt'
        if not sd_path.exists():
            sd_path = subdir / 'output' / 'size-depth.txt'
        if not sd_path.exists():
            continue
        # Ищем CSV файлы
        det_path = subdir / 'output' / detections_name
        geom_path = subdir / 'output' / geometry_name
        if not det_path.exists():
            continue

        ground_truth = _parse_size_depth_file(str(sd_path))
        if not ground_truth:
            continue

        known_sizes = {tid: vals[0] for tid, vals in ground_truth.items()}
        known_depths = {tid: vals[1] for tid, vals in ground_truth.items()}

        specs.append({
            'detections_csv': str(det_path),
            'geometry_csv': str(geom_path) if geom_path.exists() else None,
            'known_sizes': known_sizes,
            'known_depths': known_depths,
            'name': subdir.name,
        })

    return specs


def calibrate_coefficients(
    video_specs: List[dict],
    output_json: Optional[str] = None,
    frame_width: int = 3840,
    frame_height: int = 2160,
    apply_tilt_correction: bool = True,
    verbose: bool = True
) -> CameraCalibration:
    """
    Оптимизирует калибровочные коэффициенты A, B, C, D, k1, k2 по набору видео.

    Каждый элемент video_specs — словарь:
      detections_csv: путь к CSV с детекциями
      geometry_csv: путь к CSV с геометрией (или None)
      known_sizes: {track_id: size_mm}
      known_depths: {track_id: depth_m}
      name: имя видео (для отображения)

    Track ID из разных видео разделяются через namespace (video_idx * 100000 + track_id).
    """
    from scipy.optimize import differential_evolution

    if not video_specs:
        raise ValueError("Нет видео для калибровки")

    # Объединяем данные из всех видео с namespace track IDs
    all_track_pairs = {}
    known_sizes = {}
    known_depths = {}
    track_to_video = {}  # namespaced_tid -> (video_name, original_tid)

    for vid_idx, spec in enumerate(video_specs):
        if verbose:
            print(f"\n[{spec['name']}] Извлечение пар из {spec['detections_csv']}...")

        video_pairs = _extract_calibration_pairs(
            spec['detections_csv'], spec.get('geometry_csv'),
            frame_width, frame_height,
            apply_tilt_correction=apply_tilt_correction
        )

        spec_known = spec.get('known_sizes', {})
        spec_depths = spec.get('known_depths', {})

        for orig_tid, pairs in video_pairs.items():
            if orig_tid not in spec_known:
                continue
            ns_tid = vid_idx * 100000 + orig_tid
            all_track_pairs[ns_tid] = pairs
            known_sizes[ns_tid] = spec_known[orig_tid]
            if orig_tid in spec_depths:
                known_depths[ns_tid] = spec_depths[orig_tid]
            track_to_video[ns_tid] = (spec['name'], orig_tid)

        if verbose:
            for orig_tid in sorted(spec_known.keys()):
                ns_tid = vid_idx * 100000 + orig_tid
                if ns_tid in all_track_pairs:
                    n_pairs = len(all_track_pairs[ns_tid])
                    depth_info = f", глубина {spec_depths[orig_tid]:.1f} м" if orig_tid in spec_depths else ""
                    print(f"  Трек {orig_tid}: {n_pairs} пар, размер {spec_known[orig_tid]:.0f} мм{depth_info}")
                else:
                    print(f"  Трек {orig_tid}: нет пар (пропущен)")

    available_tracks = set(all_track_pairs.keys()) & set(known_sizes.keys())
    if not available_tracks:
        raise ValueError("Нет пар для калибровки ни в одном из видео")

    if verbose:
        n_total_pairs = sum(len(all_track_pairs[t]) for t in available_tracks)
        print(f"\nВсего: {len(available_tracks)} треков, {n_total_pairs} пар из {len(video_specs)} видео")

    resolution_scale = frame_width / REFERENCE_FRAME_WIDTH

    # Треки с известной глубиной (пересечение available_tracks и known_depths)
    tracks_with_depth = set()
    if known_depths:
        tracks_with_depth = available_tracks & set(known_depths.keys())

    if tracks_with_depth:
        # === Декомпозированный подход с известной глубиной ===
        # Обходим шумную формулу d = A*k^B: используем true_dist = track_depth - camera_depth
        # Фитим только p = C * d^D и дисторсию k1, k2
        if verbose:
            print(f"\nДекомпозированная калибровка ({len(tracks_with_depth)} треков с известной глубиной)...")

        # Собираем данные: true_distance, pixel_calib_true, r_norm
        all_k = []
        all_true_dist = []
        all_pixel_calib_true = []
        all_r_norm = []
        all_pixels_ref = []
        all_known_size = []

        for tid in tracks_with_depth:
            known_size = known_sizes[tid]
            track_depth = known_depths[tid]
            for p in all_track_pairs[tid]:
                true_dist = track_depth - p['depth_camera']
                if true_dist < 0.05:
                    continue
                pixels_ref = p['size_pixels'] / resolution_scale
                pixel_calib_true = pixels_ref / known_size  # px/мм при данной дистанции

                all_k.append(p['k_percent'])
                all_true_dist.append(true_dist)
                all_pixel_calib_true.append(pixel_calib_true)
                all_r_norm.append(p.get('r_norm', 0.0))
                all_pixels_ref.append(pixels_ref)
                all_known_size.append(known_size)

        all_k = np.array(all_k)
        all_true_dist = np.array(all_true_dist)
        all_pixel_calib_true = np.array(all_pixel_calib_true)
        all_r_norm = np.array(all_r_norm)
        all_pixels_ref = np.array(all_pixels_ref)
        all_known_size = np.array(all_known_size)

        if len(all_true_dist) < 2:
            raise ValueError("Недостаточно пар для калибровки")

        n_points = len(all_true_dist)
        if verbose:
            print(f"  {n_points} точек, дистанции {all_true_dist.min():.2f}–{all_true_dist.max():.2f} м")

        # Шаг 1: Фит A, B из d = A * k^B (для справки и дальнейшего использования)
        A, B = _fit_power_law(all_k, all_true_dist)
        if verbose:
            d_pred = A * (np.maximum(all_k, 1.0) ** B)
            d_err = np.abs(d_pred - all_true_dist) / all_true_dist * 100
            print(f"\n  Справка d = A*k^B: A={A:.4f}, B={B:.4f}")
            print(f"  Ошибка дистанции: медиана {np.median(d_err):.1f}%, макс {np.max(d_err):.1f}%")

        # Шаг 2: Фит C, D из p = C * d^D (без дисторсии)
        if verbose:
            print(f"\n  Фит p = C * d^D...")
        C, D = _fit_power_law(all_true_dist, all_pixel_calib_true)
        if verbose:
            p_pred = C * (all_true_dist ** D)
            p_err = np.abs(p_pred - all_pixel_calib_true) / all_pixel_calib_true * 100
            print(f"  C={C:.4f}, D={D:.4f}")
            print(f"  Ошибка pixel_calib: медиана {np.median(p_err):.1f}%, макс {np.max(p_err):.1f}%")

        # Шаг 3: Фит дисторсии k1, k2 из остатков
        # size_estimated = pixels_ref * (1 + k1*r² + k2*r⁴) / (C * d^D)
        # Хотим size_estimated ≈ known_size
        # → (1 + k1*r² + k2*r⁴) ≈ known_size * C * d^D / pixels_ref
        if verbose:
            print(f"\n  Фит дисторсии k1, k2...")

        p_fitted = C * (all_true_dist ** D)
        target_factor = all_known_size * p_fitted / all_pixels_ref
        # target_factor = 1 + k1*r² + k2*r⁴
        residuals = target_factor - 1.0
        r2 = all_r_norm**2
        r4 = all_r_norm**4
        X = np.column_stack([r2, r4])
        if len(X) >= 2:
            result_lstsq = np.linalg.lstsq(X, residuals, rcond=None)
            k1, k2 = result_lstsq[0]
        else:
            k1, k2 = 0.0, 0.0

        if verbose:
            print(f"  k1={k1:.6f}, k2={k2:.6f}")
            # Показать финальные ошибки
            distortion_factor = 1.0 + k1 * r2 + k2 * r4
            corrected_pixels = all_pixels_ref * distortion_factor
            size_est = corrected_pixels / p_fitted
            size_err = np.abs(size_est - all_known_size) / all_known_size * 100
            print(f"  Ошибка размера (с дисторсией): медиана {np.median(size_err):.1f}%, макс {np.max(size_err):.1f}%")

        # Шаг 4: Pipeline-оптимизация через эффективные параметры
        # В pipeline: size = pixels * distortion / (C * (A*k^B)^D)
        #                   = pixels * distortion / (E * k^F)
        # где E = C * A^D, F = B*D — всего 2 параметра для размера + 2 дисторсии.
        # Итеративно: фитим E, F через OLS, потом k1, k2 из остатков.
        if verbose:
            print(f"\n  Pipeline-оптимизация (эффективные параметры E, F)...")

        # Собираем данные всех пар
        all_pairs_k = []
        all_pairs_pixels_ref = []
        all_pairs_r = []
        all_pairs_known = []
        for tid in available_tracks:
            known = known_sizes[tid]
            for p in all_track_pairs[tid]:
                all_pairs_k.append(max(p['k_percent'], 1.0))
                all_pairs_pixels_ref.append(p['size_pixels'] / resolution_scale)
                all_pairs_r.append(p.get('r_norm', 0.0))
                all_pairs_known.append(known)
        all_pairs_k = np.array(all_pairs_k)
        all_pairs_pixels_ref = np.array(all_pairs_pixels_ref)
        all_pairs_r = np.array(all_pairs_r)
        all_pairs_known = np.array(all_pairs_known)

        # Итеративный фит: E, F -> k1, k2 -> повторить
        distortion_f = np.ones(len(all_pairs_k))
        for iteration in range(5):
            # OLS в log-пространстве: log(corrected_pixels / known_size) = log(E) + F*log(k)
            corrected = all_pairs_pixels_ref * distortion_f
            y = np.log(corrected / all_pairs_known)
            X_log = np.column_stack([np.ones(len(y)), np.log(all_pairs_k)])
            coefs = np.linalg.lstsq(X_log, y, rcond=None)[0]
            E = np.exp(coefs[0])
            F = coefs[1]

            # Фит дисторсии из остатков
            size_est_no_dist = all_pairs_pixels_ref / (E * all_pairs_k**F)
            target_factor = all_pairs_known / size_est_no_dist  # want: distortion * size_est = known
            # distortion = 1 / target_factor? No: size = pixels * dist / (E*k^F)
            # target_factor = known / (pixels / (E*k^F)) = known * E * k^F / pixels
            # We need: pixels * dist / (E*k^F) = known → dist = known * E * k^F / pixels = target_factor
            residuals = target_factor - 1.0
            r2_arr = all_pairs_r**2
            r4_arr = all_pairs_r**4
            X_dist = np.column_stack([r2_arr, r4_arr])
            dist_coefs = np.linalg.lstsq(X_dist, residuals, rcond=None)[0]
            k1, k2 = dist_coefs
            distortion_f = 1.0 + k1 * r2_arr + k2 * r4_arr

        # Финальная доводка через Nelder-Mead
        def _eff_loss(params):
            E_, F_, k1_, k2_ = params
            dist_f_ = 1.0 + k1_ * r2_arr + k2_ * r4_arr
            sizes = all_pairs_pixels_ref * dist_f_ / (E_ * all_pairs_k**F_)
            rel_err = (sizes - all_pairs_known) / all_pairs_known
            delta = 0.15
            abs_err = np.abs(rel_err)
            huber = np.where(abs_err <= delta, 0.5 * rel_err**2, delta * (abs_err - 0.5 * delta))
            return np.mean(huber) + 0.0005 * (k1_**2 + k2_**2)

        nm_result = minimize(
            _eff_loss, [E, F, k1, k2],
            method='Nelder-Mead',
            options={'maxiter': 100000, 'xatol': 1e-14, 'fatol': 1e-14}
        )
        E, F, k1, k2 = nm_result.x

        if verbose:
            print(f"  E={E:.6f}, F={F:.6f}, k1={k1:.6f}, k2={k2:.6f}")
            print(f"  Pipeline loss: {nm_result.fun:.8f}")

        # Декомпозируем E, F обратно в A, B, C, D
        # Для distance сохраняем исходные A, B (лучший фит k→distance)
        # Тогда C = E / A^D, но нужен D: F = B*D → D = F/B
        D = F / B if abs(B) > 1e-6 else -1.0
        C = E / (A**D) if A > 0 else E
        if verbose:
            print(f"  Декомпозиция: A={A:.4f}, B={B:.4f}, C={C:.4f}, D={D:.4f}")

    else:
        # === Совместная оптимизация без known_depths ===
        if verbose:
            print("\nСовместная оптимизация (без известной глубины)...")

        bounds = [
            (10.0, 200.0),    # A
            (-1.5, -0.3),     # B
            (1.0, 15.0),      # C
            (-2.0, -0.5),     # D
            (-0.3, 0.05),     # k1
            (-0.2, 0.2),      # k2
        ]

        de_result = differential_evolution(
            _calibration_loss,
            bounds=bounds,
            args=(all_track_pairs, known_sizes, resolution_scale, None),
            seed=42, maxiter=1000, tol=1e-8, polish=True, disp=verbose
        )

        nm_result = minimize(
            _calibration_loss, de_result.x,
            args=(all_track_pairs, known_sizes, resolution_scale, None),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        A, B, C, D, k1, k2 = nm_result.x

    # Создаём калибровку с оптимизированными коэффициентами
    calibration = CameraCalibration(
        frame_width=frame_width,
        frame_height=frame_height,
        distance_coef_A=round(A, 4),
        distance_coef_B=round(B, 4),
        pixel_calib_C=round(C, 4),
        pixel_calib_D=round(D, 4),
        distortion_k1=round(k1, 6),
        distortion_k2=round(k2, 6),
    )

    # Вспомогательная функция для отображения имени трека
    def _track_label(tid):
        if tid in track_to_video:
            vname, orig_tid = track_to_video[tid]
            return f"{vname}/t{orig_tid}"
        return str(tid)

    # Валидация
    errors_direct = []
    errors_pipeline = []
    direct_rows = []  # Собираем строки прямой валидации

    for tid in sorted(available_tracks):
        pairs = all_track_pairs[tid]
        known = known_sizes[tid]
        label = _track_label(tid)

        # Прямая оценка (через true_dist) — только для треков с известной глубиной
        if known_depths and tid in known_depths:
            track_depth = known_depths[tid]
            direct_sizes = []
            for p in pairs:
                true_d = track_depth - p['depth_camera']
                if true_d < 0.05:
                    continue
                pix_ref = p['size_pixels'] / resolution_scale
                r = p.get('r_norm', 0.0)
                dist_f = 1.0 + k1 * r**2 + k2 * r**4
                corrected = pix_ref * dist_f
                pc = C * (max(true_d, 0.1) ** D)
                direct_sizes.append(corrected / pc if pc > 0 else 0)
            if direct_sizes:
                med_direct = np.median(direct_sizes)
                err_direct = (med_direct - known) / known * 100
                errors_direct.append(abs(err_direct))
                r_meds = [p.get('r_norm', 0) for p in pairs]
                true_dists = [track_depth - p['depth_camera'] for p in pairs]
                direct_rows.append(
                    f"{label:<18}  {known:>8.1f}мм  {med_direct:>8.1f}мм  {err_direct:>+8.1f}%  {np.median(true_dists):>7.2f}м  {np.median(r_meds):>7.3f}"
                )

        # Pipeline оценка (через k → A*k^B)
        estimated = _compute_sizes_from_pairs(pairs, A, B, C, D, k1, k2, resolution_scale)
        median_est = np.median(estimated)
        rel_err = (median_est - known) / known * 100
        errors_pipeline.append(abs(rel_err))

    if verbose:
        # Прямая валидация (только если есть данные)
        if direct_rows:
            print("\n" + "=" * 85)
            print("Валидация (прямая, с известной дистанцией):")
            print(f"{'Трек':<18}  {'Истинный':>10}  {'Оценка':>10}  {'Ошибка%':>9}  {'Дист.':>8}  {'r_med':>7}")
            print("-" * 85)
            for row in direct_rows:
                print(row)
            print("-" * 85)
            print(f"Средняя ошибка (прямая): {np.mean(errors_direct):.1f}%")

        # Pipeline таблица
        print("\n" + "=" * 85)
        print("Валидация (pipeline: k -> d=A*k^B -> size):")
        print(f"{'Трек':<18}  {'Истинный':>10}  {'Оценка':>10}  {'Ошибка%':>9}  {'Дист.(м)':>9}  {'r_med':>7}")
        print("-" * 85)
        for tid in sorted(available_tracks):
            pairs = all_track_pairs[tid]
            known = known_sizes[tid]
            label = _track_label(tid)
            estimated = _compute_sizes_from_pairs(pairs, A, B, C, D, k1, k2, resolution_scale)
            median_est = np.median(estimated)
            rel_err = (median_est - known) / known * 100
            distances = [A * (max(p['k_percent'], 1.0) ** B) for p in pairs]
            r_norms = [p.get('r_norm', 0) for p in pairs]
            print(f"{label:<18}  {known:>8.1f}мм  {median_est:>8.1f}мм  {rel_err:>+8.1f}%  {np.median(distances):>8.2f}  {np.median(r_norms):>7.3f}")
        print("-" * 85)
        print(f"Средняя ошибка (pipeline): {np.mean(errors_pipeline):.1f}%")

        print(f"\nКоэффициенты:")
        print(f"  distance: d = {A:.4f} * k^({B:.4f})")
        print(f"  pixel:    p = {C:.4f} * d^({D:.4f})")
        print(f"  distortion: k1={k1:.6f}, k2={k2:.6f}")

    # Сохраняем результат
    if output_json:
        import json
        result_data = {
            'distance_coef_A': calibration.distance_coef_A,
            'distance_coef_B': calibration.distance_coef_B,
            'pixel_calib_C': calibration.pixel_calib_C,
            'pixel_calib_D': calibration.pixel_calib_D,
            'distortion_k1': calibration.distortion_k1,
            'distortion_k2': calibration.distortion_k2,
            'optical_center_x': calibration.optical_center_x,
            'optical_center_y': calibration.optical_center_y,
            'n_pairs_used': sum(len(all_track_pairs[t]) for t in available_tracks),
            'n_tracks_used': len(available_tracks),
            'n_videos_used': len(video_specs),
            'videos': [s['name'] for s in video_specs],
            'mean_error_direct_pct': round(float(np.mean(errors_direct)), 2) if errors_direct else None,
            'mean_error_pipeline_pct': round(float(np.mean(errors_pipeline)), 2),
        }
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"\nРезультат сохранён: {output_json}")

    return calibration


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
    geom.add_argument('--frame-step', type=int, default=1,
                      help='Шаг обработки кадров для optical flow (1=каждый, 2=через один). Ускоряет обработку.')
    
    # size
    size = subparsers.add_parser('size', help='Оценка размеров по трекам')
    size.add_argument('--detections', '-d', required=True)
    size.add_argument('--output', '-o')
    size.add_argument('--tracks', '-t')
    size.add_argument('--geometry', '-g', help='CSV с данными геометрии (наклон камеры)')
    size.add_argument('--width', type=int, default=3840)
    size.add_argument('--height', type=int, default=2160)
    size.add_argument('--min-depth-change', type=float, default=0.1)
    size.add_argument('--min-track-points', type=int, default=3)
    size.add_argument('--apply-tilt-correction', action='store_true', default=True,
                      help='Применять коррекцию наклона камеры (по умолчанию: вкл.)')
    size.add_argument('--no-tilt-correction', dest='apply_tilt_correction', action='store_false',
                      help='Отключить коррекцию наклона камеры')
    size.add_argument('--calibration', '-cal',
                      help='JSON файл с калибровкой (из calibrate)')

    # calibrate
    calib = subparsers.add_parser('calibrate', help='Оптимизация калибровочных коэффициентов')
    calib.add_argument('--test-dir', required=True,
                       help='Родительская папка с тестовыми видео. Каждая подпапка должна содержать '
                            'size-depth.txt и output/ball_detections.csv + output/ball_geometry.csv')
    calib.add_argument('--detections-name', default='ball_detections.csv',
                       help='Имя файла детекций в output/ (по умолчанию: ball_detections.csv)')
    calib.add_argument('--geometry-name', default='ball_geometry.csv',
                       help='Имя файла геометрии в output/ (по умолчанию: ball_geometry.csv)')
    calib.add_argument('--output', '-o', default='calibration_result.json',
                       help='JSON с результатами')
    calib.add_argument('--width', type=int, default=3840)
    calib.add_argument('--height', type=int, default=2160)
    calib.add_argument('--apply-tilt-correction', action='store_true', default=True)
    calib.add_argument('--no-tilt-correction', dest='apply_tilt_correction', action='store_false')

    # volume
    vol = subparsers.add_parser('volume', help='Расчёт осмотренного объёма воды')
    vol.add_argument('--detections', '-d', required=True)
    vol.add_argument('--tracks', '-t')
    vol.add_argument('--ctd', '-c')
    vol.add_argument('--output', '-o')
    vol.add_argument('--fov', type=float, default=156.0)
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
        process_video_geometry(args.video, args.output, args.interval,
                              frame_step=args.frame_step)
    
    elif args.command == 'size':
        output = args.output or args.detections.replace('.csv', '_with_size.csv')
        tracks = args.tracks or args.detections.replace('.csv', '_track_sizes.csv')
        cal = None
        if args.calibration:
            cal = CameraCalibration.from_json(args.calibration)
            cal.frame_width = args.width
            cal.frame_height = args.height
        process_detections_with_size(
            args.detections, output, tracks, args.geometry,
            calibration=cal,
            frame_width=args.width, frame_height=args.height,
            min_depth_change=args.min_depth_change,
            min_track_points=args.min_track_points,
            apply_tilt_correction=args.apply_tilt_correction
        )

    elif args.command == 'calibrate':
        video_specs = _discover_test_dirs(
            args.test_dir,
            detections_name=args.detections_name,
            geometry_name=args.geometry_name
        )
        if not video_specs:
            parser.error(f"Не найдено подпапок с калибровочными данными в {args.test_dir}")

        calibrate_coefficients(
            video_specs,
            output_json=args.output,
            frame_width=args.width, frame_height=args.height,
            apply_tilt_correction=args.apply_tilt_correction
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
