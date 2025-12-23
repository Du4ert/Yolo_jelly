# Модуль camera_geometry.py

Оценка геометрии камеры и реальных размеров объектов для подводного видео.

## Возможности

1. **Оценка наклона камеры** — по движению морского снега (Focus of Expansion)
2. **Расчёт реального размера объектов** — по калибровочным данным и дистанции
3. **Интеграция с детекциями** — добавление оценок размера к результатам YOLO

## Калибровка

Калибровочные данные получены для **GoPro 12 Wide 4K**:
- Константа k = 156.1 пиксель·м (по среднему диаметру Ферета)
- Коэффициент K = 2365 пикс/м на 1м дистанции
- Оптимальный диапазон измерений: 1.0 - 2.5 м
- Точность в оптимальном диапазоне: ~2.3% СКО

## Использование

### 1. Оценка наклона камеры

```bash
python src/camera_geometry.py geometry --video видео.mp4 --output output/geometry.csv
```

Параметры:
- `--video`, `-v` — путь к видео
- `--output`, `-o` — выходной CSV (по умолчанию: output/geometry.csv)
- `--interval`, `-i` — интервал оценки в кадрах (по умолчанию: 30)

Выходной CSV содержит:
- `frame_start`, `frame_end` — диапазон кадров
- `timestamp_s` — время
- `foe_x`, `foe_y` — координаты Focus of Expansion
- `tilt_horizontal_deg` — наклон по горизонтали (+ вправо)
- `tilt_vertical_deg` — наклон по вертикали (+ вниз)
- `confidence` — уверенность оценки (0-1)

### 2. Добавление размеров к детекциям

```bash
python src/camera_geometry.py size --detections output/detections.csv --output output/detections_with_size.csv
```

Параметры:
- `--detections`, `-d` — CSV с детекциями (из detect_video.py)
- `--output`, `-o` — выходной CSV
- `--geometry`, `-g` — CSV с геометрией камеры (опционально, для коррекции)
- `--width` — ширина кадра (по умолчанию: 1920)
- `--height` — высота кадра (по умолчанию: 1080)

Добавляемые колонки:
- `estimated_size_m` — оценка размера в метрах
- `estimated_distance_m` — оценка дистанции
- `size_confidence` — уверенность оценки

### 3. Использование как библиотеки

```python
from camera_geometry import (
    CameraCalibration,
    estimate_foe_from_frames,
    estimate_object_size,
    process_video_geometry,
    add_size_estimates_to_detections
)

# Калибровка (можно изменить параметры)
calibration = CameraCalibration(
    k=156.1,  # константа калибровки
    frame_width=1920,
    frame_height=1080
)

# Оценка размера объекта
from camera_geometry import estimate_object_size
result = estimate_object_size(
    bbox_size_pixels=100,  # размер bbox в пикселях
    distance_m=1.5,        # дистанция в метрах
    calibration=calibration
)
print(f"Размер: {result.size_m * 100:.1f} см")
print(f"Уверенность: {result.confidence:.0%}")

# Обработка видео
df = process_video_geometry("video.mp4", frame_interval=60)
```

## Формулы

**Размер объекта:**
```
размер_реальный (м) = размер_пикс × дистанция (м) / 2365
```

**Дистанция по известному размеру:**
```
дистанция (м) = 156.1 × (известный_размер / 0.066) / размер_пикс
```

## Ограничения

1. Калибровка выполнена для GoPro 12 Wide — для других камер нужна перекалибровка
2. Дистанция берётся из глубины CTD — это приближение (не учитывает горизонтальное смещение)
3. При сильном наклоне камеры (>30°) точность снижается
4. Оптимальный диапазон 1-2.5 м — за его пределами ошибка растёт

## Типичные размеры видов (для справки)

| Вид | Типичный размер (см) |
|-----|---------------------|
| Aurelia aurita | 10-40 (купол) |
| Rhizostoma pulmo | 20-60 (купол) |
| Beroe ovata | 5-15 |
| Mnemiopsis leidyi | 3-12 |
| Pleurobrachia pileus | 1-3 |
