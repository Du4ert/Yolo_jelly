# Модуль camera_geometry.py

Оценка геометрии камеры и реальных размеров объектов для подводного видео.

## Возможности

1. **Оценка наклона камеры** — по движению морского снега (Focus of Expansion)
2. **Расчёт реального размера объектов** — по динамике изменения размера в треке
3. **Определение абсолютной глубины объекта** — на какой глубине находится объект в толще воды

## Калибровка

Калибровочные данные получены для **GoPro 12 Wide 4K**:
- Константа k = 156.1 пиксель·м
- Коэффициент K = 2365 пикс/м на 1м дистанции
- Точность: ~2-3% в диапазоне 1-2.5 м

## Использование

### 1. Оценка размеров объектов по трекам

```bash
python src/camera_geometry.py size --detections output/detections.csv
```

Параметры:
- `--detections`, `-d` — CSV с детекциями (из detect_video.py)
- `--output`, `-o` — выходной CSV с детекциями (по умолчанию: *_with_size.csv)
- `--tracks`, `-t` — выходной CSV со статистикой треков (по умолчанию: *_track_sizes.csv)
- `--width` — ширина кадра (по умолчанию: 1920)
- `--height` — высота кадра (по умолчанию: 1080)
- `--min-depth-change` — мин. изменение глубины камеры для оценки (по умолчанию: 0.3 м)
- `--min-track-points` — мин. точек в треке (по умолчанию: 5)

### 2. Оценка наклона камеры

```bash
python src/camera_geometry.py geometry --video видео.mp4 --output output/geometry.csv
```

## Алгоритм оценки размера

Для неподвижного объекта при погружении камеры:

```
size(t) = K × real_size / distance(t)
distance(t) = |object_depth - camera_depth(t)|
```

По треку строится регрессия `1/size` от `camera_depth`:
- Наклон даёт реальный размер объекта
- Пересечение даёт глубину объекта

**Размер берётся в кадре с максимальным bbox** (объект ближе всего к камере), исключая последние кадры где объект может уходить за границы.

## Выходные файлы

### detections_with_size.csv

К исходным детекциям добавляются колонки:
- `estimated_size_cm` — размер объекта в см
- `estimated_size_m` — размер объекта в м  
- `object_depth_m` — абсолютная глубина объекта
- `distance_to_object_m` — дистанция от камеры до объекта
- `size_confidence` — уверенность оценки (0-1)

### track_sizes.csv

Статистика по каждому треку:
- `track_id`, `class_name` — идентификация
- `real_size_cm`, `real_size_m` — оценённый размер
- `object_depth_m` — глубина объекта в воде
- `distance_at_max_m` — дистанция в момент макс. размера
- `camera_depth_at_max_m` — глубина камеры в этот момент
- `max_size_frame` — номер кадра с макс. размером
- `max_size_pixels` — размер в пикселях
- `confidence` — уверенность
- `fit_r_squared` — R² линейной регрессии
- `warnings` — предупреждения

## Ограничения

1. **Требуется трекинг** — нужны данные track_id из detect_video.py с флагом `--track`
2. **Требуются данные глубины** — из CTD или `--depth-rate`
3. **Мин. изменение глубины** — для надёжной оценки нужно Δdepth ≥ 0.3 м
4. **Pleurobrachia pileus** — слишком мелкий вид, размер не оценивается
5. **Объект должен быть неподвижен** — активно плавающие объекты дадут ошибку

## Пример полного пайплайна

```bash
# 1. Детекция с трекингом
python src/detect_video.py \
    --video dive.mp4 \
    --model best.pt \
    --track \
    --ctd ctd_data.csv \
    --csv output/detections.csv

# 2. Оценка размеров
python src/camera_geometry.py size \
    --detections output/detections.csv \
    --output output/detections_with_size.csv \
    --tracks output/track_sizes.csv

# 3. Анализ
python src/analyze.py --csv output/detections_with_size.csv
```

## Типичные размеры видов (для валидации)

| Вид | Типичный размер (см) |
|-----|---------------------|
| Aurelia aurita | 10-40 (купол) |
| Rhizostoma pulmo | 20-60 (купол) |
| Beroe ovata | 5-15 |
| Mnemiopsis leidyi | 3-12 |
| Pleurobrachia pileus | 1-3 (не оценивается) |
