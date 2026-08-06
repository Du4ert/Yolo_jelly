# Принцип работы приложения

Ниже описан фактический принцип работы приложения по коду: `batch/core/processor.py`, `src/detect_video.py`, `src/camera_geometry.py`, `src/analyze.py`, `src/interactive_plot.py`.

Таск-менеджер в отчёте не рассматривается.

## Общий конвейер

Приложение обрабатывает видео в несколько этапов:

1. Детекция объектов YOLO-моделью.
2. Трекинг объектов между кадрами.
3. Привязка детекций к времени, глубине, температуре и солёности.
4. Постобработка треков: фильтрация коротких треков, стабилизация класса.
5. Оценка геометрии камеры по оптическому потоку.
6. Расчёт реальных размеров организмов.
7. Расчёт глубины объекта и дистанции до него.
8. Расчёт осмотренного объёма воды и плотности организмов.
9. Построение графиков и отчётов.
10. При необходимости рендеринг видео с подписями размеров, дистанций и наклона камеры.

## 1. Детекция объектов

Для каждого кадра видео приложение запускает модель `YOLO` из `ultralytics`.

Входные параметры по умолчанию из `config.json`:

```text
conf_threshold = 0.6
enable_tracking = true
tracker_type = bytetrack.yaml
imgsz = 1280
half = true
device = auto
```

Если рядом с моделью `.pt` есть TensorRT-файл `.engine`, приложение использует его вместо обычной модели.

Для каждого кадра считается время:

```text
timestamp_s = frame_index / fps
```

Если подключён CTD-файл, глубина и параметры среды берутся интерполяцией по времени:

```text
depth(t) = interp(t, ctd_time, ctd_depth)
temperature(t) = interp(t, ctd_time, ctd_temperature)
salinity(t) = interp(t, ctd_time, ctd_salinity)
```

Если CTD нет, но задана скорость погружения:

```text
depth(t) = t * depth_rate
```

YOLO возвращает для каждого объекта:

```text
class_id
confidence
x_center
y_center
width
height
```

Координаты и размеры bbox хранятся в нормализованном виде от `0` до `1`.

Площадь bbox:

```text
bbox_area_norm = width * height
```

Классы:

```text
0: Aurelia aurita
1: Beroe ovata
2: Mnemiopsis leidyi
3: Pleurobrachia pileus
4: Rhizostoma pulmo
```

## 2. Трекинг объектов

Если трекинг включён, вместо обычного вызова модели используется:

```text
model.track(..., persist=True, tracker="bytetrack.yaml")
```

Каждой детекции присваивается `track_id`.

Для каждого трека сохраняются:

```text
first_frame
last_frame
first_timestamp
last_timestamp
first_depth
last_depth
avg_confidence
```

Длительность трека:

```text
duration_s = last_timestamp - first_timestamp
```

Размах по кадрам:

```text
frame_span = last_frame - first_frame + 1
```

Изменение глубины за трек:

```text
depth_change_m = last_depth - first_depth
```

Если один и тот же `track_id` появляется после большого разрыва, приложение разделяет его на разные треки. Разрыв задаётся параметром:

```text
max_track_gap = 60 кадров
```

Условие разделения:

```text
if current_frame - previous_frame > max_track_gap:
    создаётся новый track_id
```

Короткие треки отбрасываются:

```text
track_length < min_track_length
```

В GUI по умолчанию:

```text
min_track_length = 5
```

Для стабилизации класса используется голосование по треку. Если в одном треке модель иногда ошибалась с видом, итоговый класс трека выбирается как наиболее частый:

```text
dominant_class = argmax_class count(class_id)
```

После этого все детекции трека получают доминирующий класс.

## 3. Геометрия камеры и FOE

Геометрия камеры используется для оценки наклона камеры и для коррекции расчёта размеров.

Приложение анализирует оптический поток между кадрами:

1. На предыдущем кадре ищутся характерные точки `goodFeaturesToTrack`.
2. Их положение на следующем кадре считается методом Лукаса-Канаде `calcOpticalFlowPyrLK`.
3. Для каждой точки получается вектор движения:

```text
vector = point_new - point_old
```

Далее оценивается `FOE`, Focus of Expansion, то есть точка, от которой визуально расходятся векторы движения при спуске камеры.

Для точки `p_i` и кандидата FOE `f` строится радиальный вектор:

```text
r_i = p_i - f
```

Нормированный радиальный вектор:

```text
r_hat_i = r_i / ||r_i||
```

Нормированный вектор оптического потока:

```text
v_hat_i = v_i / ||v_i||
```

Качество совпадения направления:

```text
dot_i = r_hat_i * v_hat_i
```

Ошибка для оптимизации:

```text
error(f) = mean(1 - dot_i)
```

Оптимизация ищет такой `f`, чтобы ошибка была минимальной.

Для устойчивости используется RANSAC:

```text
n_ransac = 150
sample_size = 30
inlier_threshold = 0.7
```

Точка считается inlier, если:

```text
dot_i > 0.7
```

Это примерно соответствует углу меньше `45°` между ожидаемым радиальным направлением и фактическим вектором.

Углы наклона камеры считаются через смещение FOE относительно центра кадра:

```text
cx = width / 2
cy = height / 2
pixels_per_degree = frame_width / fov_horizontal
```

По умолчанию для GoPro:

```text
fov_horizontal = 156°
```

Формулы:

```text
tilt_horizontal_deg = (foe_x - cx) / pixels_per_degree
tilt_vertical_deg   = (foe_y - cy) / pixels_per_degree
```

Полный наклон:

```text
tilt_total = sqrt(tilt_horizontal_deg^2 + tilt_vertical_deg^2)
```

Для коррекции размеров используется:

```text
cos_tilt = cos(tilt_total)
```

Если камера наклонена, эффективное изменение дистанции до объекта меньше, чем изменение глубины камеры:

```text
delta_distance_real = delta_depth_camera * cos(tilt_total)
```

Поэтому измеренный прирост размера корректируется:

```text
k_corrected = k_measured / cos_tilt
```

## 4. Режим оптического потока

Для каждого интервала приложение классифицирует поток:

```text
radial
parallel
ambiguous
```

Используются:

```text
foe_confidence
flow_median_speed
flow_p95_speed
flow_direction_std_deg
```

Круговое стандартное отклонение направлений считается так:

```text
angle_i = atan2(v_y, v_x)
R = |mean(exp(i * angle_i))|
flow_direction_std = sqrt(-2 * ln(R))
flow_direction_std_deg = degrees(flow_direction_std)
```

Режимы:

```text
radial:
    foe_confidence >= 0.5

parallel:
    flow_direction_std_deg < 45°
    and flow_median_speed > 2 px/frame

ambiguous:
    остальные случаи
```

`radial` нужен для основного k-метода размеров, `parallel` может использоваться для параллакс-метода.

## 5. Расчёт размера bbox в пикселях

Для каждой детекции нормализованные размеры переводятся в пиксели:

```text
w_px = width * frame_width
h_px = height * frame_height
```

Обычно размер объекта в пикселях:

```text
size_px = max(w_px, h_px)
```

Для `Rhizostoma pulmo` берётся меньшая сторона, потому что важен диаметр купола:

```text
size_px = min(w_px, h_px)
```

## 6. Основной k-метод расчёта размера

Главная идея: при погружении камеры объект становится ближе и его bbox растёт. По скорости роста bbox можно оценить дистанцию до объекта, а затем реальный размер.

Для пары кадров одного трека берутся:

```text
pixels1 = размер bbox в первом кадре
pixels2 = размер bbox во втором кадре
depth1 = глубина камеры в первом кадре
depth2 = глубина камеры во втором кадре
```

Изменение глубины:

```text
delta_d = depth2 - depth1
```

Относительное изменение размера:

```text
relative_change = (pixels2 - pixels1) / pixels1
```

Удельный прирост размера:

```text
k = relative_change / delta_d
```

В коде `k` сначала считается в долях на метр, затем переводится в проценты на метр:

```text
k_percent = k * 100
```

Если есть данные наклона камеры:

```text
k_corrected = k / cos_tilt
```

Условие для пары: размер должен увеличиться минимум на заданный процент. В коде для k-метода используется:

```text
min_size_change_pct = 10%
```

То есть:

```text
pixels2 >= pixels1 * 1.10
```

Пороги глубины разделены явно:

```text
min_track_depth_span_m = 0.3 м
min_pair_depth_change_m = 0.01 м
```

Первый параметр задаёт общий диапазон глубины трека, второй — минимальную
разницу глубины между двумя точками одной пары. Калибровочный baseline сохраняет
исторический порог диапазона трека `0.1 м` для воспроизводимости коэффициентов.

## 7. Вертикальный зазор до объекта

Модель оценивает вертикальный зазор между камерой и объектом. Историческое имя
`distance_m` сохранено для совместимости CSV и API:

```text
vertical_offset_m = distance_m = A * |k_percent|^B
```

По умолчанию:

```text
A = 80.00
B = -0.9
```

Итого:

```text
distance_m = 80.00 * |k_percent|^(-0.9)
```

Перед расчётом `k_percent` ограничивается снизу:

```text
k_percent >= 1.0
```

Надёжный диапазон дистанций по умолчанию:

```text
min_reliable_distance = 0.1 м
max_reliable_distance = 3.0 м
```

В `config.json` для UI сейчас указано:

```text
min_reliable_distance = 0.1 м
max_reliable_distance = 5.0 м
```

Если рассчитанная дистанция меньше минимальной:

```text
distance_m = min_reliable_distance
```

## 8. Пиксельная калибровка

После оценки дистанции считается масштаб изображения:

```text
pixel_calibration = C * distance_m^D
```

Где:

```text
C = 4.35
D = -1.25
```

Итого:

```text
pixel_calibration = 4.35 * distance_m^(-1.25)
```

Единица измерения:

```text
px/mm
```

То есть сколько пикселей приходится на 1 мм реального размера на данной дистанции.

## 9. Реальный размер

Перед расчётом размера пиксели нормализуются к референсному разрешению GoPro 4K:

```text
REFERENCE_FRAME_WIDTH = 3840
resolution_scale = frame_width / 3840
pixels_ref = pixels / resolution_scale
```

Реальный размер:

```text
size_mm = pixels_ref / pixel_calibration
```

В сантиметрах:

```text
size_cm = size_mm / 10
```

## 10. Глубина объекта

Так как камера смотрит вниз, объект считается глубже камеры на величину дистанции до объекта:

```text
object_depth_m = camera_depth_m + vertical_offset_m
```

Для покадровой таблицы дистанция до объекта восстанавливается обратно:

```text
distance_to_object_m = vertical_offset_m
                     = object_depth_m - camera_depth_m
```

С ограничением снизу:

```text
vertical_offset_m >= min_reliable_distance
```

Таким образом, текущие характеристики имеют разные роли:

```text
camera_depth_m       — абсолютная глубина камеры
vertical_offset_m    — вертикальный зазор от камеры до объекта
distance_m           — совместимое имя vertical_offset_m в таблице треков
distance_to_object_m — совместимое имя vertical_offset_m в таблице детекций
object_depth_m       — абсолютная глубина объекта
```

## 11. Коррекция дисторсии

В калибровке предусмотрена радиальная коррекция fisheye-дисторсии:

```text
size_corrected = size_raw * (1 + k1 * r^2 + k2 * r^4)
```

Где `r` — нормированное расстояние от оптического центра:

```text
ocx = optical_center_x * frame_width
ocy = optical_center_y * frame_height

diag_half = sqrt(frame_width^2 + frame_height^2) / 2

r = sqrt((x - ocx)^2 + (y - ocy)^2) / diag_half
```

По умолчанию:

```text
distortion_k1 = 0.0
distortion_k2 = 0.0
optical_center_x = 0.5
optical_center_y = 0.5
```

То есть фактически коррекция выключена, пока в калибровочном JSON не заданы другие коэффициенты.

## 12. Фильтрация выбросов в k-методе

Для пар кадров считается набор `k_percent`.

Медиана:

```text
k_median = median(k_percent)
```

MAD:

```text
MAD = median(|k_i - k_median|)
```

Порог:

```text
threshold = 3 * MAD * 1.4826
```

Допустимый диапазон:

```text
k_min = max(k_median - threshold, 1.0)
k_max = k_median + threshold
```

Оставляются пары:

```text
k_min <= k_i <= k_max
```

Если `MAD = 0`, верхний порог задаётся проще:

```text
k_max = k_median * 3
k_min = 1.0
```

## 13. Сглаживание размера

После расчёта размеров по парам применяется скользящая медиана:

```text
size_smoothed_i = median(size_raw в окне вокруг i)
```

Окно по умолчанию:

```text
smoothing_window = 3
```

Финальный размер выбирается как последняя стабильная точка до устойчивого уменьшения. Уменьшение считается значимым, если:

```text
size_i < size_(i-1) * 0.98
```

Если таких уменьшений подряд минимум два, финальная точка берётся перед началом падения.

Если выбранный размер слишком отличается от медианы:

```text
|size_final - median_size| > 0.5 * median_size
```

то приложение ищет ближайшую стабильную точку или берёт медиану.

## 14. Уверенность оценки размера

Базовая уверенность k-метода:

```text
confidence = 1.0
```

Дальше применяются штрафы.

Если дистанция больше надёжной:

```text
distance > max_reliable_distance:
    confidence = 0.3
```

Если дистанция больше `2 м`, но ещё не вышла за максимум:

```text
distance > 2.0:
    confidence = 0.6
```

Нестабильность `k`:

```text
cv_k = k_std / k_mean
```

Если:

```text
cv_k > 0.5
```

то:

```text
confidence *= 0.7
```

Нестабильность размеров:

```text
cv_size = std(size_smoothed) / mean(size_smoothed)
```

Если:

```text
cv_size > 0.3
```

то:

```text
confidence *= 0.8
```

Проверка на типичный диапазон вида:

```text
if size_cm < typical_min * 0.3
or size_cm > typical_max * 3:
    confidence *= 0.5
```

## 15. Фиксированный размер для Pleurobrachia pileus

Для `Pleurobrachia pileus` k-метод не применяется. В коде задан фиксированный размер:

```text
size_mm = 10.0
size_cm = 1.0
```

Вертикальный зазор принимается минимальным надёжным:

```text
vertical_offset_m = distance_m = min_reliable_distance
```

Глубина объекта:

```text
object_depth_m = camera_depth_last + vertical_offset_m
```

Метод помечается как:

```text
method = fixed
confidence = 0.5
```

## 16. Fallback по типичному размеру вида

Если k-метод не сработал, приложение может использовать типичный размер вида.

Типичные размеры в коде:

```text
Aurelia aurita:
    mean = 5 см, min = 3 см, max = 40 см

Mnemiopsis leidyi:
    mean = 3 см, min = 2 см, max = 12 см

Beroe ovata:
    mean = 5 см, min = 3 см, max = 15 см

Rhizostoma pulmo:
    mean = 30 см, min = 10 см, max = 60 см
```

Берётся средний типичный размер:

```text
typical_size_mm = typical_mean_cm * 10
```

Находится кадр с максимальным размером bbox:

```text
max_size_pix
```

Пиксельная калибровка через известный типичный размер:

```text
pixel_calibration = max_size_pix_ref / typical_size_mm
```

Затем дистанция восстанавливается из формулы:

```text
pixel_calibration = C * d^D
d = (pixel_calibration / C)^(1 / D)
d = (pixel_calibration / 4.35)^(1 / -1.25)
```

Дистанция ограничивается:

```text
min_reliable_distance <= d <= max_reliable_distance + 2.0
```

Метод помечается как:

```text
method = typical
confidence = 0.2
```

## 17. Параллакс-метод

Если оптический поток классифицирован как `parallel`, может использоваться параллакс.

Идея: ближние объекты движутся по изображению быстрее дальних.

Берётся скорость фонового потока:

```text
v_ref = median(flow_p95_speed)
```

Скорость объекта по треку:

```text
dx_i = x_(i+1) - x_i
dy_i = y_(i+1) - y_i
d_frames_i = frame_(i+1) - frame_i

speed_i = sqrt(dx_i^2 + dy_i^2) / d_frames_i
v_obj = median(speed_i)
```

Дистанция:

```text
d_obj = d_ref * (v_ref / v_obj)
```

Где:

```text
d_ref = min_reliable_distance
```

Затем:

```text
d_obj = clip(d_obj, min_reliable_distance, max_reliable_distance)
```

Размер считается стандартно:

```text
pixel_calibration = 4.35 * d_obj^(-1.25)
size_mm = max_size_pix_ref / pixel_calibration
```

В parallax fallback величина `d_obj` также трактуется как вертикальный зазор.
Глубина объекта:

```text
object_depth_m = camera_depth_m + d_obj
```

Метод:

```text
method = parallax
base confidence = 0.5
```

## 18. Покадровое назначение размеров

После оценки размера по треку приложение добавляет в CSV детекций:

```text
estimated_size_mm
estimated_size_cm
object_depth_m
distance_to_object_m
vertical_offset_m
size_confidence
size_method
```

Для k-метода размер может меняться по кадрам. Тогда используется линейная интерполяция:

```text
estimated_size_mm(frame) = interp(frame, frames_with_estimates, sizes_mm)
```

На краях `np.interp` зажимает значение к ближайшей доступной оценке.

Для остальных методов размер постоянный по всему треку:

```text
estimated_size_mm = real_size_mm
```

## 19. Осмотренный объём воды

Приложение использует цилиндрическую модель. Камера проходит диапазон глубин, а область наблюдения моделируется как цилиндр с эллиптическим сечением.

Для всех видов, кроме `P. pileus`, эффективная дистанция обнаружения `d_eff`
берётся одним из способов:

1. Если задана вручную, используется ручное значение.
2. Иначе берётся перцентиль дальних дистанций обнаружения треков.
3. Если данных нет, используется fallback по виду.

Технические fixed-дистанции `P. pileus` в эту автооценку не входят.

Для каждого трека дальняя дистанция считается как `P95` покадровых дистанций:

```text
max_detection_distance_m = quantile_0.95(distance_to_object_m по треку)
```

Потом эффективная дистанция:

```text
d_eff = percentile(max_detection_distance_m)
```

По умолчанию:

```text
percentile = 90
```

То есть:

```text
d_eff = P90(P95(distance_to_object_m внутри каждого трека))
```

С ограничением:

```text
min_reliable_distance <= d_eff <= max_reliable_distance
```

Если данных нет, используются значения:

```text
Aurelia aurita: 2.0 м
Rhizostoma pulmo: 2.5 м
Mnemiopsis leidyi: 1.0 м
Beroe ovata: 1.2 м
```

Для `P. pileus` effective distance задаётся отдельно. По умолчанию она равна
`min_reliable_distance`, выбранной оператором, и может быть заменена любой
положительной ручной величиной. Автовычисление по трекам для этого вида не
применяется. Площадь, высота цилиндра, объём и плотности `P. pileus`
рассчитываются отдельно по тем же формулам.

Ширина области обзора на дистанции `d_eff`:

```text
w = 2 * d_eff * tan(fov_h / 2)
```

Высота области обзора:

```text
h = 2 * d_eff * tan(fov_v / 2)
```

Площадь эллиптического сечения:

```text
A_eff = (pi / 4) * w * h
```

Диапазон глубин:

```text
depth_traversed = depth_max - depth_min
```

Высота цилиндра:

```text
H = depth_traversed + d_eff
```

Добавка `+ d_eff` — это “нижняя шапка”: на финальной глубине камера ещё видит объекты ниже себя.

Объём:

```text
V_total = A_eff * H
```

Скорость погружения:

```text
descent_rate = depth_traversed / duration
```

Плотность по видам:

```text
density_per_m3[class] = count[class] / V_total
```

Также сохраняется плотность на площадь сечения:

```text
density_per_m2[class] = count[class] / A_eff
```

## 20. Подсчёт организмов

Если есть таблица треков, приложение считает не детекции, а уникальные треки:

```text
count[class] = nunique(track_id)
```

Если треков нет, fallback:

```text
count[class] = number of detections
```

Это важно: для отчёта по плотности организмов предпочтительно используется количество треков, чтобы один и тот же организм не считался много раз по кадрам.

## 21. Статистика размеров по видам

Для каждого вида считаются медиана и стандартное отклонение размера.

Медиана:

```text
median_size = median(real_size_cm)
```

Для стандартного отклонения сначала отбрасываются выбросы через MAD:

```text
deviation_i = |size_i - median_size|
MAD = median(deviation_i)
threshold = 3 * MAD * 1.4826
```

Оставляются:

```text
deviation_i <= threshold
```

После этого:

```text
std_size = std(filtered_sizes)
```

## 22. Статические графики

Приложение строит три основных PNG-графика.

Вертикальное распределение:

```text
depth_bin_value = floor(depth_m / depth_bin) * depth_bin
counts[depth_bin, class_name] = number of detections
```

График строится как горизонтальная гистограмма по глубине для каждого вида.

Временная шкала:

```text
time_bin_value = floor(timestamp_s / time_bin) * time_bin
counts[time_bin, class_name] = number of detections
```

График строится как stacked bar chart по времени.

Сводка по видам:

```text
count = number of detections
conf_mean = mean(confidence)
conf_std = std(confidence)
depth_min = min(depth_m)
depth_max = max(depth_m)
depth_mean = mean(depth_m)
```

На графике показываются:

```text
количество детекций по видам
средняя уверенность ± std
```

## 23. Интерактивный график глубины

Интерактивный график строится по таблице размеров треков.

Обязательные колонки:

```text
object_depth_m
class_name
real_size_cm
```

Для каждого вида строится KDE по глубине.

Используется Gaussian KDE без внешних зависимостей.

Ширина ядра по правилу Сильвермана:

```text
bandwidth = 1.06 * std(depths) * n^(-1/5)
```

Минимум:

```text
bandwidth >= 0.25
```

Для каждой точки сетки глубины:

```text
KDE(z) = sum(exp(-0.5 * ((z - depth_i) / bandwidth)^2)) / (n * bandwidth * sqrt(2*pi))
```

Затем KDE умножается на число организмов вида:

```text
density(z) = KDE(z) * n
```

Если передана площадь сечения наблюдения:

```text
density(z) = density(z) / A_eff
```

Тогда единицы становятся:

```text
экз./м²/м глубины
```

Иначе:

```text
экз./м глубины
```

Размер маркеров на графике нормируется по реальным размерам:

```text
marker_size = 6 + 20 * (size_cm - min_size_cm) / (max_size_cm - min_size_cm)
```

Если все размеры одинаковые:

```text
marker_size = 12
```

## 24. Термоклин на интерактивном графике

Если передан CTD и выбрана температурная колонка, приложение может найти термоклин.

Берутся глубина и температура:

```text
depth_i
temp_i
```

Градиент между соседними точками:

```text
gradient_i = (temp_(i+1) - temp_i) / (depth_(i+1) - depth_i)
```

Используется модуль:

```text
gradient_abs_i = |gradient_i|
```

Глубина сегмента:

```text
segment_depth_i = (depth_i + depth_(i+1)) / 2
```

В режиме `threshold` термоклин выбирается среди участков:

```text
gradient_abs_i > thermocline_threshold
```

По умолчанию:

```text
thermocline_threshold = 0.2 °C/м
```

Из кандидатов выбирается максимальный градиент:

```text
thermocline = argmax(gradient_abs_i)
```

В режиме `maximum` выбирается максимальный градиент вообще.

## 25. Рендеринг видео с размерами

После расчёта размеров приложение может отрендерить видео, где под bbox выводится:

```text
distance_to_object_m
estimated_size_cm
```

Текст:

```text
"{distance:.2f}m | {size:.1f}cm"
```

Также может выводиться наклон камеры:

```text
Tilt H
Tilt V
Total
```

Полный наклон:

```text
total_tilt = sqrt(tilt_h^2 + tilt_v^2)
```

Цвет индикации:

```text
total_tilt < 15°: зелёный
15° <= total_tilt < 30°: жёлтый
total_tilt >= 30°: красный
```

## 26. Основные выходные файлы

После обработки обычно появляются:

```text
*_detections.csv
```

Базовые детекции по кадрам.

```text
*_tracks.csv
```

Статистика треков.

```text
detections_with_size.csv
```

Детекции с добавленными размерами, глубинами объектов и дистанциями.

```text
track_sizes.csv
```

Итоговые размеры по трекам.

```text
geometry.csv
```

FOE, наклоны камеры и параметры оптического потока.

```text
volume.csv
```

Объём, плотности и статистика размеров.

```text
vertical_distribution.png
detection_timeline.png
species_summary.png
depth_interactive.html
report.txt
```

Графики и отчёт.
