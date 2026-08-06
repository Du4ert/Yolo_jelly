# YOLO Jellyfish Detector

Система детекции и классификации желетелого макрозоопланктона на подводном видео с использованием нейросети YOLOv8.

## Целевые виды

| Класс ID | Вид | Тип |
|----------|-----|-----|
| 0 | *Aurelia aurita* | Медуза |
| 1 | *Beroe ovata* | Гребневик |
| 2 | *Mnemiopsis leidyi* | Гребневик |
| 3 | *Pleurobrachia pileus* | Гребневик |
| 4 | *Rhizostoma pulmo* | Медуза |

## Структура проекта

```
yolo_jellyfish/
├── README.md                 # Документация
├── requirements.txt          # Зависимости
├── data.yaml                 # Конфигурация датасета
├── batch_app.py              # GUI приложение для пакетной обработки
├── batch/                    # Модули пакетной обработки
│   ├── core/                 # Ядро (процессоры, менеджер задач)
│   ├── database/             # Работа с БД
│   └── ui/                   # Интерфейс
├── src/                      # Скрипты командной строки
│   ├── extract_frames.py     # Извлечение кадров из видео
│   ├── augment.py            # Аугментация данных
│   ├── train.py              # Обучение модели
│   ├── detect_video.py       # Детекция на видео с трекингом
│   ├── camera_geometry.py    # Геометрия камеры и размеры объектов
│   ├── analyze.py            # Анализ результатов (matplotlib)
│   ├── interactive_plot.py   # Интерактивная визуализация (Plotly)
│   └── utils.py              # Вспомогательные функции
├── dataset/                  # Данные для обучения
├── runs/                     # Результаты обучения
└── output/                   # Результаты детекции
```

---

## Установка

### 1. Создание виртуального окружения

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

### Тесты

```bash
python -m unittest discover -s tests -v
```

Набор содержит 13 тестов калибровочной схемы, XY-коррекции дисторсии,
формирования пар и воспроизводимости baseline. Тесты с локальным каталогом
`test_video/` пропускаются, если калибровочные данные недоступны.

### 3. Проверка GPU (опционально)

```python
import torch
print(f"CUDA доступна: {torch.cuda.is_available()}")
print(f"Устройство: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

---

## Графическое приложение (batch_app.py)

GUI приложение для пакетной обработки видео с поддержкой:
- Организации погружений по экспедициям
- Управления очередью задач
- Геометрии камеры и расчёта размеров
- Анализа и визуализации результатов

### Запуск

```bash
python batch_app.py
```

### Горячие клавиши

| Клавиша | Действие |
|---------|----------|
| `Ctrl+D` | Добавить погружение |
| `Ctrl+M` | Добавить модель |
| `Ctrl+G` | Геометрия и размеры |
| `Ctrl+A` | Анализ детекций |
| `F5` | Запустить очередь |
| `F6` | Пауза |
| `F7` | Остановить |
| `Ctrl+Q` | Выход |

### Меню "Инструменты"

#### Геометрия и размеры (Ctrl+G)

Диалог с тремя вкладками:

1. **Наклон камеры (FOE)** — анализ видео для оценки наклона камеры по морскому снегу
2. **Размеры объектов** — расчёт реальных размеров объектов по динамике изменения bbox
3. **Объём воды** — расчёт осмотренного объёма и плотности организмов

#### Анализ детекций (Ctrl+A)

Генерация графиков и отчётов:
- Вертикальное распределение по глубине
- Временная шкала детекций
- Сводка по видам
- Текстовый отчёт
- Интерактивный HTML-график (если есть файл треков с размерами)

#### Постобработка задачи → Анализ

При добавлении подзадачи «📊 Анализ и графики» через меню постобработки или при создании задачи с автопостобработкой доступно поле **«Колонки CTD»** — номера колонок CTD-файла для наложения на интерактивный график (0-based, через запятую, по умолчанию `6`).

---

## Скрипты командной строки

### 1. Извлечение кадров (extract_frames.py)

Извлечение кадров из видео для разметки.

```bash
python src/extract_frames.py --video путь/к/видео.mp4 --output dataset/raw_frames --interval 15
```

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--video`, `-v` | — | Путь к видеофайлу (обязательный) |
| `--output`, `-o` | `output/frames` | Директория для сохранения кадров |
| `--interval`, `-i` | 30 | Интервал между кадрами |

---

### 2. Аугментация данных (augment.py)

Аугментация изображений для обучения.

```bash
python src/augment.py --images dataset/images/train --labels dataset/labels/train --multiply 4
```

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--images` | — | Директория с изображениями (обязательный) |
| `--labels` | — | Директория с аннотациями (обязательный) |
| `--multiply`, `-m` | 3 | Количество аугментированных копий |

---

### 3. Обучение модели (train.py)

Обучение YOLO модели.

```bash
python src/train.py --data data.yaml --epochs 150 --model yolov8m.pt
```

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--data` | `data.yaml` | Конфигурация датасета |
| `--model` | `yolov8m.pt` | Базовая модель (n/s/m/l/x) |
| `--epochs` | 150 | Количество эпох |
| `--imgsz` | 640 | Размер входного изображения |
| `--batch` | 16 | Размер батча |
| `--device` | auto | Устройство (0, 1, cpu) |
| `--name` | `jellyfish_v1` | Имя эксперимента |

---

### 4. Детекция на видео (detect_video.py)

Детекция объектов на видео с поддержкой трекинга.

```bash
python src/detect_video.py \
    --video путь/к/видео.mp4 \
    --model runs/best/weights/best.pt \
    --track \
    --show-trails \
    --output output/результат.mp4 \
    --csv output/детекции.csv
```

#### Основные параметры

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--video`, `-v` | — | Путь к входному видео (обязательный) |
| `--model`, `-m` | — | Путь к обученной модели (обязательный) |
| `--output`, `-o` | `output/detected.mp4` | Путь к выходному видео |
| `--csv` | `output/detections.csv` | Путь к CSV с детекциями |
| `--conf` | 0.25 | Порог уверенности (0-1) |
| `--depth-rate` | None | Скорость погружения (м/с) |
| `--ctd` | None | Путь к CSV с данными CTD |
| `--no-video` | False | Не сохранять видео (только CSV) |

#### Параметры инференса (GPU/производительность)

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--device` | `auto` | Устройство: `auto`, `cpu`, `0`, `1`, ... (`auto` выбирает GPU если доступен) |
| `--imgsz` | 1280 | Размер входного изображения для YOLO |
| `--no-half` | False | Отключить FP16 (half precision) инференс |

#### Параметры трекинга

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--track` | False | Включить трекинг объектов |
| `--tracker` | `bytetrack.yaml` | Тип трекера (`bytetrack.yaml` / `botsort.yaml`) |
| `--show-trails` | False | Показывать траектории движения |
| `--trail-length` | 30 | Длина траектории в кадрах |
| `--min-track-length` | 3 | Минимальная длина трека для сохранения |
| `--no-dominant-class` | False | Не переназначать класс трека на доминирующий |

#### Экспорт для Label Studio

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--export-ls` | None | Папка для экспорта кадров и предразметки в формате Label Studio |
| `--export-ls-interval` | 15 | Интервал кадров при экспорте |
| `--export-ls-classes` | все | Классы для экспорта, через пробел: `'Aurelia aurita' 'Beroe ovata'` |

#### TensorRT

| Параметр | Описание |
|----------|----------|
| `--export-engine` | Экспортировать модель в TensorRT `.engine` и выйти (требует TensorRT и GPU) |

При наличии рядом с `.pt` файла `.engine` он будет загружен автоматически.

---

### 5. Геометрия камеры и размеры (camera_geometry.py)

Модуль для анализа геометрии камеры и расчёта размеров объектов.

#### Подкоманда: geometry

Оценка наклона камеры по Focus of Expansion.

```bash
python src/camera_geometry.py geometry \
    --video путь/к/видео.mp4 \
    --output output/geometry.csv \
    --interval 30
```

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--video`, `-v` | — | Путь к видеофайлу (обязательный) |
| `--output`, `-o` | `output/geometry.csv` | Выходной CSV |
| `--interval`, `-i` | 30 | Интервал анализа в кадрах |

**Выходной файл** (`geometry.csv`):
- `frame_start`, `frame_end` — диапазон кадров
- `timestamp_s` — время в секундах
- `foe_x`, `foe_y` — координаты Focus of Expansion
- `tilt_horizontal_deg`, `tilt_vertical_deg` — наклон камеры
- `confidence` — уверенность оценки (< 0.5 = выброс)
- `n_vectors` — количество использованных векторов

---

#### Подкоманда: size

Оценка размеров объектов по трекам.

```bash
# Базовый вызов (камера считается вертикальной)
python src/camera_geometry.py size \
    --detections output/detections.csv \
    --output output/detections_with_size.csv \
    --tracks output/track_sizes.csv

# С коррекцией наклона камеры
python src/camera_geometry.py size \
    --detections output/detections.csv \
    --geometry output/geometry.csv \
    --output output/detections_with_size.csv

# Отключить коррекцию наклона (даже если есть geometry.csv)
python src/camera_geometry.py size \
    --detections output/detections.csv \
    --geometry output/geometry.csv \
    --no-tilt-correction
```

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--detections`, `-d` | — | CSV с детекциями (обязательный) |
| `--output`, `-o` | auto | CSV с детекциями + размеры |
| `--tracks`, `-t` | auto | CSV со статистикой треков |
| `--geometry`, `-g` | None | CSV с геометрией камеры (опционально) |
| `--width` | 3840 | Ширина кадра |
| `--height` | 2160 | Высота кадра |
| `--min-depth-change` | 0.1 | Мин. изменение глубины (м) |
| `--min-track-points` | 3 | Мин. точек в треке |
| `--apply-tilt-correction` | True | Применять коррекцию наклона |
| `--no-tilt-correction` | — | Отключить коррекцию наклона |
| `--calibration` | None | JSON с откалиброванными коэффициентами (из `calibrate`) |

**Коррекция наклона камеры:**

Если камера наклонена на угол θ от вертикали, размер объекта увеличивается медленнее при погружении, так как реальное сближение с объектом составляет `Δdepth × cos(θ)`.

- **Без `--geometry`** — камера считается вертикальной (θ = 0°)
- **С `--geometry`** — применяется коррекция: `k_real = k_measured / cos(θ)`

| Наклон камеры | Коррекция k |
|----------------|----------------|
| 0° (вертикально) | +0% |
| 15° | +3.5% |
| 30° | +15.5% |
| 45° | +41.4% |
| 60° | +100% |

**Методы оценки размера:**
1. **k_method** — по удельному приросту размера (наилучший)
2. **fixed** — для мелких видов (P. pileus)
3. **typical** — по типичным размерам вида (fallback)

**Выходные колонки в детекциях:**
- `estimated_size_mm` — оценённый размер (мм)
- `estimated_size_cm` — оценённый размер (см)
- `vertical_offset_m` — вертикальный зазор между камерой и объектом
- `distance_to_object_m` — совместимое имя того же вертикального зазора
- `object_depth_m` — абсолютная глубина объекта: `depth_m + vertical_offset_m`
- `size_confidence` — уверенность оценки
- `size_method` — использованный метод

**Файл треков** (`track_sizes.csv`):
- `track_id`, `class_name` — идентификация
- `real_size_mm`, `real_size_cm` — размер объекта
- `distance_m` — совместимое имя вертикального зазора на кадре измерения
- `vertical_offset_m` — тот же зазор с явным физическим названием
- `object_depth_m` — абсолютная глубина объекта в столбе воды
- `k_mean_pct_per_m`, `k_std_pct_per_m` — статистика k (%/м)
- `method`, `confidence` — метаданные
- `warnings` — предупреждения (включая `tilt_corrected_XXdeg` если применена коррекция)

---

#### Подкоманда: calibrate

Автоматическая калибровка коэффициентов по наборам объектов с известными размерами
и глубинами. Каждая подпапка тестового каталога должна содержать `size-depth.txt`
и `output/ball_detections.csv`; `output/ball_geometry.csv` опционален.

```bash
python src/camera_geometry.py calibrate \
    --test-dir test_video \
    --output calibration_xy.json
```

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--test-dir` | — | Родительский каталог калибровочных наборов (обязательный) |
| `--detections-name` | `ball_detections.csv` | Имя CSV детекций в `output/` |
| `--geometry-name` | `ball_geometry.csv` | Имя опционального CSV геометрии в `output/` |
| `--output`, `-o` | `calibration_xy.json` | Выходной JSON с коэффициентами |
| `--width` | 3840 | Ширина кадра |
| `--height` | 2160 | Высота кадра |
| `--min-reliable-distance` | 0.1 | Нижняя граница надёжной дистанции |
| `--max-reliable-distance` | 3.0 | Верхняя граница надёжной дистанции |
| `--no-tilt-correction` | — | Отключить коррекцию наклона |
| `--min-track-depth-span` | 0.1 | Диапазон глубины калибровочного трека, м |
| `--min-pair-depth-change` | 0.01 | Минимальный шаг глубины калибровочной пары, м |
| `--min-size-change-pct` | 10 | Минимальный рост bbox калибровочной пары, % |

Формат `size-depth.txt`:

```text
track_id1:35:3.4
```

Поля: идентификатор трека, размер объекта в миллиметрах, абсолютная глубина
объекта в метрах.

**Контрольный baseline:**

```bash
python src/calibration_baseline.py \
    --test-dir test_video \
    --calibration calibration_xy.json \
    --output calibration_baseline.json
```

Отчёт фиксирует хэши входных CSV, версии библиотек, параметры выборки пар,
коэффициенты, ошибки дистанции и размера, пространственное покрытие, результаты
по каждому треку и долю оценок за пределами надёжного диапазона. Файл
`calibration_baseline.json` является контрольной точкой перед изменением формул.

Проверка порога минимального изменения глубины выполняется с исключением одного
видео из калибровки на каждой итерации:

```bash
python src/calibration_threshold_sensitivity.py \
    --test-dir test_video \
    --thresholds 0.01 0.05 0.10 \
    --output calibration_threshold_sensitivity.json
```

Порог `0.01 м` оставлен без изменений. При увеличении порога до `0.05 м`
средняя по видео ошибка медианной дистанции трека возрастает с 18.32% до 19.10%,
а ошибка размера с 13.53% до 14.86%. При `0.10 м` ошибка дистанции трека
возрастает до 28.11%.

В выходных CSV величины разделены явно:

- `vertical_offset_m` — вертикальный зазор, прибавляемый к глубине камеры;
- `object_depth_m` — абсолютная глубина объекта;

Совместимое поле `distance_to_object_m` пока совпадает с `vertical_offset_m`.

**Коррекция дисторсии:**

GoPro Wide 156° FOV вносит радиальное искажение — объекты на периферии кадра имеют завышенный пиксельный размер. Калибровка подбирает коэффициенты k1 и k2, корректирующие размер в зависимости от расстояния от оптического центра: `size_corrected = size_raw * (1 + k1*r² + k2*r⁴)`.

---

#### Использование калибровки в size

```bash
# Применить откалиброванные коэффициенты
python src/camera_geometry.py size \
    --detections output/detections.csv \
    --geometry output/geometry.csv \
    --calibration calibration_xy.json
```

Флаг `--calibration` загружает JSON и заменяет дефолтные коэффициенты (A, B, C,
D) и параметры XY-дисторсии на откалиброванные.

---

#### Подкоманда: volume

Расчёт осмотренного объёма воды и плотности организмов.

```bash
python src/camera_geometry.py volume \
    --detections output/detections_with_size.csv \
    --tracks output/track_sizes.csv \
    --ctd ctd_data.csv \
    --output output/volume.csv \
    --fov 156 \
    --percentile 90
```

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--detections`, `-d` | — | CSV с детекциями (обязательный). Рекомендуется выход подкоманды `size`. |
| `--tracks`, `-t` | None | CSV со статистикой треков с колонкой `max_detection_distance_m`. |
| `--ctd`, `-c` | None | CSV с данными CTD (для полного диапазона глубин). |
| `--output`, `-o` | auto | Выходной CSV |
| `--fov` | 156.0 | Горизонтальный FOV камеры (°) |
| `--percentile` | 90.0 | Перцентиль дальних дистанций обнаружения треков для эффективной дистанции d_eff (0–100) |
| `--detection-distance` | None | Ручное значение d_eff (м). None = автоматически по перцентилю. |
| `--near-distance` | 0.1 | Устарел, игнорируется (оставлен для совместимости CLI). |
| `--depth-min` | None | Минимальная глубина (м) |
| `--depth-max` | None | Максимальная глубина (м) |
| `--duration` | None | Длительность записи (с) |
| `--fps` | 60.0 | Частота кадров |
| `--width` | 3840 | Ширина кадра |
| `--height` | 2160 | Высота кадра |

**Цилиндрическая модель объёма**

Столб воды моделируется как цилиндр с эллиптическим сечением, вписанным в конус обзора камеры на эффективной дистанции `d_eff`:

```
V = A_eff · H
A_eff = (π/4) · w · h,  w = 2·d_eff·tan(fov_h/2),  h = 2·d_eff·tan(fov_v/2)
H     = depth_traversed + d_eff         # шапка снизу
d_eff = clamp(P_percentile(max_detection_distance_m), min_reliable, max_reliable)
```

Эффективная дистанция `d_eff` определяется эмпирически как `--percentile` (по умолчанию P90)
дальних дистанций обнаружения треков. В `track_sizes.csv` это колонка `max_detection_distance_m`,
рассчитанная как P95 покадрового `distance_to_object_m` внутри трека. Если `--tracks` без этой колонки,
дистанции агрегируются из детекций по `track_id`; затем используется типичная дистанция вида.

**Важно:** объём считается по ВСЕМУ диапазону погружения. Пустая вода учитывается для правильного расчёта плотности.

**Выходные параметры (`volume.csv`):**
- `total_volume_m3` — осмотренный объём V (м³)
- `effective_distance_m` — эффективная дистанция d_eff (м)
- `cylinder_height_m` — высота цилиндра H = depth_traversed + d_eff (м)
- `cross_section_area_m2` — площадь эллиптического сечения A_eff (м²)
- `depth_min_m`, `depth_max_m`, `depth_traversed_m` — диапазон глубин
- `fov_horizontal_deg`, `fov_vertical_deg` — углы обзора
- `duration_s`, `descent_rate_m_s` — длительность и скорость погружения
- `count_<Species>` — количество особей по видам
- `density_<Species>_per_m3` — плотность (особей/м³)

---

### 6. Анализ результатов (analyze.py)

Построение графиков и генерация отчётов.

```bash
python src/analyze.py --csv output/detections.csv --output-dir output --depth-bin 2.0
```

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--csv`, `-c` | — | Путь к CSV с детекциями (обязательный) |
| `--output-dir`, `-o` | `output/` | Директория для графиков |
| `--depth-bin` | 1.0 | Шаг биннинга по глубине (м) |
| `--report` | `report.txt` | Имя текстового отчёта |

**Генерируемые файлы:**
- `vertical_distribution.png` — профиль вертикального распределения
- `detection_timeline.png` — временная шкала детекций
- `species_summary.png` — сводка по видам
- `report.txt` — текстовый отчёт

---

### 7. Интерактивная визуализация (interactive_plot.py)

Построение интерактивного HTML-графика распределения организмов по глубине с наложением CTD-параметров. Использует Plotly.

```bash
# Базовый интерактивный HTML
python src/interactive_plot.py \
    --track-sizes output/dive001_detections_track_sizes.csv \
    --output output/depth_plot

# С CTD данными (колонки 6 и 7)
python src/interactive_plot.py \
    --track-sizes output/dive001_detections_track_sizes.csv \
    --ctd ctd_data.csv \
    --ctd-columns 6,7 \
    --output output/depth_plot

# Экспорт в SVG или PDF (требует kaleido)
python src/interactive_plot.py \
    --track-sizes output/dive001_detections_track_sizes.csv \
    --format svg \
    --output output/depth_plot

# Просмотр колонок CTD-файла
python src/interactive_plot.py --ctd ctd_data.csv --list-ctd-columns
```

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--track-sizes`, `-t` | — | CSV со статистикой треков (обязательный) |
| `--output`, `-o` | `output/depth_plot` | Путь к файлу без расширения |
| `--depth-bin` | 1.0 | Шаг биннинга по глубине (м) |
| `--title` | авто | Заголовок графика |
| `--format`, `-f` | `html` | Формат экспорта: `html`, `svg`, `pdf`, `png` |
| `--ctd` | None | CSV с данными CTD (опционально) |
| `--ctd-columns` | — | Номера колонок CTD (0-based), через запятую: `5,6,7` |
| `--list-ctd-columns` | — | Вывести список колонок CTD-файла и выйти |

> При любом формате экспорта всегда дополнительно сохраняется `.html`.
> Для экспорта в SVG/PDF/PNG установите: `pip install kaleido`

**Входной файл** (`track_sizes.csv`) — выходной файл подкоманды `size` camera_geometry.py:
- `track_id` — идентификатор трека
- `class_name` — название вида
- `real_size_cm` — размер объекта (см)
- `object_depth_m` — глубина объекта в столбе воды

**Слои на графике:**
- **Численность (N)** — линии количества по глубине для каждого вида
- **Размеры** — точки организмов (площадь маркера пропорциональна размеру)
- **CTD параметры** — пунктирные линии CTD-профилей, нормализованные к оси X

**Интерактивность (HTML):**
- Клик по элементу легенды — скрыть/показать
- Двойной клик по заголовку группы — показать только эту группу
- Колёсико мыши — масштаб
- Наведение — подробная информация

**Интеграция с batch-приложением:**
Шаг «📊 Анализ» автоматически строит интерактивный график если доступен файл треков с размерами. CTD-колонки задаются в параметрах подзадачи (по умолчанию колонка `6`).

---

## Полный цикл обработки погружения

### Через GUI (рекомендуется)

1. Запустить `python batch_app.py`
2. Добавить экспедицию: **+ Экспедиция**
3. Добавить папку с погружением: **+ Папка** → сканировать
4. Добавить модель: **Файл → Добавить модель**
5. Выбрать видео → правый клик → **Добавить в очередь**
6. Запустить очередь: **F5**
7. После обработки: **Инструменты → Геометрия и размеры**
8. Анализ: **Инструменты → Анализ детекций**

### Через командную строку

```bash
# 1. Детекция с трекингом
python src/detect_video.py \
    --video dive_001.mp4 \
    --model best.pt \
    --track \
    --ctd ctd_data.csv \
    --csv output/dive001_detections.csv

# 2. Оценка наклона камеры
python src/camera_geometry.py geometry \
    --video dive_001.mp4 \
    --output output/geometry.csv

# 3. Расчёт размеров
python src/camera_geometry.py size \
    --detections output/dive001_detections.csv \
    --geometry output/geometry.csv

# 4. Расчёт объёма и плотности
python src/camera_geometry.py volume \
    --detections output/dive001_detections_with_size.csv \
    --tracks output/dive001_detections_track_sizes.csv \
    --ctd ctd_data.csv

# 5. Анализ и графики
python src/analyze.py \
    --csv output/dive001_detections_with_size.csv \
    --output-dir output/analysis \
    --depth-bin 2.0

# 6. Интерактивная визуализация с CTD
python src/interactive_plot.py \
    --track-sizes output/dive001_detections_track_sizes.csv \
    --ctd ctd_data.csv \
    --ctd-columns 6 \
    --output output/analysis/depth_interactive
```

---

## Форматы данных

### Формат CTD

CSV с разделителем (запятая, точка с запятой или `|`):

```csv
time,depth,temperature,salinity
0.0,0.0,24.5,18.2
1.0,0.5,24.3,18.2
2.0,1.0,24.1,18.3
```

Обязательные колонки: `time`, `depth`  
Опциональные: `temperature`, `salinity`

### Формат детекций

```csv
frame,timestamp_s,depth_m,track_id,class_id,class_name,confidence,x_center,y_center,width,height
0,0.00,0.0,1,0,Aurelia aurita,0.89,0.453,0.621,0.124,0.098
```

### Формат статистики треков

```csv
track_id,class_id,class_name,first_frame,last_frame,duration_s,detections_count,avg_confidence
1,0,Aurelia aurita,0,45,1.50,42,0.893
```

---

## Решение типичных проблем

### Много ложных срабатываний на взвеси
- Увеличьте порог `--conf` до 0.4-0.5
- Добавьте класс "частица" в обучающий датасет

### Нереальные значения наклона камеры
- FOE автоматически фильтрует выбросы (confidence < 0.5)
- При малом движении камеры наклон определить невозможно

### Плохая оценка размеров
- Для k-метода нужен диапазон глубины трека не менее 0.3 м
- Минимальный шаг глубины внутри пары по умолчанию равен 0.01 м
- Bbox между точками пары должен увеличиться минимум на 10%
- Для fallback используются типичные размеры видов
- Проверьте `method` в результатах: `k_method` — лучший
- Если камера наклонена, используйте `--geometry` для коррекции
- При наклоне > 30° размеры могут быть занижены на 15%+ без коррекции

### Нехватка памяти GPU
- Уменьшите `--batch` (8 или 4)
- Используйте меньшую модель (yolov8s.pt)

---

## Системные требования

### Минимальные
- Python 3.8+
- 8 GB RAM
- CPU

### Рекомендуемые
- Python 3.10+
- 16+ GB RAM
- NVIDIA GPU с 8+ GB VRAM
- CUDA 11.8+

---

## Зависимости

```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
albumentations>=1.3.0
torch>=2.0.0
scipy>=1.10.0
sqlalchemy>=2.0.0
PyQt6>=6.4.0
```

---

## Лицензия

MIT License

## Автор

Проект для исследования вертикального распределения желетелого макрозоопланктона в Чёрном море.
