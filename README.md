# YOLO Jellyfish Detector

Система детекции и классификации желетелого макрозоопланктона на подводном видео с использованием нейросети YOLOv8.

## Целевые виды

| Класс ID | Вид | Тип |
|----------|-----|-----|
| 0 | *Aurelia aurita* | Медуза |
| 1 | *Rhizostoma pulmo* | Медуза |
| 2 | *Beroe ovata* | Гребневик |
| 3 | *Mnemiopsis leidyi* | Гребневик |
| 4 | *Pleurobrachia pileus* | Гребневик |

## Структура проекта

```
yolo_jellyfish/
├── README.md                 # Документация
├── requirements.txt          # Зависимости
├── data.yaml                 # Конфигурация датасета
├── src/                      # Исходный код
│   ├── extract_frames.py     # Извлечение кадров из видео
│   ├── augment.py            # Аугментация данных
│   ├── train.py              # Обучение модели
│   ├── detect_video.py       # Детекция на видео с трекингом
│   ├── analyze.py            # Анализ результатов
│   └── utils.py              # Вспомогательные функции
├── dataset/                  # Данные
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
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

### 3. Проверка GPU (опционально)

```python
import torch
print(f"CUDA доступна: {torch.cuda.is_available()}")
print(f"Устройство: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

---

## Подготовка данных

### Шаг 1. Извлечение кадров из видео

```bash
python src/extract_frames.py --video путь/к/видео.mp4 --output dataset/raw_frames --interval 15
```

**Параметры:**
- `--video` — путь к видеофайлу
- `--output` — директория для сохранения кадров
- `--interval` — интервал между кадрами (по умолчанию 30, при 30 fps = 1 кадр/сек)

### Шаг 2. Разметка данных

Рекомендуемые инструменты:
- **CVAT** (https://cvat.ai) — веб-интерфейс, интерполяция между кадрами
- **Label Studio** (https://labelstud.io) — гибкий, много форматов экспорта
- **LabelImg** — простой десктопный инструмент

#### Формат аннотаций YOLO

Каждому изображению `image.jpg` соответствует файл `image.txt`:

```
# class_id x_center y_center width height
# Все значения нормализованы от 0 до 1
0 0.453 0.621 0.124 0.098
2 0.234 0.445 0.067 0.089
```

#### Рекомендации по разметке

| Вид | Особенности |
|-----|-------------|
| *Aurelia aurita* | Весь купол + краевые щупальца, 4 гонады видны |
| *Rhizostoma pulmo* | Купол + массивные ротовые лопасти |
| *Beroe ovata* | Овальное тело, по максимальным габаритам |
| *Mnemiopsis leidyi* | Лопастевидные выросты включать полностью |
| *Pleurobrachia pileus* | Тело + щупальца (если видны) |

### Шаг 3. Распределение данных

Распределите размеченные данные по папкам:
- `dataset/images/train/` и `dataset/labels/train/` — 70%
- `dataset/images/val/` и `dataset/labels/val/` — 20%
- `dataset/images/test/` и `dataset/labels/test/` — 10%

### Шаг 4. Аугментация (опционально)

```bash
python src/augment.py --images dataset/images/train --labels dataset/labels/train --multiply 4
```

**Параметры:**
- `--images` — директория с изображениями
- `--labels` — директория с аннотациями
- `--multiply` — количество аугментированных копий на изображение

Аугментация включает:
- Повороты и отражения
- Изменение яркости/контраста (имитация разных глубин)
- Размытие (имитация движения камеры)
- Шум (имитация взвеси в воде)

---

## Обучение модели

### Базовое обучение

```bash
python src/train.py --data data.yaml --epochs 150 --model yolov8m.pt
```

### Параметры обучения

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--data` | `data.yaml` | Конфигурация датасета |
| `--model` | `yolov8m.pt` | Базовая модель (n/s/m/l/x) |
| `--epochs` | 150 | Количество эпох |
| `--imgsz` | 640 | Размер входного изображения |
| `--batch` | 16 | Размер батча |
| `--device` | auto | Устройство (0, 1, cpu) |
| `--name` | `jellyfish_v1` | Имя эксперимента |

### Выбор модели

| Модель | Параметры | Рекомендация |
|--------|-----------|--------------|
| `yolov8n.pt` | 3.2M | Тесты, слабое железо |
| `yolov8s.pt` | 11.2M | Быстрая работа |
| `yolov8m.pt` | 25.9M | **Рекомендуется** |
| `yolov8l.pt` | 43.7M | Много данных |
| `yolov8x.pt` | 68.2M | Максимальная точность |

### Мониторинг обучения

```bash
# Tensorboard
tensorboard --logdir runs/
```

Результаты сохраняются в `runs/jellyfish/<n>/`:
- `weights/best.pt` — лучшая модель
- `weights/last.pt` — последняя модель
- `results.csv` — метрики по эпохам
- `confusion_matrix.png` — матрица ошибок

---

## Детекция на видео

### Базовая детекция

```bash
python src/detect_video.py --video путь/к/видео.mp4 --model runs/jellyfish/best/weights/best.pt
```

### Детекция с трекингом объектов

```bash
python src/detect_video.py \
    --video путь/к/видео.mp4 \
    --model runs/jellyfish/best/weights/best.pt \
    --track \
    --show-trails \
    --output output/результат.mp4 \
    --csv output/детекции.csv
```

### Параметры детекции

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--video` | — | Путь к входному видео |
| `--model` | — | Путь к обученной модели (.pt) |
| `--output` | `output/detected.mp4` | Путь к выходному видео |
| `--csv` | `output/detections.csv` | Путь к CSV с детекциями |
| `--conf` | 0.25 | Порог уверенности (0-1) |
| `--depth-rate` | None | Скорость погружения (м/с) |
| `--ctd` | None | Путь к CSV с данными CTD |
| `--no-video` | False | Не сохранять видео |

### Параметры трекинга

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--track` | False | Включить трекинг объектов |
| `--tracker` | `bytetrack.yaml` | Тип трекера (bytetrack.yaml / botsort.yaml) |
| `--show-trails` | False | Показывать траектории движения |
| `--trail-length` | 30 | Длина траектории в кадрах |

### Форматы выходных данных

#### 1. Основной CSV с детекциями (`detections.csv`)

```csv
frame,timestamp_s,depth_m,track_id,class_id,class_name,confidence,x_center,y_center,width,height
0,0.00,0.0,1,0,Aurelia aurita,0.89,0.453,0.621,0.124,0.098
15,0.50,0.25,1,0,Aurelia aurita,0.91,0.458,0.625,0.122,0.096
30,1.00,0.50,2,2,Beroe ovata,0.76,0.234,0.445,0.067,0.089
```

**Колонки:**
- `frame` — номер кадра
- `timestamp_s` — время в секундах
- `depth_m` — глубина в метрах (если доступна)
- `track_id` — ID трека (при включенном трекинге)
- `class_id` — ID класса
- `class_name` — название вида
- `confidence` — уверенность детекции (0-1)
- `x_center, y_center, width, height` — нормализованные координаты bbox

#### 2. Статистика треков (`detections_tracks.csv`)

Создается автоматически при использовании `--track`:

```csv
track_id,class_id,class_name,first_frame,last_frame,frame_span,duration_s,detections_count,avg_confidence,depth_change_m
1,0,Aurelia aurita,0,45,46,1.50,42,0.893,0.75
2,2,Beroe ovata,30,120,91,3.00,85,0.812,1.50
```

**Колонки:**
- `track_id` — уникальный ID трека
- `class_id`, `class_name` — класс объекта
- `first_frame`, `last_frame` — первый и последний кадр трека
- `frame_span` — продолжительность в кадрах
- `duration_s` — длительность в секундах
- `first_timestamp_s`, `last_timestamp_s` — временные метки
- `first_depth_m`, `last_depth_m` — глубины начала и конца
- `detections_count` — количество детекций в треке
- `avg_confidence` — средняя уверенность
- `depth_change_m` — изменение глубины (вертикальное перемещение)

### Возможности трекинга

#### ByteTrack (по умолчанию)
- Быстрый и надежный
- Хорошо работает с перекрытиями объектов
- Рекомендуется для большинства случаев

```bash
python src/detect_video.py --video video.mp4 --model best.pt --track
```

#### BoT-SORT
- Более точный при большом количестве объектов
- Использует дополнительные признаки
- Немного медленнее

```bash
python src/detect_video.py --video video.mp4 --model best.pt --track --tracker botsort.yaml
```

#### Визуализация траекторий

С параметром `--show-trails` на видео отображаются:
- Цветные траектории движения каждого объекта
- Уникальный цвет для каждого track_id
- Точки на траектории с градиентом (новые точки ярче)
- Настраиваемая длина траектории (`--trail-length`)

```bash
python src/detect_video.py \
    --video video.mp4 \
    --model best.pt \
    --track \
    --show-trails \
    --trail-length 50
```

### Примеры использования трекинга

#### Анализ вертикальной миграции

```bash
python src/detect_video.py \
    --video dive_001.mp4 \
    --model best.pt \
    --track \
    --depth-rate 0.5 \
    --csv output/migration.csv
```

Затем анализируйте `migration_tracks.csv` для определения особей, перемещающихся вверх/вниз.

#### Подсчет количества особей

```bash
python src/detect_video.py \
    --video transect.mp4 \
    --model best.pt \
    --track \
    --no-video
```

Количество уникальных треков = количество отдельных особей в кадре.

#### Изучение поведения

```bash
python src/detect_video.py \
    --video behavior.mp4 \
    --model best.pt \
    --track \
    --show-trails \
    --trail-length 100
```

Длинные траектории позволяют визуализировать паттерны движения.

### Интеграция с CTD

Если есть данные CTD, создайте CSV файл:

```csv
time,depth,temperature,salinity
0.0,0.0,24.5,18.2
1.0,0.5,24.3,18.2
2.0,1.0,24.1,18.3
```

```bash
python src/detect_video.py --video видео.mp4 --model best.pt --ctd ctd_data.csv --track
```

---

## Анализ результатов

### Построение профилей распределения

```bash
python src/analyze.py --csv output/detections.csv --depth-bin 2.0
```

### Параметры анализа

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--csv` | — | Путь к CSV с детекциями |
| `--depth-bin` | 1.0 | Шаг биннинга по глубине (м) |
| `--output-dir` | `output/` | Директория для графиков |
| `--report` | `report.txt` | Имя текстового отчёта |

### Выходные файлы

- `vertical_distribution.png` — профиль вертикального распределения
- `detection_timeline.png` — временная шкала детекций
- `species_summary.png` — сводка по видам
- `report.txt` — текстовый отчёт

---

## Примеры использования

### Полный цикл обработки одного погружения

```bash
# 1. Извлечение кадров для разметки
python src/extract_frames.py --video dive_001.mp4 --output dataset/raw_frames --interval 30

# 2. После разметки — аугментация
python src/augment.py --images dataset/images/train --labels dataset/labels/train --multiply 3

# 3. Обучение модели
python src/train.py --data data.yaml --epochs 100 --model yolov8m.pt --name dive001_model

# 4. Детекция с трекингом на полном видео
python src/detect_video.py \
    --video dive_001.mp4 \
    --model runs/jellyfish/dive001_model/weights/best.pt \
    --track \
    --show-trails \
    --csv output/dive001_detections.csv \
    --depth-rate 0.5

# 5. Анализ результатов
python src/analyze.py --csv output/dive001_detections.csv --depth-bin 2.0
```

### Батчевая обработка нескольких видео

```bash
for video in videos/*.mp4; do
    name=$(basename "$video" .mp4)
    python src/detect_video.py \
        --video "$video" \
        --model runs/jellyfish/best/weights/best.pt \
        --track \
        --csv "output/${name}_detections.csv" \
        --no-video
done
```

### Быстрая проверка на коротком фрагменте

```bash
# Создаем 30-секундный фрагмент
ffmpeg -i full_video.mp4 -t 30 -c copy test_fragment.mp4

# Тестируем с визуализацией
python src/detect_video.py \
    --video test_fragment.mp4 \
    --model best.pt \
    --track \
    --show-trails \
    --conf 0.3
```

---

## Решение типичных проблем

### Много ложных срабатываний на взвеси

- Увеличьте порог `--conf` до 0.4-0.5
- Добавьте класс "частица" и разметьте крупные частицы
- Добавьте больше "пустых" кадров в обучающий датасет

### Путаница между Mnemiopsis и Beroe

- Увеличьте количество примеров обоих видов
- Добавьте примеры в разных состояниях (сжатые/расправленные)
- Используйте более крупную модель (yolov8l.pt)

### Не детектирует мелких Pleurobrachia

- Увеличьте разрешение: `--imgsz 1280`
- Добавьте больше примеров мелких особей
- Проверьте качество разметки

### Плохая работа на разных глубинах

- Усильте аугментацию яркости/контраста
- Убедитесь, что в обучающем датасете есть кадры с разных глубин

### Нехватка памяти GPU

- Уменьшите `--batch` (8 или 4)
- Используйте меньшую модель (yolov8s.pt)
- Уменьшите `--imgsz` (480 или 320)

### Треки обрываются слишком рано

- Используйте BoT-SORT вместо ByteTrack: `--tracker botsort.yaml`
- Уменьшите порог `--conf` для более стабильного трекинга
- Убедитесь, что объекты не перекрываются слишком сильно

### Один объект получает несколько track_id

- Увеличьте порог `--conf` чтобы избежать дублирующих детекций
- Проверьте качество модели на валидационном наборе
- Рассмотрите возможность дообучения модели

---

## Минимальные требования к датасету

| Параметр | Минимум | Рекомендуется |
|----------|---------|---------------|
| Изображений на класс | 100-200 | 500+ |
| С аугментацией | 400-800 | 2000+ |
| Соотношение train/val/test | 70/20/10 | 70/20/10 |

---

## Системные требования

### Минимальные
- Python 3.8+
- 8 GB RAM
- CPU (обучение будет медленным)

### Рекомендуемые
- Python 3.10+
- 16+ GB RAM
- NVIDIA GPU с 8+ GB VRAM (RTX 3060 и выше)
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
```

---

## Лицензия

MIT License

## Автор

Проект для исследования вертикального распределения желетелого макрозоопланктона в Чёрном море.
