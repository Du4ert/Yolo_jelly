"""
Единый источник правды для констант видов/классов.

Используется всеми скриптами src/ — detect_video.py, analyze.py,
interactive_plot.py, render_size_video.py.
"""

# Соответствие ID классов YOLO → названий видов (по data.yaml)
CLASS_NAMES = {
    0: 'Aurelia aurita',
    1: 'Beroe ovata',
    2: 'Mnemiopsis leidyi',
    3: 'Pleurobrachia pileus',
    4: 'Rhizostoma pulmo',
}

# Цвета для видов — hex, для matplotlib/plotly
SPECIES_COLORS = {
    'Aurelia aurita':       '#1f77b4',
    'Rhizostoma pulmo':     '#ff7f0e',
    'Beroe ovata':          '#2ca02c',
    'Mnemiopsis leidyi':    '#d62728',
    'Pleurobrachia pileus': '#9467bd',
}

# Цвета для видов — BGR, для OpenCV (render_size_video.py)
CLASS_COLORS_BGR = {
    'Aurelia aurita':       (255, 165, 0),   # Оранжевый
    'Beroe ovata':          (0, 255, 0),     # Зелёный
    'Mnemiopsis leidyi':    (255, 0, 255),   # Пурпурный
    'Pleurobrachia pileus': (0, 255, 255),   # Голубой
    'Rhizostoma pulmo':     (0, 0, 255),     # Красный
}

# Русские названия видов для легенды (interactive_plot.py)
SPECIES_NAMES_RU = {
    'Aurelia aurita':       'Аурелия',
    'Rhizostoma pulmo':     'Ризостома',
    'Beroe ovata':          'Берое',
    'Mnemiopsis leidyi':    'Мнемиопсис',
    'Pleurobrachia pileus': 'Плевробрахия',
}
