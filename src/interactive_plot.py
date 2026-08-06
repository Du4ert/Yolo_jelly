"""
Интерактивная визуализация распределения желетелых по глубине.

Строит векторный интерактивный график с панелью CTD и отдельными KDE-панелями
по каждому виду. Все панели используют общую ось глубины.

Использует Plotly для интерактивности и экспорта в векторные форматы (SVG, PDF).
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from constants import CLASS_NAMES, SPECIES_COLORS

# Цвета для CTD параметров
CTD_COLORS = ['#e377c2', '#17becf', '#bcbd22', '#7f7f7f', '#8c564b']


def load_ctd_data(ctd_path: str) -> pd.DataFrame:
    """Загружает данные CTD с автоматическим определением разделителя."""
    for sep in [',', ';', '|', '\t']:
        try:
            df = pd.read_csv(ctd_path, sep=sep, encoding='utf-8-sig')
            if len(df.columns) > 1:
                return df
        except:
            continue
    return pd.read_csv(ctd_path, sep=None, engine='python', encoding='utf-8-sig')


def find_depth_column(df: pd.DataFrame) -> Optional[str]:
    """Находит колонку с глубиной в DataFrame."""
    for col in df.columns:
        col_lower = str(col).lower()
        if 'depth' in col_lower or 'глуб' in col_lower:
            return col
    return None


def to_numeric_series(values: pd.Series) -> pd.Series:
    """Преобразует числа с точкой или запятой в float."""
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors='coerce')
    cleaned = values.astype(str).str.strip().str.replace(',', '.', regex=False)
    return pd.to_numeric(cleaned, errors='coerce')


def calculate_kde(depths: np.ndarray, depth_grid: np.ndarray) -> np.ndarray:
    """Считает простую Gaussian KDE без дополнительных зависимостей."""
    depths = np.asarray(depths, dtype=float)
    depths = depths[np.isfinite(depths)]

    if len(depths) < 2:
        return np.zeros_like(depth_grid, dtype=float)

    std = np.std(depths, ddof=1)
    bandwidth = 1.06 * std * (len(depths) ** (-1 / 5)) if std > 0 else 0.5
    bandwidth = max(float(bandwidth), 0.25)

    diff = (depth_grid[:, None] - depths[None, :]) / bandwidth
    kde = np.exp(-0.5 * diff ** 2).sum(axis=1)
    kde /= len(depths) * bandwidth * np.sqrt(2 * np.pi)
    return kde


def interpolate_density(depths: np.ndarray, depth_grid: np.ndarray, density: np.ndarray) -> np.ndarray:
    """Возвращает значение KDE в глубинах отдельных экземпляров."""
    if len(depth_grid) == 0 or len(density) == 0 or np.nanmax(density) <= 0:
        return np.zeros_like(depths, dtype=float)
    return np.interp(depths, depth_grid, density, left=0, right=0)


def normalize_marker_sizes(values: pd.Series) -> pd.Series:
    """Нормализует реальные размеры организмов в размеры маркеров Plotly."""
    values = pd.to_numeric(values, errors='coerce')
    valid = values.dropna()
    if len(valid) == 0:
        return pd.Series(10, index=values.index)

    min_size = valid.min()
    max_size = valid.max()
    if max_size > min_size:
        return 6 + 20 * (values.fillna(min_size) - min_size) / (max_size - min_size)
    return pd.Series(12, index=values.index)


def hex_to_rgba(color: str, alpha: float) -> str:
    """Преобразует #RRGGBB в rgba() для заливки Plotly."""
    if not isinstance(color, str) or not color.startswith('#') or len(color) != 7:
        return f"rgba(128,128,128,{alpha})"
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def axis_ticks(min_value: float, max_value: float, count: int = 5) -> List[float]:
    """Возвращает крайние и промежуточные подписи оси."""
    if not np.isfinite(min_value) or not np.isfinite(max_value):
        return []
    if max_value <= min_value:
        return [float(min_value)]
    return [float(x) for x in np.linspace(min_value, max_value, count)]


def calculate_thermocline_depth(
    ctd_df: pd.DataFrame,
    depth_col: str,
    temp_col: str,
    threshold_c_per_m: float,
    mode: str = "threshold",
    min_depth_m: float = 2.0,
) -> Optional[Tuple[float, float]]:
    """Возвращает глубину и величину максимального градиента выше порога."""
    if mode == "off" or (mode == "threshold" and threshold_c_per_m <= 0):
        return None

    profile = ctd_df[[depth_col, temp_col]].copy()
    profile[depth_col] = to_numeric_series(profile[depth_col])
    profile[temp_col] = to_numeric_series(profile[temp_col])
    profile = profile.dropna().sort_values(depth_col)
    if min_depth_m > 0:
        profile = profile[profile[depth_col] >= min_depth_m]

    depths = profile[depth_col].to_numpy(dtype=float)
    temps = profile[temp_col].to_numpy(dtype=float)
    if len(depths) < 2:
        return None

    delta_depth = np.diff(depths)
    delta_temp = np.diff(temps)
    valid = delta_depth > 0
    if not np.any(valid):
        return None

    gradients = delta_temp[valid] / delta_depth[valid]
    gradient_abs = np.abs(gradients)
    segment_depths = (depths[:-1][valid] + depths[1:][valid]) / 2
    if mode == "maximum":
        candidates = np.arange(len(gradient_abs))
    else:
        candidates = np.where(gradient_abs > threshold_c_per_m)[0]
    if len(candidates) == 0:
        return None

    strongest = candidates[np.argmax(gradient_abs[candidates])]
    return float(segment_depths[strongest]), float(gradient_abs[strongest])


def create_interactive_depth_plot(
    track_sizes_path: str,
    output_path: str,
    ctd_path: Optional[str] = None,
    ctd_columns: Optional[List[int]] = None,
    depth_bin: float = 1.0,
    cross_section_area_m2: Optional[float] = None,
    pleurobrachia_cross_section_area_m2: Optional[float] = None,
    thermocline_threshold: float = 0.2,
    thermocline_mode: str = "threshold",
    title: str = "Распределение желетелых по глубине",
    export_format: str = "html"
):
    """
    Создаёт интерактивный векторный график распределения желетелых по глубине.
    
    Раскладка слева направо:
    - CTD параметры в реальных значениях X (если передан CTD файл)
    - KDE-панель для каждого вида из CLASS_NAMES
    - Точки организмов поверх KDE, размер маркера пропорционален real_size_cm
    
    Args:
        track_sizes_path: путь к CSV с данными треков
        output_path: путь для сохранения графика
        ctd_path: путь к CSV с данными CTD (опционально)
        ctd_columns: номера колонок CTD для отображения (0-based)
        depth_bin: шаг биннинга для расчёта средних (м)
        cross_section_area_m2: площадь сечения наблюдения для нормировки KDE (м²)
        pleurobrachia_cross_section_area_m2: отдельная площадь сечения P. pileus (м²)
        thermocline_threshold: порог величины температурного градиента для термоклина (°C/м)
        thermocline_mode: режим термоклина (threshold, maximum, off)
        title: заголовок графика
        export_format: формат экспорта (html, svg, pdf, png)
    """
    if not PLOTLY_AVAILABLE:
        print(f"Ошибка: plotly не установлен для {sys.executable}")
        print(f"Установите: \"{sys.executable}\" -m pip install plotly kaleido")
        return
    
    # Загрузка данных треков
    print(f"Загрузка данных треков: {track_sizes_path}")
    df = pd.read_csv(track_sizes_path)
    
    if len(df) == 0:
        print("Предупреждение: нет данных треков")
        return
    
    # Проверяем наличие необходимых колонок
    required_cols = ['object_depth_m', 'class_name', 'real_size_cm']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Ошибка: отсутствуют необходимые колонки: {missing}")
        return
    
    df['object_depth_m'] = to_numeric_series(df['object_depth_m'])
    df['real_size_cm'] = to_numeric_series(df['real_size_cm'])

    # Фильтруем записи с глубиной
    df_depth = df[df['object_depth_m'].notna()].copy()
    
    if len(df_depth) == 0:
        print("Предупреждение: нет записей с глубиной объектов")
        return
    
    # Определяем диапазон глубин
    depth_min = float(df_depth['object_depth_m'].min())
    depth_max = float(df_depth['object_depth_m'].max())
    
    # Загружаем CTD данные
    ctd_df = None
    ctd_col_names = []
    ctd_depth_col = None
    ctd_depth_values = pd.Series(dtype=float)
    ctd_temperature_col = None
    thermocline_depth = None
    thermocline_gradient = None
    
    if ctd_path and ctd_columns:
        try:
            ctd_df = load_ctd_data(ctd_path)
            ctd_depth_col = find_depth_column(ctd_df)
            
            if ctd_depth_col:
                all_cols = list(ctd_df.columns)
                for idx in ctd_columns:
                    if 0 <= idx < len(all_cols):
                        ctd_col_names.append(all_cols[idx])
                        if idx == 6:
                            ctd_temperature_col = all_cols[idx]
                
                if ctd_col_names:
                    ctd_df[ctd_depth_col] = to_numeric_series(ctd_df[ctd_depth_col])
                    ctd_depth_values = ctd_df[ctd_depth_col].dropna()
                    if len(ctd_depth_values) > 0:
                        depth_max = float(ctd_depth_values.max())
                        depth_min = float(ctd_depth_values.min())
                        print(f"CTD колонки: {ctd_col_names}")
                        if ctd_temperature_col:
                            thermocline = calculate_thermocline_depth(
                                ctd_df,
                                ctd_depth_col,
                                ctd_temperature_col,
                                thermocline_threshold,
                                thermocline_mode,
                            )
                            if thermocline is not None:
                                thermocline_depth, thermocline_gradient = thermocline
                            if thermocline_depth is not None:
                                print(
                                    f"Термоклин: {thermocline_depth:.2f} м, "
                                    f"градиент {thermocline_gradient:.3f} °C/м"
                                )
                    else:
                        ctd_col_names = []
        except Exception as e:
            print(f"Предупреждение: не удалось загрузить CTD: {e}")
            ctd_df = None
    species_list = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]
    has_ctd_panel = ctd_df is not None and bool(ctd_col_names) and ctd_depth_col is not None
    panel_titles = (["CTD"] if has_ctd_panel else []) + species_list
    n_panels = len(panel_titles)
    col_widths = ([1.8] if has_ctd_panel else []) + [1.0] * len(species_list)

    fig = make_subplots(
        rows=1,
        cols=n_panels,
        shared_yaxes=True,
        horizontal_spacing=0.025,
        column_widths=col_widths,
    )

    depth_max = max(depth_max, depth_min + depth_bin)
    grid_step = max((depth_max - depth_min) / 400, 0.05)
    depth_grid = np.arange(depth_min, depth_max + grid_step, grid_step)
    df_depth['marker_size'] = normalize_marker_sizes(df_depth['real_size_cm'])
    area_m2 = float(cross_section_area_m2) if cross_section_area_m2 and cross_section_area_m2 > 0 else None
    if (pleurobrachia_cross_section_area_m2
            and pleurobrachia_cross_section_area_m2 > 0
            and area_m2 is None):
        raise ValueError(
            "Отдельная площадь P. pileus требует общей площади сечения"
        )
    pileus_area_m2 = (
        float(pleurobrachia_cross_section_area_m2)
        if (pleurobrachia_cross_section_area_m2
            and pleurobrachia_cross_section_area_m2 > 0)
        else area_m2
    )
    kde_axis_title = "экз./м²/м глубины" if area_m2 else "экз./м глубины"

    species_density = {}
    for species in species_list:
        species_area_m2 = (
            pileus_area_m2 if species == 'Pleurobrachia pileus' else area_m2
        )
        sp_depths = df_depth.loc[df_depth['class_name'] == species, 'object_depth_m'].values
        if len(sp_depths) >= 2:
            density = calculate_kde(sp_depths, depth_grid) * len(sp_depths)
            if species_area_m2:
                density = density / species_area_m2
        else:
            density = np.zeros_like(depth_grid)
        species_density[species] = density

    annotations = [dict(
        x=0,
        y=-0.12,
        xref='paper',
        yref='paper',
        text="Клик по легенде — скрыть/показать. Размер точек соответствует размеру экземпляров.",
        showarrow=False,
        font=dict(size=10, color='gray'),
        align='left'
    )]

    def add_panel_label(col: int, text: str, color: str = 'black') -> None:
        axis_suffix = '' if col == 1 else str(col)
        annotations.append(dict(
            x=0.5,
            y=0.925,
            xref=f"x{axis_suffix} domain",
            yref='paper',
            text=f"<b>{text}</b>",
            showarrow=False,
            font=dict(size=12, color=color),
            bgcolor='rgba(255,255,255,0.72)',
            borderpad=2,
        ))

    for col, panel_title in enumerate(panel_titles, start=1):
        label_color = 'black'
        if not (has_ctd_panel and col == 1):
            species_idx = col - (2 if has_ctd_panel else 1)
            if 0 <= species_idx < len(species_list):
                label_color = SPECIES_COLORS.get(species_list[species_idx], 'black')
        add_panel_label(col, panel_title, label_color)

    current_col = 1
    plot_domain_bottom = 0.0
    ctd_trace_count = 0
    thermocline_trace_indices = []
    temperature_trace_index = None
    thermocline_color = 'rgba(0,0,0,0.65)'
    if has_ctd_panel:
        first_ctd_trace = True
        ctd_axis_configs = []
        for i, col_name in enumerate(ctd_col_names):
            ctd_plot_df = ctd_df[[ctd_depth_col, col_name]].copy()
            ctd_plot_df[ctd_depth_col] = to_numeric_series(ctd_plot_df[ctd_depth_col])
            ctd_plot_df[col_name] = to_numeric_series(ctd_plot_df[col_name])
            ctd_plot_df = ctd_plot_df.dropna()

            if len(ctd_plot_df) > 2000:
                step = max(len(ctd_plot_df) // 2000, 1)
                ctd_plot_df = ctd_plot_df.iloc[::step]

            if len(ctd_plot_df) == 0:
                continue

            values = ctd_plot_df[col_name].values
            depths = ctd_plot_df[ctd_depth_col].values
            value_min = float(np.nanmin(values))
            value_max = float(np.nanmax(values))
            if value_max > value_min:
                pad = (value_max - value_min) * 0.03
                axis_range = [value_min - pad, value_max + pad]
                tickvals = axis_ticks(value_min, value_max, 5)
            else:
                pad = abs(value_max) * 0.05 if value_max else 1.0
                axis_range = [value_min - pad, value_max + pad]
                tickvals = [value_min]
            axis_num = None if i == 0 else n_panels + i
            trace_axis = None if axis_num is None else f"x{axis_num}"
            hover_text = [
                f"<b>{col_name}</b><br>Глубина: {d:.2f} м<br>Значение: {v:.3f}"
                for d, v in zip(depths, values)
            ]

            trace = go.Scatter(
                x=values,
                y=depths,
                mode='lines',
                name=col_name,
                legendgroup="ctd",
                legendgrouptitle_text="CTD" if first_ctd_trace else None,
                line=dict(color=CTD_COLORS[i % len(CTD_COLORS)], width=2),
                hovertemplate="%{text}<extra></extra>",
                text=hover_text,
                showlegend=True,
                meta=dict(role='temperature_ctd') if col_name == ctd_temperature_col else None,
            )
            if trace_axis:
                trace.update(xaxis=trace_axis, yaxis='y')
                fig.add_trace(trace)
            else:
                fig.add_trace(trace, row=1, col=current_col)
            if col_name == ctd_temperature_col:
                temperature_trace_index = len(fig.data) - 1
                thermocline_color = CTD_COLORS[i % len(CTD_COLORS)]

            ctd_axis_configs.append({
                'axis_num': axis_num,
                'col_name': col_name,
                'color': CTD_COLORS[i % len(CTD_COLORS)],
                'range': axis_range,
                'tickvals': tickvals,
                'ticktext': [f"{v:.2g}" for v in tickvals],
            })
            first_ctd_trace = False
            ctd_trace_count += 1

        if thermocline_depth is not None and ctd_axis_configs:
            fig.add_trace(go.Scatter(
                x=ctd_axis_configs[0]['range'],
                y=[thermocline_depth, thermocline_depth],
                mode='lines',
                name=f"Термоклин {thermocline_depth:.2f} м",
                legendgroup="thermocline",
                line=dict(color=thermocline_color, width=1.5, dash='dash'),
                hovertemplate=(
                    f"Термоклин<br>Глубина: {thermocline_depth:.2f} м<br>"
                    f"Градиент: {thermocline_gradient:.3f} °C/м<br>"
                    f"Режим: {'максимальный' if thermocline_mode == 'maximum' else 'порог'}<br>"
                    f"Порог: {thermocline_threshold:.2f} °C/м<extra></extra>"
                ),
                showlegend=False,
                meta=dict(role='thermocline'),
            ), row=1, col=current_col)
            thermocline_trace_indices.append(len(fig.data) - 1)

        if ctd_trace_count == 0:
            annotations.append(dict(
                x=0.5,
                y=0.5,
                xref='x domain',
                yref='y domain',
                text="нет CTD данных",
                showarrow=False,
                font=dict(size=12, color='gray')
            ))

        axis_gap = 0.065
        plot_domain_bottom = min(0.28, axis_gap * ctd_trace_count)
        ctd_domain = fig.layout.xaxis.domain
        for axis_idx, axis_config in enumerate(ctd_axis_configs):
            axis_position = max(plot_domain_bottom - axis_gap * (axis_idx + 1), 0.0)
            axis_layout = dict(
                title=dict(text=""),
                showgrid=axis_idx == 0,
                gridcolor='lightgray',
                showline=True,
                linecolor=axis_config['color'],
                linewidth=1,
                ticks='outside',
                tickfont=dict(color=axis_config['color']),
                range=axis_config['range'],
                tickmode='array',
                tickvals=axis_config['tickvals'],
                ticktext=axis_config['ticktext'],
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikedash='dot',
                spikecolor='rgba(0,0,0,0.45)',
                spikethickness=1,
                anchor='free',
                side='bottom',
                position=axis_position,
            )
            annotations.append(dict(
                x=0,
                y=axis_position,
                xref='paper',
                yref='paper',
                text=axis_config['col_name'],
                showarrow=False,
                xanchor='right',
                yanchor='middle',
                font=dict(size=9, color=axis_config['color']),
                xshift=-6,
            ))
            if axis_config['axis_num'] is None:
                fig.update_xaxes(axis_layout, row=1, col=current_col)
            else:
                axis_layout.update(domain=ctd_domain, overlaying='x')
                fig.update_layout({f"xaxis{axis_config['axis_num']}": axis_layout})
        current_col += 1

    np.random.seed(42)
    for species in species_list:
        color = SPECIES_COLORS.get(species, 'gray')
        sp_df = df_depth[df_depth['class_name'] == species].copy()
        depths = pd.to_numeric(sp_df['object_depth_m'], errors='coerce').dropna().values
        density = species_density[species]
        density_max = float(np.nanmax(density)) if len(density) > 0 and np.nanmax(density) > 0 else 1.0

        if len(depths) >= 2:
            x_fill = np.concatenate([density / 2, -density[::-1] / 2])
            y_fill = np.concatenate([depth_grid, depth_grid[::-1]])
            fig.add_trace(go.Scatter(
                x=x_fill,
                y=y_fill,
                mode='lines',
                name=f"KDE {species}",
                legendgroup=species,
                line=dict(color=color, width=1.5),
                fill='toself',
                fillcolor=hex_to_rgba(color, 0.38),
                hoverinfo='skip',
                showlegend=False,
            ), row=1, col=current_col)
            point_density = interpolate_density(sp_df['object_depth_m'].values, depth_grid, density)
        elif len(depths) == 1:
            fig.add_trace(go.Scatter(
                x=[-density_max * 0.18, density_max * 0.18],
                y=[depths[0], depths[0]],
                mode='lines',
                name=f"Отметка {species}",
                legendgroup=species,
                line=dict(color=color, width=2, dash='dot'),
                hovertemplate=f"{species}<br>Глубина: {depths[0]:.2f} м<extra></extra>",
                showlegend=False,
            ), row=1, col=current_col)
            point_density = np.full(len(sp_df), density_max * 0.28)
        else:
            fig.add_trace(go.Scatter(
                x=[0],
                y=[(depth_min + depth_max) / 2],
                mode='markers',
                marker=dict(size=0, opacity=0),
                hoverinfo='skip',
                showlegend=False,
            ), row=1, col=current_col)
            annotations.append(dict(
                x=0.5,
                y=0.5,
                xref=f"x{current_col if current_col > 1 else ''} domain",
                yref='paper',
                text="нет данных",
                showarrow=False,
                font=dict(size=12, color='gray')
            ))
            point_density = np.array([])

        if len(sp_df) > 0:
            point_density = np.asarray(point_density, dtype=float)
            min_width = density_max * 0.04
            point_width = np.maximum(point_density, min_width)
            x_vals = np.random.uniform(-point_width / 2, point_width / 2, len(sp_df))
            size_values = pd.to_numeric(sp_df['real_size_cm'], errors='coerce').dropna()
            if len(size_values) > 0:
                size_label = f"{species}: {size_values.min():.1f}-{size_values.max():.1f} см"
            else:
                size_label = f"{species}: размер н/д"
            hover_text = [
                f"<b>{species}</b><br>Глубина: {depth:.2f} м<br>Размер: {size:.1f} см<br>Track ID: {tid}"
                for depth, size, tid in zip(
                    sp_df['object_depth_m'],
                    sp_df['real_size_cm'],
                    sp_df['track_id'] if 'track_id' in sp_df.columns else [''] * len(sp_df)
                )
            ]
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=sp_df['object_depth_m'],
                mode='markers',
                name=size_label,
                legendgroup=species,
                marker=dict(
                    size=sp_df['marker_size'],
                    color=color,
                    opacity=0.62,
                    line=dict(width=0.6, color='black')
                ),
                hovertemplate="%{text}<extra></extra>",
                text=hover_text,
                showlegend=True,
            ), row=1, col=current_col)

        if thermocline_depth is not None:
            fig.add_trace(go.Scatter(
                x=[-density_max * 0.62, density_max * 0.62],
                y=[thermocline_depth, thermocline_depth],
                mode='lines',
                name=f"Термоклин {thermocline_depth:.2f} м",
                legendgroup="thermocline",
                line=dict(color=thermocline_color, width=1.5, dash='dash'),
                hovertemplate=(
                    f"Термоклин<br>Глубина: {thermocline_depth:.2f} м<br>"
                    f"Градиент: {thermocline_gradient:.3f} °C/м<br>"
                    f"Режим: {'максимальный' if thermocline_mode == 'maximum' else 'порог'}<br>"
                    f"Порог: {thermocline_threshold:.2f} °C/м<extra></extra>"
                ),
                showlegend=False,
                meta=dict(role='thermocline'),
            ), row=1, col=current_col)
            thermocline_trace_indices.append(len(fig.data) - 1)

        fig.update_xaxes(
            title_text=kde_axis_title,
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='gray',
            showline=True,
            linecolor='black',
            linewidth=1,
            ticks='outside',
            showticklabels=True,
            tickmode='array',
            tickvals=[
                -density_max * 0.5,
                -density_max * 0.25,
                0,
                density_max * 0.25,
                density_max * 0.5,
            ],
            ticktext=[
                f"{density_max:.2f}",
                f"{density_max * 0.5:.2f}",
                "0",
                f"{density_max * 0.5:.2f}",
                f"{density_max:.2f}",
            ],
            showspikes=False,
            range=[-density_max * 0.62, density_max * 0.62],
            row=1,
            col=current_col,
        )
        current_col += 1

    fig.update_yaxes(
        range=[depth_max, depth_min],
        domain=[plot_domain_bottom, 0.87],
        title_text='Глубина, м',
        showgrid=True,
        gridcolor='lightgray',
        showline=True,
        linecolor='black',
        ticks='outside',
        tickmode='array',
        tickvals=axis_ticks(depth_min, depth_max, 6),
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikedash='dot',
        spikecolor='rgba(0,0,0,0.45)',
        spikethickness=1,
        row=1,
        col=1,
    )
    for col in range(2, n_panels + 1):
        fig.update_yaxes(
            range=[depth_max, depth_min],
            domain=[plot_domain_bottom, 0.87],
            showgrid=True,
            gridcolor='lightgray',
            showticklabels=False,
            title_text='',
            showline=False,
            ticks='',
            showspikes=False,
            row=1,
            col=col,
        )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=16),
            yref='paper',
            y=0.995
        ),
        height=900,
        width=max(1250, 260 * n_panels),
        margin=dict(t=160, b=130 + 18 * max(ctd_trace_count - 1, 0), l=90, r=240),
        showlegend=True,
        legend=dict(
            title=dict(text="<b>CTD и размеры</b>", font=dict(size=12)),
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10),
                groupclick="toggleitem",
                tracegroupgap=8,
        ),
        hovermode='closest',
        spikedistance=-1,
        hoverdistance=80,
        annotations=annotations,
        plot_bgcolor='white'
    )
    
    # === Экспорт ===
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hoverline_post_script = """
    var plot = document.getElementById('{plot_id}');
    var temperatureTraceIndex = __TEMPERATURE_TRACE_INDEX__;
    var thermoclineTraceIndices = __THERMOCLINE_TRACE_INDICES__;
    var horizontalHoverLine = {
        type: 'line',
        xref: 'paper',
        x0: 0,
        x1: 1,
        yref: 'y',
        y0: 0,
        y1: 0,
        line: {
            color: 'rgba(0,0,0,0.45)',
            width: 1,
            dash: 'dot'
        },
        layer: 'above'
    };
    plot.on('plotly_hover', function(eventData) {
        if (!eventData.points || !eventData.points.length) {
            return;
        }
        if (!eventData.points[0].data || eventData.points[0].data.legendgroup !== 'ctd') {
            Plotly.relayout(plot, {shapes: []});
            return;
        }
        var yValue = eventData.points[0].y;
        var sourceEvent = eventData.event || (window.event || null);
        var yAxis = plot._fullLayout && plot._fullLayout.yaxis;
        if (sourceEvent && yAxis && typeof yAxis.p2d === 'function') {
            var bounds = plot.getBoundingClientRect();
            var pixelY = sourceEvent.clientY - bounds.top - yAxis._offset;
            if (Number.isFinite(pixelY)) {
                yValue = yAxis.p2d(pixelY);
            }
        }
        if (yValue === undefined || !Number.isFinite(yValue)) {
            return;
        }
        horizontalHoverLine.y0 = yValue;
        horizontalHoverLine.y1 = yValue;
        Plotly.relayout(plot, {shapes: [horizontalHoverLine]});
    });
    plot.on('plotly_unhover', function() {
        Plotly.relayout(plot, {shapes: []});
    });
    function syncThermoclineVisibility() {
        if (temperatureTraceIndex === null || thermoclineTraceIndices.length === 0) {
            return;
        }
        var tempTrace = plot.data[temperatureTraceIndex];
        var visible = !(tempTrace && (tempTrace.visible === false || tempTrace.visible === 'legendonly'));
        thermoclineTraceIndices.forEach(function(traceIndex) {
            var trace = plot.data[traceIndex];
            if (!trace) {
                return;
            }
            var targetVisible = visible ? true : 'legendonly';
            if (trace.visible !== targetVisible) {
                Plotly.restyle(plot, {visible: targetVisible}, [traceIndex]);
            }
        });
    }
    plot.on('plotly_restyle', function() {
        window.setTimeout(syncThermoclineVisibility, 0);
    });
    syncThermoclineVisibility();
    """
    hoverline_post_script = hoverline_post_script.replace(
        "__TEMPERATURE_TRACE_INDEX__",
        "null" if temperature_trace_index is None else str(temperature_trace_index),
    ).replace("__THERMOCLINE_TRACE_INDICES__", str(thermocline_trace_indices))
    
    if export_format == "html":
        fig.write_html(
            str(output_path.with_suffix('.html')),
            include_plotlyjs=True,
            full_html=True,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'toImageButtonOptions': {
                    'format': 'svg',
                    'filename': output_path.stem,
                    'scale': 2
                }
            },
            post_script=hoverline_post_script,
        )
        print(f"Интерактивный график: {output_path.with_suffix('.html')}")
        
    elif export_format == "svg":
        try:
            fig.write_image(str(output_path.with_suffix('.svg')), format='svg')
            print(f"SVG график: {output_path.with_suffix('.svg')}")
        except Exception as e:
            print(f"Ошибка SVG (pip install kaleido): {e}")
            
    elif export_format == "pdf":
        try:
            fig.write_image(str(output_path.with_suffix('.pdf')), format='pdf')
            print(f"PDF график: {output_path.with_suffix('.pdf')}")
        except Exception as e:
            print(f"Ошибка PDF (pip install kaleido): {e}")
            
    elif export_format == "png":
        try:
            fig.write_image(str(output_path.with_suffix('.png')), format='png', scale=2)
            print(f"PNG график: {output_path.with_suffix('.png')}")
        except Exception as e:
            print(f"Ошибка PNG (pip install kaleido): {e}")
    
    # Всегда сохраняем HTML
    if export_format != "html":
        html_path = output_path.with_suffix('.html')
        fig.write_html(
            str(html_path),
            include_plotlyjs=True,
            full_html=True,
            post_script=hoverline_post_script,
        )
        print(f"+ интерактивный HTML: {html_path}")


def parse_ctd_columns(value: str) -> List[int]:
    """Парсит строку с номерами колонок CTD."""
    if not value:
        return []
    return [int(p.strip()) for p in value.split(',') if p.strip().isdigit()]


def main():
    parser = argparse.ArgumentParser(
        description="Интерактивная визуализация распределения желетелых по глубине",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:

  # Интерактивный HTML
  python interactive_plot.py -t detections_track_sizes.csv -o output/plot

  # С CTD данными
  python interactive_plot.py -t detections_track_sizes.csv \\
      --ctd ctd.csv --ctd-columns 6,11,12,16 -o output/plot

  # Экспорт в SVG/PDF
  python interactive_plot.py -t detections_track_sizes.csv -f svg -o output/plot
  python interactive_plot.py -t detections_track_sizes.csv -f pdf -o output/plot

  # Посмотреть колонки CTD
  python interactive_plot.py --ctd ctd.csv --list-ctd-columns

Интерактивность (HTML):
  - Клик по легенде: скрыть/показать отдельный элемент
  - Двойной клик по заголовку группы: показать/скрыть всю группу
  - Колёсико мыши: zoom
  - Hover: подробная информация
  
Группы фильтрации:
  - Численность (N): линии количества по глубине
  - Размеры: точки организмов (размер = размер организма)
  - CTD параметры: профили температуры, солёности и др.
        """
    )
    
    parser.add_argument("--track-sizes", "-t", help="CSV с размерами треков")
    parser.add_argument("--output", "-o", default="output/depth_plot", help="Путь (без расширения)")
    parser.add_argument("--depth-bin", type=float, default=1.0, help="Шаг биннинга (м)")
    parser.add_argument("--title", default="Распределение желетелых по глубине", help="Заголовок")
    parser.add_argument("--format", "-f", choices=["html", "svg", "pdf", "png"], default="html")
    parser.add_argument("--ctd", help="CSV с данными CTD")
    parser.add_argument("--ctd-columns", type=str, default="6,11,12,16", help="Колонки CTD (0-based): 6,11,12,16")
    parser.add_argument("--cross-section-area", type=float, default=None, help="Площадь сечения наблюдения, м²")
    parser.add_argument("--pleurobrachia-cross-section-area", type=float, default=None,
                        help="Отдельная площадь сечения P. pileus, м²")
    parser.add_argument("--thermocline-threshold", type=float, default=0.2, help="Порог термоклина, °C/м")
    parser.add_argument("--thermocline-mode", choices=["threshold", "maximum", "off"], default="threshold", help="Режим термоклина")
    parser.add_argument("--list-ctd-columns", action="store_true", help="Показать колонки CTD")
    
    args = parser.parse_args()
    
    if not PLOTLY_AVAILABLE:
        print("Ошибка: pip install plotly kaleido")
        return 1
    
    if args.list_ctd_columns:
        if not args.ctd:
            print("Укажите --ctd")
            return 1
        ctd_df = load_ctd_data(args.ctd)
        print(f"\nКолонки в {args.ctd}:\n")
        for i, col in enumerate(ctd_df.columns):
            sample = ctd_df[col].dropna().head(1)
            val = str(sample.values[0])[:30] if len(sample) > 0 else "N/A"
            print(f"  {i:2d}: {col:<30} ({val})")
        return 0
    
    if not args.track_sizes:
        parser.print_help()
        return 1
    
    try:
        create_interactive_depth_plot(
            track_sizes_path=args.track_sizes,
            output_path=args.output,
            ctd_path=args.ctd,
            ctd_columns=parse_ctd_columns(args.ctd_columns) or None,
            depth_bin=args.depth_bin,
            cross_section_area_m2=args.cross_section_area,
            pleurobrachia_cross_section_area_m2=(
                args.pleurobrachia_cross_section_area
            ),
            thermocline_threshold=args.thermocline_threshold,
            thermocline_mode=args.thermocline_mode,
            title=args.title,
            export_format=args.format
        )
        return 0
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
