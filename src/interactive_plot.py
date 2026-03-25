"""
Интерактивная визуализация распределения желетелых по глубине.

Строит векторный интерактивный график с возможностью фильтрации по видам
и отображением CTD параметров. Все данные на одном холсте с общей осью глубины.

Использует Plotly для интерактивности и экспорта в векторные форматы (SVG, PDF).
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from constants import SPECIES_COLORS, SPECIES_NAMES_RU

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
        if 'depth' in col.lower():
            return col
    return None


def create_interactive_depth_plot(
    track_sizes_path: str,
    output_path: str,
    ctd_path: Optional[str] = None,
    ctd_columns: Optional[List[int]] = None,
    depth_bin: float = 1.0,
    title: str = "Распределение желетелых по глубине",
    export_format: str = "html"
):
    """
    Создаёт интерактивный векторный график распределения желетелых по глубине.
    
    Все данные на ОДНОМ холсте с совмещёнными осями X:
    - Ось Y — глубина (общая для всех данных)
    - Основная ось X — количество особей
    - Точки организмов — размер пропорционален размеру организма (отдельная фильтрация)
    - Линии количества по глубине для каждого вида (отдельная фильтрация)
    - CTD параметры — дополнительные оси X сверху
    
    Args:
        track_sizes_path: путь к CSV с данными треков
        output_path: путь для сохранения графика
        ctd_path: путь к CSV с данными CTD (опционально)
        ctd_columns: номера колонок CTD для отображения (0-based)
        depth_bin: шаг биннинга для расчёта средних (м)
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
    
    # Фильтруем записи с глубиной
    df_depth = df[df['object_depth_m'].notna()].copy()
    
    if len(df_depth) == 0:
        print("Предупреждение: нет записей с глубиной объектов")
        return
    
    # Определяем диапазон глубин
    depth_max = df_depth['object_depth_m'].max()
    
    # Загружаем CTD данные
    ctd_df = None
    ctd_col_names = []
    ctd_depth_col = None
    
    if ctd_path and ctd_columns:
        try:
            ctd_df = load_ctd_data(ctd_path)
            ctd_depth_col = find_depth_column(ctd_df)
            
            if ctd_depth_col:
                all_cols = list(ctd_df.columns)
                for idx in ctd_columns:
                    if 0 <= idx < len(all_cols):
                        ctd_col_names.append(all_cols[idx])
                
                if ctd_col_names:
                    ctd_depth_max = ctd_df[ctd_depth_col].max()
                    if ctd_depth_max > depth_max:
                        depth_max = ctd_depth_max
                    print(f"CTD колонки: {ctd_col_names}")
        except Exception as e:
            print(f"Предупреждение: не удалось загрузить CTD: {e}")
            ctd_df = None
    
    # Создаём фигуру
    fig = go.Figure()
    
    # Виды в данных
    species_list = sorted(df_depth['class_name'].unique())
    
    # Нормализация размеров маркеров
    size_min = df_depth['real_size_cm'].min()
    size_max = df_depth['real_size_cm'].max()
    
    if size_max > size_min:
        df_depth['marker_size'] = 8 + 25 * (df_depth['real_size_cm'] - size_min) / (size_max - size_min)
    else:
        df_depth['marker_size'] = 15
    
    # === Вычисляем количество по глубине ===
    depth_bins = np.arange(0, depth_max + depth_bin, depth_bin)
    bin_centers = depth_bins[:-1] + depth_bin / 2
    
    # Находим максимальное количество для нормализации
    all_counts = []
    species_counts = {}
    for species in species_list:
        sp_df = df_depth[df_depth['class_name'] == species]
        counts, _ = np.histogram(sp_df['object_depth_m'], bins=depth_bins)
        species_counts[species] = counts
        all_counts.extend(counts)
    max_count = max(all_counts) if all_counts else 1
    
    # === 1. Линии количества по глубине (группа "Численность") ===
    first_count_trace = True
    for species in species_list:
        color = SPECIES_COLORS.get(species, 'gray')
        species_ru = SPECIES_NAMES_RU.get(species, species)
        counts = species_counts[species]
        
        hover_text = [
            f"<b>{species}</b><br>"
            f"Глубина: {d:.1f}–{d+depth_bin:.1f} м<br>"
            f"Количество: {c}"
            for d, c in zip(depth_bins[:-1], counts)
        ]
        
        fig.add_trace(go.Scatter(
            x=counts,
            y=bin_centers,
            mode='lines+markers',
            name=f"{species_ru}",
            legendgroup="counts",
            legendgrouptitle_text="Численность (N)" if first_count_trace else None,
            line=dict(color=color, width=2),
            marker=dict(size=6, color=color),
            hovertemplate="%{text}<extra></extra>",
            text=hover_text
        ))
        first_count_trace = False
    
    # === 2. Scatter plot организмов (группа "Размеры") ===
    np.random.seed(42)
    first_size_trace = True
    
    for species in species_list:
        sp_df = df_depth[df_depth['class_name'] == species]
        color = SPECIES_COLORS.get(species, 'gray')
        species_ru = SPECIES_NAMES_RU.get(species, species)
        
        # X — небольшой jitter около количества в бине
        x_vals = []
        for depth in sp_df['object_depth_m']:
            bin_idx = int(depth // depth_bin)
            if bin_idx >= len(species_counts[species]):
                bin_idx = len(species_counts[species]) - 1
            base_x = species_counts[species][bin_idx] if bin_idx >= 0 else 0
            jitter = np.random.uniform(-0.3, 0.3) * max_count * 0.1
            x_vals.append(base_x + jitter)
        
        hover_text = [
            f"<b>{species}</b><br>"
            f"Глубина: {depth:.1f} м<br>"
            f"Размер: {size:.1f} см<br>"
            f"Track ID: {tid}"
            for depth, size, tid in zip(sp_df['object_depth_m'], sp_df['real_size_cm'], sp_df['track_id'])
        ]
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=sp_df['object_depth_m'],
            mode='markers',
            name=f"{species_ru}",
            legendgroup="sizes",
            legendgrouptitle_text=f"Размеры ({size_min:.1f}–{size_max:.1f} см)" if first_size_trace else None,
            marker=dict(
                size=sp_df['marker_size'],
                color=color,
                opacity=0.6,
                line=dict(width=0.5, color='black')
            ),
            hovertemplate="%{text}<extra></extra>",
            text=hover_text
        ))
        first_size_trace = False
    
    # === 3. CTD параметры (группа "CTD") ===
    n_ctd = len(ctd_col_names) if ctd_df is not None else 0
    ctd_axes_info = []
    
    if ctd_df is not None and ctd_col_names:
        first_ctd_trace = True
        for i, col_name in enumerate(ctd_col_names):
            ctd_plot_df = ctd_df[[ctd_depth_col, col_name]].dropna()
            
            if len(ctd_plot_df) > 0:
                # Прореживаем если много точек
                if len(ctd_plot_df) > 2000:
                    step = len(ctd_plot_df) // 2000
                    ctd_plot_df = ctd_plot_df.iloc[::step]
                
                values = ctd_plot_df[col_name].values
                depths = ctd_plot_df[ctd_depth_col].values
                
                # Нормализуем CTD значения в диапазон основной оси X
                v_min, v_max = values.min(), values.max()
                x_normalized = (values - v_min) / (v_max - v_min) * max_count if v_max > v_min else np.full_like(values, max_count / 2)
                
                ctd_axes_info.append({
                    'name': col_name,
                    'min': v_min,
                    'max': v_max,
                    'color': CTD_COLORS[i % len(CTD_COLORS)]
                })
                
                hover_text = [
                    f"<b>{col_name}</b><br>"
                    f"Глубина: {d:.2f} м<br>"
                    f"Значение: {v:.3f}"
                    for d, v in zip(depths, values)
                ]
                
                fig.add_trace(go.Scatter(
                    x=x_normalized,
                    y=depths,
                    mode='lines',
                    name=f"{col_name}",
                    legendgroup="ctd",
                    legendgrouptitle_text="CTD параметры" if first_ctd_trace else None,
                    line=dict(color=CTD_COLORS[i % len(CTD_COLORS)], width=2, dash='dash'),
                    hovertemplate="%{text}<extra></extra>",
                    text=hover_text
                ))
                first_ctd_trace = False
    
    # === Аннотации для шкал ===
    annotations = []
    
    # Подсказка по фильтрации
    annotations.append(dict(
        x=0,
        y=-0.12,
        xref='paper',
        yref='paper',
        text="💡 Клик по легенде — скрыть/показать. Двойной клик — показать только выбранное.",
        showarrow=False,
        font=dict(size=10, color='gray'),
        align='left'
    ))
    
    # Шкалы CTD параметров вверху
    for i, info in enumerate(ctd_axes_info):
        annotations.append(dict(
            x=0,
            y=1.0,
            xref='paper',
            yref='paper',
            text=f"<span style='color:{info['color']}'><b>{info['name']}</b>: {info['min']:.2f} — {info['max']:.2f}</span>",
            showarrow=False,
            font=dict(size=10),
            align='left',
            yshift=15 + i * 18
        ))
    
    # === Настройка layout ===
    top_margin = 80 + n_ctd * 20
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=16),
            yref='paper',
            y=0.98
        ),
        xaxis=dict(
            title='Количество особей',
            title_font=dict(size=12),
            tickfont=dict(size=11),
            gridcolor='lightgray',
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor='gray',
            zerolinewidth=1,
            range=[0, max_count * 1.1]
        ),
        yaxis=dict(
            autorange='reversed',
            title='Глубина, м',
            title_font=dict(size=14),
            tickfont=dict(size=12),
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        height=900,
        width=800,
        margin=dict(t=top_margin, b=100, l=80, r=180),
        legend=dict(
            title=dict(text="<b>Фильтры</b>", font=dict(size=12)),
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10),
            groupclick="toggleitem",  # Клик переключает отдельный элемент
            tracegroupgap=10  # Отступ между группами
        ),
        hovermode='closest',
        annotations=annotations,
        plot_bgcolor='white'
    )
    
    # === Экспорт ===
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
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
            }
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
        fig.write_html(str(html_path), include_plotlyjs=True, full_html=True)
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
      --ctd ctd.csv --ctd-columns 6,7 -o output/plot

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
    parser.add_argument("--ctd-columns", type=str, default="", help="Колонки CTD (0-based): 5,6,7")
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
