"""
Интерактивная визуализация распределения желетелых по глубине.

Строит векторный интерактивный график с возможностью фильтрации по видам
и отображением CTD параметров.

Использует Plotly для интерактивности и экспорта в векторные форматы (SVG, PDF).
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# Цвета для видов
SPECIES_COLORS = {
    'Aurelia aurita': '#1f77b4',
    'Rhizostoma pulmo': '#ff7f0e',
    'Beroe ovata': '#2ca02c',
    'Mnemiopsis leidyi': '#d62728',
    'Pleurobrachia pileus': '#9467bd'
}

# Русские названия видов для легенды
SPECIES_NAMES_RU = {
    'Aurelia aurita': 'Аурелия',
    'Rhizostoma pulmo': 'Ризостома',
    'Beroe ovata': 'Берое',
    'Mnemiopsis leidyi': 'Мнемиопсис',
    'Pleurobrachia pileus': 'Плевробрахия'
}


def load_ctd_data(ctd_path: str) -> pd.DataFrame:
    """
    Загружает данные CTD с автоматическим определением разделителя.
    
    Args:
        ctd_path: путь к CTD файлу
        
    Returns:
        DataFrame с данными CTD
    """
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
    
    Все данные отображаются на одном холсте с возможностью:
    - Фильтрации по видам через легенду (клик/двойной клик)
    - Zoom и pan
    - Hover с подробной информацией
    - Экспорта в векторные форматы (SVG, PDF)
    
    Args:
        track_sizes_path: путь к CSV с данными треков (detections_track_sizes.csv)
        output_path: путь для сохранения графика
        ctd_path: путь к CSV с данными CTD (опционально)
        ctd_columns: номера колонок CTD для отображения (0-based)
        depth_bin: шаг биннинга для расчёта средних (м)
        title: заголовок графика
        export_format: формат экспорта (html, svg, pdf, png)
    """
    if not PLOTLY_AVAILABLE:
        print("Ошибка: plotly не установлен. Установите: pip install plotly kaleido")
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
        print(f"Доступные колонки: {list(df.columns)}")
        return
    
    # Фильтруем записи с глубиной
    df_depth = df[df['object_depth_m'].notna()].copy()
    
    if len(df_depth) == 0:
        print("Предупреждение: нет записей с глубиной объектов")
        return
    
    # Определяем диапазон глубин
    depth_max = df_depth['object_depth_m'].max()
    
    # Загружаем CTD данные если указаны
    ctd_df = None
    ctd_col_names = []
    ctd_depth_col = None
    
    if ctd_path and ctd_columns:
        try:
            ctd_df = load_ctd_data(ctd_path)
            ctd_depth_col = find_depth_column(ctd_df)
            
            if ctd_depth_col is None:
                print("Предупреждение: не найдена колонка глубины в CTD данных")
                ctd_df = None
            else:
                all_cols = list(ctd_df.columns)
                for idx in ctd_columns:
                    if 0 <= idx < len(all_cols):
                        ctd_col_names.append(all_cols[idx])
                    else:
                        print(f"Предупреждение: индекс колонки {idx} вне диапазона")
                
                if ctd_col_names:
                    ctd_depth_max = ctd_df[ctd_depth_col].max()
                    if ctd_depth_max > depth_max:
                        depth_max = ctd_depth_max
                    print(f"CTD колонки для отображения: {ctd_col_names}")
                else:
                    ctd_df = None
        except Exception as e:
            print(f"Предупреждение: не удалось загрузить CTD данные: {e}")
            ctd_df = None
    
    # Определяем количество колонок для subplots
    n_ctd = len(ctd_col_names) if ctd_df is not None else 0
    n_cols = 2 + n_ctd  # scatter + means + CTD
    
    # Соотношение ширины колонок
    column_widths = [0.4, 0.3] + [0.3 / max(n_ctd, 1)] * n_ctd if n_ctd > 0 else [0.5, 0.5]
    
    # Создаём subplot
    subplot_titles = ['Организмы по глубине', 'Количество по глубине'] + ctd_col_names
    
    fig = make_subplots(
        rows=1, cols=n_cols,
        shared_yaxes=True,
        column_widths=column_widths,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.03
    )
    
    # Виды в данных
    species_list = sorted(df_depth['class_name'].unique())
    
    # Нормализация размеров для отображения
    size_min = df_depth['real_size_cm'].min()
    size_max = df_depth['real_size_cm'].max()
    
    if size_max > size_min:
        df_depth['marker_size'] = 8 + 25 * (df_depth['real_size_cm'] - size_min) / (size_max - size_min)
    else:
        df_depth['marker_size'] = 15
    
    # === Scatter plot организмов ===
    for species in species_list:
        sp_df = df_depth[df_depth['class_name'] == species]
        color = SPECIES_COLORS.get(species, 'gray')
        species_ru = SPECIES_NAMES_RU.get(species, species)
        
        # Jitter по X для разделения точек
        np.random.seed(42)  # Для воспроизводимости
        x_jitter = np.random.uniform(-0.3, 0.3, len(sp_df))
        
        # Hover текст
        hover_text = [
            f"<b>{species}</b><br>"
            f"Глубина: {depth:.1f} м<br>"
            f"Размер: {size:.1f} см<br>"
            f"Track ID: {tid}"
            for depth, size, tid in zip(
                sp_df['object_depth_m'], 
                sp_df['real_size_cm'],
                sp_df['track_id']
            )
        ]
        
        fig.add_trace(
            go.Scatter(
                x=x_jitter,
                y=sp_df['object_depth_m'],
                mode='markers',
                name=f"{species_ru} ({species})",
                legendgroup=species,
                marker=dict(
                    size=sp_df['marker_size'],
                    color=color,
                    opacity=0.7,
                    line=dict(width=0.5, color='black')
                ),
                hovertemplate="%{text}<extra></extra>",
                text=hover_text
            ),
            row=1, col=1
        )
    
    # === Линии средних по глубине ===
    depth_bins = np.arange(0, depth_max + depth_bin, depth_bin)
    bin_centers = depth_bins[:-1] + depth_bin / 2
    
    for species in species_list:
        sp_df = df_depth[df_depth['class_name'] == species]
        color = SPECIES_COLORS.get(species, 'gray')
        species_ru = SPECIES_NAMES_RU.get(species, species)
        
        # Подсчёт по бинам
        counts, _ = np.histogram(sp_df['object_depth_m'], bins=depth_bins)
        
        # Hover текст для линии
        hover_text = [
            f"<b>{species}</b><br>"
            f"Глубина: {d:.1f}-{d+depth_bin:.1f} м<br>"
            f"Количество: {c}"
            for d, c in zip(depth_bins[:-1], counts)
        ]
        
        fig.add_trace(
            go.Scatter(
                x=counts,
                y=bin_centers,
                mode='lines+markers',
                name=f"{species_ru} (среднее)",
                legendgroup=species,
                showlegend=False,
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color),
                hovertemplate="%{text}<extra></extra>",
                text=hover_text
            ),
            row=1, col=2
        )
    
    # === CTD параметры ===
    if ctd_df is not None and ctd_col_names:
        for i, col_name in enumerate(ctd_col_names):
            ctd_plot_df = ctd_df[[ctd_depth_col, col_name]].dropna()
            
            if len(ctd_plot_df) > 0:
                # Прореживаем данные для производительности (если слишком много точек)
                if len(ctd_plot_df) > 5000:
                    step = len(ctd_plot_df) // 5000
                    ctd_plot_df = ctd_plot_df.iloc[::step]
                
                hover_text = [
                    f"<b>{col_name}</b><br>"
                    f"Глубина: {d:.2f} м<br>"
                    f"Значение: {v:.3f}"
                    for d, v in zip(ctd_plot_df[ctd_depth_col], ctd_plot_df[col_name])
                ]
                
                fig.add_trace(
                    go.Scatter(
                        x=ctd_plot_df[col_name],
                        y=ctd_plot_df[ctd_depth_col],
                        mode='lines',
                        name=col_name,
                        line=dict(color='#333333', width=1.5),
                        hovertemplate="%{text}<extra></extra>",
                        text=hover_text
                    ),
                    row=1, col=3 + i
                )
    
    # === Настройка осей ===
    # Инвертируем ось Y (глубина вниз)
    fig.update_yaxes(
        autorange="reversed",
        title_text="Глубина, м",
        row=1, col=1
    )
    
    # X-ось для scatter (скрываем)
    fig.update_xaxes(
        showticklabels=False,
        title_text="",
        row=1, col=1
    )
    
    # X-ось для количества
    fig.update_xaxes(
        title_text="Количество особей",
        row=1, col=2
    )
    
    # X-оси для CTD
    for i, col_name in enumerate(ctd_col_names):
        fig.update_xaxes(
            title_text=col_name,
            row=1, col=3 + i
        )
    
    # === Общие настройки ===
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><sub>Размер маркера пропорционален размеру организма ({size_min:.1f} - {size_max:.1f} см)</sub>",
            x=0.5,
            xanchor='center'
        ),
        height=800,
        width=400 * n_cols,
        legend=dict(
            title="Виды (клик для фильтрации)",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1
        ),
        hovermode='closest',
        # Добавляем кнопки для управления
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[{"visible": True}],
                        label="Показать все",
                        method="restyle"
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=False,
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
        ]
    )
    
    # Добавляем аннотацию с инструкцией
    fig.add_annotation(
        text="💡 Клик по легенде — скрыть/показать вид. Двойной клик — показать только этот вид.",
        xref="paper", yref="paper",
        x=0, y=1.12,
        showarrow=False,
        font=dict(size=10, color="gray"),
        align="left"
    )
    
    # === Экспорт ===
    output_path = Path(output_path)
    
    if export_format == "html":
        # Интерактивный HTML
        fig.write_html(
            str(output_path.with_suffix('.html')),
            include_plotlyjs=True,
            full_html=True,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
                'toImageButtonOptions': {
                    'format': 'svg',
                    'filename': output_path.stem,
                    'height': 800,
                    'width': 400 * n_cols,
                    'scale': 2
                }
            }
        )
        print(f"Интерактивный график сохранён: {output_path.with_suffix('.html')}")
        
    elif export_format == "svg":
        try:
            fig.write_image(str(output_path.with_suffix('.svg')), format='svg')
            print(f"SVG график сохранён: {output_path.with_suffix('.svg')}")
        except Exception as e:
            print(f"Ошибка экспорта в SVG (установите kaleido: pip install kaleido): {e}")
            
    elif export_format == "pdf":
        try:
            fig.write_image(str(output_path.with_suffix('.pdf')), format='pdf')
            print(f"PDF график сохранён: {output_path.with_suffix('.pdf')}")
        except Exception as e:
            print(f"Ошибка экспорта в PDF (установите kaleido: pip install kaleido): {e}")
            
    elif export_format == "png":
        try:
            fig.write_image(str(output_path.with_suffix('.png')), format='png', scale=2)
            print(f"PNG график сохранён: {output_path.with_suffix('.png')}")
        except Exception as e:
            print(f"Ошибка экспорта в PNG (установите kaleido: pip install kaleido): {e}")
    
    # Всегда сохраняем HTML для интерактивности
    if export_format != "html":
        html_path = output_path.with_suffix('.html')
        fig.write_html(str(html_path), include_plotlyjs=True, full_html=True)
        print(f"Также сохранён интерактивный HTML: {html_path}")


def parse_ctd_columns(value: str) -> List[int]:
    """Парсит строку с номерами колонок CTD."""
    if not value:
        return []
    
    result = []
    for part in value.split(','):
        part = part.strip()
        if part.isdigit():
            result.append(int(part))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Интерактивная визуализация распределения желетелых по глубине",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Интерактивный HTML график
  python interactive_plot.py --track-sizes detections_track_sizes.csv -o output/plot

  # С CTD данными (температура - колонка 6, солёность - колонка 7)
  python interactive_plot.py --track-sizes detections_track_sizes.csv \\
                             --ctd ctd_data.csv --ctd-columns 6,7 -o output/plot

  # Экспорт в SVG (векторный)
  python interactive_plot.py --track-sizes detections_track_sizes.csv \\
                             --format svg -o output/plot

  # Экспорт в PDF
  python interactive_plot.py --track-sizes detections_track_sizes.csv \\
                             --format pdf -o output/plot

  # Посмотреть колонки в CTD файле
  python interactive_plot.py --ctd ctd_data.csv --list-ctd-columns

Интерактивные возможности (в HTML):
  - Клик по легенде: скрыть/показать вид
  - Двойной клик по легенде: показать только этот вид
  - Колёсико мыши: zoom
  - Перетаскивание: pan
  - Hover: подробная информация о точке
  - Toolbar: сохранение в SVG/PNG
        """
    )
    
    parser.add_argument(
        "--track-sizes", "-t",
        help="Путь к CSV с размерами треков (detections_track_sizes.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        default="output/depth_distribution",
        help="Путь для сохранения (без расширения)"
    )
    parser.add_argument(
        "--depth-bin",
        type=float,
        default=1.0,
        help="Шаг биннинга по глубине в метрах (по умолчанию: 1.0)"
    )
    parser.add_argument(
        "--title",
        default="Распределение желетелых по глубине",
        help="Заголовок графика"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["html", "svg", "pdf", "png"],
        default="html",
        help="Формат экспорта (по умолчанию: html)"
    )
    
    # CTD параметры
    parser.add_argument(
        "--ctd",
        help="Путь к CSV с данными CTD"
    )
    parser.add_argument(
        "--ctd-columns",
        type=str,
        default="",
        help="Номера колонок CTD через запятую (0-based), например: 5,6,7"
    )
    parser.add_argument(
        "--list-ctd-columns",
        action="store_true",
        help="Показать список колонок в CTD файле и выйти"
    )
    
    args = parser.parse_args()
    
    # Проверка plotly
    if not PLOTLY_AVAILABLE:
        print("Ошибка: plotly не установлен")
        print("Установите: pip install plotly kaleido")
        return 1
    
    # Показать колонки CTD
    if args.list_ctd_columns:
        if not args.ctd:
            print("Ошибка: укажите путь к CTD файлу через --ctd")
            return 1
        
        try:
            ctd_df = load_ctd_data(args.ctd)
            print(f"\nКолонки в файле {args.ctd}:\n")
            for i, col in enumerate(ctd_df.columns):
                sample = ctd_df[col].dropna().head(1)
                sample_str = str(sample.values[0]) if len(sample) > 0 else "N/A"
                if len(sample_str) > 30:
                    sample_str = sample_str[:27] + "..."
                print(f"  {i:2d}: {col:<30} (пример: {sample_str})")
            print()
            return 0
        except Exception as e:
            print(f"Ошибка чтения CTD файла: {e}")
            return 1
    
    # Проверка входных файлов
    if not args.track_sizes:
        print("Ошибка: укажите --track-sizes")
        parser.print_help()
        return 1
    
    try:
        # Создание директории вывода
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Парсим колонки CTD
        ctd_columns = parse_ctd_columns(args.ctd_columns)
        
        # Создаём график
        create_interactive_depth_plot(
            track_sizes_path=args.track_sizes,
            output_path=str(output_path),
            ctd_path=args.ctd,
            ctd_columns=ctd_columns if ctd_columns else None,
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
