"""
Анализ результатов детекции и построение графиков.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple


# Цвета для видов
SPECIES_COLORS = {
    'Aurelia aurita': '#1f77b4',
    'Rhizostoma pulmo': '#ff7f0e',
    'Beroe ovata': '#2ca02c',
    'Mnemiopsis leidyi': '#d62728',
    'Pleurobrachia pileus': '#9467bd'
}


def load_ctd_data(ctd_path: str) -> pd.DataFrame:
    """
    Загружает данные CTD с автоматическим определением разделителя.
    
    Args:
        ctd_path: путь к CTD файлу
        
    Returns:
        DataFrame с данными CTD
    """
    # Пробуем различные разделители
    for sep in [',', ';', '|', '\t']:
        try:
            df = pd.read_csv(ctd_path, sep=sep, encoding='utf-8-sig')
            if len(df.columns) > 1:
                return df
        except:
            continue
    
    # Последняя попытка с автоопределением
    return pd.read_csv(ctd_path, sep=None, engine='python', encoding='utf-8-sig')


def plot_depth_distribution_with_sizes(
    track_sizes_path: str,
    output_path: str,
    ctd_path: Optional[str] = None,
    ctd_columns: Optional[List[int]] = None,
    depth_bin: float = 1.0,
    title: str = "Распределение желетелых по глубине"
):
    """
    Строит график распределения желетелых по глубине с размерами и CTD данными.
    
    Ось Y — глубина (от 0 до максимальной).
    Ось X — организмы (точки) с цветом по виду и размером по estimated_size_cm.
    Линии — средние количества по глубине для каждого вида.
    Опционально — дополнительные показатели из CTD.
    
    Args:
        track_sizes_path: путь к CSV с данными треков (detections_track_sizes.csv)
        output_path: путь для сохранения графика
        ctd_path: путь к CSV с данными CTD (опционально)
        ctd_columns: номера колонок CTD для отображения (опционально, 0-based)
        depth_bin: шаг биннинга для расчёта средних (м)
        title: заголовок графика
    """
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
    depth_min = 0
    depth_max = df_depth['object_depth_m'].max()
    
    # Загружаем CTD данные если указаны
    ctd_df = None
    ctd_col_names = []
    if ctd_path and ctd_columns:
        try:
            ctd_df = load_ctd_data(ctd_path)
            # Находим колонку с глубиной (ищем Depth без учёта регистра)
            depth_col = None
            for col in ctd_df.columns:
                if 'depth' in col.lower():
                    depth_col = col
                    break
            
            if depth_col is None:
                print("Предупреждение: не найдена колонка глубины в CTD данных")
                ctd_df = None
            else:
                # Получаем имена колонок по индексам
                all_cols = list(ctd_df.columns)
                for idx in ctd_columns:
                    if 0 <= idx < len(all_cols):
                        ctd_col_names.append(all_cols[idx])
                    else:
                        print(f"Предупреждение: индекс колонки {idx} вне диапазона (0-{len(all_cols)-1})")
                
                if ctd_col_names:
                    # Обновляем максимальную глубину по CTD
                    ctd_depth_max = ctd_df[depth_col].max()
                    if ctd_depth_max > depth_max:
                        depth_max = ctd_depth_max
                    print(f"CTD колонки для отображения: {ctd_col_names}")
                else:
                    ctd_df = None
        except Exception as e:
            print(f"Предупреждение: не удалось загрузить CTD данные: {e}")
            ctd_df = None
    
    # Определяем количество подграфиков
    n_ctd_plots = len(ctd_col_names) if ctd_df is not None else 0
    n_plots = 2 + n_ctd_plots  # scatter + means + CTD колонки
    
    # Создаём сетку подграфиков
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 10), sharey=True)
    if n_plots == 1:
        axes = [axes]
    
    # Виды в данных
    species_list = df_depth['class_name'].unique()
    
    # === График 1: Scatter plot объектов по глубине ===
    ax_scatter = axes[0]
    
    # Нормализация размеров для отображения
    size_min = df_depth['real_size_cm'].min()
    size_max = df_depth['real_size_cm'].max()
    
    # Масштабируем размеры точек (от 20 до 200 пикселей)
    if size_max > size_min:
        df_depth['marker_size'] = 20 + 180 * (df_depth['real_size_cm'] - size_min) / (size_max - size_min)
    else:
        df_depth['marker_size'] = 100
    
    # Рисуем точки для каждого вида
    for species in species_list:
        sp_df = df_depth[df_depth['class_name'] == species]
        color = SPECIES_COLORS.get(species, 'gray')
        
        # Добавляем небольшой горизонтальный jitter для лучшей видимости
        x_jitter = np.random.uniform(-0.3, 0.3, len(sp_df))
        
        ax_scatter.scatter(
            x_jitter,
            sp_df['object_depth_m'],
            s=sp_df['marker_size'],
            c=color,
            alpha=0.6,
            edgecolors='black',
            linewidths=0.3,
            label=species
        )
    
    ax_scatter.set_xlabel('Организмы')
    ax_scatter.set_ylabel('Глубина, м')
    ax_scatter.set_title('Распределение по глубине\n(размер = размер организма)', fontsize=10)
    ax_scatter.set_xlim(-1, 1)
    ax_scatter.set_ylim(depth_max * 1.02, -depth_max * 0.02)  # Инвертированная ось Y
    ax_scatter.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax_scatter.grid(axis='y', alpha=0.3)
    ax_scatter.set_xticks([])
    
    # Добавляем легенду размеров
    if size_max > size_min:
        size_legend_text = f"Размер: {size_min:.1f} - {size_max:.1f} см"
        ax_scatter.text(0.02, 0.02, size_legend_text, transform=ax_scatter.transAxes,
                       fontsize=8, verticalalignment='bottom',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # === График 2: Средние количества по глубине ===
    ax_means = axes[1]
    
    # Биннинг по глубине
    depth_bins = np.arange(0, depth_max + depth_bin, depth_bin)
    df_depth['depth_bin'] = pd.cut(df_depth['object_depth_m'], bins=depth_bins, labels=depth_bins[:-1])
    
    # Подсчёт по видам и глубинам
    for species in species_list:
        sp_df = df_depth[df_depth['class_name'] == species]
        counts = sp_df.groupby('depth_bin', observed=True).size()
        
        # Заполняем нулями отсутствующие бины
        all_bins = pd.Series(0, index=depth_bins[:-1])
        for bin_val in counts.index:
            if bin_val in all_bins.index:
                all_bins[bin_val] = counts[bin_val]
        
        color = SPECIES_COLORS.get(species, 'gray')
        
        # Центры бинов для графика
        bin_centers = all_bins.index.astype(float) + depth_bin / 2
        
        ax_means.plot(
            all_bins.values,
            bin_centers,
            color=color,
            linewidth=2,
            marker='o',
            markersize=4,
            label=species
        )
    
    ax_means.set_xlabel('Количество особей')
    ax_means.set_title(f'Средние по глубине\n(бин = {depth_bin} м)', fontsize=10)
    ax_means.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax_means.grid(alpha=0.3)
    ax_means.set_xlim(left=0)
    
    # === Графики CTD параметров ===
    if ctd_df is not None and ctd_col_names:
        # Находим колонку глубины
        depth_col = None
        for col in ctd_df.columns:
            if 'depth' in col.lower():
                depth_col = col
                break
        
        for i, col_name in enumerate(ctd_col_names):
            ax_ctd = axes[2 + i]
            
            # Очищаем данные
            ctd_plot_df = ctd_df[[depth_col, col_name]].dropna()
            
            if len(ctd_plot_df) > 0:
                ax_ctd.plot(
                    ctd_plot_df[col_name],
                    ctd_plot_df[depth_col],
                    color='#333333',
                    linewidth=1.5
                )
                ax_ctd.set_xlabel(col_name)
                ax_ctd.set_title(f'{col_name}', fontsize=10)
                ax_ctd.grid(alpha=0.3)
            else:
                ax_ctd.text(0.5, 0.5, 'Нет данных', ha='center', va='center',
                           transform=ax_ctd.transAxes)
    
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"График сохранён: {output_path}")


def plot_vertical_distribution(
    df: pd.DataFrame,
    output_path: str,
    depth_bin: float = 1.0,
    title: str = "Вертикальное распределение"
):
    """
    Строит профиль вертикального распределения организмов.
    
    Args:
        df: DataFrame с детекциями
        output_path: путь для сохранения графика
        depth_bin: шаг биннинга по глубине (м)
        title: заголовок графика
    """
    if df['depth_m'].isna().all():
        print("Предупреждение: нет данных о глубине для построения профиля")
        return
    
    # Фильтруем записи с глубиной
    df_depth = df[df['depth_m'].notna()].copy()
    
    if len(df_depth) == 0:
        print("Предупреждение: нет записей с глубиной")
        return
    
    # Биннинг по глубине
    df_depth['depth_bin'] = (df_depth['depth_m'] // depth_bin) * depth_bin
    
    # Подсчёт по видам и глубинам
    counts = df_depth.groupby(['depth_bin', 'class_name']).size().unstack(fill_value=0)
    
    # Количество видов для графиков
    n_species = len(counts.columns)
    
    if n_species == 0:
        print("Предупреждение: нет данных для построения графика")
        return
    
    # Создание графика
    fig, axes = plt.subplots(1, n_species, figsize=(3.5 * n_species, 8), sharey=True)
    
    if n_species == 1:
        axes = [axes]
    
    for ax, species in zip(axes, counts.columns):
        color = SPECIES_COLORS.get(species, 'gray')
        ax.barh(
            counts.index,
            counts[species],
            height=depth_bin * 0.8,
            color=color,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        ax.set_xlabel('Количество детекций')
        ax.set_title(species, fontsize=10, style='italic')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
    
    axes[0].set_ylabel('Глубина, м')
    
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"График сохранён: {output_path}")


def plot_detection_timeline(
    df: pd.DataFrame,
    output_path: str,
    time_bin: float = 10.0,
    title: str = "Временная шкала детекций"
):
    """
    Строит временную шкалу детекций.
    
    Args:
        df: DataFrame с детекциями
        output_path: путь для сохранения графика
        time_bin: шаг биннинга по времени (сек)
        title: заголовок графика
    """
    if len(df) == 0:
        print("Предупреждение: нет данных для построения временной шкалы")
        return
    
    # Биннинг по времени
    df_copy = df.copy()
    df_copy['time_bin'] = (df_copy['timestamp_s'] // time_bin) * time_bin
    
    # Подсчёт по видам и времени
    counts = df_copy.groupby(['time_bin', 'class_name']).size().unstack(fill_value=0)
    
    # Создание графика
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bottom = np.zeros(len(counts))
    
    for species in counts.columns:
        color = SPECIES_COLORS.get(species, 'gray')
        ax.bar(
            counts.index,
            counts[species],
            width=time_bin * 0.8,
            bottom=bottom,
            color=color,
            alpha=0.7,
            label=species,
            edgecolor='black',
            linewidth=0.5
        )
        bottom += counts[species].values
    
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Количество детекций')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"График сохранён: {output_path}")


def plot_species_summary(
    df: pd.DataFrame,
    output_path: str,
    title: str = "Сводка по видам"
):
    """
    Строит сводный график по видам.
    
    Args:
        df: DataFrame с детекциями
        output_path: путь для сохранения графика
        title: заголовок графика
    """
    if len(df) == 0:
        print("Предупреждение: нет данных для построения сводки")
        return
    
    # Статистика по видам
    summary = df.groupby('class_name').agg({
        'confidence': ['count', 'mean', 'std'],
        'depth_m': ['min', 'max', 'mean']
    }).round(3)
    
    summary.columns = ['count', 'conf_mean', 'conf_std', 'depth_min', 'depth_max', 'depth_mean']
    summary = summary.sort_values('count', ascending=True)
    
    # Создание графика
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # График 1: Количество детекций
    colors = [SPECIES_COLORS.get(sp, 'gray') for sp in summary.index]
    axes[0].barh(summary.index, summary['count'], color=colors, alpha=0.7)
    axes[0].set_xlabel('Количество детекций')
    axes[0].set_title('Количество детекций по видам')
    
    # Добавляем значения на графике
    for i, v in enumerate(summary['count']):
        axes[0].text(v + 0.5, i, str(int(v)), va='center')
    
    # График 2: Средняя уверенность
    axes[1].barh(summary.index, summary['conf_mean'], xerr=summary['conf_std'], 
                  color=colors, alpha=0.7, capsize=3)
    axes[1].set_xlabel('Средняя уверенность')
    axes[1].set_title('Уверенность детекций')
    axes[1].set_xlim(0, 1)
    
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"График сохранён: {output_path}")


def generate_report(
    df: pd.DataFrame,
    output_path: str,
    video_name: Optional[str] = None,
    processing_info: Optional[dict] = None
):
    """
    Генерирует текстовый отчёт по детекциям.
    
    Args:
        df: DataFrame с детекциями
        output_path: путь для сохранения отчёта
        video_name: имя исходного видео (опционально)
        processing_info: информация об обработке (опционально)
            - detection_params: параметры детекции
            - postprocess_params: параметры постобработки
            - timing: временные метки
            - subtasks: информация о подзадачах
    """
    from datetime import datetime
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("ОТЧЁТ ПО ДЕТЕКЦИИ ЖЕЛЕТЕЛОГО МАКРОЗООПЛАНКТОНА\n")
        f.write("="*70 + "\n\n")
        
        # Дата генерации отчёта
        f.write(f"Дата отчёта: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if video_name:
            f.write(f"Исходное видео: {video_name}\n")
        
        f.write(f"Всего детекций: {len(df)}\n")
        f.write(f"Уникальных кадров с детекциями: {df['frame'].nunique()}\n")
        
        if 'timestamp_s' in df.columns:
            f.write(f"Временной диапазон: {df['timestamp_s'].min():.1f} - {df['timestamp_s'].max():.1f} с\n")
        
        if df['depth_m'].notna().any():
            f.write(f"Диапазон глубин: {df['depth_m'].min():.1f} - {df['depth_m'].max():.1f} м\n")
        
        # === КОНФИГУРАЦИЯ ОБРАБОТКИ ===
        if processing_info:
            f.write("\n" + "-"*70 + "\n")
            f.write("КОНФИГУРАЦИЯ ОБРАБОТКИ\n")
            f.write("-"*70 + "\n")
            
            # Параметры детекции
            det_params = processing_info.get('detection_params', {})
            if det_params:
                f.write("\nПараметры детекции:\n")
                f.write(f"  Модель: {det_params.get('model_name', 'N/A')}\n")
                f.write(f"  Порог уверенности: {det_params.get('conf_threshold', 'N/A')}\n")
                
                if det_params.get('enable_tracking'):
                    f.write(f"  Трекинг: включён ({det_params.get('tracker_type', 'N/A')})\n")
                    f.write(f"  Мин. длина трека: {det_params.get('min_track_length', 'N/A')} кадров\n")
                else:
                    f.write("  Трекинг: выключен\n")
                
                if det_params.get('ctd_file'):
                    f.write(f"  CTD: {det_params.get('ctd_file')}\n")
                elif det_params.get('depth_rate'):
                    f.write(f"  Скорость погружения: {det_params.get('depth_rate')} м/с\n")
            
            # Параметры постобработки
            post_params = processing_info.get('postprocess_params', {})
            if post_params:
                f.write("\nПараметры постобработки:\n")
                f.write(f"  FOV камеры: {post_params.get('fov', 'N/A')}°\n")
                f.write(f"  Ближняя дистанция: {post_params.get('near_distance', 'N/A')} м\n")
                f.write(f"  Бин глубины: {post_params.get('depth_bin', 'N/A')} м\n")
            
            # Информация о подзадачах
            subtasks = processing_info.get('subtasks', [])
            if subtasks:
                f.write("\nВыполненные операции:\n")
                for st in subtasks:
                    status_icon = "✓" if st.get('success') else "✗"
                    time_str = ""
                    if st.get('processing_time_s'):
                        time_str = f" ({st['processing_time_s']:.1f} с)"
                    result_str = ""
                    if st.get('result_text'):
                        result_str = f" → {st['result_text']}"
                    f.write(f"  [{status_icon}] {st.get('name', 'N/A')}{time_str}{result_str}\n")
            
            # Время обработки
            timing = processing_info.get('timing', {})
            if timing:
                f.write("\nВремя обработки:\n")
                if timing.get('detection_time_s'):
                    det_time = timing['detection_time_s']
                    if det_time >= 60:
                        f.write(f"  Детекция: {det_time/60:.1f} мин\n")
                    else:
                        f.write(f"  Детекция: {det_time:.1f} с\n")
                
                if timing.get('postprocess_time_s'):
                    post_time = timing['postprocess_time_s']
                    if post_time >= 60:
                        f.write(f"  Постобработка: {post_time/60:.1f} мин\n")
                    else:
                        f.write(f"  Постобработка: {post_time:.1f} с\n")
                
                if timing.get('total_time_s'):
                    total_time = timing['total_time_s']
                    if total_time >= 60:
                        f.write(f"  Общее время: {total_time/60:.1f} мин\n")
                    else:
                        f.write(f"  Общее время: {total_time:.1f} с\n")
        
        f.write("\n" + "-"*70 + "\n")
        f.write("ДЕТЕКЦИИ ПО ВИДАМ\n")
        f.write("-"*70 + "\n")
        
        for species in sorted(df['class_name'].unique()):
            sp_df = df[df['class_name'] == species]
            
            f.write(f"\n{species}\n")
            f.write(f"  Количество детекций: {len(sp_df)}\n")
            f.write(f"  Доля от общего числа: {len(sp_df)/len(df)*100:.1f}%\n")
            f.write(f"  Средняя уверенность: {sp_df['confidence'].mean():.3f}\n")
            f.write(f"  Мин/макс уверенность: {sp_df['confidence'].min():.3f} / {sp_df['confidence'].max():.3f}\n")
            
            if sp_df['depth_m'].notna().any():
                f.write(f"  Диапазон глубин: {sp_df['depth_m'].min():.1f} - {sp_df['depth_m'].max():.1f} м\n")
                f.write(f"  Средняя глубина: {sp_df['depth_m'].mean():.1f} м\n")
            
            if sp_df['temperature_c'].notna().any():
                f.write(f"  Диапазон температур: {sp_df['temperature_c'].min():.1f} - {sp_df['temperature_c'].max():.1f} °C\n")
            
            # Добавляем информацию о размерах если есть
            if 'estimated_size_mm' in sp_df.columns and sp_df['estimated_size_mm'].notna().any():
                sizes = sp_df['estimated_size_mm'].dropna()
                f.write(f"  Размеры: {sizes.min():.1f} - {sizes.max():.1f} мм (средний: {sizes.mean():.1f} мм)\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("КОНЕЦ ОТЧЁТА\n")
        f.write("="*70 + "\n")
    
    print(f"Отчёт сохранён: {output_path}")


def analyze_detections(
    csv_path: str,
    output_dir: str = "output",
    depth_bin: float = 1.0,
    report_name: str = "report.txt"
):
    """
    Полный анализ результатов детекции.
    
    Args:
        csv_path: путь к CSV с детекциями
        output_dir: директория для сохранения результатов
        depth_bin: шаг биннинга по глубине
        report_name: имя файла отчёта
    """
    # Загрузка данных
    print(f"Загрузка данных: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if len(df) == 0:
        print("Предупреждение: файл не содержит детекций")
        return
    
    # Создание директории
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nАнализ {len(df)} детекций...")
    print()
    
    # Построение графиков
    plot_vertical_distribution(
        df,
        str(output_path / "vertical_distribution.png"),
        depth_bin=depth_bin
    )
    
    plot_detection_timeline(
        df,
        str(output_path / "detection_timeline.png")
    )
    
    plot_species_summary(
        df,
        str(output_path / "species_summary.png")
    )
    
    # Генерация отчёта
    generate_report(
        df,
        str(output_path / report_name),
        video_name=Path(csv_path).stem.replace("_detections", "")
    )
    
    print()
    print("Анализ завершён!")


def parse_ctd_columns(value: str) -> List[int]:
    """
    Парсит строку с номерами колонок CTD.
    
    Args:
        value: строка вида "2,5,7" или "2, 5, 7"
        
    Returns:
        список индексов колонок
    """
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
        description="Анализ результатов детекции желетелых",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Базовый анализ детекций
  python analyze.py --csv detections.csv --output-dir output

  # График распределения по глубине с размерами
  python analyze.py --track-sizes detections_track_sizes.csv --output-dir output

  # С CTD данными (температура - колонка 6, солёность - колонка 7)
  python analyze.py --track-sizes detections_track_sizes.csv \\
                    --ctd ctd_data.csv --ctd-columns 6,7 --output-dir output

  # Посмотреть колонки в CTD файле
  python analyze.py --ctd ctd_data.csv --list-ctd-columns
        """
    )
    
    # Основные параметры
    parser.add_argument(
        "--csv", "-c",
        help="Путь к CSV с детекциями (detections.csv)"
    )
    parser.add_argument(
        "--track-sizes", "-t",
        help="Путь к CSV с размерами треков (detections_track_sizes.csv)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Директория для графиков и отчёта (по умолчанию: output)"
    )
    parser.add_argument(
        "--depth-bin",
        type=float,
        default=1.0,
        help="Шаг биннинга по глубине в метрах (по умолчанию: 1.0)"
    )
    parser.add_argument(
        "--report",
        default="report.txt",
        help="Имя файла отчёта (по умолчанию: report.txt)"
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
        help="Номера колонок CTD для отображения через запятую (0-based), например: 5,6,7"
    )
    parser.add_argument(
        "--list-ctd-columns",
        action="store_true",
        help="Показать список колонок в CTD файле и выйти"
    )
    
    args = parser.parse_args()
    
    # Показать колонки CTD
    if args.list_ctd_columns:
        if not args.ctd:
            print("Ошибка: укажите путь к CTD файлу через --ctd")
            return 1
        
        try:
            ctd_df = load_ctd_data(args.ctd)
            print(f"\nКолонки в файле {args.ctd}:\n")
            for i, col in enumerate(ctd_df.columns):
                # Показываем пример значения
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
    if not args.csv and not args.track_sizes:
        print("Ошибка: укажите --csv или --track-sizes")
        parser.print_help()
        return 1
    
    try:
        # Создание директории вывода
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Парсим колонки CTD
        ctd_columns = parse_ctd_columns(args.ctd_columns)
        
        # График распределения с размерами
        if args.track_sizes:
            plot_depth_distribution_with_sizes(
                track_sizes_path=args.track_sizes,
                output_path=str(output_path / "depth_distribution_sizes.png"),
                ctd_path=args.ctd,
                ctd_columns=ctd_columns if ctd_columns else None,
                depth_bin=args.depth_bin
            )
        
        # Стандартный анализ детекций
        if args.csv:
            analyze_detections(
                csv_path=args.csv,
                output_dir=args.output_dir,
                depth_bin=args.depth_bin,
                report_name=args.report
            )
        
        return 0
        
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
