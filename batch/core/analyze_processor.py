"""
AnalyzeProcessor - обёртка над analyze.py для интеграции с batch processing.

Поддерживает:
- Построение графиков вертикального распределения
- Временные шкалы детекций
- Сводные графики по видам
- Генерацию текстовых отчётов
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field

# Добавляем путь к src для импорта
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))


@dataclass
class AnalyzeResult:
    """Результат анализа детекций."""
    success: bool
    processing_time_s: float = 0.0
    output_dir: Optional[str] = None
    vertical_distribution_path: Optional[str] = None
    detection_timeline_path: Optional[str] = None
    species_summary_path: Optional[str] = None
    report_path: Optional[str] = None
    total_detections: int = 0
    unique_species: int = 0
    species_counts: dict = field(default_factory=dict)
    error_message: Optional[str] = None
    cancelled: bool = False


class AnalyzeProcessor:
    """
    Обработчик анализа результатов детекции.
    
    Генерирует графики и отчёты по данным детекции.
    """
    
    def __init__(
        self,
        csv_path: str,
        output_dir: str,
        depth_bin: float = 1.0,
        time_bin: float = 10.0,
        report_name: str = "report.txt",
        generate_vertical_distribution: bool = True,
        generate_timeline: bool = True,
        generate_species_summary: bool = True,
        generate_report: bool = True,
        video_name: Optional[str] = None,
    ):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.depth_bin = depth_bin
        self.time_bin = time_bin
        self.report_name = report_name
        self.generate_vertical_distribution = generate_vertical_distribution
        self.generate_timeline = generate_timeline
        self.generate_species_summary = generate_species_summary
        self.generate_report = generate_report
        self.video_name = video_name
        
        self._cancelled = False
    
    def cancel(self) -> None:
        """Отменяет выполнение."""
        self._cancelled = True
    
    def is_cancelled(self) -> bool:
        return self._cancelled
    
    def run(self) -> AnalyzeResult:
        """Запускает анализ данных."""
        self._cancelled = False
        start_time = time.time()
        
        try:
            import pandas as pd
            from analyze import (
                plot_vertical_distribution,
                plot_detection_timeline,
                plot_species_summary,
                generate_report
            )
            
            # Загружаем данные
            print(f"Загрузка данных: {self.csv_path}")
            df = pd.read_csv(self.csv_path)
            
            if len(df) == 0:
                return AnalyzeResult(
                    success=False,
                    processing_time_s=time.time() - start_time,
                    error_message="Файл не содержит детекций"
                )
            
            # Создаём выходную директорию
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            result = AnalyzeResult(
                success=True,
                output_dir=self.output_dir,
                total_detections=len(df),
                unique_species=df['class_name'].nunique()
            )
            
            # Подсчёт по видам
            result.species_counts = df['class_name'].value_counts().to_dict()
            
            # Вертикальное распределение
            if self.generate_vertical_distribution:
                if self._cancelled:
                    result.cancelled = True
                    return result
                
                vd_path = str(output_path / "vertical_distribution.png")
                try:
                    plot_vertical_distribution(df, vd_path, self.depth_bin)
                    result.vertical_distribution_path = vd_path
                except Exception as e:
                    print(f"Ошибка при построении вертикального распределения: {e}")
            
            # Временная шкала
            if self.generate_timeline:
                if self._cancelled:
                    result.cancelled = True
                    return result
                
                tl_path = str(output_path / "detection_timeline.png")
                try:
                    plot_detection_timeline(df, tl_path, self.time_bin)
                    result.detection_timeline_path = tl_path
                except Exception as e:
                    print(f"Ошибка при построении временной шкалы: {e}")
            
            # Сводка по видам
            if self.generate_species_summary:
                if self._cancelled:
                    result.cancelled = True
                    return result
                
                ss_path = str(output_path / "species_summary.png")
                try:
                    plot_species_summary(df, ss_path)
                    result.species_summary_path = ss_path
                except Exception as e:
                    print(f"Ошибка при построении сводки: {e}")
            
            # Текстовый отчёт
            if self.generate_report:
                if self._cancelled:
                    result.cancelled = True
                    return result
                
                report_path = str(output_path / self.report_name)
                video_name = self.video_name or Path(self.csv_path).stem.replace("_detections", "")
                try:
                    generate_report(df, report_path, video_name)
                    result.report_path = report_path
                except Exception as e:
                    print(f"Ошибка при генерации отчёта: {e}")
            
            result.processing_time_s = time.time() - start_time
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return AnalyzeResult(
                success=False,
                processing_time_s=time.time() - start_time,
                error_message=str(e)
            )


class BatchAnalyzeProcessor:
    """
    Пакетный анализатор для нескольких файлов.
    """
    
    def __init__(
        self,
        csv_paths: List[str],
        output_base_dir: str,
        depth_bin: float = 1.0,
        time_bin: float = 10.0,
    ):
        self.csv_paths = csv_paths
        self.output_base_dir = output_base_dir
        self.depth_bin = depth_bin
        self.time_bin = time_bin
        
        self.progress_callback: Optional[callable] = None
        self._cancelled = False
    
    def cancel(self) -> None:
        self._cancelled = True
    
    def is_cancelled(self) -> bool:
        return self._cancelled
    
    def run(self) -> List[AnalyzeResult]:
        """Запускает пакетный анализ."""
        results = []
        
        for i, csv_path in enumerate(self.csv_paths):
            if self._cancelled:
                break
            
            # Создаём подпапку для каждого файла
            csv_name = Path(csv_path).stem.replace("_detections", "")
            output_dir = os.path.join(self.output_base_dir, csv_name)
            
            processor = AnalyzeProcessor(
                csv_path=csv_path,
                output_dir=output_dir,
                depth_bin=self.depth_bin,
                time_bin=self.time_bin,
                video_name=csv_name
            )
            
            result = processor.run()
            results.append(result)
            
            if self.progress_callback:
                self.progress_callback(i + 1, len(self.csv_paths))
        
        return results
