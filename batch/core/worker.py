"""
Worker - QThread для выполнения задач и подзадач в фоновом режиме.
"""

import os
import json
from pathlib import Path
from typing import Optional
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition

from ..database import Repository, Task, SubTask, SubTaskType, TaskStatus, OutputType
from .config import get_config
from .processor import Processor, ProcessorFactory, ProcessingResult
from .geometry_processor import GeometryProcessor, SizeEstimationProcessor, VolumeEstimationProcessor
from .analyze_processor import AnalyzeProcessor


class Worker(QThread):
    """
    Воркер для выполнения задач детекции и подзадач постобработки.
    
    Signals:
        started_task: Задача начала выполняться (task_id).
        progress: Обновление прогресса (task_id, percent, current_frame, total_frames, detections, tracks).
        finished_task: Задача завершена (task_id, success, error_message).
        started_subtask: Подзадача начала выполняться (subtask_id).
        subtask_progress: Прогресс подзадачи (subtask_id, percent).
        finished_subtask: Подзадача завершена (subtask_id, success, error_message).
        all_finished: Все задачи в очереди выполнены.
    """
    
    started_task = pyqtSignal(int)
    progress = pyqtSignal(int, float, int, int, int, int)
    finished_task = pyqtSignal(int, bool, str)
    
    started_subtask = pyqtSignal(int)
    subtask_progress = pyqtSignal(int, float)
    finished_subtask = pyqtSignal(int, bool, str)
    
    all_finished = pyqtSignal()

    def __init__(self, repository: Repository, parent=None):
        super().__init__(parent)
        self.repo = repository
        self._stop_requested = False
        self._pause_requested = False
        self._current_processor: Optional[Processor] = None
        self._current_task_id: Optional[int] = None
        self._current_subtask_id: Optional[int] = None
        
        self._mutex = QMutex()
        self._pause_condition = QWaitCondition()

    def run(self):
        """Основной цикл воркера."""
        self._stop_requested = False
        IDLE_RETRIES = 10  # 10 × 500мс = 5 сек ожидания без работы
        idle_count = 0

        try:
            while not self._stop_requested:
                # Проверяем паузу
                self._mutex.lock()
                while self._pause_requested and not self._stop_requested:
                    self._pause_condition.wait(self._mutex)
                self._mutex.unlock()

                if self._stop_requested:
                    break

                # Сначала атомарно берём подзадачу завершённого родителя
                subtask_id = self.repo.claim_pending_subtask()
                if subtask_id is not None:
                    idle_count = 0
                    subtask = self.repo.get_subtask(subtask_id)
                    self._execute_subtask(subtask)
                    continue

                # Затем атомарно берём основную задачу
                task_id = self.repo.claim_pending_task()
                if task_id is not None:
                    idle_count = 0
                    task = self.repo.get_task(task_id)
                    self._execute_task(task)
                    continue

                # Нет работы — ждём: другой воркер может создать подзадачи
                idle_count += 1
                if idle_count >= IDLE_RETRIES:
                    break

                self._mutex.lock()
                self._pause_condition.wait(self._mutex, 500)
                self._mutex.unlock()

        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            self.all_finished.emit()

    def _execute_task(self, task: Task) -> None:
        """Выполняет основную задачу детекции."""
        task_id = task.id
        self._current_task_id = task_id

        # Статус RUNNING уже выставлен атомарно в claim_pending_task()
        self.started_task.emit(task_id)
        
        try:
            with self.repo.get_session() as session:
                from ..database import Task as TaskModel, VideoFile, CTDFile, Model, Dive
                from sqlalchemy.orm import joinedload
                from sqlalchemy import select
                
                stmt = (
                    select(TaskModel)
                    .options(
                        joinedload(TaskModel.video_file).joinedload(VideoFile.dive),
                        joinedload(TaskModel.ctd_file),
                        joinedload(TaskModel.model),
                    )
                    .where(TaskModel.id == task_id)
                )
                task_data = session.scalar(stmt)
                
                if not task_data:
                    raise ValueError(f"Task {task_id} not found")
                
                video_path = task_data.video_file.filepath
                model_path = task_data.model.filepath
                dive_folder = task_data.video_file.dive.folder_path
                ctd_path = task_data.ctd_file.filepath if task_data.ctd_file else None
                auto_postprocess = task_data.auto_postprocess
                auto_postprocess_params = task_data.auto_postprocess_params
                
                task_params = {
                    "conf_threshold": task_data.conf_threshold,
                    "enable_tracking": task_data.enable_tracking,
                    "tracker_type": task_data.tracker_type,
                    "show_trails": task_data.show_trails,
                    "trail_length": task_data.trail_length,
                    "min_track_length": task_data.min_track_length,
                    "depth_rate": task_data.depth_rate,
                    "save_video": task_data.save_video,
                    "export_label_studio": task_data.export_label_studio,
                    "export_ls_interval": task_data.export_ls_interval,
                    "export_ls_classes": task_data.export_ls_classes,
                    "export_ls_dir": get_config().ui.label_studio_dir,
                    # GPU-ускорение (из задачи, fallback — дефолты)
                    "device": task_data.device or "auto",
                    "imgsz": task_data.imgsz or 1280,
                    "half": task_data.half if task_data.half is not None else True,
                }
            
            processor = ProcessorFactory.from_task_data(
                video_path=video_path,
                model_path=model_path,
                dive_folder=dive_folder,
                ctd_path=ctd_path,
                **task_params,
            )
            self._current_processor = processor
            
            def on_progress(current_frame: int, total_frames: int, detections: int, tracks: int):
                if total_frames > 0:
                    percent = (current_frame / total_frames) * 100
                else:
                    percent = 0
                
                self.progress.emit(task_id, percent, current_frame, total_frames, detections, tracks)
                self.repo.update_task_progress(task_id, percent, current_frame)
                
                if self._pause_requested:
                    processor.pause()
                else:
                    processor.resume()
                
                if self._stop_requested:
                    processor.cancel()
            
            processor.progress_callback = on_progress
            result = processor.run()
            
            self._current_processor = None
            self._current_task_id = None
            
            if result.cancelled or self._stop_requested:
                self.repo.update_task_status(task_id, TaskStatus.CANCELLED, "Отменено пользователем")
                self.finished_task.emit(task_id, False, "Отменено пользователем")
                return
            
            if result.success:
                self.repo.update_task(
                    task_id,
                    detections_count=result.detections_count,
                    tracks_count=result.tracks_count,
                    processing_time_s=result.processing_time_s,
                    progress_percent=100.0,
                    class_stats_json=result.class_stats_json,
                )
                self.repo.update_task_status(task_id, TaskStatus.DONE)
                
                if result.output_video_path:
                    self.repo.add_task_output(task_id, OutputType.VIDEO, result.output_video_path)
                if result.output_csv_path:
                    self.repo.add_task_output(task_id, OutputType.CSV, result.output_csv_path)
                if result.output_tracks_path:
                    self.repo.add_task_output(task_id, OutputType.TRACKS_CSV, result.output_tracks_path)
                
                # Автопостобработка
                if auto_postprocess:
                    self._create_auto_postprocess_subtasks(task_id, auto_postprocess_params)
                
                self.finished_task.emit(task_id, True, "")
            else:
                self.repo.update_task_status(task_id, TaskStatus.ERROR, result.error_message)
                self.finished_task.emit(task_id, False, result.error_message or "Unknown error")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._current_processor = None
            self._current_task_id = None
            error_msg = str(e)
            self.repo.update_task_status(task_id, TaskStatus.ERROR, error_msg)
            self.finished_task.emit(task_id, False, error_msg)

    def _execute_subtask(self, subtask: SubTask) -> None:
        """Выполняет подзадачу постобработки."""
        subtask_id = subtask.id
        self._current_subtask_id = subtask_id

        # Не запускаем подзадачу, если уже запрошена остановка
        if self._stop_requested:
            self._current_subtask_id = None
            self.repo.update_subtask_status(subtask_id, TaskStatus.CANCELLED, "Отменено пользователем")
            self.finished_subtask.emit(subtask_id, False, "Отменено пользователем")
            return

        # Статус RUNNING уже выставлен атомарно в claim_pending_subtask()
        self.started_subtask.emit(subtask_id)
        self.subtask_progress.emit(subtask_id, 0.0)

        def on_subtask_progress(current: int, total: int) -> None:
            if total > 0:
                percent = (current / total) * 100
                self.repo.update_subtask_progress(subtask_id, percent)
                self.subtask_progress.emit(subtask_id, percent)

        try:
            parent_task = self.repo.get_task(subtask.parent_task_id)
            if not parent_task:
                raise ValueError(f"Parent task {subtask.parent_task_id} not found")
            
            video = self.repo.get_video_file(parent_task.video_id)
            dive = self.repo.get_dive(video.dive_id) if video else None
            
            if not video or not dive:
                raise ValueError("Video or dive not found")
            
            output_dir = Path(dive.folder_path) / "output"
            output_dir.mkdir(exist_ok=True)
            base_name = Path(video.filename).stem

            params = {}
            if subtask.params_json:
                params = json.loads(subtask.params_json)

            # INFERENCE — особая ветка: запускаем детекцию заново, не требуем
            # существующего detections.csv. Обрабатываем до общей логики поиска
            # путей и финализации (она применяется только к подзадачам
            # постобработки, у которых уже есть результаты детекции).
            if subtask.subtask_type == SubTaskType.INFERENCE:
                self._run_inference_subtask(
                    subtask_id, parent_task, video, dive, params,
                    on_subtask_progress,
                )
                return

            # Получаем пути к файлам
            outputs = self.repo.get_task_outputs(parent_task.id)
            detections_csv = None
            tracks_csv = None  # Треки из детекции
            track_sizes_csv = None  # Статистика размеров (с колонкой method)
            geometry_csv = None
            size_csv = None
            
            volume_csv = None
            for out in outputs:
                if out.output_type == OutputType.CSV:
                    detections_csv = out.filepath
                elif out.output_type == OutputType.TRACKS_CSV:
                    tracks_csv = out.filepath
                elif out.output_type == OutputType.TRACK_SIZES_CSV:
                    track_sizes_csv = out.filepath
                elif out.output_type == OutputType.GEOMETRY_CSV:
                    geometry_csv = out.filepath
                elif out.output_type == OutputType.SIZE_CSV:
                    size_csv = out.filepath
                elif out.output_type == OutputType.VOLUME_CSV:
                    volume_csv = out.filepath
            
            if not detections_csv or not os.path.exists(detections_csv):
                raise ValueError("Detections CSV not found")

            # Disk fallback для geometry.csv: если в БД нет записи, но файл
            # уже лежит в папке output (напр., рассчитан вне GUI или
            # после пересоздания БД) — подхватываем и регистрируем в БД,
            # чтобы последующие подзадачи тоже его видели.
            if not geometry_csv:
                candidate = output_dir / f"{base_name}_geometry.csv"
                if candidate.exists():
                    geometry_csv = str(candidate)
                    try:
                        self.repo.add_task_output(
                            parent_task.id, OutputType.GEOMETRY_CSV, geometry_csv
                        )
                    except Exception:
                        pass

            # Аналогичный fallback для _detections_with_size.csv и _track_sizes.csv,
            # чтобы SIZE_VIDEO_RENDER / VOLUME / ANALYSIS могли их найти,
            # даже если в БД записи нет.
            if not size_csv:
                candidate = output_dir / f"{base_name}_detections_with_size.csv"
                if candidate.exists():
                    size_csv = str(candidate)
                    try:
                        self.repo.add_task_output(
                            parent_task.id, OutputType.SIZE_CSV, size_csv
                        )
                    except Exception:
                        pass
            if not track_sizes_csv:
                candidate = output_dir / f"{base_name}_track_sizes.csv"
                if candidate.exists():
                    track_sizes_csv = str(candidate)
                    try:
                        self.repo.add_task_output(
                            parent_task.id, OutputType.TRACK_SIZES_CSV, track_sizes_csv
                        )
                    except Exception:
                        pass
            
            # Получаем путь к видео с детекциями
            detected_video = None
            for out in outputs:
                if out.output_type == OutputType.VIDEO:
                    detected_video = out.filepath
                    break
            
            # Выполняем в зависимости от типа
            if subtask.subtask_type == SubTaskType.GEOMETRY:
                result_value, result_text = self._run_geometry(
                    video.filepath, output_dir, base_name, video, params, parent_task.id
                )
            elif subtask.subtask_type == SubTaskType.SIZE:
                result_value, result_text = self._run_size(
                    detections_csv, geometry_csv, output_dir, base_name, video, params, parent_task.id
                )
            elif subtask.subtask_type == SubTaskType.VOLUME:
                result_value, result_text = self._run_volume(
                    detections_csv, track_sizes_csv, size_csv, geometry_csv, parent_task, output_dir, base_name, video, params
                )
            elif subtask.subtask_type == SubTaskType.ANALYSIS:
                result_value, result_text = self._run_analysis(
                    detections_csv, size_csv, track_sizes_csv, output_dir, base_name, params, parent_task.id,
                    volume_csv=volume_csv
                )
            elif subtask.subtask_type == SubTaskType.SIZE_VIDEO_RENDER:
                result_value, result_text = self._run_size_video_render(
                    detected_video, detections_csv, size_csv, geometry_csv,
                    output_dir, base_name, params, parent_task.id,
                    progress_callback=on_subtask_progress
                )
            elif subtask.subtask_type == SubTaskType.LABEL_STUDIO_EXPORT:
                result_value, result_text = self._run_label_studio_export(
                    video, detections_csv, output_dir, base_name, params, parent_task.id,
                    progress_callback=on_subtask_progress
                )
            else:
                raise ValueError(f"Unknown subtask type: {subtask.subtask_type}")

            self._current_subtask_id = None

            self.repo.update_subtask_progress(subtask_id, 100.0)
            self.subtask_progress.emit(subtask_id, 100.0)
            self.repo.update_subtask_status(
                subtask_id, TaskStatus.DONE,
                result_value=result_value,
                result_text=result_text
            )
            self.finished_subtask.emit(subtask_id, True, "")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._current_subtask_id = None
            error_msg = str(e)
            self.repo.update_subtask_status(subtask_id, TaskStatus.ERROR, error_message=error_msg)
            self.finished_subtask.emit(subtask_id, False, error_msg)

    def _run_geometry(self, video_path: str, output_dir: Path, base_name: str, video, params: dict, task_id: int):
        """Выполняет подзадачу геометрии."""
        geometry_csv = str(output_dir / f"{base_name}_geometry.csv")
        
        processor = GeometryProcessor()
        result = processor.process(
            video_path=video_path,
            output_csv=geometry_csv,
            frame_interval=params.get("frame_interval", 30),
            frame_width=video.width or 1920,
            frame_height=video.height or 1080,
            frame_step=params.get("frame_step", 1),
        )
        
        if not result.success:
            raise Exception(result.error_message or "Geometry processing failed")
        
        self.repo.add_task_output(task_id, OutputType.GEOMETRY_CSV, geometry_csv)
        
        return result.mean_tilt_deg, f"{result.mean_tilt_deg:.1f}°" if result.mean_tilt_deg else None

    def _run_size(self, detections_csv: str, geometry_csv: Optional[str], output_dir: Path, 
                  base_name: str, video, params: dict, task_id: int):
        """Выполняет подзадачу размеров."""
        size_csv = str(output_dir / f"{base_name}_detections_with_size.csv")
        track_sizes_csv = str(output_dir / f"{base_name}_track_sizes.csv")
        
        # Проверяем параметр коррекции наклона
        use_geometry = params.get("use_geometry", True)
        apply_tilt_correction = use_geometry and geometry_csv and os.path.exists(geometry_csv)

        calibration_json = params.get("calibration_json")
        min_reliable_distance = params.get("min_reliable_distance")
        max_reliable_distance = params.get("max_reliable_distance")

        processor = SizeEstimationProcessor()
        result = processor.process(
            detections_csv=detections_csv,
            output_csv=size_csv,
            tracks_csv=track_sizes_csv,
            geometry_csv=geometry_csv if apply_tilt_correction else None,
            frame_width=video.width or 1920,
            frame_height=video.height or 1080,
            apply_tilt_correction=apply_tilt_correction,
            calibration_json=calibration_json,
            min_reliable_distance=min_reliable_distance,
            max_reliable_distance=max_reliable_distance,
        )
        
        if not result.success:
            raise Exception(result.error_message or "Size estimation failed")
        
        self.repo.add_task_output(task_id, OutputType.SIZE_CSV, size_csv)
        self.repo.add_task_output(task_id, OutputType.TRACK_SIZES_CSV, track_sizes_csv)
        
        # Формируем информативный текст
        parts = []
        if result.tracks_with_k_method > 0:
            parts.append(f"{result.tracks_with_k_method} k-метод")
        if result.tracks_with_fixed > 0:
            parts.append(f"{result.tracks_with_fixed} фикс.")
        if result.tracks_with_parallax > 0:
            parts.append(f"{result.tracks_with_parallax} параллакс")
        if result.tracks_with_typical > 0:
            parts.append(f"{result.tracks_with_typical} тип.")
        
        method_info = ", ".join(parts) if parts else ""
        suffix = " (с геом.)" if result.tilt_correction_applied else ""
        
        result_text = f"{result.total_tracks} тр."
        if method_info:
            result_text += f" ({method_info})"
        result_text += suffix
        
        return float(result.total_tracks), result_text

    def _run_volume(self, detections_csv: str, track_sizes_csv: Optional[str], size_csv: Optional[str],
                    geometry_csv: Optional[str], parent_task, output_dir: Path, base_name: str, video, params: dict):
        """
        Выполняет подзадачу объёма.
        
        Args:
            detections_csv: CSV с детекциями
            track_sizes_csv: CSV со статистикой размеров (из size estimation, содержит колонку 'method')
            size_csv: CSV с детекциями + размерами
            geometry_csv: CSV с данными геометрии (наклон камеры)
            parent_task: Родительская задача
            output_dir: Папка для выходных файлов
            base_name: Базовое имя файла
            video: Объект видеофайла
            params: Параметры
        """
        volume_csv = str(output_dir / f"{base_name}_volume.csv")
        
        # Используем CSV с размерами если есть
        input_csv = size_csv if size_csv and os.path.exists(size_csv) else detections_csv
        
        # CTD
        ctd_csv = None
        if parent_task.ctd_id:
            ctd_file = self.repo.get_ctd_file(parent_task.ctd_id)
            if ctd_file:
                ctd_csv = ctd_file.filepath
        
        processor = VolumeEstimationProcessor()
        effective_distance_auto = params.get("effective_distance_auto", True)
        detection_distance = None if effective_distance_auto else params.get("detection_distance")
        fov_horizontal = params.get("fov_horizontal", params.get("fov", 95.0))
        fov_vertical = params.get("fov_vertical", 55.0)
        result = processor.process(
            detections_csv=input_csv,
            output_csv=volume_csv,
            tracks_csv=track_sizes_csv,  # Передаём статистику размеров, а не треки детекции
            ctd_csv=ctd_csv,
            fov_horizontal=fov_horizontal,
            fov_vertical=fov_vertical,
            near_distance=params.get("min_reliable_distance", 0.1),
            detection_distance=detection_distance,
            fps=video.fps or 60.0,
            frame_width=video.width or 1920,
            frame_height=video.height or 1080,
            calibration_json=params.get("calibration_json"),
            min_reliable_distance=params.get("min_reliable_distance"),
            max_reliable_distance=params.get("max_reliable_distance"),
        )
        
        if not result.success:
            raise Exception(result.error_message or "Volume estimation failed")
        
        self.repo.add_task_output(parent_task.id, OutputType.VOLUME_CSV, volume_csv)
        
        return result.total_volume_m3, f"{result.total_volume_m3:.2f} м³" if result.total_volume_m3 else None

    def _run_analysis(self, detections_csv: str, size_csv: Optional[str], track_sizes_csv: Optional[str],
                      output_dir: Path, base_name: str, params: dict, task_id: int, volume_csv: Optional[str] = None):
        """Выполняет подзадачу анализа."""
        analysis_dir = output_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)

        # Используем CSV с размерами если есть
        input_csv = size_csv if size_csv and os.path.exists(size_csv) else detections_csv

        # Собираем информацию об обработке
        processing_info = self._collect_processing_info(task_id, params)

        # Читаем плотность по видам из volume CSV
        if volume_csv and os.path.exists(volume_csv):
            try:
                import pandas as pd
                vol_df = pd.read_csv(volume_csv)
                density = {}
                cross_section_area_m2 = None
                for _, row in vol_df.iterrows():
                    param = str(row['parameter'])
                    if param == 'cross_section_area_m2':
                        cross_section_area_m2 = float(row['value'])
                    if param.startswith('density_') and param.endswith('_per_m3'):
                        species_key = param[len('density_'):-len('_per_m3')].replace('_', ' ')
                        density[species_key] = float(row['value'])
                if density:
                    processing_info['volume_density'] = density
                if cross_section_area_m2 and cross_section_area_m2 > 0:
                    processing_info['cross_section_area_m2'] = cross_section_area_m2
            except Exception as e:
                print(f"Не удалось прочитать данные объёма: {e}")

        # Путь к CTD-файлу из задачи
        ctd_path = None
        task = self.repo.get_task(task_id)
        if task and task.ctd_id:
            ctd_file = self.repo.get_ctd_file(task.ctd_id)
            if ctd_file:
                ctd_path = ctd_file.filepath

        # Колонки CTD (0-based индексы), по умолчанию 6, 11, 12, 16
        ctd_columns_raw = params.get("ctd_columns", "6,11,12,16")
        if isinstance(ctd_columns_raw, list):
            ctd_columns = [int(x) for x in ctd_columns_raw]
        elif isinstance(ctd_columns_raw, str) and ctd_columns_raw.strip():
            ctd_columns = [int(x.strip()) for x in ctd_columns_raw.split(',') if x.strip().isdigit()]
        else:
            ctd_columns = [6, 11, 12, 16]

        processor = AnalyzeProcessor()
        result = processor.process(
            csv_path=input_csv,
            output_dir=str(analysis_dir),
            depth_bin=params.get("depth_bin", 2.0),
            video_name=base_name,
            processing_info=processing_info,
            generate_interactive_plot=True,
            track_sizes_path=track_sizes_csv if track_sizes_csv and os.path.exists(track_sizes_csv) else None,
            ctd_path=ctd_path,
            ctd_columns=ctd_columns if ctd_path else None,
            cross_section_area_m2=processing_info.get('cross_section_area_m2'),
        )

        if not result.success:
            raise Exception(result.error_message or "Analysis failed")

        # Сохраняем outputs
        for plot_file in ["vertical_distribution.png", "detection_timeline.png", "species_summary.png"]:
            plot_path = analysis_dir / plot_file
            if plot_path.exists():
                self.repo.add_task_output(task_id, OutputType.ANALYSIS_PLOT, str(plot_path))

        report_path = analysis_dir / "report.txt"
        if report_path.exists():
            self.repo.add_task_output(task_id, OutputType.ANALYSIS_REPORT, str(report_path))

        if result.interactive_plot_path and Path(result.interactive_plot_path).exists():
            self.repo.add_task_output(task_id, OutputType.INTERACTIVE_PLOT, result.interactive_plot_path)

        return float(result.total_detections), f"{result.total_detections} дет."

    def _collect_processing_info(self, task_id: int, current_params: dict) -> dict:
        """
        Собирает информацию об обработке для отчёта.
        
        Args:
            task_id: ID задачи.
            current_params: Текущие параметры подзадачи.
            
        Returns:
            Словарь с информацией об обработке.
        """
        processing_info = {}
        
        try:
            # Получаем информацию о задаче
            task = self.repo.get_task(task_id)
            if not task:
                return processing_info
            
            # Параметры детекции
            model = self.repo.get_model(task.model_id)
            ctd_file = self.repo.get_ctd_file(task.ctd_id) if task.ctd_id else None
            
            detection_params = {
                'model_name': model.name if model else 'N/A',
                'conf_threshold': task.conf_threshold,
                'enable_tracking': task.enable_tracking,
                'tracker_type': task.tracker_type.replace('.yaml', '') if task.tracker_type else None,
                'min_track_length': task.min_track_length,
                'depth_rate': task.depth_rate,
                'ctd_file': ctd_file.filename if ctd_file else None,
            }
            processing_info['detection_params'] = detection_params
            
            # Параметры постобработки
            postprocess_params = {
                'fov_horizontal': current_params.get('fov_horizontal', current_params.get('fov', 95.0)),
                'fov_vertical': current_params.get('fov_vertical', 55.0),
                'near_distance': current_params.get('min_reliable_distance', 0.1),
                'min_reliable_distance': current_params.get('min_reliable_distance', 0.1),
                'max_reliable_distance': current_params.get('max_reliable_distance'),
                'effective_distance_auto': current_params.get('effective_distance_auto', True),
                'detection_distance': current_params.get('detection_distance'),
                'depth_bin': current_params.get('depth_bin', 2.0),
            }
            processing_info['postprocess_params'] = postprocess_params
            
            # Информация о выполненных подзадачах
            subtasks = self.repo.get_subtasks_for_task(task_id)
            subtasks_info = []
            total_postprocess_time = 0.0
            
            for st in subtasks:
                # Пропускаем текущую подзадачу (анализ) - она ещё выполняется
                if st.subtask_type == SubTaskType.ANALYSIS:
                    continue
                
                # Вычисляем время выполнения
                proc_time = None
                if st.started_at and st.completed_at:
                    proc_time = (st.completed_at - st.started_at).total_seconds()
                    total_postprocess_time += proc_time
                
                subtasks_info.append({
                    'name': st.type_name,
                    'success': st.status == TaskStatus.DONE,
                    'processing_time_s': proc_time,
                    'result_text': st.result_text,
                })
            
            if subtasks_info:
                processing_info['subtasks'] = subtasks_info
            
            # Временные метки
            timing = {}
            
            # Время детекции
            if task.processing_time_s:
                timing['detection_time_s'] = task.processing_time_s
            
            # Время постобработки (без анализа)
            if total_postprocess_time > 0:
                timing['postprocess_time_s'] = total_postprocess_time
            
            # Общее время
            if task.started_at and task.completed_at:
                detection_time = (task.completed_at - task.started_at).total_seconds()
                timing['total_time_s'] = detection_time + total_postprocess_time
            
            if timing:
                processing_info['timing'] = timing
                
        except Exception as e:
            print(f"Ошибка при сборе информации об обработке: {e}")
        
        return processing_info

    def _run_size_video_render(
        self,
        detected_video: Optional[str],
        detections_csv: str,
        size_csv: Optional[str],
        geometry_csv: Optional[str],
        output_dir: Path,
        base_name: str,
        params: dict,
        task_id: int,
        progress_callback=None,
    ):
        """
        Выполняет подзадачу рендеринга видео с размерами.
        
        Args:
            detected_video: Путь к видео с детекциями
            detections_csv: CSV с детекциями
            size_csv: CSV с размерами
            geometry_csv: CSV с геометрией
            output_dir: Папка для выходных файлов
            base_name: Базовое имя файла
            params: Параметры
            task_id: ID задачи
        """
        from .geometry_processor import SizeVideoRenderProcessor
        
        if not detected_video or not os.path.exists(detected_video):
            raise ValueError("Видео с детекциями не найдено")
        
        if not size_csv or not os.path.exists(size_csv):
            raise ValueError("CSV с размерами не найден. Сначала выполните оценку размеров.")
        
        output_video = str(output_dir / f"{base_name}_sized.mp4")
        
        # Проверяем параметр использования геометрии
        use_geometry = params.get("use_geometry", True)
        effective_geometry_csv = geometry_csv if (use_geometry and geometry_csv and os.path.exists(geometry_csv)) else None
        
        processor = SizeVideoRenderProcessor()
        result = processor.process(
            input_video=detected_video,
            detections_csv=detections_csv,
            size_csv=size_csv,
            output_video=output_video,
            geometry_csv=effective_geometry_csv,
            progress_callback=progress_callback,
        )
        
        if not result.success:
            raise Exception(result.error_message or "Ошибка рендеринга видео")
        
        self.repo.add_task_output(task_id, OutputType.SIZE_VIDEO, output_video)
        
        suffix = " (с геом.)" if effective_geometry_csv else ""
        return None, f"Готово{suffix}"

    def _run_label_studio_export(
        self,
        video,
        detections_csv: str,
        output_dir: Path,
        base_name: str,
        params: dict,
        task_id: int,
        progress_callback=None,
    ):
        """
        Выполняет подзадачу экспорта кадров и предразметки для Label Studio.

        Args:
            video: Объект VideoFile
            detections_csv: CSV с детекциями
            output_dir: Папка output задачи (fallback)
            base_name: Базовое имя файла
            params: Параметры из params_json (interval, classes, output_dir)
            task_id: ID задачи
            progress_callback: (current, total) → None
        """
        import sys
        import cv2
        import pandas as pd

        src_dir = str(Path(__file__).parent.parent.parent / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from label_studio_export import LabelStudioExporter

        frame_interval = params.get("frame_interval", 15)
        export_classes_list = params.get("export_classes")  # list | None
        export_classes = set(export_classes_list) if export_classes_list else None
        image_quality = params.get("image_quality", 95)

        # Папка экспорта: из параметров или fallback
        ls_output_dir = params.get("output_dir")
        if not ls_output_dir:
            dive = self.repo.get_dive(video.dive_id) if video else None
            if dive and dive.folder_path:
                ls_output_dir = str(Path(dive.folder_path) / "label_studio_export")
            else:
                ls_output_dir = str(output_dir / "label_studio_export")

        df = pd.read_csv(detections_csv)
        if df.empty:
            raise ValueError("CSV детекций пуст")

        video_name = Path(video.filepath).stem
        exporter = LabelStudioExporter(
            output_dir=ls_output_dir,
            video_name=video_name,
            frame_interval=frame_interval,
            export_classes=export_classes,
            image_quality=image_quality,
        )

        cap = cv2.VideoCapture(video.filepath)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video.filepath}")

        try:
            grouped = df.groupby("frame")
            total = len(grouped)
            prev_frame_num = -1

            for i, (frame_num, group) in enumerate(grouped):
                if self._stop_requested:
                    raise Exception("Отменено пользователем")

                self._mutex.lock()
                while self._pause_requested and not self._stop_requested:
                    self._pause_condition.wait(self._mutex)
                self._mutex.unlock()

                frame_num = int(frame_num)
                if frame_num != prev_frame_num + 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    prev_frame_num = frame_num
                    continue
                prev_frame_num = frame_num

                timestamp = float(group.iloc[0].get("timestamp_s", 0))

                detections = []
                for _, row in group.iterrows():
                    det = {
                        "class_name": row["class_name"],
                        "confidence": float(row["confidence"]),
                        "x_center": float(row["x_center"]),
                        "y_center": float(row["y_center"]),
                        "width": float(row["width"]),
                        "height": float(row["height"]),
                        "track_id": (
                            int(row["track_id"])
                            if pd.notna(row.get("track_id"))
                            else None
                        ),
                    }
                    detections.append(det)

                exporter.process_frame(frame, frame_num, timestamp, detections)

                if progress_callback:
                    progress_callback(i + 1, total)

            json_path = exporter.finalize()
        finally:
            cap.release()

        self.repo.add_task_output(task_id, OutputType.LABEL_STUDIO_JSON, json_path)

        frames_exported = len(exporter._annotations)
        return frames_exported, f"{frames_exported} кадров"

    def _create_auto_postprocess_subtasks(self, task_id: int, params_json: Optional[str]) -> None:
        """
        Создаёт подзадачи автопостобработки с учётом параметров.
        
        Args:
            task_id: ID родительской задачи.
            params_json: JSON строка с параметрами постобработки.
        """
        # Парсим параметры
        if params_json:
            params = json.loads(params_json)
        else:
            # Параметры по умолчанию (совместимость со старым кодом)
            params = {
                "geometry": True,
                "size": True,
                "size_use_geometry": True,
                "size_video": False,
                "video_use_geometry": True,
                "volume": True,
                "analysis": True,
                "fov_horizontal": 95.0,
                "fov_vertical": 55.0,
                "depth_bin": 2.0,
            }
        
        # Извлекаем флаги операций
        do_geometry = params.get("geometry", True)
        do_size = params.get("size", True)
        do_size_video = params.get("size_video", False)
        do_volume = params.get("volume", True)
        do_analysis = params.get("analysis", True)
        
        # Общие параметры для всех подзадач
        defaults = None
        try:
            from .calibration_defaults import get_calibration_defaults
            defaults = get_calibration_defaults(params.get("calibration_json"))
        except Exception:
            defaults = {}

        common_params = {
            "fov_horizontal": params.get("fov_horizontal", params.get("fov", 95.0)),
            "fov_vertical": params.get("fov_vertical", 55.0),
            "min_reliable_distance": params.get(
                "min_reliable_distance", defaults.get("min_reliable_distance", 0.1)
            ),
            "max_reliable_distance": params.get(
                "max_reliable_distance", defaults.get("max_reliable_distance", 3.0)
            ),
            "effective_distance_auto": params.get(
                "effective_distance_auto", defaults.get("effective_distance_auto", True)
            ),
            "detection_distance": params.get(
                "detection_distance", defaults.get("effective_distance")
            ),
            "depth_bin": params.get("depth_bin", 2.0),
        }
        if params.get("calibration_json"):
            common_params["calibration_json"] = params.get("calibration_json")
        try:
            from .config import get_config
            calib_path = get_config().ui.calibration_json
            if "calibration_json" not in common_params and calib_path and os.path.exists(calib_path):
                common_params["calibration_json"] = calib_path
        except Exception:
            pass
        
        position = 0
        
        # Геометрия
        if do_geometry:
            geom_params = common_params.copy()
            geom_params["frame_step"] = params.get("frame_step", 1)
            self.repo.create_subtask(
                parent_task_id=task_id,
                subtask_type=SubTaskType.GEOMETRY,
                position=position,
                params_json=json.dumps(geom_params),
            )
            position += 1
        
        # Размеры
        if do_size:
            size_params = common_params.copy()
            size_params["use_geometry"] = params.get("size_use_geometry", True)
            # Читаем файл калибровки из глобального конфига
            try:
                from .config import get_config
                calib_path = get_config().ui.calibration_json
                if calib_path and os.path.exists(calib_path):
                    size_params["calibration_json"] = calib_path
            except Exception:
                pass
            self.repo.create_subtask(
                parent_task_id=task_id,
                subtask_type=SubTaskType.SIZE,
                position=position,
                params_json=json.dumps(size_params),
            )
            position += 1
        
        # Видео с размерами
        if do_size_video:
            video_params = common_params.copy()
            video_params["use_geometry"] = params.get("video_use_geometry", True)
            self.repo.create_subtask(
                parent_task_id=task_id,
                subtask_type=SubTaskType.SIZE_VIDEO_RENDER,
                position=position,
                params_json=json.dumps(video_params),
            )
            position += 1
        
        # Объём
        if do_volume:
            self.repo.create_subtask(
                parent_task_id=task_id,
                subtask_type=SubTaskType.VOLUME,
                position=position,
                params_json=json.dumps(common_params),
            )
            position += 1
        
        # Анализ
        if do_analysis:
            analysis_params = common_params.copy()
            analysis_params["ctd_columns"] = params.get("ctd_columns", "6,11,12,16")
            self.repo.create_subtask(
                parent_task_id=task_id,
                subtask_type=SubTaskType.ANALYSIS,
                position=position,
                params_json=json.dumps(analysis_params),
            )

    def stop(self) -> None:
        """Останавливает воркер."""
        self._stop_requested = True
        
        self._mutex.lock()
        self._pause_requested = False
        self._pause_condition.wakeAll()
        self._mutex.unlock()
        
        if self._current_processor:
            self._current_processor.cancel()

    def pause(self) -> None:
        """Приостанавливает воркер."""
        self._mutex.lock()
        self._pause_requested = True
        self._mutex.unlock()
        
        if self._current_processor:
            self._current_processor.pause()

    def resume(self) -> None:
        """Возобновляет воркер."""
        self._mutex.lock()
        self._pause_requested = False
        self._pause_condition.wakeAll()
        self._mutex.unlock()
        
        if self._current_processor:
            self._current_processor.resume()

    def is_paused(self) -> bool:
        return self._pause_requested
    
    def get_current_task_id(self) -> Optional[int]:
        return self._current_task_id

    def get_current_subtask_id(self) -> Optional[int]:
        return self._current_subtask_id

    def _run_inference_subtask(
        self,
        subtask_id: int,
        parent_task,
        video,
        dive,
        params: dict,
        on_subtask_progress,
    ) -> None:
        """Запускает повторный инференс по параметрам подзадачи INFERENCE.

        params (из subtask.params_json) — overrides поверх Task'овых полей.
        None в значении = «оставить как у задачи».
        """
        from ..database import Model as ModelEntity
        from sqlalchemy.orm import joinedload
        from sqlalchemy import select as sa_select

        task_id = parent_task.id
        cascade_invalidate = bool(params.get("cascade_invalidate", True))

        try:
            with self.repo.get_session() as session:
                from ..database import Task as TaskModel, VideoFile as VF

                stmt = (
                    sa_select(TaskModel)
                    .options(
                        joinedload(TaskModel.video_file).joinedload(VF.dive),
                        joinedload(TaskModel.ctd_file),
                        joinedload(TaskModel.model),
                    )
                    .where(TaskModel.id == task_id)
                )
                task_data = session.scalar(stmt)
                if not task_data:
                    raise ValueError(f"Task {task_id} not found")

                video_path = task_data.video_file.filepath
                dive_folder = task_data.video_file.dive.folder_path
                ctd_path = task_data.ctd_file.filepath if task_data.ctd_file else None
                original_model_path = task_data.model.filepath

                # Базовые параметры — из Task; затем accumulate overrides.
                effective = {
                    "conf_threshold": task_data.conf_threshold,
                    "enable_tracking": task_data.enable_tracking,
                    "tracker_type": task_data.tracker_type,
                    "show_trails": task_data.show_trails,
                    "trail_length": task_data.trail_length,
                    "min_track_length": task_data.min_track_length,
                    "depth_rate": task_data.depth_rate,
                    "save_video": task_data.save_video,
                    "export_label_studio": task_data.export_label_studio,
                    "export_ls_interval": task_data.export_ls_interval,
                    "export_ls_classes": task_data.export_ls_classes,
                    "export_ls_dir": get_config().ui.label_studio_dir,
                    "device": task_data.device or "auto",
                    "imgsz": task_data.imgsz or 1280,
                    "half": task_data.half if task_data.half is not None else True,
                }
                override_keys = (
                    "conf_threshold", "enable_tracking", "tracker_type",
                    "show_trails", "trail_length", "min_track_length",
                    "device", "imgsz", "half",
                )
                for key in override_keys:
                    if params.get(key) is not None:
                        effective[key] = params[key]

                # Override модели — загружаем по model_id.
                model_path = original_model_path
                model_override_id = params.get("model_id")
                if model_override_id is not None and model_override_id != task_data.model_id:
                    new_model = session.get(ModelEntity, model_override_id)
                    if new_model is None:
                        raise ValueError(f"Model {model_override_id} not found")
                    model_path = new_model.filepath

            processor = ProcessorFactory.from_task_data(
                video_path=video_path,
                model_path=model_path,
                dive_folder=dive_folder,
                ctd_path=ctd_path,
                **effective,
            )
            self._current_processor = processor

            # Проксируем прогресс: подзадаче нужен percent (0..100).
            def on_progress(current_frame: int, total_frames: int,
                            detections: int, tracks: int):
                if total_frames > 0:
                    percent = (current_frame / total_frames) * 100.0
                else:
                    percent = 0.0
                on_subtask_progress(current_frame, total_frames)

                if self._pause_requested:
                    processor.pause()
                else:
                    processor.resume()

                if self._stop_requested:
                    processor.cancel()

            processor.progress_callback = on_progress
            result = processor.run()
            self._current_processor = None

            if result.cancelled or self._stop_requested:
                self._current_subtask_id = None
                self.repo.update_subtask_status(
                    subtask_id, TaskStatus.CANCELLED,
                    error_message="Отменено пользователем",
                )
                self.finished_subtask.emit(subtask_id, False, "Отменено пользователем")
                return

            if not result.success:
                self._current_subtask_id = None
                err = result.error_message or "Inference failed"
                self.repo.update_subtask_status(
                    subtask_id, TaskStatus.ERROR, error_message=err,
                )
                self.finished_subtask.emit(subtask_id, False, err)
                return

            # Транзакционно применяем результат: обновляем Output'ы, поля Task,
            # каскадно очищаем downstream при cascade_invalidate=True.
            outputs_map = {}
            if result.output_video_path:
                outputs_map[OutputType.VIDEO] = result.output_video_path
            if result.output_csv_path:
                outputs_map[OutputType.CSV] = result.output_csv_path
            if result.output_tracks_path:
                outputs_map[OutputType.TRACKS_CSV] = result.output_tracks_path

            task_overrides = {
                "model_id": model_override_id,
                "conf_threshold": params.get("conf_threshold"),
                "enable_tracking": params.get("enable_tracking"),
                "tracker_type": params.get("tracker_type"),
                "show_trails": params.get("show_trails"),
                "trail_length": params.get("trail_length"),
                "min_track_length": params.get("min_track_length"),
                "device": params.get("device"),
                "imgsz": params.get("imgsz"),
                "half": params.get("half"),
            }
            result_stats = {
                "detections_count": result.detections_count,
                "tracks_count": result.tracks_count,
                "processing_time_s": result.processing_time_s,
                "class_stats_json": result.class_stats_json,
            }
            self.repo.apply_inference_result(
                task_id=task_id,
                result_outputs=outputs_map,
                task_field_overrides=task_overrides,
                result_stats=result_stats,
                cascade_invalidate=cascade_invalidate,
            )

            self._current_subtask_id = None
            self.repo.update_subtask_progress(subtask_id, 100.0)
            self.subtask_progress.emit(subtask_id, 100.0)

            result_text = (
                f"{result.detections_count or 0} дет., "
                f"{result.tracks_count or 0} тр."
            )
            self.repo.update_subtask_status(
                subtask_id, TaskStatus.DONE,
                result_value=float(result.detections_count or 0),
                result_text=result_text,
            )
            self.finished_subtask.emit(subtask_id, True, "")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._current_processor = None
            self._current_subtask_id = None
            error_msg = str(e)
            self.repo.update_subtask_status(
                subtask_id, TaskStatus.ERROR, error_message=error_msg,
            )
            self.finished_subtask.emit(subtask_id, False, error_msg)
