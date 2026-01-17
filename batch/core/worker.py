"""
Worker - QThread для выполнения задач и подзадач в фоновом режиме.
"""

import os
import json
from pathlib import Path
from typing import Optional
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition

from ..database import Repository, Task, SubTask, SubTaskType, TaskStatus, OutputType
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
        
        while not self._stop_requested:
            # Проверяем паузу
            self._mutex.lock()
            while self._pause_requested and not self._stop_requested:
                self._pause_condition.wait(self._mutex)
            self._mutex.unlock()
            
            if self._stop_requested:
                break
            
            # Сначала ищем ожидающие подзадачи (у завершённых родителей)
            pending_subtasks = self.repo.get_pending_subtasks()
            if pending_subtasks:
                subtask = pending_subtasks[0]
                parent = self.repo.get_task(subtask.parent_task_id)
                if parent and parent.status == TaskStatus.DONE:
                    self._execute_subtask(subtask)
                    continue
            
            # Если нет подзадач, ищем основные задачи
            pending_tasks = self.repo.get_pending_tasks()
            if not pending_tasks:
                break
            
            task = pending_tasks[0]
            self._execute_task(task)
        
        self.all_finished.emit()

    def _execute_task(self, task: Task) -> None:
        """Выполняет основную задачу детекции."""
        task_id = task.id
        self._current_task_id = task_id
        
        self.repo.update_task_status(task_id, TaskStatus.RUNNING)
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
        
        self.repo.update_subtask_status(subtask_id, TaskStatus.RUNNING)
        self.started_subtask.emit(subtask_id)
        
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
            
            # Получаем пути к файлам
            outputs = self.repo.get_task_outputs(parent_task.id)
            detections_csv = None
            tracks_csv = None  # Треки из детекции
            track_sizes_csv = None  # Статистика размеров (с колонкой method)
            geometry_csv = None
            size_csv = None
            
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
            
            if not detections_csv or not os.path.exists(detections_csv):
                raise ValueError("Detections CSV not found")
            
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
                    detections_csv, size_csv, output_dir, base_name, params, parent_task.id
                )
            elif subtask.subtask_type == SubTaskType.SIZE_VIDEO_RENDER:
                result_value, result_text = self._run_size_video_render(
                    detected_video, detections_csv, size_csv, geometry_csv,
                    output_dir, base_name, params, parent_task.id
                )
            else:
                raise ValueError(f"Unknown subtask type: {subtask.subtask_type}")
            
            self._current_subtask_id = None
            
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
        
        processor = SizeEstimationProcessor()
        result = processor.process(
            detections_csv=detections_csv,
            output_csv=size_csv,
            tracks_csv=track_sizes_csv,
            geometry_csv=geometry_csv if apply_tilt_correction else None,
            frame_width=video.width or 1920,
            frame_height=video.height or 1080,
            apply_tilt_correction=apply_tilt_correction,
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
        result = processor.process(
            detections_csv=input_csv,
            output_csv=volume_csv,
            tracks_csv=track_sizes_csv,  # Передаём статистику размеров, а не треки детекции
            ctd_csv=ctd_csv,
            fov=params.get("fov", 100.0),
            near_distance=params.get("near_distance", 0.3),
            fps=video.fps or 60.0,
            frame_width=video.width or 1920,
            frame_height=video.height or 1080,
        )
        
        if not result.success:
            raise Exception(result.error_message or "Volume estimation failed")
        
        self.repo.add_task_output(parent_task.id, OutputType.VOLUME_CSV, volume_csv)
        
        return result.total_volume_m3, f"{result.total_volume_m3:.2f} м³" if result.total_volume_m3 else None

    def _run_analysis(self, detections_csv: str, size_csv: Optional[str], output_dir: Path, 
                      base_name: str, params: dict, task_id: int):
        """Выполняет подзадачу анализа."""
        analysis_dir = output_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        # Используем CSV с размерами если есть
        input_csv = size_csv if size_csv and os.path.exists(size_csv) else detections_csv
        
        # Собираем информацию об обработке
        processing_info = self._collect_processing_info(task_id, params)
        
        processor = AnalyzeProcessor()
        result = processor.process(
            csv_path=input_csv,
            output_dir=str(analysis_dir),
            depth_bin=params.get("depth_bin", 2.0),
            video_name=base_name,
            processing_info=processing_info,
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
                'fov': current_params.get('fov', 100.0),
                'near_distance': current_params.get('near_distance', 0.3),
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
        task_id: int
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
        )
        
        if not result.success:
            raise Exception(result.error_message or "Ошибка рендеринга видео")
        
        self.repo.add_task_output(task_id, OutputType.SIZE_VIDEO, output_video)
        
        suffix = " (с геом.)" if effective_geometry_csv else ""
        return None, f"Готово{suffix}"

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
                "fov": 100.0,
                "near_distance": 0.3,
                "depth_bin": 2.0,
            }
        
        # Извлекаем флаги операций
        do_geometry = params.get("geometry", True)
        do_size = params.get("size", True)
        do_size_video = params.get("size_video", False)
        do_volume = params.get("volume", True)
        do_analysis = params.get("analysis", True)
        
        # Общие параметры для всех подзадач
        common_params = {
            "fov": params.get("fov", 100.0),
            "near_distance": params.get("near_distance", 0.3),
            "depth_bin": params.get("depth_bin", 2.0),
        }
        
        position = 0
        
        # Геометрия
        if do_geometry:
            self.repo.create_subtask(
                parent_task_id=task_id,
                subtask_type=SubTaskType.GEOMETRY,
                position=position,
                params_json=json.dumps(common_params),
            )
            position += 1
        
        # Размеры
        if do_size:
            size_params = common_params.copy()
            size_params["use_geometry"] = params.get("size_use_geometry", True)
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
            self.repo.create_subtask(
                parent_task_id=task_id,
                subtask_type=SubTaskType.ANALYSIS,
                position=position,
                params_json=json.dumps(common_params),
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
