"""
Repository - CRUD операции для работы с базой данных.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
from sqlalchemy import create_engine, select, update, delete, func
from sqlalchemy.orm import Session, sessionmaker

from .models import (
    Base,
    Catalog,
    Dive,
    VideoFile,
    CTDFile,
    Model,
    Task,
    SubTask,
    SubTaskType,
    TaskOutput,
    TaskStatus,
    OutputType,
)


class Repository:
    """
    Репозиторий для работы с базой данных.
    Предоставляет CRUD операции для всех сущностей.
    """

    def __init__(self, db_path: str):
        """
        Инициализация репозитория.
        
        Args:
            db_path: Путь к файлу SQLite базы данных.
        """
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Создаём таблицы, если их нет
        Base.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        """Создаёт новую сессию."""
        return self.SessionLocal()

    # ========== CATALOG OPERATIONS ==========

    def create_catalog(
        self,
        name: str,
        description: Optional[str] = None,
        color: Optional[str] = None,
    ) -> Catalog:
        """Создаёт новый каталог (экспедицию)."""
        with self.get_session() as session:
            # Определяем позицию
            stmt = select(func.max(Catalog.position))
            max_pos = session.scalar(stmt) or 0
            
            catalog = Catalog(
                name=name,
                description=description,
                color=color,
                position=max_pos + 1,
            )
            session.add(catalog)
            session.commit()
            session.refresh(catalog)
            return catalog

    def get_catalog(self, catalog_id: int) -> Optional[Catalog]:
        """Получает каталог по ID."""
        with self.get_session() as session:
            return session.get(Catalog, catalog_id)

    def get_all_catalogs(self) -> List[Catalog]:
        """Получает все каталоги."""
        with self.get_session() as session:
            stmt = select(Catalog).order_by(Catalog.position)
            return list(session.scalars(stmt))

    def update_catalog(self, catalog_id: int, **kwargs) -> Optional[Catalog]:
        """Обновляет каталог."""
        with self.get_session() as session:
            catalog = session.get(Catalog, catalog_id)
            if catalog:
                for key, value in kwargs.items():
                    if hasattr(catalog, key):
                        setattr(catalog, key, value)
                session.commit()
                session.refresh(catalog)
            return catalog

    def delete_catalog(self, catalog_id: int) -> bool:
        """Удаляет каталог. Погружения остаются без каталога."""
        with self.get_session() as session:
            catalog = session.get(Catalog, catalog_id)
            if catalog:
                # Отвязываем погружения
                stmt = update(Dive).where(Dive.catalog_id == catalog_id).values(catalog_id=None)
                session.execute(stmt)
                session.delete(catalog)
                session.commit()
                return True
            return False

    def get_dives_by_catalog(self, catalog_id: Optional[int]) -> List[Dive]:
        """Получает погружения каталога. None = погружения без каталога."""
        with self.get_session() as session:
            if catalog_id is None:
                stmt = select(Dive).where(Dive.catalog_id.is_(None)).order_by(Dive.position)
            else:
                stmt = select(Dive).where(Dive.catalog_id == catalog_id).order_by(Dive.position)
            return list(session.scalars(stmt))

    def move_dive_to_catalog(self, dive_id: int, catalog_id: Optional[int]) -> bool:
        """Перемещает погружение в каталог. None = без каталога."""
        with self.get_session() as session:
            dive = session.get(Dive, dive_id)
            if dive:
                dive.catalog_id = catalog_id
                session.commit()
                return True
            return False

    # ========== DIVE OPERATIONS ==========

    def create_dive(
        self,
        name: str,
        folder_path: str,
        date: Optional[datetime] = None,
        location: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Dive:
        """
        Создаёт новое погружение.
        
        Args:
            name: Название погружения.
            folder_path: Путь к папке с данными.
            date: Дата погружения.
            location: Место погружения.
            notes: Заметки.
        
        Returns:
            Созданный объект Dive.
        """
        with self.get_session() as session:
            dive = Dive(
                name=name,
                folder_path=str(Path(folder_path).resolve()),
                date=date,
                location=location,
                notes=notes,
            )
            session.add(dive)
            session.commit()
            session.refresh(dive)
            return dive

    def get_dive(self, dive_id: int) -> Optional[Dive]:
        """Получает погружение по ID."""
        with self.get_session() as session:
            return session.get(Dive, dive_id)

    def get_dive_by_path(self, folder_path: str) -> Optional[Dive]:
        """Получает погружение по пути к папке."""
        with self.get_session() as session:
            stmt = select(Dive).where(
                Dive.folder_path == str(Path(folder_path).resolve())
            )
            return session.scalar(stmt)

    def get_all_dives(self) -> List[Dive]:
        """Получает все погружения."""
        with self.get_session() as session:
            stmt = select(Dive).order_by(Dive.created_at.desc())
            return list(session.scalars(stmt))

    def update_dive(self, dive_id: int, **kwargs) -> Optional[Dive]:
        """Обновляет погружение."""
        with self.get_session() as session:
            dive = session.get(Dive, dive_id)
            if dive:
                for key, value in kwargs.items():
                    if hasattr(dive, key):
                        setattr(dive, key, value)
                session.commit()
                session.refresh(dive)
            return dive

    def delete_dive(self, dive_id: int) -> bool:
        """Удаляет погружение и все связанные данные."""
        with self.get_session() as session:
            dive = session.get(Dive, dive_id)
            if dive:
                session.delete(dive)
                session.commit()
                return True
            return False

    # ========== VIDEO FILE OPERATIONS ==========

    def add_video_file(
        self,
        dive_id: int,
        filepath: str,
        extract_metadata: bool = True,
    ) -> Optional[VideoFile]:
        """
        Добавляет видеофайл к погружению.
        
        Args:
            dive_id: ID погружения.
            filepath: Путь к видеофайлу.
            extract_metadata: Извлечь метаданные из видео.
        
        Returns:
            Созданный объект VideoFile или None.
        """
        filepath = str(Path(filepath).resolve())
        
        if not os.path.exists(filepath):
            return None

        with self.get_session() as session:
            # Проверяем, что погружение существует
            dive = session.get(Dive, dive_id)
            if not dive:
                return None

            # Проверяем, не добавлен ли уже этот файл
            stmt = select(VideoFile).where(VideoFile.filepath == filepath)
            existing = session.scalar(stmt)
            if existing:
                return existing

            # Извлекаем метаданные
            metadata = {}
            if extract_metadata:
                metadata = self._extract_video_metadata(filepath)

            video = VideoFile(
                dive_id=dive_id,
                filename=os.path.basename(filepath),
                filepath=filepath,
                **metadata,
            )
            session.add(video)
            session.commit()
            session.refresh(video)
            return video

    def _extract_video_metadata(self, filepath: str) -> Dict[str, Any]:
        """Извлекает метаданные из видеофайла."""
        metadata = {}
        try:
            cap = cv2.VideoCapture(filepath)
            if cap.isOpened():
                metadata["fps"] = cap.get(cv2.CAP_PROP_FPS)
                metadata["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                metadata["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                metadata["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if metadata["fps"] > 0:
                    metadata["duration_s"] = metadata["frame_count"] / metadata["fps"]
                
                # Размер файла
                metadata["filesize_mb"] = os.path.getsize(filepath) / (1024 * 1024)
                
                cap.release()
        except Exception:
            pass
        return metadata

    def get_video_file(self, video_id: int) -> Optional[VideoFile]:
        """Получает видеофайл по ID."""
        with self.get_session() as session:
            return session.get(VideoFile, video_id)

    def get_videos_by_dive(self, dive_id: int) -> List[VideoFile]:
        """Получает все видеофайлы погружения."""
        with self.get_session() as session:
            stmt = select(VideoFile).where(VideoFile.dive_id == dive_id)
            return list(session.scalars(stmt))

    def delete_video_file(self, video_id: int) -> bool:
        """Удаляет видеофайл из базы."""
        with self.get_session() as session:
            video = session.get(VideoFile, video_id)
            if video:
                session.delete(video)
                session.commit()
                return True
            return False

    # ========== CTD FILE OPERATIONS ==========

    def add_ctd_file(
        self,
        dive_id: int,
        filepath: str,
        extract_metadata: bool = True,
    ) -> Optional[CTDFile]:
        """
        Добавляет файл CTD к погружению.
        
        Args:
            dive_id: ID погружения.
            filepath: Путь к файлу CTD.
            extract_metadata: Извлечь метаданные из файла.
        
        Returns:
            Созданный объект CTDFile или None.
        """
        filepath = str(Path(filepath).resolve())
        
        if not os.path.exists(filepath):
            return None

        with self.get_session() as session:
            # Проверяем, что погружение существует
            dive = session.get(Dive, dive_id)
            if not dive:
                return None

            # Проверяем, не добавлен ли уже этот файл
            stmt = select(CTDFile).where(CTDFile.filepath == filepath)
            existing = session.scalar(stmt)
            if existing:
                return existing

            # Извлекаем метаданные
            metadata = {}
            if extract_metadata:
                metadata = self._extract_ctd_metadata(filepath)

            ctd = CTDFile(
                dive_id=dive_id,
                filename=os.path.basename(filepath),
                filepath=filepath,
                **metadata,
            )
            session.add(ctd)
            session.commit()
            session.refresh(ctd)
            return ctd

    def _extract_ctd_metadata(self, filepath: str) -> Dict[str, Any]:
        """Извлекает метаданные из файла CTD."""
        metadata = {}
        try:
            import pandas as pd
            
            # Определяем разделитель
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline()
            
            for sep in ['|', ';', '\t', ',']:
                if sep in first_line:
                    delimiter = sep
                    break
            else:
                delimiter = ','
            
            df = pd.read_csv(filepath, sep=delimiter)
            df.columns = df.columns.str.lower()
            
            metadata["records_count"] = len(df)
            
            if "depth" in df.columns:
                metadata["max_depth"] = float(df["depth"].max())
                metadata["min_depth"] = float(df["depth"].min())
            
            if "time" in df.columns:
                metadata["duration_s"] = float(df["time"].max() - df["time"].min())
                
        except Exception:
            pass
        return metadata

    def get_ctd_file(self, ctd_id: int) -> Optional[CTDFile]:
        """Получает файл CTD по ID."""
        with self.get_session() as session:
            return session.get(CTDFile, ctd_id)

    def get_ctd_by_dive(self, dive_id: int) -> List[CTDFile]:
        """Получает все файлы CTD погружения."""
        with self.get_session() as session:
            stmt = select(CTDFile).where(CTDFile.dive_id == dive_id)
            return list(session.scalars(stmt))

    def delete_ctd_file(self, ctd_id: int) -> bool:
        """Удаляет файл CTD из базы."""
        with self.get_session() as session:
            ctd = session.get(CTDFile, ctd_id)
            if ctd:
                session.delete(ctd)
                session.commit()
                return True
            return False

    # ========== MODEL OPERATIONS ==========

    def add_model(
        self,
        name: str,
        filepath: str,
        description: Optional[str] = None,
        base_model: Optional[str] = None,
        classes_count: Optional[int] = None,
    ) -> Optional[Model]:
        """
        Добавляет модель в базу.
        
        Args:
            name: Название модели.
            filepath: Путь к файлу модели (.pt).
            description: Описание модели.
            base_model: Базовая архитектура (yolov8n, yolov8m, etc.).
            classes_count: Количество классов.
        
        Returns:
            Созданный объект Model или None.
        """
        filepath = str(Path(filepath).resolve())
        
        if not os.path.exists(filepath):
            return None

        with self.get_session() as session:
            # Проверяем, не добавлена ли уже эта модель
            stmt = select(Model).where(Model.filepath == filepath)
            existing = session.scalar(stmt)
            if existing:
                return existing

            model = Model(
                name=name,
                filepath=filepath,
                description=description,
                base_model=base_model,
                classes_count=classes_count,
            )
            session.add(model)
            session.commit()
            session.refresh(model)
            return model

    def get_model(self, model_id: int) -> Optional[Model]:
        """Получает модель по ID."""
        with self.get_session() as session:
            return session.get(Model, model_id)

    def get_all_models(self) -> List[Model]:
        """Получает все модели."""
        with self.get_session() as session:
            stmt = select(Model).order_by(Model.created_at.desc())
            return list(session.scalars(stmt))

    def update_model(self, model_id: int, **kwargs) -> Optional[Model]:
        """Обновляет модель."""
        with self.get_session() as session:
            model = session.get(Model, model_id)
            if model:
                for key, value in kwargs.items():
                    if hasattr(model, key):
                        setattr(model, key, value)
                session.commit()
                session.refresh(model)
            return model

    def delete_model(self, model_id: int) -> bool:
        """Удаляет модель из базы."""
        with self.get_session() as session:
            model = session.get(Model, model_id)
            if model:
                session.delete(model)
                session.commit()
                return True
            return False

    # ========== TASK OPERATIONS ==========

    def create_task(
        self,
        video_id: int,
        model_id: int,
        ctd_id: Optional[int] = None,
        **params,
    ) -> Optional[Task]:
        """
        Создаёт новую задачу.
        
        Args:
            video_id: ID видеофайла.
            model_id: ID модели.
            ctd_id: ID файла CTD (опционально).
            **params: Параметры детекции.
        
        Returns:
            Созданный объект Task или None.
        """
        with self.get_session() as session:
            # Проверяем существование связанных сущностей
            video = session.get(VideoFile, video_id)
            model = session.get(Model, model_id)
            
            if not video or not model:
                return None
            
            if ctd_id:
                ctd = session.get(CTDFile, ctd_id)
                if not ctd:
                    return None

            # Определяем позицию в очереди
            stmt = select(func.max(Task.position))
            max_position = session.scalar(stmt) or 0

            task = Task(
                video_id=video_id,
                model_id=model_id,
                ctd_id=ctd_id,
                position=max_position + 1,
                **params,
            )
            session.add(task)
            session.commit()
            session.refresh(task)
            return task

    def get_task(self, task_id: int) -> Optional[Task]:
        """Получает задачу по ID."""
        with self.get_session() as session:
            return session.get(Task, task_id)

    def get_all_tasks(self) -> List[Task]:
        """Получает все задачи."""
        with self.get_session() as session:
            stmt = select(Task).order_by(Task.position)
            return list(session.scalars(stmt))

    def get_pending_tasks(self) -> List[Task]:
        """Получает задачи в ожидании."""
        with self.get_session() as session:
            stmt = (
                select(Task)
                .where(Task.status == TaskStatus.PENDING)
                .order_by(Task.position)
            )
            return list(session.scalars(stmt))

    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """Получает задачи по статусу."""
        with self.get_session() as session:
            stmt = select(Task).where(Task.status == status).order_by(Task.position)
            return list(session.scalars(stmt))

    def update_task(self, task_id: int, **kwargs) -> Optional[Task]:
        """Обновляет задачу."""
        with self.get_session() as session:
            task = session.get(Task, task_id)
            if task:
                for key, value in kwargs.items():
                    if hasattr(task, key):
                        setattr(task, key, value)
                session.commit()
                session.refresh(task)
            return task

    def update_task_status(
        self,
        task_id: int,
        status: TaskStatus,
        error_message: Optional[str] = None,
    ) -> Optional[Task]:
        """Обновляет статус задачи."""
        with self.get_session() as session:
            task = session.get(Task, task_id)
            if task:
                task.status = status
                task.error_message = error_message
                
                if status == TaskStatus.RUNNING:
                    task.started_at = datetime.now()
                elif status in (TaskStatus.DONE, TaskStatus.ERROR, TaskStatus.CANCELLED):
                    task.completed_at = datetime.now()
                
                session.commit()
                session.refresh(task)
            return task

    def update_task_progress(
        self,
        task_id: int,
        progress_percent: float,
        current_frame: int,
    ) -> None:
        """Обновляет прогресс задачи."""
        with self.get_session() as session:
            stmt = (
                update(Task)
                .where(Task.id == task_id)
                .values(progress_percent=progress_percent, current_frame=current_frame)
            )
            session.execute(stmt)
            session.commit()

    def move_task(self, task_id: int, new_position: int) -> bool:
        """Перемещает задачу в очереди."""
        with self.get_session() as session:
            task = session.get(Task, task_id)
            if not task or task.status != TaskStatus.PENDING:
                return False

            old_position = task.position
            
            if new_position > old_position:
                # Двигаем вниз
                stmt = (
                    update(Task)
                    .where(Task.position > old_position)
                    .where(Task.position <= new_position)
                    .values(position=Task.position - 1)
                )
            else:
                # Двигаем вверх
                stmt = (
                    update(Task)
                    .where(Task.position >= new_position)
                    .where(Task.position < old_position)
                    .values(position=Task.position + 1)
                )
            
            session.execute(stmt)
            task.position = new_position
            session.commit()
            return True

    def delete_task(self, task_id: int) -> bool:
        """Удаляет задачу."""
        with self.get_session() as session:
            task = session.get(Task, task_id)
            if task:
                session.delete(task)
                session.commit()
                return True
            return False

    # ========== TASK OUTPUT OPERATIONS ==========

    def add_task_output(
        self,
        task_id: int,
        output_type: OutputType,
        filepath: str,
    ) -> Optional[TaskOutput]:
        """
        Добавляет выходной файл задачи.
        
        Args:
            task_id: ID задачи.
            output_type: Тип выходного файла.
            filepath: Путь к файлу.
        
        Returns:
            Созданный объект TaskOutput или None.
        """
        filepath = str(Path(filepath).resolve())

        with self.get_session() as session:
            task = session.get(Task, task_id)
            if not task:
                return None

            # Размер файла
            filesize_mb = None
            if os.path.exists(filepath):
                filesize_mb = os.path.getsize(filepath) / (1024 * 1024)

            output = TaskOutput(
                task_id=task_id,
                output_type=output_type,
                filepath=filepath,
                filename=os.path.basename(filepath),
                filesize_mb=filesize_mb,
            )
            session.add(output)
            session.commit()
            session.refresh(output)
            return output

    def get_task_outputs(self, task_id: int) -> List[TaskOutput]:
        """Получает выходные файлы задачи."""
        with self.get_session() as session:
            stmt = select(TaskOutput).where(TaskOutput.task_id == task_id)
            return list(session.scalars(stmt))

    # ========== HELPER METHODS FOR UI ==========

    def get_video_files_for_dive(self, dive_id: int) -> List[VideoFile]:
        """Alias for get_videos_by_dive."""
        return self.get_videos_by_dive(dive_id)

    def get_ctd_files_for_dive(self, dive_id: int) -> List[CTDFile]:
        """Alias for get_ctd_by_dive."""
        return self.get_ctd_by_dive(dive_id)

    # ========== STATISTICS ==========

    def get_statistics(self) -> Dict[str, Any]:
        """Получает общую статистику."""
        with self.get_session() as session:
            stats = {
                "dives_count": session.scalar(select(func.count(Dive.id))),
                "videos_count": session.scalar(select(func.count(VideoFile.id))),
                "models_count": session.scalar(select(func.count(Model.id))),
                "tasks_total": session.scalar(select(func.count(Task.id))),
                "tasks_pending": session.scalar(
                    select(func.count(Task.id)).where(Task.status == TaskStatus.PENDING)
                ),
                "tasks_done": session.scalar(
                    select(func.count(Task.id)).where(Task.status == TaskStatus.DONE)
                ),
                "tasks_error": session.scalar(
                    select(func.count(Task.id)).where(Task.status == TaskStatus.ERROR)
                ),
            }
            return stats

    # ========== POST-PROCESSING OPERATIONS ==========

    def get_task_with_outputs(self, task_id: int) -> Optional[Task]:
        """
        Получает задачу с загруженными outputs.
        
        Args:
            task_id: ID задачи.
            
        Returns:
            Задача с загруженными связями или None.
        """
        with self.get_session() as session:
            from sqlalchemy.orm import joinedload
            stmt = (
                select(Task)
                .where(Task.id == task_id)
                .options(joinedload(Task.outputs))
            )
            return session.scalar(stmt)

    def get_completed_tasks(self) -> List[Task]:
        """Получает завершённые задачи детекции."""
        with self.get_session() as session:
            stmt = (
                select(Task)
                .where(Task.status == TaskStatus.DONE)
                .order_by(Task.completed_at.desc())
            )
            return list(session.scalars(stmt))

    def update_postprocess_status(
        self,
        task_id: int,
        process_type: str,
        status: "PostProcessStatus",
        error: Optional[str] = None,
        **extra_fields,
    ) -> Optional[Task]:
        """
        Обновляет статус постобработки.
        
        Args:
            task_id: ID задачи.
            process_type: Тип постобработки (geometry, size, volume, analysis).
            status: Новый статус.
            error: Сообщение об ошибке.
            **extra_fields: Дополнительные поля для обновления.
            
        Returns:
            Обновлённая задача или None.
        """
        from .models import PostProcessStatus
        
        with self.get_session() as session:
            task = session.get(Task, task_id)
            if not task:
                return None
            
            # Обновляем статус
            status_field = f"{process_type}_status"
            error_field = f"{process_type}_error"
            
            if hasattr(task, status_field):
                setattr(task, status_field, status)
            if hasattr(task, error_field):
                setattr(task, error_field, error)
            
            # Обновляем дополнительные поля
            for key, value in extra_fields.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            
            session.commit()
            session.refresh(task)
            return task

    def get_task_output_by_type(
        self,
        task_id: int,
        output_type: OutputType,
    ) -> Optional[TaskOutput]:
        """Получает выходной файл задачи по типу."""
        with self.get_session() as session:
            stmt = (
                select(TaskOutput)
                .where(TaskOutput.task_id == task_id)
                .where(TaskOutput.output_type == output_type)
            )
            return session.scalar(stmt)

    def remove_task_output_by_type(
        self,
        task_id: int,
        output_type: OutputType,
    ) -> bool:
        """Удаляет выходной файл задачи по типу."""
        with self.get_session() as session:
            stmt = (
                delete(TaskOutput)
                .where(TaskOutput.task_id == task_id)
                .where(TaskOutput.output_type == output_type)
            )
            result = session.execute(stmt)
            session.commit()
            return result.rowcount > 0

    # ========== SUBTASK OPERATIONS ==========

    def create_subtask(
        self,
        parent_task_id: int,
        subtask_type: "SubTaskType",
        position: int = 0,
        params_json: Optional[str] = None,
    ) -> Optional["SubTask"]:
        """
        Создаёт подзадачу постобработки.
        
        Args:
            parent_task_id: ID родительской задачи.
            subtask_type: Тип подзадачи.
            position: Позиция в очереди.
            params_json: Параметры в формате JSON.
            
        Returns:
            Созданная подзадача или None.
        """
        from .models import SubTask, SubTaskType
        
        with self.get_session() as session:
            task = session.get(Task, parent_task_id)
            if not task:
                return None
            
            subtask = SubTask(
                parent_task_id=parent_task_id,
                subtask_type=subtask_type,
                position=position,
                params_json=params_json,
            )
            session.add(subtask)
            session.commit()
            session.refresh(subtask)
            return subtask

    def get_subtask(self, subtask_id: int) -> Optional["SubTask"]:
        """Получает подзадачу по ID."""
        from .models import SubTask
        
        with self.get_session() as session:
            return session.get(SubTask, subtask_id)

    def get_subtasks_for_task(self, task_id: int) -> List["SubTask"]:
        """Получает все подзадачи для задачи."""
        from .models import SubTask
        
        with self.get_session() as session:
            stmt = (
                select(SubTask)
                .where(SubTask.parent_task_id == task_id)
                .order_by(SubTask.position)
            )
            return list(session.scalars(stmt))

    def get_pending_subtasks(self) -> List["SubTask"]:
        """Получает все ожидающие подзадачи."""
        from .models import SubTask
        
        with self.get_session() as session:
            stmt = (
                select(SubTask)
                .where(SubTask.status == TaskStatus.PENDING)
                .order_by(SubTask.parent_task_id, SubTask.position)
            )
            return list(session.scalars(stmt))

    def update_subtask(
        self,
        subtask_id: int,
        **kwargs,
    ) -> Optional["SubTask"]:
        """Обновляет подзадачу."""
        from .models import SubTask
        
        with self.get_session() as session:
            subtask = session.get(SubTask, subtask_id)
            if subtask:
                for key, value in kwargs.items():
                    if hasattr(subtask, key):
                        setattr(subtask, key, value)
                session.commit()
                session.refresh(subtask)
            return subtask

    def update_subtask_status(
        self,
        subtask_id: int,
        status: TaskStatus,
        error_message: Optional[str] = None,
        result_value: Optional[float] = None,
        result_text: Optional[str] = None,
    ) -> Optional["SubTask"]:
        """Обновляет статус подзадачи."""
        from .models import SubTask
        from datetime import datetime
        
        with self.get_session() as session:
            subtask = session.get(SubTask, subtask_id)
            if subtask:
                subtask.status = status
                subtask.error_message = error_message
                
                if result_value is not None:
                    subtask.result_value = result_value
                if result_text is not None:
                    subtask.result_text = result_text
                
                if status == TaskStatus.RUNNING:
                    subtask.started_at = datetime.now()
                elif status in (TaskStatus.DONE, TaskStatus.ERROR, TaskStatus.CANCELLED):
                    subtask.completed_at = datetime.now()
                
                session.commit()
                session.refresh(subtask)
            return subtask

    def delete_subtask(self, subtask_id: int) -> bool:
        """Удаляет подзадачу."""
        from .models import SubTask
        
        with self.get_session() as session:
            subtask = session.get(SubTask, subtask_id)
            if subtask:
                session.delete(subtask)
                session.commit()
                return True
            return False

    def create_postprocess_subtasks(
        self,
        task_id: int,
        geometry: bool = True,
        size: bool = True,
        size_video_render: bool = False,
        volume: bool = True,
        analysis: bool = True,
        params_json: Optional[str] = None,
    ) -> List["SubTask"]:
        """
        Создаёт набор подзадач постобработки для задачи.
        
        Args:
            task_id: ID родительской задачи.
            geometry: Создать подзадачу геометрии.
            size: Создать подзадачу размеров.
            size_video_render: Создать подзадачу рендеринга видео с размерами.
            volume: Создать подзадачу объёма.
            analysis: Создать подзадачу анализа.
            params_json: Общие параметры.
            
        Returns:
            Список созданных подзадач.
        """
        from .models import SubTask, SubTaskType
        
        subtasks = []
        position = 0
        
        # Порядок важен: геометрия -> размеры -> видео с размерами -> объём -> анализ
        types_to_create = []
        if geometry:
            types_to_create.append(SubTaskType.GEOMETRY)
        if size:
            types_to_create.append(SubTaskType.SIZE)
        if size_video_render:
            types_to_create.append(SubTaskType.SIZE_VIDEO_RENDER)
        if volume:
            types_to_create.append(SubTaskType.VOLUME)
        if analysis:
            types_to_create.append(SubTaskType.ANALYSIS)
        
        for st_type in types_to_create:
            st = self.create_subtask(
                parent_task_id=task_id,
                subtask_type=st_type,
                position=position,
                params_json=params_json,
            )
            if st:
                subtasks.append(st)
                position += 1
        
        return subtasks

    def get_task_with_subtasks(self, task_id: int) -> Optional[Task]:
        """Получает задачу с загруженными подзадачами."""
        with self.get_session() as session:
            from sqlalchemy.orm import joinedload
            stmt = (
                select(Task)
                .where(Task.id == task_id)
                .options(joinedload(Task.subtasks))
            )
            return session.scalar(stmt)
