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
    Dive,
    VideoFile,
    CTDFile,
    Model,
    Task,
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
