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
    CANONICAL_SUBTASK_ORDER,
    SUBTASK_OUTPUT_TYPES,
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
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Создаём таблицы, если их нет
        Base.metadata.create_all(self.engine)

        # WAL-режим для поддержки параллельных читателей
        from sqlalchemy import text as _text
        with self.engine.connect() as _conn:
            _conn.execute(_text("PRAGMA journal_mode=WAL"))

        # Миграция: добавляем новые колонки к существующим таблицам
        self._migrate()

    def _migrate(self):
        """Добавляет недостающие колонки в существующие таблицы."""
        from sqlalchemy import text, inspect
        insp = inspect(self.engine)

        # Колонки GPU-ускорения для таблицы tasks
        if 'tasks' in insp.get_table_names():
            existing = {c['name'] for c in insp.get_columns('tasks')}
            migrations = [
                ("device", "VARCHAR(20) DEFAULT 'auto'"),
                ("imgsz", "INTEGER DEFAULT 1280"),
                ("half", "BOOLEAN DEFAULT 1"),
                ("is_skipped", "BOOLEAN DEFAULT 0"),
                ("class_stats_json", "TEXT"),
                ("export_label_studio", "BOOLEAN DEFAULT 0"),
                ("export_ls_interval", "INTEGER DEFAULT 15"),
                ("export_ls_classes", "TEXT"),
            ]
            with self.engine.begin() as conn:
                for col_name, col_def in migrations:
                    if col_name not in existing:
                        conn.execute(text(
                            f"ALTER TABLE tasks ADD COLUMN {col_name} {col_def}"
                        ))

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

    def get_model_tasks_count(self, model_id: int) -> int:
        """Возвращает количество задач, использующих модель."""
        with self.get_session() as session:
            stmt = select(func.count()).select_from(Task).where(Task.model_id == model_id)
            return session.scalar(stmt) or 0

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

            # Убираем параметры, не являющиеся полями модели Task
            task_params = {k: v for k, v in params.items()
                          if hasattr(Task, k)}
            task = Task(
                video_id=video_id,
                model_id=model_id,
                ctd_id=ctd_id,
                position=max_position + 1,
                **task_params,
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
            from sqlalchemy.orm import joinedload
            stmt = (
                select(Task)
                .options(joinedload(Task.outputs))
                .order_by(Task.position)
            )
            return list(session.scalars(stmt).unique())

    def get_pending_tasks(self) -> List[Task]:
        """Получает задачи в ожидании (не пропущенные)."""
        with self.get_session() as session:
            stmt = (
                select(Task)
                .where(Task.status == TaskStatus.PENDING)
                .where(Task.is_skipped == False)  # noqa: E712
                .order_by(Task.position)
            )
            return list(session.scalars(stmt))

    def claim_pending_task(self) -> Optional[int]:
        """Атомарно захватывает первую PENDING-задачу, переводя её в RUNNING.

        Использует условный UPDATE (WHERE status=PENDING) для защиты от race condition:
        если другой воркер уже захватил эту задачу, rowcount=0 и мы пробуем следующую.

        Returns:
            task_id захваченной задачи, или None если нет доступных.
        """
        while True:
            with self.get_session() as session:
                task_id = session.scalar(
                    select(Task.id)
                    .where(Task.status == TaskStatus.PENDING)
                    .where(Task.is_skipped == False)  # noqa: E712
                    .order_by(Task.position)
                    .limit(1)
                )
                if task_id is None:
                    return None

                # Захватываем только если ещё PENDING — защита от двойного захвата
                result = session.execute(
                    update(Task)
                    .where(Task.id == task_id)
                    .where(Task.status == TaskStatus.PENDING)
                    .values(status=TaskStatus.RUNNING, started_at=datetime.now())
                )
                session.commit()

                if result.rowcount == 1:
                    return task_id
                # Другой воркер успел захватить эту задачу — ищем следующую

    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """Получает задачи по статусу."""
        with self.get_session() as session:
            stmt = select(Task).where(Task.status == status).order_by(Task.position)
            return list(session.scalars(stmt))

    def reset_stale_running_tasks(self) -> int:
        """Сбрасывает задачи и подзадачи в статусе RUNNING в ERROR.

        Вызывается при старте приложения, чтобы устранить задачи,
        зависшие после аварийного завершения предыдущего сеанса.
        Возвращает количество сброшенных записей.
        """
        count = 0
        with self.get_session() as session:
            result = session.execute(
                update(Task)
                .where(Task.status == TaskStatus.RUNNING)
                .values(
                    status=TaskStatus.ERROR,
                    error_message="Прервано (аварийное завершение приложения)",
                )
            )
            count += result.rowcount
            result = session.execute(
                update(SubTask)
                .where(SubTask.status == TaskStatus.RUNNING)
                .values(
                    status=TaskStatus.ERROR,
                    error_message="Прервано (аварийное завершение приложения)",
                )
            )
            count += result.rowcount
            session.commit()
        return count

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

    def set_task_skipped(self, task_id: int, skipped: bool) -> bool:
        """Устанавливает флаг пропуска для PENDING-задачи.

        Returns:
            True если флаг успешно изменён, False если задача не найдена или не PENDING.
        """
        with self.get_session() as session:
            task = session.get(Task, task_id)
            if task and task.status == TaskStatus.PENDING:
                task.is_skipped = skipped
                session.commit()
                return True
        return False

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

    def update_subtask_progress(self, subtask_id: int, progress_percent: float) -> None:
        """Обновляет прогресс подзадачи."""
        from .models import SubTask
        with self.get_session() as session:
            stmt = (
                update(SubTask)
                .where(SubTask.id == subtask_id)
                .values(progress_percent=progress_percent)
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

    def move_task_within_expedition(self, task_id: int, direction: int) -> bool:
        """Перемещает задачу вверх (direction=-1) или вниз (+1) в рамках своей экспедиции.

        Находит соседнюю задачу в том же каталоге и меняет с ней позиции местами.
        Задачи других экспедиций не затрагиваются.
        """
        with self.get_session() as session:
            task = session.get(Task, task_id)
            if not task or task.status != TaskStatus.PENDING:
                return False

            # Определяем catalog_id через цепочку video → dive → catalog
            video = task.video_file
            catalog_id = video.dive.catalog_id if video and video.dive else None

            # Получаем все задачи той же экспедиции, отсортированные по position
            if catalog_id is None:
                stmt = (
                    select(Task)
                    .join(Task.video_file)
                    .join(VideoFile.dive)
                    .where(Dive.catalog_id.is_(None))
                    .order_by(Task.position)
                )
            else:
                stmt = (
                    select(Task)
                    .join(Task.video_file)
                    .join(VideoFile.dive)
                    .where(Dive.catalog_id == catalog_id)
                    .order_by(Task.position)
                )
            expedition_tasks = list(session.scalars(stmt))

            # Находим индекс текущей задачи
            idx = next((i for i, t in enumerate(expedition_tasks) if t.id == task_id), None)
            if idx is None:
                return False

            swap_idx = idx + direction
            if swap_idx < 0 or swap_idx >= len(expedition_tasks):
                return False

            neighbor = expedition_tasks[swap_idx]

            # Меняем позиции местами
            task.position, neighbor.position = neighbor.position, task.position
            session.commit()
            return True

    def get_tasks_by_catalog(self, catalog_id: int) -> List[Task]:
        """Возвращает все завершённые задачи детекции в указанной экспедиции."""
        from sqlalchemy.orm import joinedload

        with self.get_session() as session:
            stmt = (
                select(Task)
                .join(Task.video_file)
                .join(VideoFile.dive)
                .where(Dive.catalog_id == catalog_id)
                .where(Task.status == TaskStatus.DONE)
                .options(
                    joinedload(Task.video_file).joinedload(VideoFile.dive),
                    joinedload(Task.outputs),
                    joinedload(Task.subtasks),
                )
                .order_by(Task.position)
            )
            return list(session.scalars(stmt).unique())

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

    def claim_pending_subtask(self) -> Optional[int]:
        """Атомарно захватывает первую PENDING-подзадачу.

        Для всех типов кроме INFERENCE требует, чтобы родитель был в DONE
        (детекция уже выполнена → постобработка имеет смысл). INFERENCE
        сам выполняет детекцию, поэтому может запускаться у не-DONE задач
        тоже (PENDING/ERROR/CANCELLED — но не RUNNING).

        Гарантирует последовательность: не берёт подзадачу, если у того же
        родителя уже есть RUNNING-подзадача (INFERENCE → GEOMETRY → SIZE →
        SIZE_VIDEO_RENDER → VOLUME → ANALYSIS).

        Returns:
            subtask_id захваченной подзадачи, или None если нет доступных.
        """
        from .models import SubTask

        with self.get_session() as session:
            candidates = session.scalars(
                select(SubTask)
                .join(Task, SubTask.parent_task_id == Task.id)
                .where(SubTask.status == TaskStatus.PENDING)
                .where(Task.status != TaskStatus.RUNNING)
                .order_by(SubTask.position)
            ).all()

            for subtask in candidates:
                # INFERENCE может запускаться у не-DONE задач; остальные
                # типы требуют завершённую детекцию.
                if subtask.subtask_type != SubTaskType.INFERENCE:
                    parent = session.get(Task, subtask.parent_task_id)
                    if parent is None or parent.status != TaskStatus.DONE:
                        continue

                running_sibling = session.scalar(
                    select(SubTask)
                    .where(SubTask.parent_task_id == subtask.parent_task_id)
                    .where(SubTask.status == TaskStatus.RUNNING)
                )
                if running_sibling is not None:
                    continue

                # Захватываем только если ещё PENDING
                result = session.execute(
                    update(SubTask)
                    .where(SubTask.id == subtask.id)
                    .where(SubTask.status == TaskStatus.PENDING)
                    .values(status=TaskStatus.RUNNING, started_at=datetime.now())
                )
                session.commit()

                if result.rowcount == 1:
                    return subtask.id
                # Другой воркер успел — продолжаем поиск среди кандидатов
        return None

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

    # ========== POSTPROCESS DIALOG v2 HELPERS ==========

    def delete_subtask_with_outputs(self, subtask_id: int) -> bool:
        """Удаляет подзадачу и все связанные TaskOutput-записи в одной транзакции.

        Файлы на диске НЕ удаляются — процессоры перезапишут их при следующем
        запуске. Бросает RuntimeError, если подзадача в статусе RUNNING.

        Returns:
            True, если что-то удалено; False — если подзадача не найдена.
        """
        with self.get_session() as session:
            subtask = session.get(SubTask, subtask_id)
            if subtask is None:
                return False
            if subtask.status == TaskStatus.RUNNING:
                raise RuntimeError(
                    f"Нельзя удалить выполняющуюся подзадачу {subtask_id}"
                )

            output_types = SUBTASK_OUTPUT_TYPES.get(subtask.subtask_type, [])
            if output_types:
                session.execute(
                    delete(TaskOutput)
                    .where(TaskOutput.task_id == subtask.parent_task_id)
                    .where(TaskOutput.output_type.in_(output_types))
                )

            session.delete(subtask)
            session.commit()
            return True

    def delete_outputs_for_subtask_types(
        self,
        task_id: int,
        subtask_types: List[SubTaskType],
    ) -> int:
        """Массово удаляет TaskOutput-записи для перечня типов подзадач.

        Используется при каскадной инвалидации после INFERENCE.
        Файлы на диске НЕ удаляются.

        Returns:
            Количество удалённых строк.
        """
        output_types: List[OutputType] = []
        for st_type in subtask_types:
            output_types.extend(SUBTASK_OUTPUT_TYPES.get(st_type, []))
        if not output_types:
            return 0

        with self.get_session() as session:
            result = session.execute(
                delete(TaskOutput)
                .where(TaskOutput.task_id == task_id)
                .where(TaskOutput.output_type.in_(output_types))
            )
            session.commit()
            return result.rowcount or 0

    def delete_finished_subtasks_of_types(
        self,
        task_id: int,
        subtask_types: List[SubTaskType],
    ) -> int:
        """Удаляет DONE/ERROR/CANCELLED-подзадачи указанных типов.

        PENDING/RUNNING не трогаются. Используется при каскадной инвалидации
        после INFERENCE: устаревшие результаты удаляются, новые подзадачи
        будут созданы пользователем при необходимости.
        """
        if not subtask_types:
            return 0

        with self.get_session() as session:
            result = session.execute(
                delete(SubTask)
                .where(SubTask.parent_task_id == task_id)
                .where(SubTask.subtask_type.in_(subtask_types))
                .where(SubTask.status.in_((
                    TaskStatus.DONE,
                    TaskStatus.ERROR,
                    TaskStatus.CANCELLED,
                )))
            )
            session.commit()
            return result.rowcount or 0

    def reset_subtask_positions(self, task_id: int) -> None:
        """Переприсваивает position всем подзадачам Task по каноническому порядку.

        Внутри одного типа сохраняется порядок по id (стабильность). Не трогает
        подзадачи в RUNNING — их position остаётся прежней (но в практике
        канонический порядок enforced через тип, а не position).
        """
        with self.get_session() as session:
            subtasks = session.scalars(
                select(SubTask).where(SubTask.parent_task_id == task_id)
            ).all()

            order_index = {t: i for i, t in enumerate(CANONICAL_SUBTASK_ORDER)}

            def sort_key(st: SubTask):
                return (order_index.get(st.subtask_type, 999), st.id)

            for new_pos, st in enumerate(sorted(subtasks, key=sort_key)):
                if st.status == TaskStatus.RUNNING:
                    continue
                st.position = new_pos
            session.commit()

    def create_postprocess_subtasks_v2(
        self,
        task_id: int,
        ops: List[tuple],
        force_overwrite_types: Optional[set] = None,
    ) -> List["SubTask"]:
        """Создаёт набор подзадач постобработки с поддержкой пересчёта.

        Args:
            task_id: ID родительской задачи.
            ops: Список (SubTaskType, params_json:str). Порядок создания
                определяется CANONICAL_SUBTASK_ORDER, а не порядком в списке.
            force_overwrite_types: Множество типов, для которых нужно удалить
                существующую DONE/ERROR/CANCELLED-подзадачу и связанные
                TaskOutput'ы перед созданием новой. Если у типа есть
                PENDING/RUNNING-подзадача — она не трогается, новая не создаётся
                (избегаем дубликатов в очереди).

        Returns:
            Список созданных подзадач.

        Raises:
            RuntimeError: если попытка пересчитать подзадачу в RUNNING.
        """
        force_overwrite_types = force_overwrite_types or set()
        created: List[SubTask] = []

        with self.get_session() as session:
            task = session.get(Task, task_id)
            if task is None:
                return []

            # Сортируем ops по канону для предсказуемого порядка создания.
            order_index = {t: i for i, t in enumerate(CANONICAL_SUBTASK_ORDER)}
            ops_sorted = sorted(
                ops, key=lambda op: order_index.get(op[0], 999)
            )

            for st_type, params_json in ops_sorted:
                existing = session.scalars(
                    select(SubTask)
                    .where(SubTask.parent_task_id == task_id)
                    .where(SubTask.subtask_type == st_type)
                ).all()

                # Если уже есть PENDING или RUNNING — не создаём дубликат.
                in_queue = [
                    e for e in existing
                    if e.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
                ]
                if in_queue:
                    continue

                # Если выбран принудительный пересчёт и есть DONE/ERROR/CANCELLED —
                # удаляем старую подзадачу с её Output'ами.
                if st_type in force_overwrite_types:
                    finished = [
                        e for e in existing
                        if e.status in (
                            TaskStatus.DONE, TaskStatus.ERROR, TaskStatus.CANCELLED
                        )
                    ]
                    for old in finished:
                        output_types = SUBTASK_OUTPUT_TYPES.get(old.subtask_type, [])
                        if output_types:
                            session.execute(
                                delete(TaskOutput)
                                .where(TaskOutput.task_id == task_id)
                                .where(TaskOutput.output_type.in_(output_types))
                            )
                        session.delete(old)
                    session.flush()
                else:
                    # Не пересчитываем, есть DONE-результат — пропускаем.
                    has_done = any(e.status == TaskStatus.DONE for e in existing)
                    if has_done:
                        continue

                # Если создаём INFERENCE для не-DONE задачи — отключаем её
                # из обычной очереди, чтобы воркер не запустил параллельно
                # старую детекцию через claim_pending_task.
                if st_type == SubTaskType.INFERENCE and task.status != TaskStatus.DONE:
                    if task.status == TaskStatus.PENDING:
                        task.is_skipped = True

                new_st = SubTask(
                    parent_task_id=task_id,
                    subtask_type=st_type,
                    position=0,  # будет переприсвоен ниже
                    params_json=params_json,
                )
                session.add(new_st)
                session.flush()
                created.append(new_st)

            # Переприсваиваем position всем подзадачам Task'а по канону.
            all_subtasks = session.scalars(
                select(SubTask).where(SubTask.parent_task_id == task_id)
            ).all()
            for new_pos, st in enumerate(sorted(
                all_subtasks,
                key=lambda s: (order_index.get(s.subtask_type, 999), s.id),
            )):
                if st.status == TaskStatus.RUNNING:
                    continue
                st.position = new_pos

            session.commit()
            for st in created:
                session.refresh(st)

        return created

    def reset_task_for_reinference(self, task_id: int, params: dict) -> None:
        """Сбрасывает задачу для повторной детекции без создания INFERENCE-подзадачи.

        - Обновляет параметры детекции (модель, порог, трекинг, save_video и т.д.).
        - Удаляет старые первичные Output'ы (VIDEO, CSV, TRACKS_CSV).
        - Если cascade_invalidate=True: удаляет downstream Output'ы и подзадачи.
        - Удаляет все существующие INFERENCE-подзадачи.
        - Сбрасывает Task в статус PENDING.
        """
        cascade = bool(params.get("cascade_invalidate", True))

        with self.get_session() as session:
            task = session.get(Task, task_id)
            if task is None:
                raise ValueError(f"Task {task_id} не найдена")

            if task.status == TaskStatus.RUNNING:
                raise RuntimeError(f"Задача #{task_id} сейчас выполняется — нельзя сбросить")

            model_id = params.get("model_id")
            if model_id is not None:
                model_obj = session.get(Model, model_id)
                if model_obj is None:
                    raise ValueError(f"Модель {model_id} не найдена")
                task.model_id = model_id

            for key in (
                "conf_threshold", "enable_tracking", "tracker_type",
                "show_trails", "trail_length", "min_track_length",
                "device", "imgsz", "half", "save_video",
            ):
                if params.get(key) is not None:
                    setattr(task, key, params[key])

            depth_rate = params.get("depth_rate")
            if depth_rate is not None:
                task.depth_rate = depth_rate

            primary_types = SUBTASK_OUTPUT_TYPES[SubTaskType.INFERENCE]
            session.execute(
                delete(TaskOutput)
                .where(TaskOutput.task_id == task_id)
                .where(TaskOutput.output_type.in_(primary_types))
            )

            downstream = [
                SubTaskType.GEOMETRY,
                SubTaskType.SIZE,
                SubTaskType.SIZE_VIDEO_RENDER,
                SubTaskType.VOLUME,
                SubTaskType.ANALYSIS,
            ]

            if cascade:
                downstream_outputs: List[OutputType] = []
                for t in downstream:
                    downstream_outputs.extend(SUBTASK_OUTPUT_TYPES.get(t, []))
                if downstream_outputs:
                    session.execute(
                        delete(TaskOutput)
                        .where(TaskOutput.task_id == task_id)
                        .where(TaskOutput.output_type.in_(downstream_outputs))
                    )
                session.execute(
                    delete(SubTask)
                    .where(SubTask.parent_task_id == task_id)
                    .where(SubTask.subtask_type.in_(downstream))
                )

            session.execute(
                delete(SubTask)
                .where(SubTask.parent_task_id == task_id)
                .where(SubTask.subtask_type == SubTaskType.INFERENCE)
                .where(SubTask.status != TaskStatus.RUNNING)
            )

            task.status = TaskStatus.PENDING
            task.is_skipped = False
            task.auto_postprocess = False  # постобработка управляется подзадачами из диалога
            task.progress_percent = 0.0
            task.current_frame = 0
            task.detections_count = None
            task.tracks_count = None
            task.processing_time_s = None
            task.error_message = None
            task.started_at = None
            task.completed_at = None
            task.class_stats_json = None

            session.commit()

    def apply_inference_result(
        self,
        task_id: int,
        result_outputs: Dict[OutputType, str],
        task_field_overrides: Dict[str, Any],
        result_stats: Dict[str, Any],
        cascade_invalidate: bool,
    ) -> None:
        """Транзакционно применяет результат INFERENCE-подзадачи.

        - Удаляет старые TaskOutput типов VIDEO/CSV/TRACKS_CSV.
        - Добавляет новые TaskOutput из result_outputs.
        - Перезаписывает поля Task (overrides + статистика).
        - Устанавливает Task.status = DONE и снимает is_skipped.
        - Если cascade_invalidate=True: удаляет downstream-Output'ы и
          DONE/ERROR-подзадачи downstream-типов.

        Args:
            task_id: ID Task.
            result_outputs: { OutputType.VIDEO: path, ... }
            task_field_overrides: { "model_id": 7, "conf_threshold": 0.5, ... }
            result_stats: { "detections_count": 1234, "tracks_count": 56,
                            "processing_time_s": 12.3, "class_stats_json": "..." }
            cascade_invalidate: если True — каскадно очистить downstream.
        """
        with self.get_session() as session:
            task = session.get(Task, task_id)
            if task is None:
                raise ValueError(f"Task {task_id} не найдена")

            # 1. Удаляем старые INFERENCE-Output'ы.
            inference_outputs = SUBTASK_OUTPUT_TYPES[SubTaskType.INFERENCE]
            session.execute(
                delete(TaskOutput)
                .where(TaskOutput.task_id == task_id)
                .where(TaskOutput.output_type.in_(inference_outputs))
            )

            # 2. Добавляем новые Output'ы.
            for out_type, filepath in result_outputs.items():
                if not filepath:
                    continue
                session.add(TaskOutput(
                    task_id=task_id,
                    output_type=out_type,
                    filepath=filepath,
                    filename=os.path.basename(filepath),
                    filesize_mb=(
                        os.path.getsize(filepath) / (1024 * 1024)
                        if os.path.exists(filepath) else None
                    ),
                ))

            # 3. Перезаписываем поля Task: overrides + статистика.
            for key, value in task_field_overrides.items():
                if value is None:
                    continue
                if hasattr(task, key):
                    setattr(task, key, value)
            for key, value in result_stats.items():
                if hasattr(task, key) and value is not None:
                    setattr(task, key, value)

            task.status = TaskStatus.DONE
            task.is_skipped = False
            task.error_message = None
            task.progress_percent = 100.0
            task.completed_at = datetime.now()

            # 4. Каскадная инвалидация downstream-результатов.
            if cascade_invalidate:
                downstream = [
                    SubTaskType.GEOMETRY,
                    SubTaskType.SIZE,
                    SubTaskType.SIZE_VIDEO_RENDER,
                    SubTaskType.VOLUME,
                    SubTaskType.ANALYSIS,
                ]
                downstream_outputs: List[OutputType] = []
                for t in downstream:
                    downstream_outputs.extend(SUBTASK_OUTPUT_TYPES.get(t, []))
                if downstream_outputs:
                    session.execute(
                        delete(TaskOutput)
                        .where(TaskOutput.task_id == task_id)
                        .where(TaskOutput.output_type.in_(downstream_outputs))
                    )
                session.execute(
                    delete(SubTask)
                    .where(SubTask.parent_task_id == task_id)
                    .where(SubTask.subtask_type.in_(downstream))
                    .where(SubTask.status.in_((
                        TaskStatus.DONE,
                        TaskStatus.ERROR,
                        TaskStatus.CANCELLED,
                    )))
                )

            session.commit()
