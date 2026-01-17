"""
SQLAlchemy модели для базы данных batch processing.
"""

import enum
from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    Boolean,
    DateTime,
    Date,
    ForeignKey,
    Enum,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


class Base(DeclarativeBase):
    """Базовый класс для всех моделей."""
    pass


class TaskStatus(enum.Enum):
    """Статусы задачи."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


class SubTaskType(enum.Enum):
    """Типы подзадач постобработки."""
    GEOMETRY = "geometry"
    SIZE = "size"
    VOLUME = "volume"
    ANALYSIS = "analysis"
    SIZE_VIDEO_RENDER = "size_video_render"


class OutputType(enum.Enum):
    """Типы выходных файлов."""
    VIDEO = "video"
    CSV = "csv"
    TRACKS_CSV = "tracks_csv"
    GEOMETRY_CSV = "geometry_csv"
    SIZE_CSV = "size_csv"
    TRACK_SIZES_CSV = "track_sizes_csv"
    VOLUME_CSV = "volume_csv"
    SIZE_VIDEO = "size_video"  # Видео с размерами
    ANALYSIS_PLOT = "analysis_plot"
    ANALYSIS_REPORT = "analysis_report"


class Catalog(Base):
    """Каталог для группировки погружений."""
    __tablename__ = "catalogs"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    color: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    position: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    dives: Mapped[List["Dive"]] = relationship(back_populates="catalog")

    def __repr__(self) -> str:
        return f"<Catalog(id={self.id}, name='{self.name}')>"


class Dive(Base):
    """Погружение - папка с видео и данными CTD."""
    __tablename__ = "dives"

    id: Mapped[int] = mapped_column(primary_key=True)
    catalog_id: Mapped[Optional[int]] = mapped_column(ForeignKey("catalogs.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    folder_path: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    date: Mapped[Optional[datetime]] = mapped_column(Date, nullable=True)
    location: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    position: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    catalog: Mapped[Optional["Catalog"]] = relationship(back_populates="dives")
    video_files: Mapped[List["VideoFile"]] = relationship(back_populates="dive", cascade="all, delete-orphan")
    ctd_files: Mapped[List["CTDFile"]] = relationship(back_populates="dive", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Dive(id={self.id}, name='{self.name}')>"


class VideoFile(Base):
    """Видеофайл."""
    __tablename__ = "video_files"

    id: Mapped[int] = mapped_column(primary_key=True)
    dive_id: Mapped[int] = mapped_column(ForeignKey("dives.id"), nullable=False)
    
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    filepath: Mapped[str] = mapped_column(String(1024), nullable=False)
    
    duration_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    frame_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    filesize_mb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)

    dive: Mapped["Dive"] = relationship(back_populates="video_files")
    tasks: Mapped[List["Task"]] = relationship(back_populates="video_file")

    def __repr__(self) -> str:
        return f"<VideoFile(id={self.id}, filename='{self.filename}')>"


class CTDFile(Base):
    """Файл CTD."""
    __tablename__ = "ctd_files"

    id: Mapped[int] = mapped_column(primary_key=True)
    dive_id: Mapped[int] = mapped_column(ForeignKey("dives.id"), nullable=False)
    
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    filepath: Mapped[str] = mapped_column(String(1024), nullable=False)
    
    records_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    max_depth: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    min_depth: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    duration_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)

    dive: Mapped["Dive"] = relationship(back_populates="ctd_files")
    tasks: Mapped[List["Task"]] = relationship(back_populates="ctd_file")

    def __repr__(self) -> str:
        return f"<CTDFile(id={self.id}, filename='{self.filename}')>"


class Model(Base):
    """Обученная модель YOLO."""
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    filepath: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    base_model: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    classes_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)

    tasks: Mapped[List["Task"]] = relationship(back_populates="model")

    def __repr__(self) -> str:
        return f"<Model(id={self.id}, name='{self.name}')>"


class Task(Base):
    """Задача на обработку видео."""
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    
    video_id: Mapped[int] = mapped_column(ForeignKey("video_files.id"), nullable=False)
    ctd_id: Mapped[Optional[int]] = mapped_column(ForeignKey("ctd_files.id"), nullable=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), nullable=False)
    
    # Основной статус детекции
    status: Mapped[TaskStatus] = mapped_column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False)
    position: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    # Параметры детекции
    conf_threshold: Mapped[float] = mapped_column(Float, default=0.25, nullable=False)
    enable_tracking: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    tracker_type: Mapped[str] = mapped_column(String(50), default="bytetrack.yaml", nullable=False)
    show_trails: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    trail_length: Mapped[int] = mapped_column(Integer, default=30, nullable=False)
    min_track_length: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    depth_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    save_video: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # Прогресс детекции
    progress_percent: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    current_frame: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Результаты детекции
    detections_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tracks_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    processing_time_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Флаг автозапуска постобработки и параметры
    auto_postprocess: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    auto_postprocess_params: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON с параметрами
    
    # Временные метки
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    video_file: Mapped["VideoFile"] = relationship(back_populates="tasks")
    ctd_file: Mapped[Optional["CTDFile"]] = relationship(back_populates="tasks")
    model: Mapped["Model"] = relationship(back_populates="tasks")
    outputs: Mapped[List["TaskOutput"]] = relationship(back_populates="task", cascade="all, delete-orphan")
    subtasks: Mapped[List["SubTask"]] = relationship(back_populates="parent_task", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Task(id={self.id}, status={self.status.value})>"
    
    @property
    def has_detections_csv(self) -> bool:
        """Проверяет наличие CSV с детекциями."""
        return any(o.output_type == OutputType.CSV for o in self.outputs)
    
    @property
    def has_tracks_csv(self) -> bool:
        """Проверяет наличие CSV с треками."""
        return any(o.output_type == OutputType.TRACKS_CSV for o in self.outputs)
    
    @property 
    def detections_csv_path(self) -> Optional[str]:
        """Возвращает путь к CSV с детекциями."""
        for o in self.outputs:
            if o.output_type == OutputType.CSV:
                return o.filepath
        return None
    
    @property
    def tracks_csv_path(self) -> Optional[str]:
        """Возвращает путь к CSV с треками."""
        for o in self.outputs:
            if o.output_type == OutputType.TRACKS_CSV:
                return o.filepath
        return None
    
    def get_subtask(self, subtask_type: SubTaskType) -> Optional["SubTask"]:
        """Возвращает подзадачу по типу."""
        for st in self.subtasks:
            if st.subtask_type == subtask_type:
                return st
        return None
    
    def get_postprocess_summary(self) -> str:
        """Возвращает краткую сводку по постобработке."""
        icons = {
            TaskStatus.PENDING: "○",
            TaskStatus.RUNNING: "▶",
            TaskStatus.DONE: "✓",
            TaskStatus.ERROR: "✗",
            TaskStatus.CANCELLED: "−",
            None: "·",  # Нет подзадачи
        }
        
        def get_icon(subtask_type: SubTaskType) -> str:
            st = self.get_subtask(subtask_type)
            return icons.get(st.status if st else None, "·")
        
        g = get_icon(SubTaskType.GEOMETRY)
        s = get_icon(SubTaskType.SIZE)
        r = get_icon(SubTaskType.SIZE_VIDEO_RENDER)
        v = get_icon(SubTaskType.VOLUME)
        a = get_icon(SubTaskType.ANALYSIS)
        
        return f"{g}{s}{r}{v}{a}"
    
    @property
    def has_pending_subtasks(self) -> bool:
        """Проверяет наличие ожидающих подзадач."""
        return any(st.status == TaskStatus.PENDING for st in self.subtasks)


class SubTask(Base):
    """Подзадача постобработки."""
    __tablename__ = "subtasks"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    parent_task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id"), nullable=False)
    
    subtask_type: Mapped[SubTaskType] = mapped_column(Enum(SubTaskType), nullable=False)
    status: Mapped[TaskStatus] = mapped_column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False)
    position: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    # Прогресс
    progress_percent: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Результаты (зависят от типа)
    result_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # tilt_deg / volume_m3 / tracks_count
    result_text: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # Описание результата
    
    # Параметры (JSON-like строка для гибкости)
    params_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Временные метки
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Relationship
    parent_task: Mapped["Task"] = relationship(back_populates="subtasks")
    
    def __repr__(self) -> str:
        return f"<SubTask(id={self.id}, type={self.subtask_type.value}, status={self.status.value})>"
    
    @property
    def type_name(self) -> str:
        """Человекочитаемое название типа."""
        names = {
            SubTaskType.GEOMETRY: "Геометрия",
            SubTaskType.SIZE: "Размеры",
            SubTaskType.VOLUME: "Объём",
            SubTaskType.ANALYSIS: "Анализ",
            SubTaskType.SIZE_VIDEO_RENDER: "Видео с размерами",
        }
        return names.get(self.subtask_type, "???")
    
    @property
    def type_icon(self) -> str:
        """Иконка типа."""
        icons = {
            SubTaskType.GEOMETRY: "📐",
            SubTaskType.SIZE: "📏",
            SubTaskType.VOLUME: "📦",
            SubTaskType.ANALYSIS: "📊",
            SubTaskType.SIZE_VIDEO_RENDER: "🎬",
        }
        return icons.get(self.subtask_type, "?")


class TaskOutput(Base):
    """Выходной файл задачи."""
    __tablename__ = "task_outputs"

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id"), nullable=False)
    
    output_type: Mapped[OutputType] = mapped_column(Enum(OutputType), nullable=False)
    filepath: Mapped[str] = mapped_column(String(1024), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    filesize_mb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)

    task: Mapped["Task"] = relationship(back_populates="outputs")

    def __repr__(self) -> str:
        return f"<TaskOutput(id={self.id}, type={self.output_type.value})>"


def create_database(db_path: str) -> None:
    """Создаёт базу данных и все таблицы."""
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
