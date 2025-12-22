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
    PENDING = "pending"      # Ожидает в очереди
    RUNNING = "running"      # Выполняется
    PAUSED = "paused"        # Приостановлена
    DONE = "done"            # Завершена успешно
    ERROR = "error"          # Завершена с ошибкой
    CANCELLED = "cancelled"  # Отменена


class OutputType(enum.Enum):
    """Типы выходных файлов."""
    VIDEO = "video"          # Видео с детекциями
    CSV = "csv"              # CSV с детекциями
    TRACKS_CSV = "tracks_csv"  # CSV со статистикой треков


class Dive(Base):
    """
    Погружение - папка с видео и данными CTD.
    """
    __tablename__ = "dives"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    folder_path: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    date: Mapped[Optional[datetime]] = mapped_column(Date, nullable=True)
    location: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )

    # Relationships
    video_files: Mapped[List["VideoFile"]] = relationship(
        back_populates="dive", cascade="all, delete-orphan"
    )
    ctd_files: Mapped[List["CTDFile"]] = relationship(
        back_populates="dive", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Dive(id={self.id}, name='{self.name}')>"


class VideoFile(Base):
    """
    Видеофайл, принадлежащий погружению.
    """
    __tablename__ = "video_files"

    id: Mapped[int] = mapped_column(primary_key=True)
    dive_id: Mapped[int] = mapped_column(ForeignKey("dives.id"), nullable=False)
    
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    filepath: Mapped[str] = mapped_column(String(1024), nullable=False)
    
    # Метаданные видео
    duration_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    frame_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    filesize_mb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )

    # Relationships
    dive: Mapped["Dive"] = relationship(back_populates="video_files")
    tasks: Mapped[List["Task"]] = relationship(back_populates="video_file")

    def __repr__(self) -> str:
        return f"<VideoFile(id={self.id}, filename='{self.filename}')>"


class CTDFile(Base):
    """
    Файл с данными CTD зонда.
    """
    __tablename__ = "ctd_files"

    id: Mapped[int] = mapped_column(primary_key=True)
    dive_id: Mapped[int] = mapped_column(ForeignKey("dives.id"), nullable=False)
    
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    filepath: Mapped[str] = mapped_column(String(1024), nullable=False)
    
    # Метаданные CTD
    records_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    max_depth: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    min_depth: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    duration_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )

    # Relationships
    dive: Mapped["Dive"] = relationship(back_populates="ctd_files")
    tasks: Mapped[List["Task"]] = relationship(back_populates="ctd_file")

    def __repr__(self) -> str:
        return f"<CTDFile(id={self.id}, filename='{self.filename}')>"


class Model(Base):
    """
    Обученная модель YOLO.
    """
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    filepath: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Метаданные модели (опционально)
    base_model: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # yolov8n, yolov8m, etc.
    classes_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )

    # Relationships
    tasks: Mapped[List["Task"]] = relationship(back_populates="model")

    def __repr__(self) -> str:
        return f"<Model(id={self.id}, name='{self.name}')>"


class Task(Base):
    """
    Задача на обработку видео.
    """
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    
    # Связи с другими сущностями
    video_id: Mapped[int] = mapped_column(ForeignKey("video_files.id"), nullable=False)
    ctd_id: Mapped[Optional[int]] = mapped_column(ForeignKey("ctd_files.id"), nullable=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), nullable=False)
    
    # Статус и позиция в очереди
    status: Mapped[TaskStatus] = mapped_column(
        Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False
    )
    position: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    # Параметры детекции
    conf_threshold: Mapped[float] = mapped_column(Float, default=0.25, nullable=False)
    enable_tracking: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    tracker_type: Mapped[str] = mapped_column(
        String(50), default="bytetrack.yaml", nullable=False
    )
    show_trails: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    trail_length: Mapped[int] = mapped_column(Integer, default=30, nullable=False)
    min_track_length: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    depth_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    save_video: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # Прогресс и результаты
    progress_percent: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    current_frame: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Статистика результатов
    detections_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tracks_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    processing_time_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Временные метки
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    video_file: Mapped["VideoFile"] = relationship(back_populates="tasks")
    ctd_file: Mapped[Optional["CTDFile"]] = relationship(back_populates="tasks")
    model: Mapped["Model"] = relationship(back_populates="tasks")
    outputs: Mapped[List["TaskOutput"]] = relationship(
        back_populates="task", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Task(id={self.id}, status={self.status.value})>"


class TaskOutput(Base):
    """
    Выходной файл задачи.
    """
    __tablename__ = "task_outputs"

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id"), nullable=False)
    
    output_type: Mapped[OutputType] = mapped_column(Enum(OutputType), nullable=False)
    filepath: Mapped[str] = mapped_column(String(1024), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    filesize_mb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )

    # Relationships
    task: Mapped["Task"] = relationship(back_populates="outputs")

    def __repr__(self) -> str:
        return f"<TaskOutput(id={self.id}, type={self.output_type.value})>"


def create_database(db_path: str) -> None:
    """
    Создаёт базу данных и все таблицы.
    
    Args:
        db_path: Путь к файлу SQLite базы данных.
    """
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
