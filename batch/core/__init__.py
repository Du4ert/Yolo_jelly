"""
Модуль ядра приложения.
"""

from .config import Config, get_config, init_config, save_config
from .processor import Processor, ProcessorFactory, ProcessingResult
from .task_manager import TaskManager
from .worker import Worker
from .geometry_processor import (
    GeometryProcessor,
    GeometryResult,
    SizeEstimationProcessor,
    SizeEstimationResult,
    VolumeEstimationProcessor,
    VolumeEstimationResult,
    SizeVideoRenderProcessor,
    SizeVideoRenderResult,
)
from .analyze_processor import (
    AnalyzeProcessor,
    AnalyzeResult,
    BatchAnalyzeProcessor,
)

__all__ = [
    # Config
    "Config",
    "get_config",
    "init_config",
    "save_config",
    # Detection processor
    "Processor",
    "ProcessorFactory",
    "ProcessingResult",
    # Task management
    "TaskManager",
    "Worker",
    # Geometry & Size
    "GeometryProcessor",
    "GeometryResult",
    "SizeEstimationProcessor",
    "SizeEstimationResult",
    "VolumeEstimationProcessor",
    "VolumeEstimationResult",
    "SizeVideoRenderProcessor",
    "SizeVideoRenderResult",
    # Analysis
    "AnalyzeProcessor",
    "AnalyzeResult",
    "BatchAnalyzeProcessor",
]
