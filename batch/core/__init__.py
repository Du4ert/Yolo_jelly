"""
Модуль ядра приложения.
"""

from .config import Config, get_config, init_config, save_config
from .processor import Processor, ProcessorFactory, ProcessingResult
from .task_manager import TaskManager
from .worker import Worker

__all__ = [
    "Config",
    "get_config",
    "init_config",
    "save_config",
    "Processor",
    "ProcessorFactory",
    "ProcessingResult",
    "TaskManager",
    "Worker",
]
