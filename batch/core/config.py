"""
Управление конфигурацией приложения.

Конфигурация хранится в config.json и содержит:
- Путь к базе данных
- Параметры детекции по умолчанию
- Настройки UI
"""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class DetectionParams:
    """Параметры детекции по умолчанию."""
    conf_threshold: float = 0.25
    enable_tracking: bool = True
    tracker_type: str = "bytetrack.yaml"
    show_trails: bool = False
    trail_length: int = 30
    min_track_length: int = 3
    save_video: bool = True


@dataclass
class UISettings:
    """Настройки интерфейса."""
    window_geometry: Optional[str] = None  # Сохранённая геометрия окна
    window_state: Optional[str] = None     # Состояние окна (maximized и т.д.)
    last_browse_path: Optional[str] = None  # Последний путь в диалоге выбора
    last_model_id: Optional[int] = None     # Последняя выбранная модель
    splitter_sizes: Optional[list] = None   # Размеры сплиттеров


@dataclass
class Config:
    """
    Основная конфигурация приложения.
    """
    database_path: str = "batch.db"
    default_detection_params: DetectionParams = field(default_factory=DetectionParams)
    ui: UISettings = field(default_factory=UISettings)

    @classmethod
    def load(cls, config_path: str) -> "Config":
        """
        Загружает конфигурацию из файла.
        
        Args:
            config_path: Путь к файлу конфигурации.
        
        Returns:
            Объект Config.
        """
        if not os.path.exists(config_path):
            # Создаём конфигурацию по умолчанию
            config = cls()
            config.save(config_path)
            return config

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return cls._from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            # При ошибке возвращаем конфигурацию по умолчанию
            return cls()

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Создаёт Config из словаря."""
        detection_data = data.get("default_detection_params", {})
        ui_data = data.get("ui", {})
        
        return cls(
            database_path=data.get("database_path", "batch.db"),
            default_detection_params=DetectionParams(
                conf_threshold=detection_data.get("conf_threshold", 0.25),
                enable_tracking=detection_data.get("enable_tracking", True),
                tracker_type=detection_data.get("tracker_type", "bytetrack.yaml"),
                show_trails=detection_data.get("show_trails", False),
                trail_length=detection_data.get("trail_length", 30),
                min_track_length=detection_data.get("min_track_length", 3),
                save_video=detection_data.get("save_video", True),
            ),
            ui=UISettings(
                window_geometry=ui_data.get("window_geometry"),
                window_state=ui_data.get("window_state"),
                last_browse_path=ui_data.get("last_browse_path"),
                last_model_id=ui_data.get("last_model_id"),
                splitter_sizes=ui_data.get("splitter_sizes"),
            ),
        )

    def save(self, config_path: str) -> None:
        """
        Сохраняет конфигурацию в файл.
        
        Args:
            config_path: Путь к файлу конфигурации.
        """
        data = {
            "database_path": self.database_path,
            "default_detection_params": asdict(self.default_detection_params),
            "ui": asdict(self.ui),
        }
        
        # Создаём директорию, если нужно
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует конфигурацию в словарь."""
        return {
            "database_path": self.database_path,
            "default_detection_params": asdict(self.default_detection_params),
            "ui": asdict(self.ui),
        }


# ========== ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР ==========

_config: Optional[Config] = None
_config_path: Optional[str] = None


def init_config(config_path: str) -> Config:
    """
    Инициализирует глобальную конфигурацию.
    
    Args:
        config_path: Путь к файлу конфигурации.
    
    Returns:
        Объект Config.
    """
    global _config, _config_path
    _config_path = config_path
    _config = Config.load(config_path)
    return _config


def get_config() -> Config:
    """
    Возвращает глобальную конфигурацию.
    
    Returns:
        Объект Config.
    
    Raises:
        RuntimeError: Если конфигурация не инициализирована.
    """
    if _config is None:
        raise RuntimeError(
            "Configuration not initialized. Call init_config() first."
        )
    return _config


def save_config() -> None:
    """Сохраняет глобальную конфигурацию."""
    if _config is not None and _config_path is not None:
        _config.save(_config_path)


def get_config_path() -> Optional[str]:
    """Возвращает путь к файлу конфигурации."""
    return _config_path
