#!/usr/bin/env python
"""
YOLO Jellyfish Batch Processor - Точка входа.

Запуск:
    python batch_app.py
"""

import sys
from pathlib import Path

# Добавляем корневую директорию в PYTHONPATH
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))


def get_app_paths():
    """Возвращает пути к файлам приложения."""
    app_dir = Path(__file__).parent
    return {
        "config": app_dir / "config.json",
        "database": app_dir / "batch.db",
    }


def check_dependencies():
    """Проверяет наличие зависимостей."""
    missing = []
    
    try:
        import sqlalchemy
    except ImportError:
        missing.append("SQLAlchemy")
    
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        missing.append("PyQt6")
    
    if missing:
        print("=" * 60)
        print("ERROR: Missing dependencies!")
        print(f"Please install: {', '.join(missing)}")
        print("")
        print("Run: pip install " + " ".join(missing))
        print("Or:  pip install -r requirements.txt")
        print("=" * 60)
        return False
    
    return True


def main():
    """Главная функция приложения."""
    # Проверяем зависимости
    if not check_dependencies():
        return 1
    
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    
    from batch.core.config import init_config, save_config
    from batch.database import Repository
    from batch.ui import MainWindow
    
    paths = get_app_paths()
    
    # Инициализация конфигурации
    print(f"Loading config: {paths['config']}")
    config = init_config(str(paths['config']))
    
    # Путь к БД
    db_path = paths['database']
    print(f"Database: {db_path}")
    
    # Инициализация репозитория
    repo = Repository(str(db_path))
    
    # Создание приложения
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO Jellyfish Batch Processor")
    app.setOrganizationName("YoloJellyfish")
    
    # Стиль (опционально)
    app.setStyle("Fusion")
    
    # Создание и показ главного окна
    window = MainWindow(repo)
    window.show()
    
    # Сохраняем конфиг при выходе
    app.aboutToQuit.connect(save_config)
    
    print("Application started.")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
