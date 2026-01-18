"""
Обучение модели YOLO для детекции желетелых.

Параметры обучения загружаются из train_config.yaml
"""

import argparse
import torch
import yaml
from pathlib import Path
from typing import Any, Optional
from ultralytics import YOLO


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "train_config.yaml"


def load_config(config_path: Optional[Path] = None) -> dict:
    """
    Загружает конфигурацию обучения из YAML файла.
    
    Args:
        config_path: путь к конфигурационному файлу
        
    Returns:
        Словарь с параметрами конфигурации
    """
    path = config_path or DEFAULT_CONFIG_PATH
    
    if not path.exists():
        raise FileNotFoundError(f"Конфигурационный файл не найден: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_param(config: dict, *keys: str, default: Any = None) -> Any:
    """
    Получает параметр из вложенной структуры конфига.
    
    Args:
        config: словарь конфигурации
        keys: путь к параметру (например, 'optimizer', 'lr0')
        default: значение по умолчанию
        
    Returns:
        Значение параметра или default
    """
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def train_model(
    config_path: Optional[Path] = None,
    # Переопределения через CLI
    data_yaml: Optional[str] = None,
    model_name: Optional[str] = None,
    epochs: Optional[int] = None,
    imgsz: Optional[int] = None,
    batch: Optional[int] = None,
    device: Optional[str] = None,
    project: Optional[str] = None,
    name: Optional[str] = None,
    resume: bool = False
) -> str:
    """
    Обучает модель YOLO на датасете желетелых.
    
    Параметры загружаются из конфигурационного файла, но могут быть
    переопределены через аргументы функции.
    
    Args:
        config_path: путь к конфигурационному файлу
        data_yaml: путь к конфигурации датасета (переопределяет конфиг)
        model_name: имя базовой модели (переопределяет конфиг)
        epochs: количество эпох (переопределяет конфиг)
        imgsz: размер входного изображения (переопределяет конфиг)
        batch: размер батча (переопределяет конфиг)
        device: устройство для обучения (переопределяет конфиг)
        project: директория проекта (переопределяет конфиг)
        name: имя эксперимента (переопределяет конфиг)
        resume: продолжить обучение с чекпоинта
    
    Returns:
        Путь к лучшей модели
    """
    # Загружаем конфигурацию
    config = load_config(config_path)
    
    # Основные параметры (с возможностью переопределения через CLI)
    _data_yaml = data_yaml or get_param(config, 'base', 'data_yaml', default='data.yaml')
    _model_name = model_name or get_param(config, 'base', 'model', default='yolov8m.pt')
    _project = project or get_param(config, 'base', 'project', default='runs/jellyfish')
    _name = name or get_param(config, 'base', 'name', default='experiment')
    
    # Параметры обучения
    _epochs = epochs if epochs is not None else get_param(config, 'training', 'epochs', default=150)
    _imgsz = imgsz if imgsz is not None else get_param(config, 'training', 'imgsz', default=640)
    _batch = batch if batch is not None else get_param(config, 'training', 'batch', default=16)
    _device = device or get_param(config, 'training', 'device', default='auto')
    _workers = get_param(config, 'training', 'workers', default=4)
    _patience = get_param(config, 'training', 'patience', default=30)
    _save_period = get_param(config, 'training', 'save_period', default=10)
    
    # Оптимизатор
    _optimizer = get_param(config, 'optimizer', 'name', default='AdamW')
    _lr0 = get_param(config, 'optimizer', 'lr0', default=0.001)
    _lrf = get_param(config, 'optimizer', 'lrf', default=0.01)
    _momentum = get_param(config, 'optimizer', 'momentum', default=0.937)
    _weight_decay = get_param(config, 'optimizer', 'weight_decay', default=0.0005)
    
    # Регуляризация
    _dropout = get_param(config, 'regularization', 'dropout', default=0.1)
    
    # Аугментация
    _augment = get_param(config, 'augmentation', 'augment', default=True)
    _hsv_h = get_param(config, 'augmentation', 'hsv_h', default=0.015)
    _hsv_s = get_param(config, 'augmentation', 'hsv_s', default=0.7)
    _hsv_v = get_param(config, 'augmentation', 'hsv_v', default=0.4)
    _degrees = get_param(config, 'augmentation', 'degrees', default=180)
    _translate = get_param(config, 'augmentation', 'translate', default=0.1)
    _scale = get_param(config, 'augmentation', 'scale', default=0.5)
    _shear = get_param(config, 'augmentation', 'shear', default=0.0)
    _perspective = get_param(config, 'augmentation', 'perspective', default=0.0)
    _flipud = get_param(config, 'augmentation', 'flipud', default=0.3)
    _fliplr = get_param(config, 'augmentation', 'fliplr', default=0.5)
    _mosaic = get_param(config, 'augmentation', 'mosaic', default=0.8)
    _mixup = get_param(config, 'augmentation', 'mixup', default=0.1)
    _copy_paste = get_param(config, 'augmentation', 'copy_paste', default=0.0)
    
    # Продвинутые параметры
    _close_mosaic = get_param(config, 'advanced', 'close_mosaic', default=10)
    _box = get_param(config, 'advanced', 'box', default=7.5)
    _cls = get_param(config, 'advanced', 'cls', default=0.5)
    _dfl = get_param(config, 'advanced', 'dfl', default=1.5)
    _nbs = get_param(config, 'advanced', 'nbs', default=64)
    
    # Валидация и логирование
    _val = get_param(config, 'validation', 'val', default=True)
    _plots = get_param(config, 'validation', 'plots', default=True)
    _verbose = get_param(config, 'logging', 'verbose', default=True)
    
    # Определяем устройство
    if _device == "auto":
        _device = 0 if torch.cuda.is_available() else "cpu"
    
    # Вывод конфигурации
    print("=" * 60)
    print("ОБУЧЕНИЕ YOLO ДЛЯ ДЕТЕКЦИИ ЖЕЛЕТЕЛЫХ")
    print("=" * 60)
    print(f"Конфигурация: {config_path or DEFAULT_CONFIG_PATH}")
    print()
    print("ОСНОВНЫЕ ПАРАМЕТРЫ:")
    print(f"  Датасет: {_data_yaml}")
    print(f"  Модель: {_model_name}")
    print(f"  Эпох: {_epochs}")
    print(f"  Размер изображения: {_imgsz}")
    print(f"  Размер батча: {_batch}")
    print(f"  Устройство: {_device}")
    print()
    print("ОПТИМИЗАТОР:")
    print(f"  {_optimizer}, lr0={_lr0}, lrf={_lrf}")
    print(f"  momentum={_momentum}, weight_decay={_weight_decay}")
    print()
    print("АУГМЕНТАЦИЯ:")
    print(f"  HSV: h={_hsv_h}, s={_hsv_s}, v={_hsv_v}")
    print(f"  Геометрия: degrees={_degrees}, scale={_scale}, translate={_translate}")
    print(f"  Отражения: flipud={_flipud}, fliplr={_fliplr}")
    print(f"  Mosaic={_mosaic}, MixUp={_mixup}")
    print()
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    print()
    
    # Загрузка модели
    model = YOLO(_model_name)
    
    # Обучение
    results = model.train(
        data=_data_yaml,
        epochs=_epochs,
        imgsz=_imgsz,
        batch=_batch,
        
        # Оптимизатор
        optimizer=_optimizer,
        lr0=_lr0,
        lrf=_lrf,
        momentum=_momentum,
        weight_decay=_weight_decay,
        
        # Аугментация
        augment=_augment,
        hsv_h=_hsv_h,
        hsv_s=_hsv_s,
        hsv_v=_hsv_v,
        degrees=_degrees,
        translate=_translate,
        scale=_scale,
        shear=_shear,
        perspective=_perspective,
        flipud=_flipud,
        fliplr=_fliplr,
        mosaic=_mosaic,
        mixup=_mixup,
        copy_paste=_copy_paste,
        
        # Регуляризация
        dropout=_dropout,
        
        # Продвинутые
        close_mosaic=_close_mosaic,
        box=_box,
        cls=_cls,
        dfl=_dfl,
        nbs=_nbs,
        
        # Сохранение
        project=_project,
        name=_name,
        save=True,
        save_period=_save_period,
        
        # Валидация и логирование
        val=_val,
        plots=_plots,
        verbose=_verbose,
        
        # Разное
        patience=_patience,
        device=_device,
        workers=_workers,
        resume=resume,
    )
    
    # Путь к лучшей модели
    best_model_path = Path(_project) / _name / "weights" / "best.pt"
    
    print()
    print("=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 60)
    print(f"Лучшая модель: {best_model_path}")
    print()
    
    # Валидация на тестовом наборе
    print("Валидация модели...")
    metrics = model.val()
    
    print()
    print("МЕТРИКИ:")
    print(f"  mAP50: {metrics.box.map50:.3f}")
    print(f"  mAP50-95: {metrics.box.map:.3f}")
    print()
    print("AP50 по классам:")
    for i, ap in enumerate(metrics.box.ap50):
        print(f"  Класс {i}: {ap:.3f}")
    
    return str(best_model_path)


def main():
    parser = argparse.ArgumentParser(
        description="Обучение YOLO для детекции желетелых",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Обучение с параметрами из конфига
  python train.py
  
  # Обучение с другим конфигом
  python train.py --config my_config.yaml
  
  # Переопределение параметров конфига через CLI
  python train.py --epochs 200 --batch 32
  
  # Продолжение обучения
  python train.py --resume
        """
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help=f"Путь к конфигурационному файлу (по умолчанию: {DEFAULT_CONFIG_PATH})"
    )
    parser.add_argument(
        "--data", "-d",
        default=None,
        help="Путь к конфигурации датасета (переопределяет конфиг)"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Базовая модель: yolov8n/s/m/l/x.pt (переопределяет конфиг)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=None,
        help="Количество эпох (переопределяет конфиг)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Размер входного изображения (переопределяет конфиг)"
    )
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=None,
        help="Размер батча (переопределяет конфиг)"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Устройство: 0, 1, cpu, auto (переопределяет конфиг)"
    )
    parser.add_argument(
        "--name", "-n",
        default=None,
        help="Имя эксперимента (переопределяет конфиг)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Продолжить обучение с последнего чекпоинта"
    )
    
    args = parser.parse_args()
    
    try:
        train_model(
            config_path=args.config,
            data_yaml=args.data,
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            name=args.name,
            resume=args.resume
        )
        return 0
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        return 1
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
