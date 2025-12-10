"""
Обучение модели YOLO для детекции желетелых.
"""

import argparse
import torch
from pathlib import Path
from ultralytics import YOLO


def train_model(
    data_yaml: str,
    model_name: str = "yolov8m.pt",
    epochs: int = 150,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "auto",
    project: str = "runs/jellyfish",
    name: str = "experiment",
    resume: bool = False
) -> str:
    """
    Обучает модель YOLO на датасете желетелых.
    
    Args:
        data_yaml: путь к конфигурации датасета
        model_name: имя базовой модели или путь к чекпоинту
        epochs: количество эпох
        imgsz: размер входного изображения
        batch: размер батча
        device: устройство для обучения
        project: директория проекта
        name: имя эксперимента
        resume: продолжить обучение с чекпоинта
    
    Returns:
        Путь к лучшей модели
    """
    # Определяем устройство
    if device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    
    print("="*60)
    print("ОБУЧЕНИЕ YOLO ДЛЯ ДЕТЕКЦИИ ЖЕЛЕТЕЛЫХ")
    print("="*60)
    print(f"Конфигурация датасета: {data_yaml}")
    print(f"Базовая модель: {model_name}")
    print(f"Эпох: {epochs}")
    print(f"Размер изображения: {imgsz}")
    print(f"Размер батча: {batch}")
    print(f"Устройство: {device}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*60)
    print()
    
    # Загрузка модели
    model = YOLO(model_name)
    
    # Параметры обучения
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        
        # Оптимизатор
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # Аугментация (встроенная в YOLO)
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=180,
        translate=0.1,
        scale=0.5,
        flipud=0.3,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.1,
        
        # Регуляризация
        dropout=0.1,
        
        # Сохранение
        project=project,
        name=name,
        save=True,
        save_period=10,
        
        # Разное
        patience=30,
        device=device,
        workers=4,
        verbose=True,
        resume=resume,
    )
    
    # Путь к лучшей модели
    best_model_path = Path(project) / name / "weights" / "best.pt"
    
    print()
    print("="*60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("="*60)
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
        description="Обучение YOLO для детекции желетелых"
    )
    parser.add_argument(
        "--data", "-d",
        default="data.yaml",
        help="Путь к конфигурации датасета (по умолчанию: data.yaml)"
    )
    parser.add_argument(
        "--model", "-m",
        default="yolov8m.pt",
        help="Базовая модель: yolov8n/s/m/l/x.pt (по умолчанию: yolov8m.pt)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=150,
        help="Количество эпох (по умолчанию: 150)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Размер входного изображения (по умолчанию: 640)"
    )
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=16,
        help="Размер батча (по умолчанию: 16)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Устройство: 0, 1, cpu, auto (по умолчанию: auto)"
    )
    parser.add_argument(
        "--name", "-n",
        default="jellyfish_v1",
        help="Имя эксперимента (по умолчанию: jellyfish_v1)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Продолжить обучение с последнего чекпоинта"
    )
    
    args = parser.parse_args()
    
    try:
        train_model(
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
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
