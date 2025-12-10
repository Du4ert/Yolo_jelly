"""
Вспомогательные функции для работы с данными.
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple
import random


def split_dataset(
    images_dir: str,
    labels_dir: str,
    output_base: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[int, int, int]:
    """
    Разбивает датасет на train/val/test.
    
    Args:
        images_dir: директория с изображениями
        labels_dir: директория с аннотациями
        output_base: базовая директория для выходных данных
        train_ratio: доля train
        val_ratio: доля val
        test_ratio: доля test
        seed: seed для воспроизводимости
    
    Returns:
        Tuple (train_count, val_count, test_count)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
        "Сумма долей должна быть равна 1"
    
    random.seed(seed)
    
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_base)
    
    # Собираем список изображений с аннотациями
    image_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        image_files.extend(images_path.glob(ext))
    
    # Фильтруем только те, для которых есть аннотации
    valid_files = []
    for img_file in image_files:
        label_file = labels_path / img_file.with_suffix('.txt').name
        if label_file.exists():
            valid_files.append(img_file)
    
    print(f"Найдено {len(valid_files)} изображений с аннотациями")
    
    # Перемешиваем
    random.shuffle(valid_files)
    
    # Разбиваем
    n_total = len(valid_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = valid_files[:n_train]
    val_files = valid_files[n_train:n_train + n_val]
    test_files = valid_files[n_train + n_val:]
    
    # Создаём директории и копируем файлы
    for split_name, split_files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        img_out = output_path / 'images' / split_name
        lbl_out = output_path / 'labels' / split_name
        
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        
        for img_file in split_files:
            label_file = labels_path / img_file.with_suffix('.txt').name
            
            shutil.copy(img_file, img_out / img_file.name)
            shutil.copy(label_file, lbl_out / label_file.name)
    
    print(f"Разбиение завершено:")
    print(f"  Train: {len(train_files)}")
    print(f"  Val: {len(val_files)}")
    print(f"  Test: {len(test_files)}")
    
    return len(train_files), len(val_files), len(test_files)


def count_annotations(labels_dir: str) -> dict:
    """
    Подсчитывает количество аннотаций по классам.
    
    Args:
        labels_dir: директория с аннотациями
    
    Returns:
        Словарь {class_id: count}
    """
    labels_path = Path(labels_dir)
    counts = {}
    
    for label_file in labels_path.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    counts[class_id] = counts.get(class_id, 0) + 1
    
    return counts


def validate_dataset(dataset_dir: str) -> bool:
    """
    Проверяет корректность структуры датасета.
    
    Args:
        dataset_dir: корневая директория датасета
    
    Returns:
        True если датасет валиден
    """
    dataset_path = Path(dataset_dir)
    
    required_dirs = [
        'images/train',
        'images/val',
        'labels/train',
        'labels/val'
    ]
    
    errors = []
    
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        if not dir_path.exists():
            errors.append(f"Отсутствует директория: {dir_name}")
    
    if errors:
        print("Ошибки валидации:")
        for err in errors:
            print(f"  - {err}")
        return False
    
    # Проверяем соответствие изображений и аннотаций
    for split in ['train', 'val', 'test']:
        img_dir = dataset_path / 'images' / split
        lbl_dir = dataset_path / 'labels' / split
        
        if not img_dir.exists():
            continue
        
        images = set(f.stem for f in img_dir.glob('*') if f.suffix in ['.jpg', '.png', '.jpeg'])
        labels = set(f.stem for f in lbl_dir.glob('*.txt'))
        
        missing_labels = images - labels
        if missing_labels:
            print(f"Предупреждение: {split} - изображения без аннотаций: {len(missing_labels)}")
    
    print("Датасет валиден")
    return True


def convert_cvat_to_yolo(xml_path: str, output_dir: str, class_mapping: dict) -> int:
    """
    Конвертирует CVAT XML в формат YOLO.
    
    Args:
        xml_path: путь к XML файлу CVAT
        output_dir: директория для .txt файлов
        class_mapping: словарь {имя_класса: индекс}
    
    Returns:
        Количество конвертированных изображений
    """
    import xml.etree.ElementTree as ET
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    converted = 0
    
    for image in root.findall('.//image'):
        img_name = image.get('name')
        img_width = float(image.get('width'))
        img_height = float(image.get('height'))
        
        txt_name = Path(img_name).stem + '.txt'
        txt_path = output_path / txt_name
        
        annotations = []
        
        for box in image.findall('box'):
            label = box.get('label')
            if label not in class_mapping:
                continue
            
            class_id = class_mapping[label]
            
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            
            x_center = (xtl + xbr) / 2 / img_width
            y_center = (ytl + ybr) / 2 / img_height
            width = (xbr - xtl) / img_width
            height = (ybr - ytl) / img_height
            
            annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        with open(txt_path, 'w') as f:
            f.write('\n'.join(annotations))
        
        converted += 1
    
    print(f"Конвертировано {converted} изображений")
    return converted
