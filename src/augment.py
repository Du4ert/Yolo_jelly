"""
Аугментация данных для подводных изображений.
"""

import albumentations as A
import cv2
import os
import argparse
from pathlib import Path
from typing import List, Tuple


def create_augmentation_pipeline() -> A.Compose:
    """
    Создаёт пайплайн аугментации для подводных изображений.
    
    Returns:
        Albumentations Compose pipeline
    """
    return A.Compose([
        # Геометрические трансформации
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=180, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.Affine(scale=(0.8, 1.2), p=0.3),
        
        # Имитация условий подводной съёмки
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.MotionBlur(blur_limit=7, p=1),
        ], p=0.3),
        
        # Изменение освещённости (глубина влияет на свет)
        A.RandomBrightnessContrast(
            brightness_limit=(-0.3, 0.2),
            contrast_limit=(-0.2, 0.2),
            p=0.5
        ),
        
        # Цветовые искажения (поглощение красного на глубине)
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.3,
            hue=0.1,
            p=0.4
        ),
        
        # Имитация взвеси/мутности
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        
        # Имитация частиц в воде
        A.CoarseDropout(
            max_holes=20,
            max_height=8,
            max_width=8,
            fill_value=200,
            p=0.2
        ),
        
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3
    ))


def read_yolo_annotations(label_path: str) -> Tuple[List[List[float]], List[int]]:
    """
    Читает аннотации в формате YOLO.
    
    Args:
        label_path: путь к файлу аннотаций
    
    Returns:
        Tuple (bboxes, class_labels)
    """
    bboxes = []
    class_labels = []
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_labels.append(int(parts[0]))
                    bboxes.append([float(x) for x in parts[1:5]])
    
    return bboxes, class_labels


def write_yolo_annotations(label_path: str, bboxes: List, class_labels: List[int]):
    """
    Записывает аннотации в формате YOLO.
    
    Args:
        label_path: путь к файлу аннотаций
        bboxes: список bounding boxes
        class_labels: список меток классов
    """
    with open(label_path, 'w') as f:
        for cls, bbox in zip(class_labels, bboxes):
            bbox_str = ' '.join(f"{x:.6f}" for x in bbox)
            f.write(f"{cls} {bbox_str}\n")


def augment_dataset(
    images_dir: str,
    labels_dir: str,
    output_images: str = None,
    output_labels: str = None,
    augmentations_per_image: int = 3
) -> int:
    """
    Применяет аугментацию к датасету.
    
    Args:
        images_dir: директория с изображениями
        labels_dir: директория с аннотациями
        output_images: директория для аугментированных изображений (если None — та же)
        output_labels: директория для аугментированных аннотаций (если None — та же)
        augmentations_per_image: количество аугментированных копий на изображение
    
    Returns:
        Количество созданных аугментированных изображений
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    # Если выходные директории не указаны, используем входные
    out_images = Path(output_images) if output_images else images_path
    out_labels = Path(output_labels) if output_labels else labels_path
    
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    
    transform = create_augmentation_pipeline()
    
    # Собираем список изображений
    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    
    print(f"Найдено {len(image_files)} изображений")
    print(f"Создание {augmentations_per_image} аугментаций на изображение...")
    
    created_count = 0
    
    for img_file in image_files:
        # Определяем путь к аннотациям
        label_file = labels_path / img_file.with_suffix('.txt').name
        
        # Читаем изображение
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"  Пропуск (не удалось прочитать): {img_file.name}")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Читаем аннотации
        bboxes, class_labels = read_yolo_annotations(str(label_file))
        
        # Создаём аугментированные версии
        for i in range(augmentations_per_image):
            try:
                if bboxes:
                    transformed = transform(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                    new_bboxes = transformed['bboxes']
                    new_labels = transformed['class_labels']
                else:
                    # Если нет аннотаций, применяем только к изображению
                    simple_transform = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.RandomBrightnessContrast(p=0.5),
                    ])
                    transformed = simple_transform(image=image)
                    new_bboxes = []
                    new_labels = []
                
                # Генерируем имена файлов
                aug_img_name = f"{img_file.stem}_aug{i}{img_file.suffix}"
                aug_label_name = f"{img_file.stem}_aug{i}.txt"
                
                # Сохраняем изображение
                aug_image = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_images / aug_img_name), aug_image)
                
                # Сохраняем аннотации
                write_yolo_annotations(
                    str(out_labels / aug_label_name),
                    new_bboxes,
                    new_labels
                )
                
                created_count += 1
                
            except Exception as e:
                print(f"  Ошибка аугментации {img_file.name} #{i}: {e}")
                continue
        
        if (image_files.index(img_file) + 1) % 50 == 0:
            print(f"  Обработано {image_files.index(img_file) + 1}/{len(image_files)} изображений...")
    
    print(f"\nГотово! Создано {created_count} аугментированных изображений")
    return created_count


def main():
    parser = argparse.ArgumentParser(
        description="Аугментация данных для обучения YOLO"
    )
    parser.add_argument(
        "--images", "-i",
        required=True,
        help="Директория с изображениями"
    )
    parser.add_argument(
        "--labels", "-l",
        required=True,
        help="Директория с аннотациями"
    )
    parser.add_argument(
        "--output-images",
        default=None,
        help="Директория для аугментированных изображений (по умолчанию: та же)"
    )
    parser.add_argument(
        "--output-labels",
        default=None,
        help="Директория для аугментированных аннотаций (по умолчанию: та же)"
    )
    parser.add_argument(
        "--multiply", "-m",
        type=int,
        default=3,
        help="Количество аугментированных копий на изображение (по умолчанию: 3)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.images):
        print(f"Ошибка: директория не найдена: {args.images}")
        return 1
    
    if not os.path.exists(args.labels):
        print(f"Ошибка: директория не найдена: {args.labels}")
        return 1
    
    try:
        augment_dataset(
            args.images,
            args.labels,
            args.output_images,
            args.output_labels,
            args.multiply
        )
        return 0
    except Exception as e:
        print(f"Ошибка: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
