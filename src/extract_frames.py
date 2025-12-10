"""
Извлечение кадров из видео для последующей разметки.
"""

import cv2
import os
import argparse
from pathlib import Path


def extract_frames(video_path: str, output_dir: str, frame_interval: int = 30) -> int:
    """
    Извлекает кадры из видео погружения.
    
    Args:
        video_path: путь к видеофайлу
        output_dir: директория для сохранения кадров
        frame_interval: интервал между кадрами (при 30 fps, interval=30 даёт 1 кадр/сек)
    
    Returns:
        Количество сохранённых кадров
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Видео: {video_path}")
    print(f"  FPS: {fps:.1f}")
    print(f"  Всего кадров: {total_frames}")
    print(f"  Длительность: {duration:.1f} сек")
    print(f"  Интервал извлечения: {frame_interval} кадров")
    print(f"  Ожидаемое количество: ~{total_frames // frame_interval} кадров")
    print()
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps if fps > 0 else 0
            filename = f"frame_{saved_count:05d}_t{timestamp:.2f}s.jpg"
            filepath = output_path / filename
            
            cv2.imwrite(str(filepath), frame)
            saved_count += 1
            
            if saved_count % 50 == 0:
                print(f"  Сохранено {saved_count} кадров...")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\nГотово! Сохранено {saved_count} кадров в {output_dir}")
    return saved_count


def main():
    parser = argparse.ArgumentParser(
        description="Извлечение кадров из видео для разметки"
    )
    parser.add_argument(
        "--video", "-v",
        required=True,
        help="Путь к видеофайлу"
    )
    parser.add_argument(
        "--output", "-o",
        default="dataset/raw_frames",
        help="Директория для сохранения кадров (по умолчанию: dataset/raw_frames)"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=30,
        help="Интервал между кадрами (по умолчанию: 30)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Ошибка: файл не найден: {args.video}")
        return 1
    
    try:
        extract_frames(args.video, args.output, args.interval)
        return 0
    except Exception as e:
        print(f"Ошибка: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
