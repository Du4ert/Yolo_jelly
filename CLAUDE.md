# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLO Jellyfish Detector — a Python system for detecting, classifying, and measuring jellyfish/ctenophores in underwater video from Black Sea expeditions. Uses YOLOv8 (Ultralytics) for detection, OpenCV for video processing, PyQt6 for GUI, and SQLAlchemy for task management.

Documentation and comments are in Russian.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run GUI application
python batch_app.py

# Train model
python src/train.py --config train_config.yaml

# Run detection on video
python src/detect_video.py --video input.mp4 --model best.pt --track --csv output.csv

# Camera geometry & size estimation
python src/camera_geometry.py geometry --video input.mp4 --output geometry.csv
python src/camera_geometry.py size --detections detections.csv --geometry geometry.csv
python src/camera_geometry.py volume --detections detections.csv --tracks tracks.csv

# Analysis & visualization
python src/analyze.py --csv detections.csv --output-dir output/
```

There is no test suite in this project.

## Architecture

### Two interfaces to the same pipeline

1. **CLI scripts** (`src/`) — standalone scripts for each pipeline stage (training → detection → geometry → analysis)
2. **GUI app** (`batch_app.py` + `batch/`) — PyQt6 application that orchestrates the same pipeline with task queuing

### Batch processing system (`batch/`)

- **`database/models.py`** — SQLAlchemy ORM: Catalog → Dive → Video hierarchy, plus Model, Task, SubTask, Output entities. Task status: PENDING → RUNNING → DONE/ERROR
- **`database/repository.py`** — data access layer wrapping all DB queries
- **`core/task_manager.py`** — manages the task queue; `core/worker.py` runs tasks in a background QThread
- **`core/processor.py`** — wraps `detect_video.py` logic; `geometry_processor.py` and `analyze_processor.py` wrap their respective scripts
- **`ui/`** — MainWindow with DivePanel, ModelPanel, TaskTable; dialogs for adding dives/models, configuring geometry, viewing analysis

### Camera geometry pipeline (`src/camera_geometry.py`)

The largest and most complex module (~2000 lines). Three subcommands:
- `geometry` — estimates camera tilt from marine snow motion (Focus of Expansion)
- `size` — calculates real object sizes from pixel measurements + distance, corrected for tilt
- `volume` — computes surveyed water volume and organism density

Calibrated for GoPro 12 Wide 4K (K=2365 px/m at 1m). Size formula: `size_real = (size_pixels × distance) / K`

### Detection classes (5 species)

Defined in `data.yaml` and `dataset/classes.txt`:
0: Aurelia aurita, 1: Beroe ovata, 2: Mnemiopsis leidyi, 3: Pleurobrachia pileus, 4: Rhizostoma pulmo

Class IDs are in alphabetical order, defined authoritatively in `data.yaml`.

## Key Configuration Files

- **`config.json`** — app settings: DB path, default detection params (conf=0.75, ByteTrack tracker), UI state
- **`data.yaml`** — YOLO dataset config (paths, 5 classes)
- **`train_config.yaml`** — training hyperparameters (epochs=150, batch=4, imgsz=1920, AdamW)
- **`batch.db`** — SQLite database for the batch processing system
