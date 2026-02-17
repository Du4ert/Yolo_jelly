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

### CLI scripts (`src/`)

| File | Lines | Complexity | Purpose |
|------|-------|------------|---------|
| `camera_geometry.py` | ~1700 | Very high | Camera tilt (FOE), k-method sizing, volume estimation |
| `detect_video.py` | ~700 | High | YOLO detection + tracking + CTD interpolation |
| `interactive_plot.py` | ~520 | Medium-high | Interactive Plotly depth distribution plots |
| `analyze.py` | ~450 | Medium | Distribution plots (matplotlib) + text report |
| `render_size_video.py` | ~450 | Medium | Overlay size/distance info on detected video |
| `train.py` | ~400 | Medium | Model training with YAML config |
| `augment.py` | ~270 | Medium | Dataset augmentation (Albumentations) |
| `utils.py` | ~230 | Low | Dataset split, CVAT→YOLO conversion |
| `extract_frames.py` | ~110 | Low | Extract frames from video for annotation |

### Batch processing system (`batch/`)

- **`database/models.py`** — SQLAlchemy ORM: Catalog → Dive → Video hierarchy, plus Model, Task, SubTask, Output entities. Task status: PENDING → RUNNING → DONE/ERROR. SubTask types: GEOMETRY, SIZE, VOLUME, ANALYSIS, SIZE_VIDEO_RENDER
- **`database/repository.py`** — data access layer wrapping all DB queries (CRUD for all entities, task queue management, output tracking)
- **`core/task_manager.py`** — QObject managing the task queue with Qt signals; `core/worker.py` runs tasks in a background QThread with pause/resume via QMutex+QWaitCondition
- **`core/processor.py`** — wraps `detect_video.py` logic with progress callbacks and cancellation; `geometry_processor.py` wraps camera_geometry.py and render_size_video.py; `analyze_processor.py` wraps analyze.py
- **`core/config.py`** — dataclasses (Config, DetectionParams, UISettings) loaded from config.json
- **`ui/main_window.py`** — MainWindow with DivePanel (catalog→dive→video tree), ModelPanel, TaskTable (tree with subtasks), StatusBarWidget
- **`ui/dialogs/`** — NewTaskDialog, EditTaskDialog, PostProcessDialog, AddDiveDialog, AddModelDialog, CatalogDialog, GeometryDialog, AnalyzeDialog

### Camera geometry pipeline (`src/camera_geometry.py`)

The largest and most complex module (~1700 lines). Three subcommands:
- `geometry` — estimates camera tilt from marine snow motion (Focus of Expansion via Nelder-Mead optimization)
- `size` — calculates real object sizes via k-method: k = (Δpixels/pixels₁) / Δdepth, with MAD outlier filtering, moving median smoothing, tilt correction
- `volume` — computes surveyed water volume (frustum model) and organism density

Key formulas: distance `d = 24.68 × |k|^(-0.644)`, pixel calibration `p = 2.432 × d^(-1.0334)`, size `size_real = (size_pixels × distance) / K`. Calibrated for GoPro 12 Wide 4K (K=2365 px/m at 1m).

### Detection classes (5 species)

Defined in `data.yaml` and `dataset/classes.txt`:
0: Aurelia aurita, 1: Beroe ovata, 2: Mnemiopsis leidyi, 3: Pleurobrachia pileus, 4: Rhizostoma pulmo

Class IDs are in alphabetical order, defined authoritatively in `data.yaml`.

## Key Configuration Files

- **`config.json`** — app settings: DB path, default detection params (conf=0.75, ByteTrack tracker), UI state
- **`data.yaml`** — YOLO dataset config (paths, 5 classes)
- **`train_config.yaml`** — training hyperparameters (epochs=150, batch=4, imgsz=1920, AdamW)
- **`batch.db`** — SQLite database for the batch processing system

## Known Issues and Technical Debt

### Critical: Class names hardcoded in 5 places
Class names are duplicated in `data.yaml`, `dataset/classes.txt`, `src/detect_video.py` (CLASS_NAMES dict), `batch/core/processor.py` (CLASS_NAMES dict copy), and `src/analyze.py`/`src/interactive_plot.py` (SPECIES_COLORS/SPECIES_NAMES_RU). Changing class order in `data.yaml` for retraining will break detection scripts. Should be centralized.

### Config parameter mismatches
| Parameter | config.json | Code default | README |
|-----------|-------------|-------------|--------|
| conf_threshold | 0.75 | 0.25 | 0.25 |
| min_track_length | 5 | 3 | 3 |
| imgsz (training) | 1920 (train_config.yaml) | 640 | 640 |
| batch (training) | 4 (train_config.yaml) | 16 | 16 |

### Other issues
- **DB thread safety**: Worker (QThread) and UI both access SQLite; sessions created per-operation but no thread isolation guarantee
- **No transaction atomicity**: Task + SubTask creation not atomic; partial failure leaves inconsistent state
- **Camera calibration hardcoded**: K=2365 for GoPro 12 Wide 4K in camera_geometry.py; different camera requires code change
- **Path resolution inconsistent**: train.py uses PROJECT_ROOT (robust), detect_video.py assumes CWD (fragile), batch_app.py uses absolute paths (robust)
- **No logging**: only traceback.print_exc(); no structured logging
- **No input validation**: CSV files not checked for required columns; YAML/JSON configs not validated against schema
- **camera_geometry.py edge cases**: division risk when delta_depth ≈ 0 (0.01 threshold), MAD=0 fallback, hardcoded smoothing window=3, magic number thresholds (2%, 0.3/0.6/0.2 confidence)
- **No test suite**
- **requirements.txt**: only lower version bounds, no upper limits
