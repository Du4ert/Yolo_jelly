# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLO-based jellyfish detection and classification system for underwater video analysis. Targets 5 species of gelatinous macrozooplankton in the Black Sea. Dual-mode architecture: CLI tools for individual processing steps and GUI batch processor with database-backed task queue.

**Target Species (0-4):**
- Aurelia aurita (jellyfish)
- Beroe ovata (ctenophore)
- Mnemiopsis leidyi (ctenophore)
- Pleurobrachia pileus (ctenophore)
- Rhizostoma pulmo (jellyfish)

## Development Environment

**Virtual Environment:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

**Dependencies:**
```bash
pip install -r requirements.txt
```

**Check GPU (optional but recommended):**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Application Entry Points

### GUI Batch Processor (Primary Tool)
```bash
python batch_app.py
```
- PyQt6-based interface for managing expeditions, dives, and processing queue
- Database: `batch.db` (SQLAlchemy)
- Config: `config.json`
- Keyboard shortcuts: Ctrl+D (add dive), Ctrl+M (add model), Ctrl+G (geometry), Ctrl+A (analyze), F5/F6/F7 (queue controls)

### CLI Tools (src/)
Individual processing steps for custom workflows:
```bash
# Extract frames for annotation
python src/extract_frames.py --video path/to/video.mp4 --interval 15

# Train model
python src/train.py --data data.yaml --epochs 150 --model yolov8m.pt

# Detect with tracking
python src/detect_video.py --video input.mp4 --model best.pt --track --csv output.csv

# Camera geometry (subcommands: geometry/size/calibrate/volume)
python src/camera_geometry.py geometry --video input.mp4 --output geometry.csv
python src/camera_geometry.py size --detections detections.csv --geometry geometry.csv
python src/camera_geometry.py volume --detections size.csv --tracks track_sizes.csv

# Analysis and visualization
python src/analyze.py --csv detections.csv --output-dir output/
python src/interactive_plot.py --track-sizes track_sizes.csv --ctd ctd.csv --ctd-columns 6
```

## Architecture

### Two-Layer Design

**Layer 1: CLI Tools (src/)** - Stateless, single-purpose processors:
- `detect_video.py` - YOLO inference + ByteTrack/BoTSORT tracking
- `camera_geometry.py` - Multi-mode: FOE tilt estimation, size calculation, calibration, volume estimation
- `analyze.py` - Matplotlib static plots
- `interactive_plot.py` - Plotly interactive depth profiles with CTD overlay
- `train.py`, `augment.py`, `extract_frames.py` - Training pipeline utilities

**Layer 2: Batch Processor (batch/)** - Stateful orchestrator:
- `batch_app.py` - Entry point, dependency checker
- `batch/ui/main_window.py` - Main GUI controller
- `batch/core/processor.py` - Wraps `detect_video.py` with progress callbacks
- `batch/core/task_manager.py` - Queue execution, subtask dependency resolution
- `batch/database/` - SQLAlchemy models and repository pattern
- `batch/ui/dialogs/` - Task configuration, geometry tools, analysis dialogs

### Scientific Workflow Pipeline

Standard processing sequence (enforced by `CANONICAL_SUBTASK_ORDER`):

1. **INFERENCE** - YOLO detection + tracking → `detections.csv`, `tracks.csv`
2. **GEOMETRY** - Focus of Expansion analysis → `geometry.csv` (camera tilt)
3. **SIZE** - Optical growth method → `detections_with_size.csv`, `track_sizes.csv`
4. **SIZE_VIDEO_RENDER** - Visual size overlay (optional QC)
5. **VOLUME** - Cylindrical model → `volume.csv` (density per species)
6. **ANALYSIS** - Plots + report + interactive HTML

### Database Schema (batch/database/models.py)

**Core entities:**
- `Expedition` → `Dive` → `Video` (hierarchical organization)
- `YoloModel` (model registry with metadata)
- `Task` → `SubTask` (processing queue with dependencies)
- `TaskOutput` (file tracking with `OutputType` enum)

**Task lifecycle:**
- Status: `PENDING → RUNNING → DONE` (or `ERROR/CANCELLED/PAUSED`)
- SubTask types: `INFERENCE, GEOMETRY, SIZE, VOLUME, ANALYSIS, SIZE_VIDEO_RENDER, LABEL_STUDIO_EXPORT`

### Camera Geometry & Size Estimation (src/camera_geometry.py)

**Critical scientific module** with four subcommands:

**1. `geometry` - Camera tilt via Focus of Expansion (FOE)**
- Analyzes marine snow particle motion
- Outputs: `foe_x`, `foe_y`, `tilt_horizontal_deg`, `tilt_vertical_deg`, `confidence`
- Low confidence (< 0.5) = outlier/insufficient motion

**2. `size` - Object size via optical growth method**
- Primary method: k-method (growth rate: `k = (Δpixels/pixels₁) / Δdepth`)
- Tilt correction: `k_real = k_measured / cos(θ)` (15° tilt → +3.5%, 30° → +15.5%, 45° → +41.4%)
- Fallback: fixed sizes for small species (P. pileus = 10mm), typical sizes
- Outputs: `estimated_size_mm`, `distance_to_object_m`, `object_depth_m`, `size_confidence`, `size_method`

**3. `calibrate` - Calibration coefficient fitting**
- Decomposes 6 parameters: A,B (distance), C,D (pixel), k1,k2 (distortion)
- Modes: with `--known-depth` (decomposed, robust) or without (joint optimization)
- GoPro fisheye correction: `size_corrected = size_raw * (1 + k1*r² + k2*r⁴)`

**4. `volume` - Water volume via cylindrical model**
- `V = A_eff · H`, where `A_eff = (π/4) · w · h` at effective distance `d_eff`
- `d_eff` = P90 of max detection distances (empirical visibility range)
- Outputs: `total_volume_m3`, `density_<Species>_per_m3`

**Calibration constants (CameraCalibration dataclass):**
- GoPro 12 Wide 4K (3840×2160, 156° FOV)
- Distance formula: `d = 80.00 * |k|^(-0.9)` meters
- Pixel calibration: `p = 4.35 * d^(-1.25)` px/mm
- Reliable range: 0.1–3.0 m

### Tracking & ID Normalization (src/tracking_utils.py)

**Critical for multi-segment processing:**
- `normalize_track_ids()` - Renumbers sparse IDs to dense sequence (1,2,3...)
- `patch_ultralytics_track_ids()` - Injects custom ID state into tracker
- `rebuild_track_state()` - Reconstructs tracker state from CSV (for resume/merge)

**Why needed:** ByteTrack/BoTSORT create sparse ID sequences (1, 5, 9, 143...). Database foreign keys and analysis tools expect dense sequences.

### Video I/O (src/video_utils.py)

- `ThreadedVideoCapture` - Async frame reading (avoids blocking on HDD/network)
- `NvencVideoWriter` - Hardware-accelerated H.264/H.265 encoding (NVIDIA NVENC)

### CTD Integration

**Format:** CSV with `time` and `depth` columns (comma/semicolon/pipe delimiters)
- `detect_video.py` interpolates depth to frame timestamps
- `interactive_plot.py` overlays normalized CTD profiles (temperature, salinity, etc.)
- CTD columns specified by 0-based index: `--ctd-columns 6,7`

## Common Tasks

### Full Processing Pipeline (CLI)
```bash
# 1. Detection with tracking
python src/detect_video.py --video dive.mp4 --model best.pt --track --ctd ctd.csv --csv detections.csv

# 2. Camera tilt estimation
python src/camera_geometry.py geometry --video dive.mp4 --output geometry.csv

# 3. Size estimation with tilt correction
python src/camera_geometry.py size --detections detections.csv --geometry geometry.csv

# 4. Volume and density calculation
python src/camera_geometry.py volume --detections detections_with_size.csv --tracks track_sizes.csv --ctd ctd.csv

# 5. Analysis and visualization
python src/analyze.py --csv detections_with_size.csv --output-dir output/
python src/interactive_plot.py --track-sizes track_sizes.csv --ctd ctd.csv --ctd-columns 6 --output plot
```

### Training Workflow
```bash
# 1. Extract frames for annotation
python src/extract_frames.py --video raw.mp4 --output dataset/raw_frames --interval 15

# 2. Augment training data
python src/augment.py --images dataset/images/train --labels dataset/labels/train --multiply 4

# 3. Train model
python src/train.py --data data.yaml --epochs 150 --model yolov8m.pt --batch 16

# Trained weights: runs/train/jellyfish_v1/weights/best.pt
```

### Calibration Workflow
```bash
# Record test video with known-size objects at known depth
# Detect + track
python src/detect_video.py --video calibration.mp4 --model best.pt --track --csv calib_detections.csv

# Geometry analysis
python src/camera_geometry.py geometry --video calibration.mp4 --output calib_geometry.csv

# Calibrate (example: track_id=1 at 67mm and 67.076m depth)
python src/camera_geometry.py calibrate \
    --detections calib_detections.csv \
    --geometry calib_geometry.csv \
    --known-size 1:67.0 \
    --known-depth 1:67.076 \
    --output calibration.json

# Apply calibration
python src/camera_geometry.py size \
    --detections detections.csv \
    --geometry geometry.csv \
    --calibration calibration.json
```

### Database Operations (batch/)

**Direct SQL access:**
```python
from batch.database.models import Base, Task, Dive, Video
from batch.database.repository import Repository
from sqlalchemy import create_engine

engine = create_engine('sqlite:///batch.db')
repo = Repository(engine)

# Query pending tasks
pending = repo.session.query(Task).filter(Task.status == TaskStatus.PENDING).all()

# Add dive programmatically
dive = Dive(name="Dive_001", expedition_id=1, date=datetime.now().date())
repo.session.add(dive)
repo.session.commit()
```

**Subtask dependency resolution:** TaskManager automatically orders subtasks by `CANONICAL_SUBTASK_ORDER` and blocks dependent tasks until prerequisites complete.

## Code Patterns & Conventions

### Progress Callbacks
All processors accept `progress_callback: Optional[Callable[[int, int, str], None]]`:
```python
def progress_callback(current: int, total: int, message: str):
    print(f"[{current}/{total}] {message}")

processor.run(video_path, model_path, progress_callback=progress_callback)
```

### Cancellation & Pause
Processors check `self._cancelled` and `self._paused` flags:
```python
class VideoProcessor:
    def __init__(self):
        self._cancelled = False
        self._paused = False

    def cancel(self):
        self._cancelled = True

    def pause(self):
        self._paused = True
```

### Result Dataclasses
Use typed results instead of tuples:
```python
@dataclass
class ProcessingResult:
    success: bool
    detections_count: int
    error_message: Optional[str] = None
```

### Output File Naming Convention
```
<base>_detections.csv                 # Raw detections
<base>_detections_tracks.csv          # Track stats
<base>_detections_geometry.csv        # Camera geometry
<base>_detections_with_size.csv       # Detections + size estimates
<base>_detections_track_sizes.csv     # Track-level sizes
<base>_detections_volume.csv          # Volume calculation
```

## Critical Implementation Details

### Track ID Continuity
**Problem:** YOLO trackers create sparse IDs (1, 5, 9, 143...) which break foreign key constraints.

**Solution:** Always call `normalize_track_ids(df)` on detection DataFrames before writing to CSV or database.

### Tilt Correction is Optional But Important
Size estimates assume vertical camera movement. For tilted cameras (> 15°):
1. Run `camera_geometry.py geometry` to get tilt
2. Pass `--geometry geometry.csv` to `camera_geometry.py size`
3. Without correction, 30° tilt → 15% size underestimate

### GPU Memory Management
**Detection:** Use `--imgsz 1280` (default) for 8GB VRAM. If OOM → reduce to 960 or 640.

**Training:** Default batch=16. If OOM → reduce to 8 or 4.

**TensorRT:** For production inference, export to `.engine`:
```bash
python src/detect_video.py --video dummy.mp4 --model best.pt --export-engine
# Creates best.engine, auto-loaded in future runs
```

### CSV Encoding & Delimiters
CTD files support multiple delimiters: `|`, `;`, `\t`, `,` (auto-detected). Always use UTF-8 encoding.

### Distortion Correction
GoPro Wide 156° FOV introduces radial distortion (objects appear larger at frame edges). Calibration fits k1, k2 coefficients. Without calibration, edge objects may have 10-20% size errors.

## File Locations & Data Flow

**Input:**
- Videos: User-specified (GUI: organized by expedition/dive)
- YOLO models: `runs/train/<name>/weights/best.pt` or custom paths
- CTD data: Expedition-specific CSV files
- Dataset: `dataset/images/{train,val,test}` + `dataset/labels/{train,val,test}`

**Output:**
- Batch processing: `<video_dir>/<video_name>_<suffix>.<ext>`
- CLI tools: `output/` directory (configurable)
- Training: `runs/train/<experiment_name>/`
- Database: `batch.db` (SQLite, root directory)
- Config: `config.json` (model registry, paths)

**Temporary:**
- ThreadedVideoCapture cache: Frame buffers in memory
- Label Studio export: Frames written to `--export-ls` directory

## Troubleshooting

**High false positive rate (marine snow):**
- Increase `--conf` threshold to 0.4-0.5
- Add "particle" class to training data as negative examples

**Unrealistic tilt angles:**
- FOE requires sufficient camera motion (> 0.5 m/s descent)
- Check `confidence` column in geometry.csv (< 0.5 = unreliable)

**Poor size estimates:**
- Verify depth change > 0.1 m during track lifetime
- Check `size_method` in output: `k_method` is best, `typical` is fallback
- Apply tilt correction if camera angle > 15°

**GPU OOM during detection:**
- Reduce `--imgsz` to 960 or 640
- Use smaller model (yolov8s.pt instead of yolov8m.pt)
- Disable half precision with `--no-half` (doubles memory but may help with driver issues)

**Task stuck in RUNNING state:**
- Check worker logs for exceptions
- Restart batch_app.py (TaskManager recovers gracefully)
- Manual DB fix: `UPDATE tasks SET status='pending' WHERE id=X`

## Testing & Quality Control

**No formal test suite** - validation via:
1. Training metrics: `runs/train/<name>/results.png` (mAP, precision, recall)
2. Visual QC: Output videos with bounding boxes + track IDs
3. Size validation: Compare `track_sizes.csv` against manual measurements
4. Volume sanity check: Density values vs. literature ranges

**Verification UI (batch/ui/dialogs/verify_dialog.py):**
- Manual review of detections with frame-by-frame navigation
- Delete false positives, reclassify misidentifications
- Outputs `_verified.csv` with cleaned detections

## Performance Optimization

**Detection speed (RTX 3060, 4K video):**
- FP16 inference: ~30-40 FPS
- TensorRT engine: ~60-80 FPS (2x speedup)
- CPU-only: ~2-5 FPS

**Batch processing parallelization:**
- GUI processes tasks sequentially (one worker thread)
- For parallel processing, run multiple instances of CLI tools with different videos

**I/O bottlenecks:**
- Use SSD for video input (HDD → 50% speed penalty)
- `ThreadedVideoCapture` mitigates HDD latency via async reads
- `NvencVideoWriter` offloads encoding to GPU (vs. CPU-bound OpenCV)

## Language & Documentation

**Primary language:** Russian (README.md, UI labels, comments)

**English used for:** Code identifiers, Git commits, technical terms

**When modifying UI:** Maintain Russian labels for consistency with existing interface.
