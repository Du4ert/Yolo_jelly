# AGENTS.md

## Quick start

```bash
pip install -r requirements.txt
python batch_app.py                    # GUI
python src/detect_video.py --video ... --model ... --track --csv out.csv   # CLI
python src/train.py --config train_config.yaml
```

No test suite, no linter, no formatter, no CI.

## Architecture

Two parallel interfaces, same pipeline:

- **GUI entrypoint**: `batch_app.py` — PyQt6 app. Imports from `batch/`. Uses `sys.path.insert(0, ...)` at top.
- **CLI scripts**: `src/detect_video.py`, `src/train.py`, `src/camera_geometry.py`, etc.
- **`batch/core/processor.py`** — reimplements detection/tracking logic in parallel with `src/detect_video.py`. Imports from `src/` via `sys.path.insert(0, ...)`.

## Class names (5 species)

**Source of truth**: `data.yaml` for ID→name mapping.
**Centralized for `src/`**: `src/constants.py` exports `CLASS_NAMES`, `SPECIES_COLORS`, `SPECIES_NAMES_RU`, `CLASS_PRIORITY`, `CLASS_COLORS_BGR`.
**Still duplicated in**: `batch/core/processor.py` (lines 49-55) — its own `CLASS_NAMES` dict. If you change class order in `data.yaml`, update both.

```yaml
0: Aurelia aurita     1: Beroe ovata        2: Mnemiopsis leidyi
3: Pleurobrachia pileus  4: Rhizostoma pulmo
```

## Config files (source of truth vs doc)

| File | Role | Key actual values |
|------|------|-------------------|
| `config.json` | App/UI settings | conf_threshold=0.6, ByteTrack, imgsz=1280 |
| `data.yaml` | Dataset + class names | 5 classes, alphabetical order |
| `train_config.yaml` | Training hyperparams | epochs=180, batch=4, imgsz=1280, AdamW |

README.md and CLAUDE.md have stale defaults — trust the YAML/JSON files.

## Camera geometry

Calibrated for GoPro 12 Wide 4K (3840×2160, FOV 156°). Default calibration constants in `CameraCalibration` dataclass in `src/camera_geometry.py`. Override via `--calibration calibration.json`.

## Gotchas

- All comments and documentation in Russian.
- `dataset/`, `output/`, `runs/`, `test_video/` are gitignored — not in the repo.
- `batch.db` (SQLite) is gitignored — created on first GUI run.
- Training batch size 4 fits ~8 GB VRAM at imgsz=1280. For lower VRAM, reduce imgsz or batch in `train_config.yaml`.
