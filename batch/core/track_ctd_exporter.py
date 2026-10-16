"""Подготовка источников экспорта треков с CTD из базы задач."""

from pathlib import Path

from .expedition_exporter import _get_export_tasks
from ..database.models import OutputType
from tracks_ctd import export_tracks_ctd_csv


def export_task_tracks_ctd(repo, output_path, catalog_id=None, task_ids=None,
                           dive_id=None, ctd_columns=None, all_parameters=False):
    """Экспортирует готовые измерения погружения, выбранных задач или экспедиции."""
    tasks = (repo.get_completed_tasks_by_dive(dive_id) if dive_id is not None
             else _get_export_tasks(repo, catalog_id, task_ids))
    dive_ids = {task.video_file.dive_id for task in tasks if task.video_file}
    single_dive = len(dive_ids) == 1
    sources, warnings = [], []
    seen_videos = set()
    for task in tasks:
        video = task.video_file
        dive = video.dive if video else None
        if not dive or video.id in seen_videos:
            continue
        output = repo.get_task_output_by_type(task.id, OutputType.TRACK_SIZES_CSV)
        path = Path(output.filepath) if output and output.filepath else None
        if path is None or not path.is_file():
            path = Path(dive.folder_path) / "output" / f"{Path(video.filename).stem}_track_sizes.csv"
        if not path.is_file():
            warnings.append(f"{dive.name} / {video.filename}: нет готового CSV размеров треков.")
            continue
        seen_videos.add(video.id)
        ctd = repo.get_ctd_file(task.ctd_id) if task.ctd_id else None
        layer_output = repo.get_task_output_by_type(task.id, OutputType.LAYER_DENSITY_CSV)
        layers = Path(layer_output.filepath) if layer_output and layer_output.filepath else None
        if layers is None or not layers.is_file():
            layers = Path(dive.folder_path) / "output" / "analysis" / "layers.csv"
        catalog = repo.get_catalog(dive.catalog_id) if dive.catalog_id is not None else None
        expedition = catalog.name if catalog else "Без экспедиции"
        sources.append(dict(
            track_sizes_path=str(path), station=dive.name,
            ctd_path=ctd.filepath if ctd else None, layers_path=str(layers),
            id_prefix="" if single_dive else (
                f"{expedition}_{dive.name}_{video.filename}_d{dive.id}_v{video.id}_"
            ),
        ))
    count, messages = export_tracks_ctd_csv(sources, output_path, ctd_columns, all_parameters)
    if not tasks:
        warnings.append("Нет завершённых задач для экспорта.")
    return count, warnings + messages
