"""
Диалоги экспорта экспедиции или выбранных задач — сохранение и сбор CSV.
"""

from PyQt6.QtWidgets import QFileDialog, QMessageBox


def _get_export_scope(repo, catalog_id, task_ids):
    """Возвращает имя файла, описание выборки и сообщение для пустой выборки."""
    if task_ids is not None:
        return (
            "selected_dives",
            "выбранных погружений",
            "Среди выбранных задач нет завершённых.",
        )
    catalog = repo.get_catalog(catalog_id)
    if not catalog:
        return None
    return catalog.name, f'экспедиции «{catalog.name}»', "В экспедиции нет завершённых задач."


def export_expedition_data(
    parent, repo, catalog_id: int | None = None, task_ids: list[int] | None = None,
) -> None:
    scope = _get_export_scope(repo, catalog_id, task_ids)
    if scope is None:
        return
    name, description, empty_message = scope

    default_name = f"{name}_summary.csv"
    filepath, _ = QFileDialog.getSaveFileName(
        parent,
        f"Экспорт {description}",
        default_name,
        "CSV (*.csv)",
    )
    if not filepath:
        return

    try:
        from ...core.expedition_exporter import export_expedition_csv

        total, with_data = export_expedition_csv(repo, catalog_id, filepath, task_ids)

        if total == 0:
            QMessageBox.information(
                parent,
                "Экспорт",
                empty_message,
            )
        elif total == with_data:
            QMessageBox.information(
                parent,
                "Экспорт",
                f"Экспортировано {total} задач в {filepath}",
            )
        else:
            zero_count = total - with_data
            QMessageBox.information(
                parent,
                "Экспорт",
                f"Экспортировано {total} задач в {filepath}\n"
                f"({zero_count} задач без данных о объёме)",
            )
    except Exception as e:
        QMessageBox.warning(
            parent,
            "Ошибка экспорта",
            f"Не удалось выполнить экспорт:\n{e}",
        )


def export_expedition_tracks(
    parent, repo, catalog_id: int | None = None, task_ids: list[int] | None = None,
) -> None:
    scope = _get_export_scope(repo, catalog_id, task_ids)
    if scope is None:
        return
    name, description, empty_message = scope

    default_name = f"{name}_tracks.csv"
    filepath, _ = QFileDialog.getSaveFileName(
        parent,
        f"Экспорт треков {description}",
        default_name,
        "CSV (*.csv)",
    )
    if not filepath:
        return

    try:
        from ...core.expedition_exporter import export_expedition_tracks_csv

        total, files_with_data, rows_count = export_expedition_tracks_csv(
            repo, catalog_id, filepath, task_ids
        )

        if total == 0:
            QMessageBox.information(
                parent,
                "Экспорт треков",
                empty_message,
            )
        elif rows_count == 0:
            QMessageBox.information(
                parent,
                "Экспорт треков",
                "Не найдено данных _track_sizes.csv для экспорта.",
            )
        else:
            QMessageBox.information(
                parent,
                "Экспорт треков",
                f"Экспортировано {rows_count} треков из {files_with_data} файлов в {filepath}",
            )
    except Exception as e:
        QMessageBox.warning(
            parent,
            "Ошибка экспорта",
            f"Не удалось выполнить экспорт треков:\n{e}",
        )


def export_expedition_layers(
    parent, repo, catalog_id: int | None = None, task_ids: list[int] | None = None,
) -> None:
    scope = _get_export_scope(repo, catalog_id, task_ids)
    if scope is None:
        return
    name, description, empty_message = scope

    default_name = f"{name}_layers.csv"
    filepath, _ = QFileDialog.getSaveFileName(
        parent,
        f"Экспорт слоёв {description}",
        default_name,
        "CSV (*.csv)",
    )
    if not filepath:
        return

    try:
        from ...core.expedition_exporter import export_expedition_layers_csv

        total, files_with_data, rows_count = export_expedition_layers_csv(
            repo, catalog_id, filepath, task_ids
        )

        if total == 0:
            QMessageBox.information(
                parent,
                "Экспорт слоёв",
                empty_message,
            )
        elif rows_count == 0:
            QMessageBox.information(
                parent,
                "Экспорт слоёв",
                "Не найдено файлов layers.csv для экспорта.",
            )
        else:
            QMessageBox.information(
                parent,
                "Экспорт слоёв",
                f"Экспортировано {rows_count} строк из "
                f"{files_with_data} погружений в {filepath}",
            )
    except Exception as e:
        QMessageBox.warning(
            parent,
            "Ошибка экспорта",
            f"Не удалось выполнить экспорт слоёв:\n{e}",
        )
