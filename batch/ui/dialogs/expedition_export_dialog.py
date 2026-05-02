"""
Диалог экспорта данных экспедиции — открывает Save As и вызывает сбор CSV.
"""

from PyQt6.QtWidgets import QFileDialog, QMessageBox


def export_expedition_data(parent, repo, catalog_id: int) -> None:
    catalog = repo.get_catalog(catalog_id)
    if not catalog:
        return

    default_name = f"{catalog.name}_summary.csv"
    filepath, _ = QFileDialog.getSaveFileName(
        parent,
        f'Экспорт экспедиции «{catalog.name}»',
        default_name,
        "CSV (*.csv)",
    )
    if not filepath:
        return

    try:
        from ...core.expedition_exporter import export_expedition_csv

        total, with_data = export_expedition_csv(repo, catalog_id, filepath)

        if total == 0:
            QMessageBox.information(
                parent,
                "Экспорт",
                "В экспедиции нет завершённых задач.",
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
