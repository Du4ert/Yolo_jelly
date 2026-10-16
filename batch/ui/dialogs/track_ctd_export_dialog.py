"""Параметры экспорта готовых треков с данными CTD."""

from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QLabel, QLineEdit, QMessageBox, QVBoxLayout,
)


class TrackCTDExportDialog(QDialog):
    """Выбирает параметры CTD и сохраняет общую таблицу организмов."""

    def __init__(self, repo, parent=None, catalog_id=None, task_ids=None, dive_id=None):
        super().__init__(parent)
        self.repo = repo
        self.scope = dict(catalog_id=catalog_id, task_ids=task_ids, dive_id=dive_id)
        self.default_name = "selected_dives_tracks_ctd.csv"
        title = "Экспорт треков с CTD выбранных погружений"
        if dive_id is not None:
            dive = repo.get_dive(dive_id)
            if dive:
                title = f'Экспорт треков с CTD погружения «{dive.name}»'
                self.default_name = f"{dive.name}_tracks_ctd.csv"
        elif catalog_id is not None:
            catalog = repo.get_catalog(catalog_id)
            if catalog:
                title = f'Экспорт треков с CTD экспедиции «{catalog.name}»'
                self.default_name = f"{catalog.name}_tracks_ctd.csv"
        self.setWindowTitle(title)
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)
        info = QLabel("Глубины и размеры берутся из готовых расчётов.\n"
                      "Слои — из layers.csv; при отсутствии файла поле layer пустое.")
        info.setWordWrap(True)
        layout.addWidget(info)
        form = QFormLayout()
        self.columns = QLineEdit("6,11,12,16")
        self.columns.setToolTip("Индексы колонок CTD с нуля, как при построении графика.")
        form.addRow("Колонки CTD:", self.columns)
        self.all_parameters = QCheckBox("Все параметры")
        self.all_parameters.toggled.connect(lambda checked: self.columns.setEnabled(not checked))
        form.addRow(self.all_parameters)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._export)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _export(self):
        from ...core.track_ctd_exporter import export_task_tracks_ctd
        from tracks_ctd import parse_ctd_columns

        try:
            columns = None if self.all_parameters.isChecked() else parse_ctd_columns(self.columns.text())
        except ValueError as error:
            QMessageBox.warning(self, "Параметры CTD", str(error))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, self.windowTitle(), self.default_name, "CSV (*.csv)"
        )
        if not path:
            return
        try:
            count, warnings = export_task_tracks_ctd(
                self.repo, path, ctd_columns=columns,
                all_parameters=self.all_parameters.isChecked(), **self.scope,
            )
        except Exception as error:
            QMessageBox.warning(self, "Ошибка экспорта", str(error))
            return
        message = QMessageBox(self)
        message.setWindowTitle("Экспорт треков с CTD")
        message.setText(f"Экспортировано организмов: {count}\nФайл: {path}")
        if warnings:
            message.setInformativeText("Есть замечания к исходным данным. Откройте подробности.")
            message.setDetailedText("\n".join(warnings))
        message.exec()
        self.accept()
