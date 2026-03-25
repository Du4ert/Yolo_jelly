"""
Вспомогательная функция для выбора нескольких папок в одном диалоге.
"""

from typing import List

from PyQt6.QtWidgets import QFileDialog, QListView, QTreeView, QAbstractItemView


def select_multiple_directories(parent=None, title: str = "Выберите папки", start_path: str = "") -> List[str]:
    """
    Открывает диалог выбора папки с поддержкой множественного выделения
    (Ctrl+клик, Shift+клик).

    Qt не поддерживает мульти-выбор директорий в нативном диалоге, поэтому
    используется не-нативный режим с ручным переключением внутренних view в
    ExtendedSelection.

    Возвращает список выбранных путей (пустой, если пользователь отменил).
    """
    dialog = QFileDialog(parent, title, start_path)
    dialog.setFileMode(QFileDialog.FileMode.Directory)
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)

    for view in (
        dialog.findChild(QListView, "listView"),
        dialog.findChild(QTreeView),
    ):
        if view is not None:
            view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

    if dialog.exec():
        return dialog.selectedFiles()
    return []
