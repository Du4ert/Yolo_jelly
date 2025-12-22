"""
Диалоговые окна.
"""

from .new_task import NewTaskDialog
from .add_dive import AddDiveDialog
from .add_model import AddModelDialog
from .edit_task import EditTaskDialog
from .edit_dive import EditDiveDialog
from .catalog_dialog import CatalogDialog

__all__ = [
    "NewTaskDialog",
    "AddDiveDialog",
    "AddModelDialog",
    "EditTaskDialog",
    "EditDiveDialog",
    "CatalogDialog",
]
