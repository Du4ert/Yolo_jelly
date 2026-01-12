"""
Диалоговые окна.
"""

from .new_task import NewTaskDialog
from .add_dive import AddDiveDialog
from .add_model import AddModelDialog
from .edit_task import EditTaskDialog
from .edit_dive import EditDiveDialog
from .catalog_dialog import CatalogDialog
from .geometry_dialog import GeometryDialog
from .analyze_dialog import AnalyzeDialog

__all__ = [
    "NewTaskDialog",
    "AddDiveDialog",
    "AddModelDialog",
    "EditTaskDialog",
    "EditDiveDialog",
    "CatalogDialog",
    "GeometryDialog",
    "AnalyzeDialog",
]
