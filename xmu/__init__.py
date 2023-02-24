"""Reads and writes XML for Axiell EMu"""
from .containers import EMuColumn, EMuConfig, EMuGrid, EMuRow, EMuRecord, EMuSchema
from .io import EMuReader, write_group, write_import
from .types import EMuDate, EMuFloat, EMuLatitude, EMuLongitude, EMuTime, EMuType
from .utils import (
    get_mod,
    has_mod,
    is_nesttab,
    is_nesttab_inner,
    is_ref,
    is_ref_tab,
    is_tab,
    strip_mod,
    strip_tab,
)

__version__ = "0.1b5"
__author__ = "Adam Mansur"
__credits__ = "Smithsonian National Museum of Natural History"
