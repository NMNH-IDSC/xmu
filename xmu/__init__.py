"""Reads and writes XML for Axiell EMu"""

from .api import (
    EMuAPI,
    EMuAPIParser,
    EMuAPIResponse,
    contains,
    emu_escape,
    emu_unescape,
    exact,
    exists,
    is_not_null,
    is_null,
    order,
    phrase,
    phonetic,
    proximity,
    range_,
    regex,
    stemmed,
)
from .containers import (
    EMuColumn,
    EMuConfig,
    EMuEncoder,
    EMuGrid,
    EMuRow,
    EMuRecord,
    EMuSchema,
)
from .io import EMuReader, clean_xml, write_csv, write_group, write_import, write_xml
from .types import EMuDate, EMuFloat, EMuLatitude, EMuLongitude, EMuTime, EMuType
from .utils import (
    flatten,
    get_mod,
    get_ref,
    get_tab,
    has_mod,
    is_nesttab,
    is_nesttab_inner,
    is_ref,
    is_ref_tab,
    is_tab,
    split_field,
    strip_mod,
    strip_tab,
    to_ref,
)

__version__ = "0.1b9"
__author__ = "Adam Mansur"
__credits__ = "Smithsonian National Museum of Natural History"
