"""Functions to assess whether EMu field names are tables, references, etc."""
import re
from functools import lru_cache

#: tuple : suffixes that designate tables in EMu
TAB_SUFFIXES = ("0", "_nesttab", "_nesttab_inner", "_tab")

#: tuple : suffixes that designate references in EMu
REF_SUFFIXES = ("Ref", "Ref_nesttab", "Ref_nesttab_inner", "Ref_tab")

#: tuple : suffixes that designate nested tables in EMu
NESTTAB_SUFFIXES = ("_nesttab",)

#: tuple : suffixes that designate inner nested tables in EMu
NESTTAB_INNER_SUFFIXES = ("_nesttab_inner",)

#: str : pattern that matches table suffixes
TAB_PATTERN = "(" + "|".join(TAB_SUFFIXES) + ")$"

#: str : pattern that matches reference suffixes
REF_PATTERN = "(" + "|".join(REF_SUFFIXES) + ")$"

#: str : pattern that matches update modifiers
MOD_PATTERN = r"\((\d+=|[\+\-])\)$"


@lru_cache(maxsize=None)
def is_tab(field):
    """Checks if a field name is a table

    Parameters
    ----------
    field : str
        field name

    Returns
    -------
    bool
        True if field name is a table, False if not
    """
    return strip_mod(field).endswith(TAB_SUFFIXES)


@lru_cache(maxsize=None)
def is_nesttab(field):
    """Checks if a field name is a nested table

    Parameters
    ----------
    field : str
        field name

    Returns
    -------
    bool
        True if field name is a nested table, False if not
    """
    return strip_mod(field).endswith(NESTTAB_SUFFIXES)


@lru_cache(maxsize=None)
def is_nesttab_inner(field):
    """Checks if a field name is an inner nested table

    Parameters
    ----------
    field : str
        field name

    Returns
    -------
    bool
        True if field name is an inner nested table, False if not
    """
    return strip_mod(field).endswith(NESTTAB_INNER_SUFFIXES)


@lru_cache(maxsize=None)
def is_ref_tab(field):
    """Checks if a field name is a reference table

    Parameters
    ----------
    field : str
        field name

    Returns
    -------
    bool
        True if field name is a reference table, False if not
    """
    return is_tab(field) and is_ref(field)


@lru_cache(maxsize=None)
def is_ref(field):
    """Checks if a field name is a reference

    Parameters
    ----------
    field : str
        field name

    Returns
    -------
    bool
        True if field name is a reference, False if not
    """
    return strip_mod(field).endswith(REF_SUFFIXES)


@lru_cache(maxsize=None)
def has_mod(field):
    """Checks if a field name ends with an update modifier

    Parameters
    ----------
    field : str
        field name

    Returns
    -------
    bool
        True if field name ends with an update modifier, False if not
    """
    result = bool(re.search(MOD_PATTERN, field))
    if result and not is_tab(field):
        raise ValueError(f"Update modifier found on an atomic field: {field}")
    return result


@lru_cache(maxsize=None)
def strip_tab(field):
    """Strips table suffixes from a field name

    Parameters
    ----------
    field : str
        field name

    Returns
    -------
    str
        field name without a table suffix
    """
    return re.sub(TAB_PATTERN, "", strip_mod(field))


@lru_cache(maxsize=None)
def strip_mod(field):
    """Strips update modifier from a field name

    Parameters
    ----------
    field : str
        field name

    Returns
    -------
    str
        field name without an update modifier
    """
    return field.rsplit("(", 1)[0]


@lru_cache(maxsize=None)
def get_mod(field):
    """Gets the update modifier from a field name

    Parameters
    ----------
    field : str
        field name

    Returns
    -------
    str
        a modifier if found, otherwise an empty string
    """
    if not field.endswith(")"):
        return ""
    mod = "(" + field.rsplit("(", 1)[-1]
    if not re.match(MOD_PATTERN, mod):
        raise ValueError(f"Invalid modifier: {mod}")
    return mod.strip("()")
