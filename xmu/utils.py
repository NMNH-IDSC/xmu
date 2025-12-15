"""Functions to assess whether EMu field names are tables, references, etc."""

import re
from functools import cache

#: tuple : suffixes that designate tables in EMu
TAB_SUFFIXES = ("0", "_nesttab", "_nesttab_inner", "_tab")

#: tuple : suffixes that designate references in EMu
REF_SUFFIXES = ("RefLocal", "Ref", "Ref_nesttab", "Ref_nesttab_inner", "Ref_tab")

#: tuple : suffixes that designate nested tables in EMu
NESTTAB_SUFFIXES = ("_nesttab",)

#: tuple : suffixes that designate inner nested tables in EMu
NESTTAB_INNER_SUFFIXES = ("_nesttab_inner",)

#: str : pattern that matches table suffixes
TAB_PATTERN = "(" + "|".join(TAB_SUFFIXES) + ")$"

#: str : pattern that matches reference suffixes
REF_PATTERN = "(" + "|".join(REF_SUFFIXES) + ")$"

#: str : pattern that matches update modifiers
MOD_PATTERN = r"\(\d*[=\+\-]\)$"


@cache
def is_tab(field: str) -> bool:
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
    return strip_mod(field).endswith(TAB_SUFFIXES) and not field[-2:].isnumeric()


@cache
def is_nesttab(field: str) -> bool:
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


@cache
def is_nesttab_inner(field: str) -> bool:
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


@cache
def is_ref_tab(field: str) -> bool:
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


@cache
def is_ref(field: str) -> bool:
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


@cache
def has_mod(field: str) -> bool:
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


@cache
def split_field(field: str) -> tuple[str]:
    """Splits field into components

    Parameters
    ----------
    field : str
        field name

    Returns
    -------
    tuple[str]
        field, ref, tab, mod
    """
    mod = get_mod(field)
    if mod:
        mod = f"({mod})"
    return (re.sub("Ref$", "", strip_tab(field)), get_ref(field), get_tab(field), mod)


@cache
def to_ref(field: str) -> str:
    """Converts field to reference

    Parameters
    ----------
    field : str
        field name

    Returns
    -------
    tuple
        field name with reference signifier
    """
    orig = field
    field, ref, tab, mod = split_field(field)
    if tab in ("0", "_nesttab_inner"):
        raise ValueError(f"Invalid reference: {orig}")
    if not ref:
        ref = "Ref"
    return f"{field}{ref}{tab}({mod})".replace("()", "")


@cache
def strip_tab(field: str) -> str:
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


@cache
def strip_mod(field: str) -> str:
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


@cache
def strip_ref(field: str) -> str:
    """Strips reference suffixes from a field name

    Parameters
    ----------
    field : str
        field name

    Returns
    -------
    str
        field name without a reference suffix
    """
    return re.sub("(Ref(Local)?$|Ref(Local)?_)", "_", field).rstrip("_")


@cache
def get_ref(field: str) -> str:
    """Gets the reference signifier from a field name

    Parameters
    ----------
    field : str
        field name

    Returns
    -------
    str
        a reference signifier if present, otherwise an empty string
    """
    try:
        return re.search(REF_PATTERN, strip_tab(field)).group()
    except AttributeError:
        return ""


@cache
def get_tab(field: str) -> str:
    """Gets the table signifier from a field name

    Parameters
    ----------
    field : str
        field name

    Returns
    -------
    str
        a table signifier if present, otherwise an empty string
    """
    try:
        return re.search(TAB_PATTERN, strip_mod(field)).group()
    except AttributeError:
        return ""


@cache
def get_mod(field: str) -> str:
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


def flatten(obj: dict, path: list = None, result: dict = None) -> dict:
    """Flattens a record to a one-level dict

    Parameters
    ----------
    obj : dict
        an EMu record
    path : list, omit
        the path to the current key. Users should omit when calling.
    result: dict, optional
        the flattened object. Defaults to empty dict. Users should generally omit
        when calling.

    Returns
    -------
    list
        records flattened to one level
    """
    if path is None:
        path = []
        result = {}
    if isinstance(obj, dict):
        for key, val in obj.items():
            path.append(key)
            flatten(val, path, result)
            path.pop()
    elif isinstance(obj, (list, tuple)):
        for i, val in enumerate(obj):
            path.append(f"{i + 1}.{strip_tab(path[-1])}")
            flatten(val, path, result)
            path.pop()
    else:
        result[".".join(path)] = obj
    return result
