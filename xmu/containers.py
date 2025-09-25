"""Defines containers to read and write various EMu objects"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
from collections.abc import (
    Callable,
    Generator,
    Hashable,
    MutableMapping,
    MutableSequence,
)
from ctypes import c_uint64
from functools import lru_cache
from pathlib import Path
from pprint import pformat
from textwrap import wrap
from typing import Any
from warnings import warn

from lxml import etree
import yaml

from .io import EMuReader
from .types import EMuDate, EMuFloat, EMuLatitude, EMuLongitude, EMuTime
from .utils import (
    is_ref,
    is_nesttab,
    is_nesttab_inner,
    is_tab,
    get_mod,
    has_mod,
    strip_mod,
    strip_tab,
)


logger = logging.getLogger(__name__)


class EMuConfig(MutableMapping):
    """Reads and writes a configuration file

    Automatically loaded when EMuRecord is first accessed. The current configuration
    can be accessed using the config attribute on each of the EMu classes.

    Parameters
    ----------
    path : str
        path to the config file. If omitted, checks the current and home
        directories for the file.

    Attributes
    ----------
    path : str
        path to the config file
    title : str
        title to write at the top of the config file
    filename : str
        default filename for config file
    classes : list
        list of classes to add the config object to
    """

    def __init__(self, path: str = None):
        self.path = path
        self.title = "YAML configuration file for python xmu package"
        self.filename = ".xmurc"
        self.classes = [
            EMuSchema,
            EMuReader,
            EMuRecord,
            EMuColumn,
            EMuGrid,
            EMuRow,
        ]
        self._config = None

        # Options as key: (default, comment)
        self._options = {
            "schema_path": (
                "",
                (
                    "Path to a schema.pl file. A JSON copy of the schema will be"
                    " created in the same directory the first time xmu is run."
                ),
            ),
            "groups": (
                {},
                (
                    "Additional groups not defined in the schema. Schema groups"
                    " correspond to grids, so groups that include non-grid fields"
                    " fields (usually tabs, like the Lat/Long tab in Collection"
                    " Events, where the non-grid content changes depending on"
                    " which row is selected) do not include a complete list of"
                    " fields that should be included when updating the group."
                ),
            ),
            "make_visible": (
                [],
                (
                    "Path as 'module.field' to fields missing an ItemName entry in the"
                    " schema that should resolve even if schema.visible_only is True."
                ),
            ),
            "lookup_no_autopopulate": (
                [],
                (
                    "Path as 'module.field' to fields that should not be populated"
                    " when filling a lookup hierarchy during import."
                ),
            ),
            "reverse_attachments": (
                {},
                (
                    "Reverse attachments as {module: {other_field: other_module}}."
                    " These are records that attach to the current record and can be"
                    " included in a report using the Attach Module function in the"
                    " field selection window. The field name is based on the attachment"
                    " field in the other record and does not appear in the schema."
                ),
            ),
        }

        self.load_rcfile()

        # Set config parameter on all classes
        for cl in self.classes:
            cl.config = self

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({pformat(self._config)})"

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, key) -> Any:
        return self._config[key]

    def __setitem__(self, key, val) -> None:
        self._config[key] = val

    def __delitem__(self, key) -> None:
        del self._config[key]

    def __len__(self) -> int:
        return len(self._config)

    def __iter__(self) -> Generator:
        return iter(self._config)

    def load_rcfile(self, path: str = None) -> dict:
        """Loads a configuration file

        Parameters
        ----------
        path : str, optional
            path to the rcfile. If not given, checks the current then home
            directory for the filename.

        Returns
        -------
        dict
            either a custom configuration loaded from a file or the
            default configuration defined in this function
        """

        if path is None:
            path = self.path

        # Check the current then home directories if path not given
        default_paths = [".", os.path.expanduser("~")]
        paths = default_paths if path is None else [path]

        # Create a default configuration based on _options attribute
        self._config = {k: v[0] for k, v in self._options.items()}

        # Check each location for the rcfile
        for path in paths:
            # Use a default filename if none given
            if os.path.isdir(path):
                path = os.path.join(path, self.filename)

            try:
                with open(path, encoding="utf-8") as f:
                    self.update(yaml.safe_load(f))
                self.path = path
                break
            except FileNotFoundError:
                pass

        return self._config

    def save_rcfile(self, path: str = None, overwrite: bool = False) -> None:
        """Saves a configuration file

        Parameters
        ----------
        path : str
            path for the rcfile. If a directory, adds the filename.
            Defaults to the user's home directory.
        overwrite : bool
            whether to overwrite the file if it exists
        """

        # Default to user home directory
        if path is None:
            path = os.path.expanduser("~")

        # Use a default filename if none given
        if os.path.isdir(path):
            path = os.path.join(path, self.filename)

        # Check if a file already exists at the path
        try:
            with open(path, encoding="utf-8") as f:
                pass
            if overwrite:
                raise FileNotFoundError
            raise IOError(
                f"'{path}' already exists. Use overwrite=True to overwrite it."
            )
        except FileNotFoundError:
            pass

        # Write a commented YAML file. Comments aren't supported by pyyaml
        # and have to be hacked in.
        content = [f"# {self.title}"]
        for line in yaml.dump(self._config, sort_keys=False).splitlines():
            try:
                comment = self._options[line.split(":")[0]][1]
                wrapped = "\n".join([f"# {l}" for l in wrap(comment)])
                content.extend(["", wrapped, line])
            except KeyError:
                # Catches keys that are not top-level options
                content.append(line)

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))

    def update(self, obj: dict, path: list = None) -> None:
        """Recusrively updates configuration from dict

        Parameters
        ----------
        obj : dict
            configuration object. User should always supply a dict, although
            the function itself may pass a variety of object types here.
        path : list, optional
            path to the current item. User should omit.

        Returns
        -------
        None
        """
        if path is None:
            path = []

        if path:
            config = self._config
            for key in path[:-1]:
                config = config.setdefault(key, {})

        if isinstance(obj, dict):
            for key, val in obj.items():
                path.append(key)
                self.update(val, path)
                path.pop()
        elif isinstance(obj, list):
            config[path[-1]] = []
            for i, val in enumerate(obj):
                config[path[-1]].append(type(val)())
                path.append(i)
                self.update(val, path)
                path.pop()
        else:
            if isinstance(obj, str) and obj.startswith("~"):
                obj = os.path.realpath(os.path.expanduser(obj))
            try:
                config[path[-1]] = obj
            except IndexError:
                config.append(obj)


class EMuSchema(dict):
    """Reads and queries an EMu schema file

    Sets the schema class attribute on all classes defined in this module
    when called. If a schema_path is specified in the config file, the
    schema is loaded automatically the first time an EMuRecord is created.

    Parameters
    ----------
    args, kwargs :
        any arguments that can be used to create a dict. If a single string,
        tries to load the dict from the path represented by that string. If
        omitted, will check the config attribute for a schema path.

    Attributes
    ----------
    path : str
        path to the schema file, if used
    visible_only : bool
        whether to resolve fields without an ItemName attribute in the schema
    validate_paths : bool
        whether to validate paths as they are added to an EMuRecord
    """

    #: EMuConfig : module-wide configuration parameters. Set automatically
    #: when an EMuConfig object is created.
    #:
    #: :meta hide-value:
    config = None

    def __init__(self, *args, **kwargs):
        # Load a config file from one of the default locations if empty
        if self.config is None:
            EMuConfig()

        # Disable both checks for the initial read
        self.visible_only = False
        self.validate_paths = False

        if not args or kwargs:
            try:
                args = [self.config["schema_path"]]
            except TypeError:
                pass

        self.path = None
        if len(args) == 1 and isinstance(args[0], (str, Path)):
            self.from_file(args[0])
        elif args or kwargs:
            super().__init__(*args, **kwargs)

        # Enable both checks by default
        self.visible_only = True
        self.validate_paths = True

        # Set schema parameter on all classes
        EMuReader.schema = self
        EMuRecord.schema = self
        EMuColumn.schema = self
        EMuGrid.schema = self
        EMuRow.schema = self

        # Tweak the schema based on the config file
        if self.config is not None:
            # Add custom groups from config file. This needs to come after the
            # assignment of the class attributes because _get_field_info() uses
            # one of those to access the schema.
            for module, groups in self.config["groups"].items():
                for fields in groups.values():
                    try:
                        self.define_group(module, fields)
                    except KeyError as exc:
                        warn(f"Could not define custom group: {str(exc)}")

            # Add entries for reverse attachment fields to the schema
            for mod, fields in self.config["reverse_attachments"].items():
                for field, refmod in fields.items():
                    # The field takes the name of the attachment field in the
                    # linking record, which may not be tabular. Because attachment
                    # fields are always tabular (that is, more than one record can
                    # link to a given record), the field name is updated to use
                    # EMu's tab suffix. This allows code elsewhere in this package
                    # that relies on field names to work as expected.
                    if not is_tab(field):
                        field += "_tab"
                    self["Schema"][mod]["columns"][field] = {
                        "ColumnName": field,
                        "DataKind": "dkTable",
                        "RefKey": "irn",
                        "RefPrompt": field.replace("Ref_", ""),
                        "RefTable": refmod,
                    }
                    self.config.setdefault("make_visible", []).append(f"{mod}.{field}")
            self.config["make_visible"] = sorted(set(self.config["make_visible"]))

    def __getitem__(self, path) -> Any:
        path = _split_path(path)
        obj = super().__getitem__(path[0])
        try:
            for key in path[1:]:
                obj = obj[key]
        except KeyError as exc:
            try:
                similar = self._get_similar_keys(path)
            except KeyError:
                raise KeyError(f"Path not found: {path} (failed at {key})") from exc
            else:
                raise KeyError(
                    f"Path not found: {path} (failed at {key}, similar keys include {similar})"
                ) from exc
        return obj

    @property
    def modules(self) -> list[str]:
        """Gets the list of modules in the schema"""
        return sorted(self["Schema"].keys())

    def get(self, key: str, default: Any = None) -> Any:
        """Overrides the native get method to support paths

        Parameters
        ----------
        key : str
            key to retrieve
        default : Any, optional
           default value to return if key not found

        Returns
        -------
        Any
            value for the key or default if not found
        """
        try:
            return self[key]
        except KeyError:
            return default

    def from_file(self, path: str) -> None:
        """Loads schema from a file, creating a JSON version if not found

        Parameters
        ----------
        path : str
            path to a schema filenumpy docstring returns
        """
        self.path = path
        path = os.path.splitext(path)[0]
        try:
            self.from_json(f"{path}.json")
        except FileNotFoundError:
            self.from_pl(f"{path}.pl")

            # Map group definitions to fields prior to saving the JSON file
            EMuRecord.schema = self
            for module, data in self["Schema"].items():
                for key, fields in data.get("groups", {}).items():
                    try:
                        self.define_group(module, fields)
                    except KeyError as exc:
                        warn(
                            f"The schema file includes an invalid group: {module}, {fields} (exc={exc})"
                        )
                    else:
                        # Capture the original group defined by the schema, mapping
                        # any view fields to the underlying reference field
                        fields = list({_map_view(module, f): 1 for f in fields})
                        data["groups"][key] = fields
                        for field in fields:
                            self.get_field_info(module, field)["GroupFieldsOrig"] = (
                                fields[:]
                            )

            self.to_json(f"{path}.json")

    def from_pl(self, path: str) -> None:
        """Loads schema from an EMu schema.pl file

        Parameters
        ----------
        path : str
            path to a schema file
        """
        self.update(self._read_schema_pl(path))

    def from_json(self, path: str) -> None:
        """Loads schema from JSON

        Parameters
        ----------
        path : str
            path to a JSON schema file
        """
        with open(path, encoding="utf-8") as f:
            self.update(json.load(f))

    def to_json(self, path: str, **kwargs) -> None:
        """Saves schema to JSON

        Parameters
        ----------
        path : str
            path to a JSON schema file
        kwargs :
            keyword arguments to pass to json.dump() to control the format
            of the JSON file. Method saves the JSON compactly by default.
        """
        params = {
            "ensure_ascii": False,
            "indent": None,
            "sort_keys": False,
            "separators": (",", ":"),
        }
        params.update(**kwargs)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self, f, **params)

    def iterfields(self) -> Generator[str, str, dict]:
        """Iterates over all fields in the schema

        Yields
        ------
        tuple
            (module, field, field info) for each field
        """
        for module in self.modules:
            cols = self[("Schema", module, "columns")]
            for field, info in cols.items():
                yield module, field, info

    def define_group(self, module: str, fields: list) -> None:
        """Maps a group definition to each member field

        Groups are read from a schema if possible but can also be defined manually.

        Parameters
        ----------
        module : str
            backend module name
        fields : list
            list of fields in the group
        """
        fields_ = {}
        for field in fields:
            try:
                info = self.get_field_info(module, field)
            except KeyError as exc:
                if "is a view of the data in" not in str(exc):
                    raise
                # Map views to the attachment field
                field = str(exc).split(" ")[-1].strip("'")
            else:
                field = info.get("RefLink", field)
            fields_[field] = 1

        fields = list(fields_)
        for field in fields[:]:
            # Combine groups that share one or more fields
            info = self.get_field_info(module, field)
            fields_ = info.get("GroupFields", [])
            if fields_:
                a = set(fields)
                b = set(fields_)
                if not (a.issubset(b) or b.issubset(a)):
                    warn(f"Combined groups:\n- {fields}\n- {fields_}")
                fields.extend([f for f in fields_ if f not in fields])
                fields_.extend([f for f in fields if f not in fields_])
            else:
                info["GroupFields"] = fields

        # Field definitions modified, so clear the cache
        _get_field_info.cache_clear()

    @staticmethod
    def get_field_info(module: str, path: str, visible_only: bool = None) -> dict:
        """Gets data about the field specified by a path

        Parameters
        ----------
        module : str
            backend module name
        path : str
            path to the field in EMu
        visible_only : bool
            whether to resolve fields that do not appear in the client

        Returns
        -------
        dict
            information about the field (names, data types, etc.)
        """
        return _get_field_info(module, path, visible_only=visible_only)

    def _read_schema_pl(self, path: str) -> dict:
        """Reads an EMu schema file

        Parameters
        ----------
        path : str
            path to a schema.pl file

        Returns
        -------
        dict
            EMu schema
        """
        schema = {"Schema": {}}
        dct = schema["Schema"]
        keypath = ["Schema"]
        items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip().strip(",")

                # Open new child dictionary when line ends with an arrow
                if line.endswith("=>"):
                    key = line.rsplit("=>")[0].strip().strip("'")
                    keypath.append(key)
                    dct[key] = {}
                    dct = dct[key]

                # Add new key-val when arrow occurs mid-line
                elif "=>" in line:
                    key, val = [s.strip().strip("'") for s in line.split("=>")]
                    dct[key] = self._parse_value(val)

                # Open a new list
                elif line == "[":
                    items.append([])

                # Set list as the value for the current key
                elif line == "]" and len(items) == 1:
                    last = keypath.pop(-1)
                    dct = schema
                    for key in keypath:
                        dct = dct[key]
                    dct[last] = items.pop(-1)

                # Append nested list to previous item in the lists container
                elif line == "]":
                    items[-2].append(items.pop(-1))

                # Append value to the current list
                elif items:
                    vals = self._parse_value(line)
                    if line.startswith("[") or isinstance(vals, (str, int)):
                        items[-1].append(vals)
                    else:
                        items[-1].extend(vals)

                # Go up one level in the dictionary
                elif line == "}":
                    keypath.pop(-1)
                    dct = schema
                    for key in keypath:
                        dct = dct[key]

        return schema

    def _get_similar_keys(self, path: str) -> list[str]:
        """Finds fields similar to the one at the end of the given path"""
        last = path[-1][:4]
        obj = super().__getitem__(path[0])
        for key in path[1:-1]:
            obj = obj[key]
        return [k for k in obj if k.startswith(last)]

    @staticmethod
    def _parse_value(val: str) -> str | list:
        """Parses strings, lists, and integers from perl file"""
        vals = val.strip("[]").split(",")
        try:
            vals = [int(s.strip()) for s in vals]
        except ValueError:
            vals = [s.strip("'").strip() for s in vals]
        return vals if "," in val else vals[0]


class EMuColumn(list):
    """Reads and writes data in a table field

    Parameters
    ----------
    vals : iterable
        values for the column
    module : str
        backend name of an EMu module
    field : str
        name of an EMu field
    dict_class : EMuRecord
        class to use for dicts
    list_class : EMuColumn
        class to use for lists

    Attributes
    ----------
    module : str
        backend name of an EMu module
    field : str
        name of an EMu field
    dict_class : EMuRecord
        class to use for dicts
    list_class : EMuColumn
        class to use for lists
    """

    #: EMuConfig : module-wide configuration parameters. Set automatically
    #: when an EMuConfig object is created.
    #:
    #: :meta hide-value:
    config = None

    #: EMuSchema : info about a specific EMu configuration. Set automatically
    #: when an EMuSchema object is created.
    #:
    #: :meta hide-value:
    schema = None

    def __init__(
        self,
        vals: list = None,
        module: str = None,
        field: str = None,
        dict_class: Callable = None,
        list_class: Callable = None,
    ):
        self.module = module
        self.field = field
        self.dict_class = dict_class if dict_class else DEFAULT_RECORD
        self.list_class = list_class if list_class else DEFAULT_COLUMN
        if self.schema and not self.module:
            raise ValueError(
                f"Must provide module when schema is used (one of {self.schema.modules})"
            )
        super().__init__()
        if vals:
            self.extend(vals)

    def __str__(self) -> str:
        return f"EMuColumn({super().__str__()})"

    def __setitem__(self, i: int, val: Any) -> None:
        # Catch non-integer indices to avoid problems with coercing values
        if not isinstance(i, int):
            raise TypeError("list indices must be integers or slices, not str")
        super().__setitem__(i, _coerce_values(self, val))

    def __add__(self, obj: Any) -> EMuColumn:
        return self.__class__(
            super().__add__(obj), module=self.module, field=self.field
        )

    def __iadd__(self, obj: Any) -> EMuColumn:
        self.extend(obj)
        return self

    def insert(self, i: int, val: Any) -> None:
        super().insert(i, _coerce_values(self, val))

    def append(self, val: Any) -> None:
        super().append(_coerce_values(self, val))

    def extend(self, vals: list) -> None:
        super().extend([_coerce_values(self, v) for v in vals])

    def copy(self) -> "EMuColumn":
        """Overrides the native list.copy method to return an object of this class"""
        return pickle.loads(pickle.dumps(self))

    def to_xml(
        self,
        root: etree.Element | etree.SubElement = None,
        kind: str = None,
        row_ids: tuple = None,
    ) -> etree.Element:
        """Converts column to XML formatted for EMu

        Normally called without specifying arguments.

        Parameters
        ----------
        root : lxml.etree.Element | lxml.etree.SubElement
            parent element in the XML tree
        kind : str
           kind of XML file. One of "import", "update", or "emu".
        row_ids : tuple
            list of values for the tuple group attribute

        Returns
        -------
        lxml.etree.Element or SubElement
            table as XML
        """
        name = strip_mod(self.field)
        mod = get_mod(self.field)

        if root is None:
            root = etree.Element("table")
            root.set("name", name)
        elif root.tag != "table":
            root = etree.SubElement(root, "table")
            root.set("name", name)

        for (
            i,
            child,
        ) in enumerate(self):
            tup = etree.SubElement(root, "tuple")

            # Add group and row indicators for updates
            if mod:
                tup.set("row", mod)
                if row_ids and mod == "+":
                    tup.set("group", row_ids[i])
            # Otherwise explicitly number the table rows. This prevents EMu from
            # skipping empty nested table cells when reading an import file.
            else:
                tup.set("row", str(i + 1))

            try:
                child.to_xml(tup, kind=kind)
            except AttributeError:
                # In an EMu export, empty rows in an outer nested table appear
                # as an empty tuple. Otherwise tuples always contain one or more
                # atomic fields, including an empty irn field for references.
                if _is_not_blank(child) or not is_nesttab(self.field):
                    # Interpret an atomic value inside a reference table as an irn
                    name = "irn" if is_ref(self.field) else strip_tab(self.field)
                    atom = etree.SubElement(tup, "atom")
                    atom.set("name", name)

                    # Set text, deferring to the emu_str method if it exists
                    atom.text = ""
                    if _is_not_blank(child):
                        try:
                            atom.text = child.emu_str()
                        except AttributeError:
                            atom.text = str(child)

        return root


class EMuRow(MutableMapping):
    """Reads and writes data in a grid row

    Changes to the row are reflected in the original EMuRecord

    Parameters
    ----------
    rec : EMuRecord
        the EMu record the grid is from
    path : str
        path to a field that is part of the grid
    index : int
        the index of the row
    fill_value : Any
        value used when deleting an item from the row

    Attributes
    ----------
    group : tuple
        names for all columns that are part of the parent grid, whether they appear
        in the current record or not
    fill_value : Any
        value used when deleting an item from the row
    """

    #: EMuConfig : module-wide configuration parameters. Set automatically
    #: when an EMuConfig object is created.
    #:
    #: :meta hide-value:
    config = None

    #: EMuSchema : info about a specific EMu configuration. Set automatically
    #: when an EMuSchema object is created.
    #:
    #: :meta hide-value:
    schema = None

    def __init__(self, rec: EMuRecord, path: str, index: int, fill_value: Any = None):
        module = _get_module(rec)
        self.group = tuple(
            self.schema.get_field_info(module, path).get("GroupFields", [])
        )
        if not self.group:
            raise KeyError(f"{module}.{path} is not part of a group")
        self.fill_value = fill_value
        self.index = index

        # Use path to drill down to the correct parent record
        path = _split_path(path)[:-1]
        if path:
            rec = rec[path]
        self._rec = rec

    def __str__(self) -> str:
        try:
            row = {c: self._rec[c][self.index] for c in self.columns}
        except IndexError:
            raise IndexError(
                "One or more columns has no data for this row. Use pad() on the parent grid to prevent this error."
            )
        return f"{self.__class__.__name__}({str(row)})"

    def __repr__(self) -> str:
        return str(self)

    def __iter__(self) -> Generator:
        return iter(self.columns)

    def __len__(self) -> int:
        return len(self.columns)

    def __setitem__(self, key: Hashable, val: Any) -> None:
        self._rec[key][self.index] = val

    def __getitem__(self, key: Hashable) -> Any:
        return self._rec[key][self.index]

    def __delitem__(self, key: Hashable) -> Any:
        self[key] = self.fill_value

    @property
    def columns(self) -> list[str]:
        """List of columns in the row that exist in the record"""
        cols = [c for c in self._rec if strip_mod(c) in set(self.group)]
        if (
            any((c.endswith(")") for c in cols))
            and len({c.rsplit("(", 1)[-1] for c in cols}) > 1
        ):
            raise ValueError(f"Inconsistent modifier within grid: {cols}")
        return cols

    @property
    def replace_mod(self) -> str:
        """Modifier needed to replace a cell in this row in an import"""
        return f"{self.index + 1}="

    def row_id(self) -> str:
        """Calculates an identifier based on the index and content of a row"""
        val = str(self.index) + str(self)
        return c_uint64(hash(val)).value.to_bytes(8, "big").hex()


class EMuGrid(MutableSequence):
    """Reads and writes data in a grid

    Changes to the grid are reflected in the original EMuRecord

    Parameters
    ----------
    rec : EMuRecord
        the EMu record the grid is from
    path : str
        path to a field that is part of the grid
    fill_value : Any
        value used when padding columns
    pad : bool
        whether to pad the columns to the same length

    Attributes
    ----------
    group : tuple
        names for all columns that are part of this grid, whether they appear
        in the current record or not
    fill_value : Any
        value to use when padding the grid or deleting an item from an EMuRow
        object created from this grid
    columns
    """

    #: EMuConfig : module-wide configuration parameters. Set automatically
    #: when an EMuConfig object is created.
    #:
    #: :meta hide-value:
    config = None

    #: EMuSchema : info about a specific EMu configuration. Set automatically
    #: when an EMuSchema object is created.
    #:
    #: :meta hide-value:
    schema = None

    def __init__(self, rec: EMuRecord, path: str, fill_value: Any = None):
        module = _get_module(rec)
        info = self.schema.get_field_info(module, path)
        self.group = tuple(info.get("GroupFields", []))
        if not self.group:
            raise KeyError(f"{module}.{path} is not part of a group")
        self.orig_group = tuple(info.get("GroupFieldsOrig", []))
        self.fill_value = fill_value

        # Use path to drill down to the correct parent record
        path = _split_path(path)[:-1]
        if path:
            rec = rec[path]
        self._rec = rec

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({list(self)})"

    def __repr__(self) -> str:
        return str(self)

    def __iter__(self) -> Generator:
        for i in range(len(self)):
            yield EMuRow(self._rec, self.columns[0], i, fill_value=self.fill_value)

    def __len__(self) -> int:
        try:
            return max([len(self._rec[c]) for c in self.columns])
        except ValueError:
            return 0

    def __getitem__(self, key: Hashable) -> Any:
        if isinstance(key, int):
            return list(self)[key]

        if isinstance(key, dict):
            matches = []
            for row in self:
                for key_, val in key.items():
                    try:
                        row_val = self._transform(row.get(key_))
                    except IndexError:
                        raise IndexError(
                            f"Key in query does not appear in all rows: {repr(key_)}"
                        )
                    match_val = self._transform(val)
                    # Match if values are equal or if row_val matches one of
                    # multiple options given for match_val
                    if row_val == match_val or (
                        row_val and match_val and row_val in match_val.split("|")
                    ):
                        continue
                    break
                else:
                    matches.append(row)
            return matches

        return self._rec[key]

    def __setitem__(self, *args) -> None:
        # Required by MutableSequence but does not make sense to implement
        raise NotImplementedError(
            "Cannot set items on an EMuGrid. Use the main EMuRecord object or"
            " an individual EMuRow instead."
        )

    def __delitem__(self, i: int) -> None:
        for col in self.columns:
            del self._rec[col][i]

    def __eq__(self, other: Any) -> bool:
        return list(self) == list(other)

    @property
    def columns(self) -> list[str]:
        """List of columns in the grid that exist in the record"""
        cols = [c for c in self._rec if strip_mod(c) in set(self.group)]
        if (
            any((c.endswith(")") for c in cols))
            and len({c.rsplit("(", 1)[-1] for c in cols}) > 1
        ):
            raise ValueError(f"Inconsistent modifier within grid: {cols}")
        return cols

    def verify(self) -> None:
        """Checks if any fields in the grid are missing from the record

        Returns
        -------
        None

        Raises
        ------
        ValueError
            if fields are missing
        """
        missing = set(self.group) - set(self.columns)
        if missing:
            raise ValueError(
                f"Grid including '{self.columns[0]}' is missing fields: {missing}."
                f" Verify that all grid fields are present in import, then run"
                f" grid.add_columns().pad() to complete the grid."
            )

    def insert(self, *args) -> None:
        # Required by MutableSequence but does not make sense to implement
        raise NotImplementedError(
            "Cannot insert into an EMuGrid. Use the main EMuRecord object instead."
        )

    def items(self) -> Generator:
        for col in self.columns:
            yield col, self._rec[col]

    def add_columns(self, cols: list = None, fill_value: Any = None) -> EMuGrid:
        """Adds missing columns to the grid

        Parameters
        ----------
        cols : list-like
            columns to add. If not given, adds all columns in the group attribute
            that do not already appear in the record.
        fill_value : Any
            the value used to pad a column. Defaults to fill_value attribute
            of the instance if not given.

        Returns
        -------
        self
        """
        if fill_value is None:
            fill_value = self.fill_value
        mod = get_mod(self.columns[0]) if self.columns else None
        if cols is None:
            cols = self.group
        if mod:
            cols = [f"{c}({mod})" if not has_mod(c) else c for c in cols]
        for col in set(cols) - set(self.columns):
            self._rec.setdefault(col, [fill_value for _ in range(len(self))])
        return self

    def pad(self, fill_value: Any = None) -> EMuGrid:
        """Pads all columns in the table to the same length

        Parameters
        ----------
        fill_value : Any
            the value used to pad a column. Defaults to fill_value attribute
            if not given.

        Returns
        -------
        EMuGrid
            the instance of EMuGrid from which this method was called
        """
        if fill_value is None:
            fill_value = self.fill_value
        for col in self.columns:
            diff = len(self) - len(self._rec[col])
            self._rec[col].extend([fill_value for _ in range(diff)])
        return self

    def filter(self, field: str = None, where: dict = None) -> list[Any]:
        """Filters the grid

        Parameters
        ----------
        field : str
            a specific column to return. If empty, the whole row is returned.
        where : dict
            the query as a dict. A row must match all criteria to be returned.

        Returns
        -------
        list[Any]
            list of matching rows or values
        """
        results = []
        for row in self[where]:
            results.append(row[field] if field else row)
        return results

    def sort(self, key: int | str | Callable, reverse: bool = False):
        """Sorts the grid by a single field

        Parameters
        ----------
        key : int | str | Callable
            the key or function to use to sort the grid
        reverse : bool
            whether to sort the grid in reverse

        Returns
        -------
        None
        """
        func = key if callable(key) else lambda r: r[key]
        vals = {}
        for row in sorted(list(self.pad()), key=func, reverse=reverse):
            for key, val in list(row.items()):
                vals.setdefault(key, []).append(val)
        for key, vals in vals.items():
            self._rec[key] = vals

    @staticmethod
    def _transform(val: Any) -> str:
        if not isinstance(val, (list, tuple)):
            val = [val]
        return "|".join([str(s) if s is not None else "" for s in val]).lower()


class EMuRecord(dict):
    """Reads and writes data in an EMu record

    Parameters
    ----------
    rec : str | dict
        record as a mapping or iterable
    module : str
        backend name of an EMu module
    field : str
        name of an EMu field
    dict_class : Callable
        class to use for dicts
    list_class : Callable
        class to use for lists

    Attributes
    ----------
    module : str
        backend name of an EMu module
    field : str
        name of an EMu field
    dict_class : Callable
        class to use for dicts. Defaults to DEFAULT_RECORD constant if not provided.
    list_class : Callable
        class to use for lists. Defaults to DEFAULT_COLUMN constant if not provided.
    """

    #: EMuConfig : module-wide configuration parameters. Set automatically
    #: when an EMuConfig object is created.
    #:
    #: :meta hide-value:
    config = None

    #: EMuSchema : info about a specific EMu configuration. Set automatically
    #: when an EMuSchema object is created.
    #:
    #: :meta hide-value:
    schema = None

    def __init__(
        self,
        rec: str | dict = None,
        module: str = None,
        field: str = None,
        dict_class: Callable = None,
        list_class: Callable = None,
    ):
        self.module = module
        self.field = field
        self.dict_class = dict_class if dict_class else DEFAULT_RECORD
        self.list_class = list_class if list_class else DEFAULT_COLUMN

        if self.schema and not self.module:
            raise ValueError(
                f"Must provide module when schema is used (one of {self.schema.modules})"
            )

        super().__init__()
        if rec:
            if isinstance(rec, str):
                rec = json.loads(rec)
            elif isinstance(rec, self.__class__):
                rec = rec.copy()
            self.update(rec)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({pformat(self)})"

    def __getitem__(self, path: str | list | tuple) -> Any:
        path = _split_path(path)
        try:
            if len(path) > 1:
                obj = self
                for key in path:
                    try:
                        obj = obj[key]
                    except TypeError:
                        if obj is None:
                            raise KeyError
                        raise
                return obj
            key = path[0]
            return super().__getitem__(key)
        except KeyError as exc:
            # Check path against schema if key not found
            module = _get_module(self)
            dotpath = ".".join(path)
            if module and self.schema is not None and self.schema.validate_paths:
                try:
                    self.schema.get_field_info(module, path)
                except KeyError:
                    raise KeyError(
                        f"Invalid path: {dotpath} (module={module})"
                    ) from exc
                else:
                    raise KeyError(
                        f"Path not found but valid: {dotpath} (module={module})"
                    ) from exc
            raise KeyError(
                f"Path not found: {dotpath} (module={module}) (failed at {key})"
            ) from exc

    def __setitem__(self, key: Hashable, val: Any) -> None:
        # Catch a key containing illegal characters
        if not re.match(r"\w+$", strip_mod(key)):
            raise ValueError(f"Invalid key: {key} (module={_get_module(self)})")
        super().__setitem__(key, _coerce_values(self, val, key))

    def get(self, key: Hashable, default: Any = None) -> Any:
        """Overrides the native dict.get method to map unrecognized terms"""
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key: Hashable, val: Any) -> Any:
        """Overrides the native dict.setdefault method to use the subclass setter"""
        try:
            return self[key]
        except KeyError:
            self[key] = val
            return self[key]

    def update(self, *args, **kwargs) -> None:
        """Overrides the native dict.update method to use the subclass setter"""
        for key, val in dict(*args, **kwargs).items():
            self[key] = val

    def copy(self) -> EMuRecord:
        """Overrides the native dict.copy method to return an object of this class"""
        return pickle.loads(pickle.dumps(self))

    def json(self, **kwargs) -> str:
        """Converts record to JSON

        Parameters
        ----------
        kwargs :
            any kwarg accepted by json.dumps

        Returns
        -------
        str
            record as JSON string
        """
        kwargs.setdefault("cls", EMuEncoder)
        kwargs.setdefault("ensure_ascii", False)
        return json.dumps(dict(self), **kwargs)

    def grid(self, field: str, **kwargs) -> EMuGrid:
        """Returns the EMuGrid object containing the given field

        Parameters
        ----------
        field : str
            any field name that appears in a grid
        kwargs :
            keyword agruments for EMuGrid

        Returns
        -------
        EMuGrid
            the grid from the current record
        """
        return EMuGrid(self, field, **kwargs)

    def to_xml(
        self, root: etree.Element | etree.SubElement = None, kind: str = None
    ) -> etree.Element:
        """Converts record to XML formatted for EMu

        Normally called without specifying arguments.

        Parameters
        ----------
        root : lxml.etree.Element or lxml.etree.SubElement
            parent element in the XML tree
        kind : str
           kind of XML file. One of "import", "update", or "emu". If not
           given, assigns "update" if top-level records have irns and "import"
           if not.

        Returns
        -------
        lxml.etree.Element or SubElement
            record as XML
        """
        kinds = {None, "emu", "import", "update"}
        if kind not in {None, "emu", "import", "update"}:
            raise ValueError(f"kind must be one of {kinds}")

        if root is None:
            root = etree.Element("tuple")
        elif root.get("name") == self.module:
            root = etree.SubElement(root, "tuple")

        # Records containing irns in the top level of the dict are updates
        if kind is None:
            kind = "update" if "irn" in self else "import"

        # Fill in grids and cache row IDs so grids are only checked once
        grids = {}
        for key in list(self):
            if kind == "update":
                try:
                    grids[key]
                except KeyError:
                    try:
                        grid = self.grid(key)
                    except KeyError:
                        pass
                    else:
                        # Include all columns when appending or prepending
                        if key.endswith(("(+)", "(-)")):
                            grid.add_columns()
                        grid.pad()
                        row_ids = [r.row_id() for r in grid]
                        for col in grid.columns:
                            grids[col] = row_ids

            # Populate unfilled lookup list parent fields
            module = _get_module(self)
            if self.schema is not None and self.schema.validate_paths:
                while True:
                    field_info = self.schema.get_field_info(module, key)
                    try:
                        lookup_parent = field_info["LookupParent"]
                    except KeyError:
                        break

                    # Break on SecLookupRoot and any other fields specified in config
                    if (
                        lookup_parent == "SecLookupRoot"
                        or f"{self.module}.{lookup_parent}"
                        not in self.config["lookup_no_autopopulate"]
                    ):
                        break

                    try:
                        self[lookup_parent]
                    except KeyError:
                        logger.debug(f"Filled parent in lookup: {lookup_parent}")
                        self[lookup_parent] = None

                    key = lookup_parent

        for key, val in self.items():
            if is_tab(key):
                # If field is part of a grid, pass row identifiers to the
                # EMuColumn to_xml() method. These will be used to populate the
                # group attribute in each tuple tag for appends and prepends.
                val.to_xml(root, kind=kind, row_ids=grids.get(key, None))
            elif is_ref(key):
                if isinstance(val, int):
                    # The module does not matter here, so just use the parent's
                    val = self.__class__({"irn": val}, module=self.module)
                ref_tup = etree.SubElement(root, "tuple")
                ref_tup.set("name", key)
                if val is not None:
                    val.to_xml(ref_tup, kind=kind)
            else:
                atom = etree.SubElement(root, "atom")
                atom.set("name", key)

                # Set text, deferring to the emu_str method if it exists
                atom.text = ""
                if _is_not_blank(val):
                    try:
                        atom.text = val.emu_str()
                    except AttributeError:
                        atom.text = str(val)
        return root


class EMuEncoder(json.JSONEncoder):
    """Encodes objects using EMuRecord and EMuColumn"""

    def default(self, o: Any) -> str:
        if isinstance(o, dict):
            return dict(o)
        elif isinstance(o, list):
            return list(o)
        return str(o)


def _coerce_values(parent: EMuRecord | EMuColumn, child: Any, key: str = None) -> Any:
    """Coerces child containers and values to specific classes"""

    # Pickled objects are missing the instance attributes required to
    # use this function, but since these objects have already been
    # coerced, they can be returned as is.
    try:
        parent.module
    except AttributeError:
        return child

    if isinstance(parent, dict):
        dict_class = parent.dict_class
        list_class = parent.list_class
        field = key
        module = _get_module(parent)

    # List items inherit the field attribute from their parent
    elif isinstance(parent, (list, tuple)):
        dict_class = parent.dict_class
        list_class = parent.list_class
        field = parent.field
        module = parent.module

    else:
        dict_class = None
        list_class = None
        field = None
        module = None

    # Validate field if schema has been loaded
    field_info = None
    if parent.schema and parent.schema.validate_paths:
        field_info = parent.schema.get_field_info(module, key if key else field)

    # Label inner nested tables
    if is_nesttab(field) and not isinstance(parent, dict_class):
        field = f"{strip_mod(field)}_inner"

    # Tables must be list-like
    if field != parent.field and is_tab(field) and not isinstance(child, (list, tuple)):
        if child is None:
            child = []
        else:
            raise TypeError(
                f"Columns must be lists ({repr(child)} was assigned to {field})"
            )

    # Simplify IRN-only references
    if is_ref(field):
        # Simplify IRN-only references to integers
        if isinstance(child, dict) and list(child) == ["irn"]:
            child = child["irn"]

        # Interpret integers in reference fields as IRNs
        try:
            child_as_int = int(child)
            if isinstance(child_as_int, int):
                return child_as_int
        except (TypeError, ValueError):
            pass

    # References must be dicts or ints (which are interpreted as IRNs)
    if (
        is_ref(field)
        and not is_tab(field)
        and not isinstance(child, dict)
        and child is not None
    ):
        raise TypeError(
            f"References must be dicts ({repr(child)} was assigned to {field})"
        )

    # Sequences must only be used in tables
    if isinstance(child, (list, tuple)) and not is_tab(field):
        raise TypeError(
            f"Sequence assigned to atomic field ({repr(child)} was assigned to {field})"
        )

    # References should use the target module and drop the field
    if isinstance(child, dict) and not isinstance(child, dict_class):
        child = dict_class(child, module=_get_module(parent, field), field=None)

    # Coerce columns to given list class
    elif isinstance(child, (list, tuple)) and not isinstance(child, list_class):
        child = list_class(child, module=module, field=field)

    # Coerce non-list, non-dict data to an appropriate type if a schema is defined
    elif field_info and not isinstance(child, (dict, list)):
        # Coerce common NAs to None
        if str(child) in {"nan", "<NA>", "NaT"}:
            child = None

        # Coerce empty values to empty strings in Text fields. Exclude
        # inner nested tables so that empty rows can be signified by None.
        dtype = field_info["DataType"]
        if (
            dtype in ("Text", "String")
            and child is None
            and not is_nesttab_inner(field)
        ):
            child = ""

        elif child is not None:
            try:
                child = {
                    "Currency": str,
                    "Date": EMuDate,
                    "Float": EMuFloat,
                    "Integer": int,
                    "Latitude": EMuLatitude,
                    "Longitude": EMuLongitude,
                    "String": str,
                    "Text": str,
                    "Time": EMuTime,
                    "UserName": str,
                    "UserId": str,
                }[dtype](child)
            except (TypeError, ValueError) as exc:
                # Handle integers with decimals or commas
                if (
                    dtype == "Integer"
                    and isinstance(child, str)
                    and (
                        re.search(r"\.0+$", child)
                        or re.match(r"-?\d{1,3}(,\d{3})+", child)
                    )
                ):
                    child = int(EMuFloat(child))
                else:
                    raise TypeError(
                        f"Could not coerce to {dtype} ({field}={repr(child)})"
                    ) from exc

    # Evaluate nesting within tables
    if (
        isinstance(parent, dict)
        and is_tab(field)
        and not is_nesttab_inner(field)
        and child
    ):
        if is_nesttab(field):
            if any((not isinstance(c, list) for c in child if c is not None)):
                raise TypeError(f"Too few levels in a nested table ({field})")
            elif any(
                (
                    any((isinstance(c, list) for c in c if c is not None))
                    for c in child
                    if c is not None
                )
            ):
                raise TypeError(f"Too many levels in a nested table ({field})")
        elif any((isinstance(c, list) for c in child)):
            raise TypeError(f"Too many levels in a table ({field})")

    return child


@lru_cache(maxsize=None)
def _get_field_info(
    module: str, path: str | list[str], visible_only: bool = None
) -> dict:
    """Gets field info from a schema for a given module and path

    Moved outside of EMuSchema to allow use of lru_cache.
    """
    schema = EMuRecord.schema

    if visible_only is None:
        visible_only = schema.visible_only

    segments = _split_path(path)
    modules = [module]
    for seg in segments:
        obj = schema[
            ("Schema", modules[-1], "columns", strip_mod(seg).replace("_inner", ""))
        ]
        modules.append(obj.get("RefTable", modules[-1]))

    # Views are data from a single attachment that appear in multiple fields in
    # the linking module. They should not be read or written to.
    try:
        is_view = obj["RefLink"] != obj["ColumnName"]
    except KeyError:
        pass
    else:
        if is_view:
            # NOTE: The content of this error message is used in define_group()
            raise KeyError(
                f"{module}.{seg} is a view of the data in {obj['RefLink']}. Access"
                f" that data through the main attachment field instead."
            )

    # The schema may include fields that are not visible in the client. ItemName
    # *appears* to be populated only for fields that actually appear in the client.
    module = modules[-2]
    if (
        visible_only
        and not obj.get("ItemName")
        and not ".".join([module] + list(segments)) in EMuRecord.config["make_visible"]
    ):
        raise KeyError(f"{module}.{seg} appears in the schema but is not visible")

    return obj


def _get_module(obj: EMuRecord | EMuColumn, field: str = None) -> str:
    """Gets module name"""
    if field is None:
        field = obj.field
    if obj.schema is not None and field is not None and is_ref(field):
        return obj.schema.get_field_info(obj.module, field)["RefTable"]
    return obj.module


def _is_not_blank(val: Any) -> bool:
    """Tests if value is not blank"""
    return val or val == 0


def _split_path(path: str) -> tuple[str]:
    """Splits path into segments"""
    if isinstance(path, tuple):
        return path
    if isinstance(path, str):
        return tuple(re.split(r"[./]", path))
    if isinstance(path, list):
        return tuple(path)
    raise ValueError(f"Invalid path format: {path}")


DEFAULT_RECORD = EMuRecord
DEFAULT_COLUMN = EMuColumn

# Load config and schema, if provided
try:
    EMuSchema()
except (FileNotFoundError, TypeError):
    pass
