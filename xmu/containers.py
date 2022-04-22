"""Defines containers to read and write various EMu objects"""
import json
import logging
import os
import re
from collections.abc import MutableMapping, MutableSequence
from ctypes import c_uint64
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from pprint import pformat
from textwrap import wrap
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
    """

    def __init__(self, path=None):
        self.path = path
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
        }

        self.load_rcfile()

        # Set config parameter on all classes
        EMuSchema.config = self
        EMuReader.config = self
        EMuRecord.config = self
        EMuColumn.config = self
        EMuGrid.config = self
        EMuRow.config = self

    def __str__(self):
        return f"{self.__class__.__name__}({pformat(self._config)})"

    def __repr__(self):
        return repr(self._config)

    def __getitem__(self, key):
        return self._config[key]

    def __setitem__(self, key, val):
        self._config[key] = val

    def __delitem__(self, key):
        del self._config[key]

    def __len__(self):
        return len(self._config)

    def __iter__(self):
        return iter(self._config)

    def load_rcfile(self, path=None):
        """Loads a configuration file

        Parameters
        ----------
        path : str
            path to the rcfile. If not given, checks the current then home
            directory for an .xmurc file.

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
        config = {k: v[0] for k, v in self._options.items()}

        # Check each location for the rcfile
        for path in paths:

            # Use a default filename if none given
            if os.path.isdir(path):
                path = os.path.join(path, ".xmurc")

            try:
                with open(path, encoding="utf-8") as f:
                    config.update(yaml.safe_load(f))
                self.path = path
            except (FileNotFoundError, TypeError):
                pass

        self._config = config
        return config

    def save_rcfile(self, path=None, overwrite=False):
        """Saves a configuration file

        Parameters
        ----------
        path : str
            path for the rcfile. If a directory, adds .xmurc as the filename.
            Defaults to the user's home directory.
        overwrite : bool
            whether to overwrite the file if it exists
        """

        # Default to user home directory
        if path is None:
            path = os.path.expanduser("~")

        # Use a default filename if none given
        if os.path.isdir(path):
            path = os.path.join(path, ".xmurc")

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
        content = ["# YAML configuration file for python xmu package"]
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
    config = None

    def __init__(self, *args, **kwargs):

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

        # Add custom groups from config file. This needs to come after the
        # assignment of the class attributes because _get_field_info() uses
        # one of those to access the schema.
        if self.config is not None:
            for module, groups in self.config["groups"].items():
                for fields in groups.values():
                    self.define_group(module, fields, overwrite=True)

    def __getitem__(self, path):
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
    def modules(self):
        """Gets the list of modules in the schema"""
        return sorted(self["Schema"].keys())

    def get(self, key, default=None):
        """Overrides the native get method to support paths

        Parameters
        ----------
        key : mixed
            key to retrieve
        default : mixed
           default value to return if key not found

        Returns
        -------
        mixed
            value for the key or default if not found
        """
        try:
            return self[key]
        except KeyError:
            return default

    def from_file(self, path):
        """Loads schema from a file, creating a JSON version if not found

        Parameters
        ----------
        path : str
            path to a schema file
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
                for fields in data.get("groups", {}).values():
                    self.define_group(module, fields)

            self.to_json(f"{path}.json")

    def from_pl(self, path):
        """Loads schema from an EMu schema.pl file

        Parameters
        ----------
        path : str
            path to a schema file
        """
        self.update(self._read_schema_pl(path))

    def from_json(self, path):
        """Loads schema from JSON

        Parameters
        ----------
        path : str
            path to a JSON schema file
        """
        with open(path, encoding="utf-8") as f:
            self.update(json.load(f))

    def to_json(self, path, **kwargs):
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

    def iterfields(self):
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

    def define_group(self, module, fields, overwrite=False):
        """Maps a group definition to each member field

        Groups are read from a schema if possible but can also be defined manually.

        Parameters
        ----------
        module : str
            backend module name
        fields : list
            list of fields in the group
        overwrite : bool
            whether to overwrite an existing group
        """
        for field in fields:

            info = self.get_field_info(module, field)

            # Group definitions in the schema point to the client field that
            # manages a reference (e.g., ModField_tab instead of ModFieldRef_tab).
            # Use the reference table instead.
            try:
                fields[fields.index(field)] = info["RefLink"]
                info = self.get_field_info(module, info["RefLink"])
            except KeyError:
                pass

            if info.get("GroupFields") and not overwrite:
                warn(f"{module}.{field} is assigned to multiple groups")
            else:
                info["GroupFields"] = fields

        # Field definitions modified, so clear the cache
        _get_field_info.cache_clear()

    @staticmethod
    def get_field_info(module, path, visible_only=None):
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
        if isinstance(path, list):
            path = ".".join(path)
        return _get_field_info(module, path, visible_only=visible_only)

    def _read_schema_pl(self, path):
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

    def _get_similar_keys(self, path):
        """Finds fields similar to the one at the end of the given path"""
        last = path[-1][:4]
        obj = super().__getitem__(path[0])
        for key in path[1:-1]:
            obj = obj[key]
        return [k for k in obj if k.startswith(last)]

    @staticmethod
    def _parse_value(val):
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

    Attributes
    ----------
    module : str
        backend name of an EMu module
    field : str
        name of an EMu field
    dict_class : class
        class to apply to dicts that are children of this element
    """

    #: EMuConfig : module-wide configuration parameters. Set automatically
    #: when an EMuConfig object is created.
    config = None

    #: EMuSchema : info about a specific EMu configuration. Set automatically
    #: when an EMuSchema object is created.
    schema = None

    def __init__(self, vals=None, module=None, field=None):
        """Loads a vocabulary from a file or dict"""
        self.module = module
        self.field = field
        self.dict_class = EMuRecord
        if self.schema and not self.module:
            raise ValueError(
                f"Must provide module when schema is used (one of {self.schema.modules})"
            )
        super().__init__()
        if vals:
            self.extend(vals)

    def __str__(self):
        return f"EMuColumn({super().__str__()})"

    def __setitem__(self, i, val):
        # Catch non-integer indices to avoid problems with coercing values
        if not isinstance(i, int):
            raise TypeError("list indices must be integers or slices, not str")
        super().__setitem__(i, _coerce_values(self, val))

    def __add__(self, obj):
        return self.__class__(
            super().__add__(obj), module=self.module, field=self.field
        )

    def __iadd__(self, obj):
        self.extend(obj)
        return self

    def insert(self, i, val):
        super().insert(i, _coerce_values(self, val))

    def append(self, val):
        super().append(_coerce_values(self, val))

    def extend(self, vals):
        super().extend([_coerce_values(self, v) for v in vals])

    def to_xml(self, root=None, kind=None, row_ids=None):
        """Converts column to XML formatted for EMu

        Normally called without specifying arguments.

        Parameters
        ----------
        root : lxml.etree.Element or SubElement
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
                    atom.text = str(child) if _is_not_blank(child) else ""

        return root


class EMuRow(MutableMapping):
    """Reads and writes data in a grid row

    Changes to the row are reflected in the original EMuRecord

    Parameters
    ----------
    rec : XMuRecord
        the EMu record the grid is from
    path : str
        path to a field that is part of the grid
    index : int
        the index of the row
    fill_value : mixed
        value used when deleting an item from the row

    Attributes
    ----------
    group : tuple
        names for all columns that are part of the parent grid, whether they appear
        in the current record or not
    fill_value : mixed
        value used when deleting an item from the row
    """

    #: EMuConfig : module-wide configuration parameters. Set automatically
    #: when an EMuConfig object is created.
    config = None

    #: EMuSchema : info about a specific EMu configuration. Set automatically
    #: when an EMuSchema object is created.
    schema = None

    def __init__(self, rec, path, index, fill_value=None):
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

    def __str__(self):
        try:
            row = {c: self._rec[c][self.index] for c in self.columns}
        except IndexError:
            raise IndexError(
                "One or more columns has no data for this row. Use pad() on the parent grid to prevent this error."
            )
        return f"{self.__class__.__name__}({row})"

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(self.columns)

    def __len__(self):
        return len(self.columns)

    def __setitem__(self, key, val):
        self._rec[key][self.index] = val

    def __getitem__(self, key):
        return self._rec[key][self.index]

    def __delitem__(self, key):
        self[key] = self.fill_value

    @property
    def columns(self):
        """Lists columns in the row that exist in the record"""
        cols = [c for c in self._rec if strip_mod(c) in set(self.group)]
        if (
            any((c.endswith(")") for c in cols))
            and len({c.rsplit("(", 1)[-1] for c in cols}) > 1
        ):
            raise ValueError(f"Inconsistent modifier within grid: {cols}")
        return cols

    @property
    def mod(self):
        """Returns the modifier suffix needed to update this row in an import"""
        return f"{self.index + 1}="

    def row_id(self):
        """Calculates an identifier based on the index and content of a row"""
        val = str(self.index) + str(self)
        return c_uint64(hash(val)).value.to_bytes(8, "big").hex()


class EMuGrid(MutableSequence):
    """Reads and writes data in a grid

    Changes to the grid are reflected in the original EMuRecord

    Parameters
    ----------
    rec : XMuRecord
        the EMu record the grid is from
    path : str
        path to a field that is part of the grid
    fill_value : mixed
        value used when padding columns
    pad : bool
        whether to pad the columns to the same length

    Attributes
    ----------
    group : tuple
        names for all columns that are part of this grid, whether they appear
        in the current record or not
    fill_value : mixed
        value to use when padding the grid or deleting an item from an EMuRow
        object created from this grid
    """

    #: EMuConfig : module-wide configuration parameters. Set automatically
    #: when an EMuConfig object is created.
    config = None

    #: EMuSchema : info about a specific EMu configuration. Set automatically
    #: when an EMuSchema object is created.
    schema = None

    def __init__(self, rec, path, fill_value=None):
        module = _get_module(rec)
        self.group = tuple(
            self.schema.get_field_info(module, path).get("GroupFields", [])
        )
        if not self.group:
            raise KeyError(f"{module}.{path} is not part of a group")
        self.fill_value = fill_value

        # Use path to drill down to the correct parent record
        path = _split_path(path)[:-1]
        if path:
            rec = rec[path]
        self._rec = rec

    def __str__(self):
        return f"{self.__class__.__name__}({list(self)})"

    def __repr__(self):
        return str(self)

    def __iter__(self):
        for i in range(len(self)):
            yield EMuRow(self._rec, self.columns[0], i, fill_value=self.fill_value)

    def __len__(self):
        try:
            return max([len(self._rec[c]) for c in self.columns])
        except ValueError:
            return 0

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self)[key]

        if isinstance(key, dict):
            matches = []
            for i, row in enumerate(self):
                if all(
                    (
                        self._transform(row.get(k)) in self._transform(v)
                        for k, v in key.items()
                    )
                ):
                    matches.append((i, row))
            return matches

        return self._rec[key]

    def __setitem__(self, i, vals):
        # Required by MutableSequence but does not make sense to implement
        raise NotImplementedError(
            "Cannot set items on an EMuGrid. Use the main EMuRecord object or"
            " an individual EMuRow instead."
        )

    def __delitem__(self, i):
        for col in self.columns:
            del self._rec[col][i]

    def __eq__(self, other):
        return list(self) == list(other)

    @property
    def columns(self):
        """Lists columns in the grid that exist in the record"""
        cols = [c for c in self._rec if strip_mod(c) in set(self.group)]
        if (
            any((c.endswith(")") for c in cols))
            and len({c.rsplit("(", 1)[-1] for c in cols}) > 1
        ):
            raise ValueError(f"Inconsistent modifier within grid: {cols}")
        return cols

    def insert(self, index, value):
        # Required by MutableSequence but does not make sense to implement
        raise NotImplementedError(
            "Cannot insert into an EMuGrid. Use the main EMuRecord object instead."
        )

    def add_columns(self, cols=None, fill_value=None):
        """Adds missing columns to the grid

        Parameters
        ----------
        cols : list-like
            columns to add. If not given, adds all columns in the group attribute
            that do not already appear in the record.
        fill_value : mixed
            the value used to pad a column. Defaults to fill_value attribute
            if not given.

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

    def pad(self, fill_value=None):
        """Pads all columns in the table to the same length

        Parameters
        ----------
        fill_value : mixed
            the value used to pad a column. Defaults to fill_value attribute
            if not given.

        Returns
        -------
        self
        """
        if fill_value is None:
            fill_value = self.fill_value
        for col in self.columns:
            diff = len(self) - len(self._rec[col])
            self._rec[col].extend([fill_value for _ in range(diff)])
        return self

    def query(self, field=None, where=None):
        """Queries the grid

        Parameters
        ----------
        field : str
            a specific column to return. If empty, the whole row is returned.
        where : dict
            the query as a dict. A row must match all criteria to be returned.

        Returns
        -------
        list
            list of matching rows or values
        """
        results = []
        for _, row in self[where]:
            results.append(row[field] if field else row)
        return results

    @staticmethod
    def _transform(val):
        if not isinstance(val, (list, tuple)):
            val = [val]
        return "|".join([str(s) if s is not None else "" for s in val]).lower()


class EMuRecord(dict):
    """Reads and writes data in an EMu record

    Parameters
    ----------
    rec : mapping or iterable
        record as a mapping or iterable
    module : str
        backend name of an EMu module
    field : str
        name of an EMu field
    list_class : list-like
        class to use for lists in the dict

    Attributes
    ----------
    module : str
        backend name of an EMu module
    field : str
        name of an EMu field
    list_class : str
        class to use for lists in the dict
    """

    #: EMuConfig : module-wide configuration parameters. Set automatically
    #: when an EMuConfig object is created.
    config = None

    #: EMuSchema : info about a specific EMu configuration. Set automatically
    #: when an EMuSchema object is created.
    schema = None

    def __init__(self, rec=None, module=None, field=None, list_class=EMuColumn):
        self.module = module
        self.field = field
        self.list_class = list_class

        # Load a config file from one of the default locations if empty
        if self.config is None:
            EMuConfig()

        # Load schema specified in the config file if empty
        if self.schema is None:
            try:
                EMuSchema(self.config["schema_path"])
            except (FileNotFoundError, TypeError):
                pass

        if self.schema and not self.module:
            raise ValueError(
                f"Must provide module when schema is used (one of {self.schema.modules})"
            )

        super().__init__()
        if rec:
            self.update(rec)

    def __str__(self):
        return f"{self.__class__.__name__}({pformat(self)})"

    def __getitem__(self, path):
        path = _split_path(path)
        try:
            if len(path) > 1:
                obj = self
                for key in path:
                    obj = obj[key]
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
                    warn(f"Invalid path: {dotpath} (module={module})")
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

    def __setitem__(self, key, val):
        super().__setitem__(key, _coerce_values(self, val, key))

    def get(self, key, default=None):
        """Overrides the native dict.get method to map unrecognized terms"""
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key, val):
        """Overrides the native dict.setdefault method to use the subclass setter"""
        try:
            return self[key]
        except KeyError:
            self[key] = val
            return self[key]

    def update(self, *args, **kwargs):
        """Overrides the native dict.update method to use the subclass setter"""
        for key, val in dict(*args, **kwargs).items():
            self[key] = val

    def copy(self):
        """Overrides the native dict.copy method to return an objet of this class"""
        return self.__class__(deepcopy(dict(self)), module=_get_module(self))

    def grid(self, field, **kwargs):
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

    def to_xml(self, root=None, kind=None):
        """Converts record to XML formatted for EMu

        Normally called without specifying arguments.

        Parameters
        ----------
        root : lxml.etree.Element or SubElement
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
        if kind == "update":
            for key in list(self):
                try:
                    grids[key]
                except KeyError:
                    try:
                        grid = self.grid(key)
                    except KeyError:
                        pass
                    else:
                        # Include all columns when appending
                        if key.endswith("(+)"):
                            grid.add_columns()
                        grid.pad()
                        row_ids = [r.row_id() for r in grid]
                        for col in grid.columns:
                            grids[col] = row_ids

        for key, val in self.items():
            if is_tab(key):
                # If field is part of a grid, pass row identifiers to the
                # EMuColumn to_xml() method. These will be used to populate the
                # group attribute in each tuple tag for appends and prepends.
                val.to_xml(root, kind=kind, row_ids=grids.get(key, None))
            elif is_ref(key):
                ref_tup = etree.SubElement(root, "tuple")
                ref_tup.set("name", key)
                val.to_xml(ref_tup, kind=kind)
            elif (
                _is_not_blank(val)
                or kind in {"emu", "update"}
                or is_ref(self.field if self.field else "")
            ):
                atom = etree.SubElement(root, "atom")
                atom.set("name", key)
                atom.text = str(val) if _is_not_blank(val) else ""
        return root


def _coerce_values(parent, child, key=None):
    """Coerces child containers and values to specific classes"""

    if isinstance(parent, dict):
        dict_class = parent.__class__
        list_class = parent.list_class
        module = _get_module(parent)
        field = key

    # Lists inherit the field attribute from their parent
    elif isinstance(parent, (list, tuple)):
        dict_class = parent.dict_class
        list_class = parent.__class__
        module = parent.module
        field = parent.field

    # Validate field if schema has been loaded
    field_info = None
    if parent.schema and parent.schema.validate_paths:
        field_info = parent.schema.get_field_info(module, key if key else field)

    # Label inner nested tables
    if is_nesttab(field) and not isinstance(parent, dict_class):
        field = f"{strip_mod(field)}_inner"

    # Simplify IRN-only references
    if is_ref(field):
        # Simplify IRN-only references to integers
        if isinstance(child, dict) and list(child) == ["irn"]:
            child = child["irn"]

        # Interpret integers in reference fields as IRNs
        if isinstance(child, int):
            return child

    # Tables must be list-like
    if (
        field != parent.field
        and is_tab(field)
        and not isinstance(child, (list, tuple))
        and child is not None
    ):
        raise TypeError(f"Columns must be lists ({child} was assigned to {field})")

    # References must be dicts or ints (which are interpreted as IRNS
    if (
        is_ref(field)
        and not is_tab(field)
        and not isinstance(child, dict)
        and child is not None
    ):
        raise TypeError(f"References must be dicts ({child} was assigned to {field})")

    # Coerce containers to the proper types
    if isinstance(child, dict) and not isinstance(child, dict_class):
        child = dict_class(child, module=module, field=field)

    elif isinstance(child, (list, tuple)) and not isinstance(child, list_class):
        child = list_class(child, module=module, field=field)

    # Coerce non-list, non-dict data to an appropriate type if a schema is defined
    elif field_info and not isinstance(child, (dict, list)):
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
def _get_field_info(module, path, visible_only=None):
    """Gets field info from a schema for a given module and path

    Moved outside of EMuSchema to allow use of lru_cache.
    """
    schema = EMuRecord.schema

    if visible_only is None:
        visible_only = schema.visible_only

    for seg in _split_path(path):
        obj = schema[
            ("Schema", module, "columns", strip_mod(seg).replace("_inner", ""))
        ]
        module = obj.get("RefTable", module)

    # ItemName *appears* to be populated only for fields that appear in the client
    if visible_only and not obj.get("ItemName"):
        raise KeyError(f"{module}.{seg} is valid but not not visible")

    return obj


def _get_module(obj):
    """Gets module name"""
    if obj.schema is not None and obj.field is not None and is_ref(obj.field):
        return obj.schema.get_field_info(obj.module, obj.field)["RefTable"]
    return obj.module


def _is_not_blank(val):
    """Tests if value is not blank"""
    return val or val == 0


def _split_path(path):
    """Splits path into segments"""
    if isinstance(path, str):
        path = re.split("[./]", path)
    elif not isinstance(path, (list, tuple)):
        path = [path]
    return path
