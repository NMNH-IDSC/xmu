"""Defines tools to work with the EMu API"""

import getpass
import json
import logging
import re
from typing import Any, Callable
from urllib.parse import unquote_plus, urljoin

import requests

from .types import EMuDate, EMuLatitude, EMuLongitude, EMuTime
from .utils import is_ref


logger = logging.getLogger(__name__)


TIMEOUT = 30


class EMuAPI:
    """Connects to and queries the EMu API

    Parameters
    ----------
    url : str
        the url for the EMu API, including tenant
    username : str, optional
        an EMu username. If omitted, defaults to the current OS username.
    password : str, optional
        the password for the given username, If omitted, the user will be
        prompted for the password when the class is initiated.

    Attributes
    ----------
    module : str
        the backend name of an EMu module, for example, ecatalogue or eparties
    use_emu_syntax : bool
        specifies whether to use the EMu client syntax when parsing search terms.
        Clients searches escape control characters using a backslash.
    """

    schema = None

    def __init__(
        self,
        url: str,
        username: str = None,
        password: str = None,
        rec_class: Callable = dict,
    ):
        self.base_url = url.rstrip("/") + "/"
        self.use_emu_syntax = True
        self.rec_class = rec_class

        # Get token
        self._token = requests.post(
            urljoin(self.base_url, "tokens"),
            json={
                "username": getpass.getuser() if username is None else username,
                "password": getpass.getpass() if password is None else password,
            },
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT,
        ).headers["Authorization"]

        self._session = None

    @property
    def session(self):
        if self._session is None:
            self._session = requests.Session()
        return self._session

    @session.setter
    def session(self, val):
        self._session = val

    def get(self, *args, **kwargs):
        """Performs a GET operation with the proper authorization header"""
        headers = kwargs.setdefault("headers", {})
        headers["Authorization"] = f"{self._token}"
        headers.setdefault("Prefer", "representation=none")

        # Add the HTTP method override per recommendation at
        # https://help.emu.axiell.com/emurestapi/3.1.2/05-Appendices-Override.html
        headers["X-HTTP-Method-Override"] = "GET"
        headers["Content-Type"] = "application/x-www-form-urlencoded"

        return EMuAPIResponse(
            self.session.post(*args, **kwargs), api=self, rec_class=self.rec_class
        )

    def retrieve(self, module: str, irn: str | int, select: list[str] = None) -> None:
        """Retrieves a single record from an irn

        Parameters
        ----------
        module: str
            the module to query
        irn : str | int
            the IRN for the EMu record to retrieve
        select : list[str], optional
            the list of EMu fields to return. If omitted, returns the record id.

        Returns
        -------
        ????
        """
        url = self.base_url
        for part in [module, str(irn)]:
            url = urljoin(url, part).rstrip("/") + "/"
        params = self._prep_query(module=module, select=select)
        resp = self.get(url.rstrip("/"), data=params)
        resp.select = select
        return resp

    def search(
        self,
        module: str,
        select: list[str] = None,
        sort_: dict = None,
        filter_: dict = None,
        limit: int = 10,
        cursor_type: str = "server",
    ):
        """Searches EMu based on the provided filter

        Parameters
        ----------
        module: str
            the module to query
        select : list[str], optional
            the list of EMu fields to return. If omitted, returns the record id.
        sort_ : dict, optional
            criteria by which to order the results. Each key must have the value
            "asc" or "desc".
        filter_: dict, optional
            search filter. Each key-value pair consists of a field name and value.
            Complex searches can be made using the helper functions included in
            this module (contains, phrase, etc.) Lists are expanded as OR searches.
            Values that have not been converted to the API syntax will be parsed
            using a set of rules modeled on EMu client searches.
        limit: int, default=10
            the number of records to return per page
        cursor_type: strc, default="server"

        Yields
        ------
        ????
        """
        params = self._prep_query(
            module=module,
            select=select,
            sort=sort_,
            filter=filter_,
            limit=limit,
            cursor_type=cursor_type,
        )
        resp = self.get(urljoin(self.base_url, module).rstrip("/"), data=params)
        resp.select = select
        return resp

    def _prep_query(self, **kwargs):

        params = {}

        if kwargs.get("select"):
            params["select"] = _prep_select(kwargs["select"])

        if kwargs.get("sort"):
            params["sort"] = _prep_sort(kwargs["sort"])

        if kwargs.get("filter"):
            params["filter"] = _prep_filter(
                kwargs["module"], kwargs["filter"], self.use_emu_syntax
            )

        for key in ("limit", "cursorType"):
            if kwargs.get(key):
                params[key] = kwargs[key]

        return params


class EMuAPIResponse:

    def __init__(self, response, api, rec_class):
        self._response = response
        self._api = api
        self._rec_class = rec_class
        self.select = None

    def __getattr__(self, attr):
        return getattr(self._response, attr)

    def __iter__(self):
        try:
            # yield self.json()["data"]
            rec = _parse_api(self.module, self.json()["data"])
            yield rec
            # if self._rec_class != dict:
            #    rec = self._rec_class(rec, module=self.module)
            # yield self.resolve_attachments(rec)
        except KeyError:
            resp = self
            while True:
                try:
                    for match in resp.json()["matches"]:
                        # yield match["data"]
                        rec = _parse_api(resp.module, match["data"])
                        yield rec
                        # if resp._rec_class != dict:
                        #    rec = resp._rec_class(rec, module=resp.module)
                        # yield resp.resolve_attachments(rec)
                except Exception as exc:
                    try:
                        raise ValueError(f"Could not parse match: {match}") from exc
                    except NameError:
                        raise ValueError(
                            f"No records found: {repr(resp.text)} ({resp.request.url})"
                        ) from exc
                else:
                    try:
                        resp = resp.next_page()
                    except ValueError:
                        break

    @property
    def module(self):
        try:
            return self.json()["id"].split("/")[-2]
        except KeyError:
            return self.json()["matches"][0]["id"].split("/")[-2]

    @property
    def params(self):
        params = {}
        for param in self.request.body.split("&"):
            key, val = param.split("=", 1)
            params.setdefault(key, []).append(json.loads(unquote_plus(val)))
        return params

    def records(self):
        return {r["irn"]: r for r in self}

    def first(self):
        for rec in iter(self):
            return rec

    def next_page(self):
        limit = int(self.params.get(b"limit", [10])[0])
        if len(json.loads(self.headers["Next-Offsets"])) % limit:
            raise ValueError("Last page")
        return self._api.get(
            self.url,
            data=self.request.body,
            headers={"Next-Search": self.headers["Next-Search"]},
        )

    def resolve_attachments(self, rec):
        if not self.select:
            return rec
        for key, val in rec.items():
            if is_ref(key):
                field_info = self._api.schema.get_field_info(self.module, key)
                try:
                    select = self.select[key]
                except (KeyError, TypeError):
                    pass
                else:
                    if isinstance(val, list):
                        for i, val in enumerate(val):
                            if val:
                                val = self._api.retrieve(
                                    field_info["RefTable"], val, select=select
                                ).first()
                            rec[key][i] = val
                    else:
                        rec[key] = self._api.retrieve(
                            field_info["RefTable"], val, select=select
                        ).first()
        return rec


def and_(clauses: list[dict]) -> dict:
    """Combines a list of clauses with AND

    Parameters
    ----------
    clauses : list[dict]
        list of clauses

    Returns
    -------
    dict
        {"AND": clauses}
    """
    return {"AND": clauses}


def or_(clauses: list[dict]) -> dict:
    """Combines a list of clauses with OR

    Parameters
    ----------
    clauses : list[dict]
        list of clauses

    Returns
    -------
    dict
        {"OR": clauses}
    """
    return {"OR": clauses}


def not_(clause: dict) -> dict:
    """Negates a clause

    Parameters
    ----------
    clauses : dict
        a clause

    Returns
    -------
    dict
        {"NOT": clauses}
    """
    return {"NOT": clause}


def contains(val: str | list[str], col: str = None) -> dict:
    """Builds a clause to match fields containing a value in the EMu API

    Paramters
    ---------
    val : str | list[str]
        the text to search for

    Returns
    -------
    dict
        an EMu API contains clause
    """
    return _build_multivalue_clause(val, col=col, op="contains")


def range_(
    gt: str | int | float = None,
    lt: str | int | float = None,
    gte: str | int | float = None,
    lte: str | int | float = None,
    mode: str = None,
    col: str = None,
) -> dict:
    """Builds a clause to match a range of values

    At least one of gt, lt, gte, and lte must be provided. Only one of gt and gte
    can be provided, and only one of lt and lte can be provided.

    Paramters
    ---------
    gt: str | float | int
        the lower bound of the search, not inclusive
    lt: str | float | int
        the upper bound of the search, inclusive
    gte: str | float | int
        the lower bound of the search, inclusive
    lte: str | float | int
        the upper bound of the search, inclusive
    mode : str
        one of date, time, latitude, or longitude. If omitted, will try to guess
        based on the column or value.

    Returns
    -------
    dict
        an EMu API phonetic clause
    """
    kwargs = {"gt": gt, "lt": lt, "gte": gte, "lte": lte}
    op = {k: v for k, v in kwargs.items() if v is not None}
    if not op:
        raise ValueError("Must provide at least one of gt, lt, gte, or lte")
    if "gt" in op and "gte" in op:
        raise ValueError("Can only provide one of gt and gte")
    if "lt" in op and "lte" in op:
        raise ValueError("Can only provide one of lt and lte")
    # Infer mode from type of data
    if mode is None:
        mode = _infer_mode(list(op.values())[0])
        if mode:
            op["mode"] = mode
    return _build_clause(None, col=col, op="range", **op)


def exact(val: str | float | int, col: str = None, mode: str = None) -> dict:
    """Builds a clause to match fields exactly matching a value in the EMu API

    Paramters
    ---------
    val : str | float | int | list[str] | list[float] | list[int]
        the value or list of values to match
    mode : str
        TKTK

    Returns
    -------
    dict
        an EMu API exact clause
    """
    if mode is None:
        mode = _infer_mode(val)
    return _build_clause(val, col=col, op="exact", mode=mode)


def exists(val: bool, col: str = None) -> dict:
    """Builds a clause to test whether a field is populated in the EMu API

    Paramters
    ---------
    val : bool
        whether the field is populated. True returns records where the specified
        field is populated, False returns records where it is empty.

    Returns
    -------
    dict
        an EMu API exists clause
    """
    return _build_clause(val, col=col, op="exists")


def phonetic(val: str | list[str], col: str = None) -> dict:
    """Builds a clause to ... value in the EMu API

    Paramters
    ---------
    val : str | list[str]
        the text to search for

    Returns
    -------
    dict
        an EMu API phonetic clause
    """
    return _build_multivalue_clause(val, col=col, op="phonetic")


def phrase(val: str | list[str], col: str = None) -> dict:
    """Builds a clause to ... value in the EMu API

    Paramters
    ---------
    val : str | list[str]
        the text to search for

    Returns
    -------
    dict
        an EMu API phrase clause
    """
    return _build_clause(val, col=col, op="phrase")


def proximity(val: str | list[str], col: str = None, distance: int = 3) -> dict:
    """Builds a clause to search for words within a certain distance of each other

    Equivalent to \\'\\(the \\"black cat\\"\\) <= 5 words\\' in the EMu client. The
    client supports more complex operations (for example, searching in order) that do
    not appear to be supported by the API.

    Paramters
    ---------
    val : str | list[str]
        a string of two or more words or a list of such strings
    distance : int
        the maximum distance between words

    Returns
    -------
    dict
        an EMu API phonetic clause
    """
    return _build_clause(val, col=col, op="proximity", distance=distance)


def regex(val: str | list[str], col: str = None) -> dict:
    """Builds a clause to ... value in the EMu API

    Paramters
    ---------
    val : str | list[str]
        the pattern to search for

    Returns
    -------
    dict
        an EMu API regex clause
    """
    return _build_clause(val, col=col, op="regex")


def stemmed(val: str | list[str], col: str = None) -> dict:
    """Builds a clause to ... value in the EMu API

    Paramters
    ---------
    val : str | list[str]
        the text to search for

    Returns
    -------
    dict
        an EMu API stemmed clause
    """
    return _build_multivalue_clause(val, col=col, op="stemmed")


def is_not_null(col: str = None) -> dict:
    """Builds a clause that matches a non-empty field in the EMu API

    Alias for exists(True).

    Returns
    -------
    dict
        an EMu API exists=True clause
    """
    return exists(True, col=col)


def is_null(col: str = None) -> dict:
    """Builds a clause that matches an empty field in the EMu API

    Returns
    -------
    dict
        an EMu API exists=False clause
    """
    return exists(False, col=col)


def order(val: str = "asc", col: str = None) -> dict:
    """Builds a clause to sort in the given direction

    Paramters
    ---------
    val : str
        sort direction. Must be either "asc" or "desc".

    Returns
    -------
    dict
        an EMu API order clause
    """
    return _build_clause(val, col=col, op="order")


def emu_escape(val: str) -> str:
    """Escapes a string according to EMu escape syntax"""
    for item in ['"', "'", "!", "[", "]", "^", "$"]:
        val = val.replace(item, rf"\{item}")
    val = val.replace("!*", r"!\*")
    val = val.replace("!+", r"!\+")
    return val


def emu_unescape(val: str) -> str:
    """Unescapes a string that uses the EMu escape syntax"""
    for item in ['"', "'", "!", "[", "]", "^", "$"]:
        val = val.replace(rf"\{item}", item)
    val = val.replace(r"!\*", "!*")
    val = val.replace(r"!\+", "!+")
    return val


def _infer_mode(val: Any) -> str | None:
    """Infers mode based on value"""
    classes = [
        (float, None),
        (EMuDate, "date"),
        (EMuTime, "time"),
        (EMuLatitude, "latitude"),
        (EMuLongitude, "longitude"),
    ]
    for cls_, mode in classes:
        if isinstance(val, cls_):
            return mode
    for cls_, mode in classes:
        try:
            cls_(val)
        except (IndexError, TypeError, ValueError):
            pass
        else:
            return mode
    return None


def _isinstance(val: Any, obj: object) -> bool:
    """Tests whether value or first value in an iterable is an instance of obj"""
    return isinstance(val[0] if val and isinstance(val, (list, tuple)) else val, obj)


def _prep_field(val: str) -> str:
    """Formats field names to use data.EmuField syntax"""
    if val == "id" or val.startswith("data."):
        return val
    return f"data.{val}"


def _prep_select(select: dict | list = None) -> str:
    """Expands list of fields to the format used by the EMu API"""
    if select is None:
        select = []
    select = list(select)
    if "id" not in select:
        select.insert(0, "id")
    param = ",".join([_prep_field(f) for f in select])
    logger.debug(f"select={repr(param)}")
    return param


def _prep_sort(sort_: dict) -> str:
    """Expands a simple sort to the format used by the EMu API"""
    if isinstance(sort_, list):
        sort_ = {c: "asc" for c in sort_}
    clauses = []
    for col, val in sort_.items():
        if not isinstance(val, dict):
            val = order(val, col=col)
        clauses.append(val)
    param = json.dumps(clauses)
    logger.debug(f"sort={repr(param)}")
    return param


def _prep_filter(module: str, filter_: dict, use_emu_syntax: bool = True) -> str:
    """Expands a simple filter to the format used by the EMu API"""
    stmts = []
    for col, val in filter_.items():

        # Add column name to individual clauses if not already there
        if isinstance(val, dict):
            for bool_op, vals in val.items():
                val[bool_op] = [{col: v} for v in vals]

        else:
            # Infer operator based on data type in the schema if provided
            if EMuAPI.schema:
                data_type = EMuAPI.schema.get_field_info(module, col)["DataType"]
                val = _val_to_query(
                    col, val, use_emu_syntax=use_emu_syntax, data_type=data_type
                )

            # Otherwise base the clause on the type of data supplied
            elif _isinstance(val, bool):
                val = exists(val, col=col)
            elif _isinstance(val, (float, int)):
                val = exact(val, col=col)
            else:
                val = _val_to_query(col, val, use_emu_syntax=use_emu_syntax)

        val = val.get("AND", val)
        if not isinstance(val, list):
            val = [val]
        if len(val) > 1:
            stmts.append(and_(val))
        else:
            stmts.append(val[0])

    param = json.dumps(and_(stmts) if len(stmts) > 1 else stmts[0])
    logger.debug(f"filter={repr(param)}")
    return param


def _build_clause(val: Any, op: str, col: str = None, **kwargs) -> dict:
    """Helper function to build a clause for the EMu API"""

    if col:
        col = _prep_field(col)

    # Omit empty kwargs
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if op == "range":

        gt_key = None
        gt = None
        lt_key = None
        lt = None
        for key, val in kwargs.items():
            if key.startswith("gt"):
                gt_key = key
                gt = val
            elif key.startswith("lt"):
                lt_key = key
                lt = val

        if isinstance(gt, (list, tuple)) or isinstance(lt, (list, tuple)):

            if gt and lt:
                # If both gt and lt are defined, they must have the same
                # type and same number of items
                if type(gt) != type(lt):
                    raise ValueError(f"{gt_key} and {lt_key} must have the same type")
                if isinstance(gt, (list, tuple)) and len(gt) != len(lt):
                    raise ValueError(
                        f"{gt_key} and {lt_key} must have the same number of items"
                    )
                vals = []
                for gt, lt in zip(gt, lt):
                    kwargs[gt_key] = gt
                    kwargs[lt_key] = lt
                    vals.append(_build_clause(None, op, col=col, **kwargs))
                return or_(vals)

            elif gt:
                for gt in gt:
                    kwargs[gt_key] = gt
                    vals.append(_build_clause(None, op, col=col, **kwargs))
                return or_(vals)

            elif lt:
                for lt in lt:
                    kwargs[lt_key] = lt
                    vals.append(_build_clause(None, op, col=col, **kwargs))
                return or_(vals)

        else:
            return {"range": kwargs} if col is None else {col: {"range": kwargs}}

    elif isinstance(val, (list, tuple)):
        if len(val) > 1:
            return or_([_build_clause(v, op, col=col, **kwargs) for v in val])
        val = vals[0]

    if op != "order":
        val = {"value": val}
        val.update(kwargs)


def _build_multivalue_clause(val: Any, op: str, col: str = None):
    """Builds clauses for operations that should be split by word"""
    clauses = []
    for val in [val] if isinstance(val, str) else val:
        clause = _build_clause(val.split(" "), col=col, op=op)
        try:
            clause = {"AND": clause.pop("OR")}
        except KeyError:
            pass
        clauses.append(clause)
    return or_(clauses) if len(clauses) > 1 else clauses[0]


def _val_to_query(
    col: str, val: str | list, use_emu_syntax: bool = True, data_type: str = None
) -> dict:
    """Converts a search string to a query based on EMu client

    Parameters
    ----------
    col: str
       the EMu column name
    val : str | list
        the value to convert
    use_emu_syntax : bool
        whether the value uses EMu escape syntax
    data_type : str
        the EMu data type. Used to ensure that range searches use the correct
        data type.

    Returns
    -------
    dict
        a query corresponding to the given value
    """
    # FIXME: Implement regex

    # Process multiple values
    if isinstance(val, list):
        if len(val) > 1:
            return or_([_val_to_query(col, v) for v in val])
        else:
            val = val[0]

    # Coerce to numeric type if a data_type hint is numeric
    to_type = {"Float": float, "Integer": int}.get(data_type, str)

    # Simple numeric values can be returned with exact
    if (not data_type or to_type in (float, int)) and isinstance(val, (float, int)):
        return exact(val, col=col)

    # EMuType classes map to exact
    for cls_, mode_ in (
        (EMuDate, "date"),
        (EMuTime, "time"),
        (EMuLatitude, "latitude"),
        (EMuLongitude, "longitude"),
    ):
        if isinstance(val, cls_):
            return exact(str(val), col=col, mode=mode_)

    # The mode argument controls how the exact and range clauses handle comparisons
    mode = {
        "Date": "date",
        "Time": "time",
        "Latitude": "latitude",
        "Longitude": "longitude",
    }.get(data_type)

    # Operators that can be used in client searches
    ops = ["!", ">=", ">", "<=", "<"]
    if use_emu_syntax:
        ops = [emu_escape(o) for o in ops]
    ops = "(" + "|".join([re.escape(o) for o in ops]) + ")"

    clauses = []

    # Not nulls
    not_nulls = ["!*", "!+"]
    if use_emu_syntax:
        not_nulls = [emu_escape(n) for n in not_nulls]
    not_nulls = "(" + "|".join([re.escape(n) for n in not_nulls]) + ")"
    if re.search(not_nulls, val):
        clauses.append(exists(True, col=col))
    val = re.sub(not_nulls, "", val).strip()

    # Phrases
    if use_emu_syntax:
        pattern = rf"{ops}?(\\'(?:.*?)\\'|\\\"(?:.*?)\\\")"
    else:
        pattern = rf"{ops}?('(?:.*?)'|\"(?:.*?)\")"
    for op, val_ in re.findall(pattern, val):
        clause = phrase(val_.strip("\"'\\"), col=col)
        clauses.append(clause if op.lstrip("\\") != "!" else not_(clause))
    val = re.sub(pattern, "", val).strip()

    # Words and numbers
    pattern = f"{ops}?(.*)"
    ranges = {}
    for val in re.split(f" +", val):
        op, val = re.findall(pattern, val)[0]
        op = op.lstrip("\\")
        if "<" in op or ">" in op:
            ranges[op] = val
        else:
            if to_type != str or mode:
                clause = exact(to_type(val), col=col, mode=mode)
            else:
                clause = contains(to_type(val), col=col)
            clauses.append(clause if op.lstrip("\\") != "!" else not_(clause))

    # Ranges
    if ranges:
        mapping = {">=": "gte", ">": "gt", "<=": "lte", "<": "lt"}
        kwargs = {mapping[k]: to_type(v) for k, v in ranges.items()}
        kwargs["mode"] = mode
        clauses.append(range_(col=col, **kwargs))

    return and_(clauses) if len(clauses) > 1 else clauses[0]


def _parse_api(module, val, key=None, mapped=None):

    if mapped is None:
        mapped = {}

    if key:
        print(key)
        try:
            key = EMuAPI.schema.map_short_name(module, key)
        except KeyError:
            pass

    # Iterate dicts
    if isinstance(val, dict):
        for key, val in val.items():
            _parse_api(module, val, key, mapped)

    # Map tables
    elif key.endswith("_grp"):

        keys = []
        for row in val:
            keys.extend(row)
        keys = set(keys)

        grid = {}
        for row in val:
            for key in keys:
                grid.setdefault(key, []).append(row.get(key))

        for key, vals in grid.items():
            if any(vals):
                _parse_api(module, vals, key, mapped)

    # Map nested tables
    elif key.endswith("_subgrp"):
        keys = []
        for row in val:
            for inner_row in row:
                keys.extend(inner_row)
        keys = set(keys)

        grid = {k: [] for k in keys}
        for row in val:
            for val in grid.values():
                val.append([])
            for inner_row in row:
                for key in keys:
                    grid[key][-1].append(inner_row.get(key))

        for key, vals in grid.items():
            if any(vals):
                _parse_api(module, vals, key, mapped)

    # Simplify IRNs
    elif val and key == "irn" or is_ref(key):
        if isinstance(val, str) and val.startswith("emu:"):
            mapped[key] = int(val.split("/")[-1])
        elif isinstance(val, list):
            mapped[key] = [
                int(s.split("/")[-1]) if isinstance(s, str) else s for s in val
            ]

    else:
        mapped[key] = val

    return mapped
