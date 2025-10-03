"""Defines tools to work with the EMu API"""

import getpass
import json
import logging
import re
from typing import Any
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
    parser : EMuAPIParser, optional
        the parser object used to parse individual records. The default EMuAPIParser
        class returns a close approximation of the format used by EMuRecord. If None,
        records will be returned as formatted by the API.
    autopage : bool = True
        whether to automatically page through results if the total number of results
        exceeds the limit of a given request

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
        parser: EMuAPIParser = None,
        autopage: bool = True,
    ):
        self.base_url = url.rstrip("/") + "/"
        self.use_emu_syntax = True
        self.parser = parser

        # The autopage parameter is passed to EMuAPIResponse but it is cleaner
        # to implement it here
        self.autopage = autopage

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

        select = kwargs.pop("select", None)

        # Redact authorization before logging
        redacted = re.sub(
            "'Authorization': '.*?'", "'Authorization': '[REDACTED]'", str(kwargs)
        )

        logger.debug(f"Making GET request: {args[0]} (params={redacted})")
        return EMuAPIResponse(
            self.session.post(*args, **kwargs),
            api=self,
            select=select,
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
        EMuAPIResponse
            the query response
        """
        url = self.base_url
        for part in [module, str(irn)]:
            url = urljoin(url, part).rstrip("/") + "/"
        params = self._prep_query(module=module, select=select)
        return self.get(url.rstrip("/"), data=params, select=select)

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
        EMuAPIResponse
            the query response
        """
        params = self._prep_query(
            module=module,
            select=select,
            sort=sort_,
            filter=filter_,
            limit=limit,
            cursor_type=cursor_type,
        )
        return self.get(
            urljoin(self.base_url, module).rstrip("/"), data=params, select=select
        )

    def _prep_query(self, **kwargs):
        """Format the query for the EMu API"""

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
    """Wraps a response from the EMu API response"""

    cache = {}

    def __init__(
        self,
        response: requests.Response,
        api: EMuAPI,
        select: list[str] | dict[dict] = None,
    ):
        self._response = response
        self._api = api
        self._select = select

    def __getattr__(self, attr):
        return getattr(self._response, attr)

    def __len__(self):
        return len(json.loads(self.headers["Next-Offsets"]))

    def __iter__(self):
        try:
            yield self._get(self.json()["data"])
        except json.JSONDecodeError:
            raise ValueError(
                f"Response cannot be decoded: {repr(self.text)} (status_code={self.status_code})"
            )
        except KeyError:
            resp = self
            while True:
                try:
                    for match in resp.json()["matches"]:
                        yield self._get(match["data"], resp)
                except Exception as exc:
                    try:
                        raise ValueError(
                            f"Could not parse match: {match} from {repr(resp.text)}"
                        ) from exc
                    except NameError:
                        raise ValueError(
                            f"No records found: {repr(resp.text)} ({resp.request.url})"
                        ) from exc
                else:
                    # Get
                    if self._api.autopage:
                        try:
                            resp = resp.next_page()
                        except ValueError:
                            break
                        else:
                            if hasattr(resp, "from_cache") and resp.from_cache:
                                logger.debug("Response is from cache")
                            else:
                                logger.debug("Response is from server")

    @property
    def module(self):
        """The EMu module queried to create the response"""
        try:
            return self.json()["id"].split("/")[-2]
        except KeyError:
            return self.json()["matches"][0]["id"].split("/")[-2]

    @property
    def params(self):
        """The query parameters used to make the request"""
        body = self.request.body
        # Decode the request body if using requests_cache
        try:
            body = body.decode("utf-8")
        except AttributeError:
            pass
        params = {}
        for param in body.split("&"):
            key, val = param.split("=", 1)
            val = unquote_plus(val)
            try:
                val = json.loads(val)
            except json.JSONDecodeError:
                pass
            params.setdefault(key, []).append(val)
        return params

    def records(self):
        """Gets a mapping of all records in the result set by IRN

        Returns
        -------
        dict
            dict that maps irns to records
        """
        return {r["irn"]: r for r in self}

    def first(self):
        """Gets the first record from the result set

        Returns
        -------
        dict
            the first record. If a rec_class is specified, the record will use that
            class.
        """
        for rec in iter(self):
            return rec

    def next_page(self):
        """Gets the next pages of results in the result set

        Returns
        -------
        EMuAPIResponse
            the result from the next page
        """
        limit = int(self.params.get("limit", [10])[0])
        if len(self) != limit:
            raise ValueError(f"Last page (num_results={len(self)}, limit={limit})")
        return self._api.get(
            self.url,
            data=self.request.body,
            headers={"Next-Search": self.headers["Next-Search"]},
        )

    def _get(self, rec, resp=None):
        """Reads and parses a single record from a response"""
        if resp is None:
            resp = self
        key = rec["irn"]
        try:
            return self.__class__.cache[key]
        except KeyError:
            if resp._api.parser is not None:
                rec = self._api.parser.parse(
                    rec, module=resp.module, select=resp._select
                )
            self.__class__.cache[key] = rec
            return rec


class EMuAPIParser:
    """Parses responses from the EMu API"""

    def __init__(self, rec_class=dict):
        self._rec_class = rec_class

    def parse(self, rec: dict, module: str, select: list | dict[dict] = None):
        """Parses a record returned by the EMu API

        Only attachments mapped in the original select parameter are resolved.

        Parameters
        ----------
        rec : dict
            a record retrieved from the EMu API
        module : str
            the backend name of the EMu module
        select : list | dict
            the fields to return

        Returns
        -------
        dict
            the record with all attachments resolved
        """
        parsed = _parse_api(rec, module)
        if self._rec_class != dict:
            parsed = self._rec_class(parsed, module=module)
        if select:
            parsed = self.resolve_attachments(parsed, select=select)
        return parsed

    def resolve_attachments(self, rec: dict, select: list | dict[dict] = None):
        """Resolves attachments in a record returned by the EMu API

        Only attachments mapped in the select parameter are resolved.

        Parameters
        ----------
        rec : dict
            a record returned by the API
        select : list | dict

        Returns
        -------
        dict
            the record with all attachments resolved
        """
        for key, val in rec.items():
            if is_ref(key):
                field_info = self._api.schema.get_field_info(self.module, key)
                try:
                    select_ = select[key]
                except (KeyError, TypeError):
                    pass
                else:
                    if isinstance(val, list):
                        for i, val in enumerate(val):
                            if val:
                                val = self._api.retrieve(
                                    field_info["RefTable"], val, select=select_
                                ).first()
                            rec[key][i] = val
                    else:
                        rec[key] = self._api.retrieve(
                            field_info["RefTable"], val, select=select_
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
    """Builds a clause to match fields containing a value

    Equivalent to the basic, text-only search in the EMu client.

    Paramters
    ---------
    val : str | list[str]
        the text to search for or a list of such strings

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
    """Builds a clause to match the complete contents of a column exactly

    Equivalent to \\^\\"hello world\\"\\$ in the EMu client. Case insensitive.

    Paramters
    ---------
    val : str | float | int | list[str] | list[float] | list[int]
        the value or list of values to match
    mode : str
        one of date, time, latitude, or longitude. If omitted, will try to guess
        based on the column or value.

    Returns
    -------
    dict
        an EMu API exact clause
    """
    if mode is None:
        mode = _infer_mode(val)
    return _build_clause(val, col=col, op="exact", mode=mode)


def exists(val: bool, col: str = None) -> dict:
    """Builds a clause to test whether a field is populated

    Equivalent to \\* \\+ in the EMu client if True. Equivalent to \\!\\* or \\!\\+
    if False or None.

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
    """Builds a clause to perform a phonetic search

    Equivalent to \\@smythe in the EMu client.

    Paramters
    ---------
    val : str | list[str]
        the text to search for or a list of such strings

    Returns
    -------
    dict
        an EMu API phonetic clause
    """
    return _build_multivalue_clause(val, col=col, op="phonetic")


def phrase(val: str | list[str], col: str = None) -> dict:
    """Builds a clause to search for a phrase

    Equvalent to \\"the black cat\"" in the EMu client.

    Paramters
    ---------
    val : str | list[str]
        a multiword phrase or a list of such phrases

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
        an EMu API phrase clause
    """
    return _build_clause(val, col=col, op="proximity", distance=distance)


def regex(val: str | list[str], col: str = None) -> dict:
    """Builds a clause to perform a regular expression search

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
    """Builds a clause to search for words matching the same root

    Equivalent to \\~locate in the EMu client

    Paramters
    ---------
    val : str | list[str]
        the root word to search for. For example, elect would match election,
        elected, electioneering, elects but would not match electricity

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
    """Escapes a string according to EMu escape syntax

    For example, the regular expression ^Hello world$ will be escaped as
    \\^Hello world\\$.

    Paramters
    ---------
    val : str
        the text to escape

    Returns
    -------
    str
        the escaped text
    """
    for item in ['"', "'", "!", "[", "]", "^", "$", "*", "+", "~", "@", "=", "=="]:
        val = val.replace(item, rf"\{item}")
    val = val.replace(r"=\=", "==")
    return val


def emu_unescape(val: str) -> str:
    """Unescapes a string that uses the EMu escape syntax

    For example, the regular expression \\^Hello world\\$ will be escaped as
    ^Hello world$.

    Paramters
    ---------
    val : str
        the text to unescape

    Returns
    -------
    str
        the unescaped text
    """
    for item in ['"', "'", "!", "[", "]", "^", "$", "*", "+", "~", "@", "=", "=="]:
        val = val.replace(rf"\{item}", item)
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
    logger.debug(f"Prepped select as {repr(param)}")
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
    logger.debug(f"Prepped sort as {repr(param)}")
    return param


def _prep_filter(module: str, filter_: dict, use_emu_syntax: bool = True) -> str:
    """Expands a simple filter to the format used by the EMu API"""
    stmts = []
    for col, val in filter_.items():
        # Add column name to individual clauses if not already there
        if isinstance(val, dict):
            for key in list(val):
                vals = val[key]
                if key in ("AND", "OR", "NOT"):
                    val[key] = [{_prep_field(col): v} for v in vals]
                else:
                    val[_prep_field(col)] = {key: vals}
                    del val[key]

        else:
            # Infer operator based on data type in the schema if provided
            if EMuAPI.schema:
                data_type = EMuAPI.schema.get_field_info(module, col)["DataType"]
                val = _val_to_query(
                    col, val, use_emu_syntax=use_emu_syntax, data_type=data_type
                )

            # Otherwise base the clause on the type of data supplied
            elif val is None:
                val = exists(False, col=col)
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

    # Filter must include a boolean operator even if there is only one element
    param = json.dumps(and_(stmts))
    logger.debug(f"Prepped filter as {repr(param)}")
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
                clause = or_(vals)
                logger.debug(f"Built range clause: {clause}")
                return clause

            elif gt:
                for gt in gt:
                    kwargs[gt_key] = gt
                    vals.append(_build_clause(None, op, col=col, **kwargs))
                clause = or_(vals)
                logger.debug(f"Built range clause: {clause}")
                return clause

            elif lt:
                for lt in lt:
                    kwargs[lt_key] = lt
                    vals.append(_build_clause(None, op, col=col, **kwargs))
                clause = or_(vals)
                logger.debug(f"Built range clause: {clause}")
                return clause

        else:
            clause = {"range": kwargs} if col is None else {col: {"range": kwargs}}
            logger.debug(f"Built range clause: {clause}")
            return clause

    elif isinstance(val, (list, tuple)):
        if len(val) > 1:
            return or_([_build_clause(v, op, col=col, **kwargs) for v in val])
        val = val[0]

    if op != "order":
        val = {"value": val}
        val.update(kwargs)
    clause = {op: val} if col is None else {col: {op: val}}
    logger.debug(f"Built {op} clause: {clause}")
    return clause


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

    # Map booleans and None using exists
    if isinstance(val, bool) or val is None:
        return exists(bool(val), col=col)

    # Coerce to numeric type if a data_type hint is numeric
    to_type = {"Float": float, "Integer": int}.get(data_type, str)

    # Simple numeric values can be returned with exact
    if not data_type or to_type in (float, int):
        if isinstance(val, (float, int)):
            return exact(val, col=col)
        try:
            return exact(to_type(val), col=col)
        except ValueError:
            pass

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

    # Search for empty fields (null search)
    chars = ["!*", "!+"]
    if use_emu_syntax:
        chars = [emu_escape(n) for n in chars]
    pattern = r"(^|\b)(" + "|".join([re.escape(n) for n in chars]) + r")(\b|$)"
    if re.search(pattern, val):
        clauses.append(exists(False, col=col))
    val = re.sub(pattern, "", val).strip()
    if not val:
        return and_(clauses) if len(clauses) > 1 else clauses[0]

    # Search for populated fields
    chars = ["*", "+"]
    if use_emu_syntax:
        chars = [emu_escape(n) for n in chars]
    pattern = r"(^|\b)(" + "|".join([re.escape(n) for n in chars]) + r")(\b|$)"
    if re.search(pattern, val):
        clauses.append(exists(True, col=col))
    val = re.sub(pattern, "", val).strip()
    if not val:
        return and_(clauses) if len(clauses) > 1 else clauses[0]

    # Search by stem
    chars = ["~"]
    if use_emu_syntax:
        chars = [emu_escape(n) for n in chars]
    pattern = r"(^|\b)(" + "|".join([re.escape(n) for n in chars]) + r")([-\w]+)"
    match = re.search(pattern, val)
    if match:
        clauses.append(stemmed(match.group(3), col=col))
    val = re.sub(pattern, "", val).strip()
    if not val:
        return and_(clauses) if len(clauses) > 1 else clauses[0]

    # Search phonetically
    chars = ["@"]
    if use_emu_syntax:
        chars = [emu_escape(n) for n in chars]
    pattern = r"(^|\b)(" + "|".join([re.escape(n) for n in chars]) + r")([-\w]+)"
    match = re.search(pattern, val)
    if match:
        clauses.append(phonetic(match.group(3), col=col))
    val = re.sub(pattern, "", val).strip()
    if not val:
        return and_(clauses) if len(clauses) > 1 else clauses[0]

    # Search case- and diacritic-sensitively
    chars = ["=", "=="]
    if use_emu_syntax:
        chars = [emu_escape(n) for n in chars]
    pattern = r"(^|\b)(" + "|".join([re.escape(n) for n in chars]) + r")([-\w]+)"
    if re.search(pattern, val):
        raise ValueError(
            "Case- and diacritic-sensitive searches are not supported by the API"
        )

    # Phrases
    if use_emu_syntax:
        pattern = rf"{ops}?(\\'(?:.*?)\\'|\\\"(?:.*?)\\\")"
    else:
        pattern = rf"{ops}?('(?:.*?)'|\"(?:.*?)\")"
    for op, val_ in re.findall(pattern, val):
        clause = phrase(val_.strip("\"'\\"), col=col)
        clauses.append(clause if op.lstrip("\\") != "!" else not_(clause))
    val = re.sub(pattern, "", val).strip()
    if not val:
        return and_(clauses) if len(clauses) > 1 else clauses[0]

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


def _parse_api(val, module, key=None, mapped=None):
    """Parses API response to remove field groupings"""

    if mapped is None:
        mapped = {}

    if key:
        try:
            key = EMuAPI.schema.map_short_name(module, key)
        except KeyError:
            pass

    # Iterate dicts
    if isinstance(val, dict):
        for key, val in val.items():
            _parse_api(val, module, key, mapped)

    # Map tables. Groups are based on definitions in the schema.
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
                _parse_api(vals, module, key, mapped)

    # Map nested tables
    elif key.endswith("_subgrp"):
        keys = []
        for row in val:
            if row:
                for inner_row in row:
                    keys.extend(inner_row)
        keys = set(keys)

        grid = {k: [] for k in keys}
        for row in val:
            for val in grid.values():
                val.append([])
            if row:
                for inner_row in row:
                    for key in keys:
                        grid[key][-1].append(inner_row.get(key))

        for key, vals in grid.items():
            if any(vals):
                _parse_api(vals, module, key, mapped)

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
