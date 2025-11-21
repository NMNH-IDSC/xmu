"""Defines tools to work with the EMu API"""

import json
import logging
import re
import time
import tomllib
from functools import cache, cached_property
from pathlib import Path
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
    url : str, optional
        the url for the EMu API, including tenant
    username : str, optional
        an EMu username. If omitted, defaults to the current OS username.
    password : str, optional
        the password for the given username, If omitted, the user will be
        prompted for the password when the class is initiated.
    autopage : bool = True
        whether to automatically page through results if the total number of results
        exceeds the limit of a given request
    config_path : str | Path
        path to a TOML config file used to set url, username, password, and autopage
    parser : EMuAPIParser, optional
        the parser object used to parse individual records. The default EMuAPIParser
        class returns a close approximation of the format used by EMuRecord. If None,
        records will be returned as formatted by the API.

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
        url: str = None,
        username: str = None,
        password: str = None,
        autopage: bool = None,
        config_path: str | Path = "emuapi.toml",
        parser: "EMuAPIParser" = None,
    ):
        self.config_path = config_path
        try:
            with open(self.config_path, "rb") as f:
                config = tomllib.load(f)["params"]
        except FileNotFoundError:
            self.base_url = url.rstrip("/") + "/"
        else:
            if not url:
                url = config["url"]
            if not username:
                username = config["username"]
            if not password:
                password = config["password"]
            if autopage is None:
                autopage = config["autopage"]

        self.base_url = url.rstrip("/") + "/"
        self.use_emu_syntax = True

        # Parse must be assigned when the instance is created
        self._parser = None
        self.parser = parser

        # The autopage parameter is passed to EMuAPIResponse but it is cleaner
        # to implement it here
        self.autopage = autopage

        # Get token
        self._token = None
        self.get_token(username=username, password=password)

        self._session = None

    @property
    def parser(self):
        """The parser object used to parse records returned by the API"""
        return self._parser

    @parser.setter
    def parser(self, val):
        self._parser = val
        if val:
            self._parser.api = self

    @property
    def session(self):
        """The session object to use for API queries"""
        if self._session is None:
            self._session = requests.Session()
        return self._session

    @session.setter
    def session(self, val):
        self._session = val

    def get_token(self, refresh=False, **kwargs):
        """Retrieves a token from the server to authorize requests

        Parameters
        ----------
        kwargs :
            username and password if no config file is found

        Returns
        -------
        str
            the authorization token need to make API requests
        """
        # Token requests sometimes fail, particularly if several are done quickly.
        # To prevent this, the token is cached to a file in the working directory when
        # it is read.
        if refresh:
            print("Refreshing token")
        else:
            try:
                with open("token") as f:
                    self._token = f.read().strip()
                return self._token
            except FileNotFoundError:
                pass

        if not kwargs:
            with open(self.config_path, "rb") as f:
                kwargs = tomllib.load(f)["params"]

        resp = requests.post(
            urljoin(self.base_url, "tokens"),
            json={
                "username": kwargs["username"],
                "password": kwargs["password"],
            },
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT,
        )

        try:
            self._token = resp.headers["Authorization"]
        except KeyError:
            raise ValueError(
                f"Token request failed: {resp.url} (status_code={resp.status_code})"
            )
        else:
            with open("token", "w") as f:
                f.write(self._token)
            return self._token

    def get(self, *args, select=None, **kwargs):
        """Performs a GET operation with the proper authorization header

        Most requests should use either retrieve or search instead of calling this
        method directly.

        Parameters
        ----------
        args:
            Any arg accepted by request.get()
        select : list[str] | dict[dict], optional
            A container with fields to include in the returned records. Fields from
            other modules can be included using a dict formatted as follows:
            {
                "EMuField": None,
                "EMuFieldRef": {
                    "EMuFieldInAnotherModule": None,
                }
            }

        kwargs:
            Any kwarg accepted by request.get(). By default, the headers kwarg
            includes {"Prefer": "representation=none", "X-HTTP-Method-Override" = "GET",
            "Content-Type": "application/x-www-form-urlencoded"}. The latter two keys
            are used to implement the HTTP method override recommended by Axiell.

        Returns
        -------
        EMuAPIResponse
            the response returned for the request
        """
        headers = kwargs.setdefault("headers", {})
        headers["Authorization"] = f"{self._token}"
        headers.setdefault("Prefer", "representation=none")

        # Add the HTTP method override per recommendation at
        # https://help.emu.axiell.com/emurestapi/3.1.2/05-Appendices-Override.html
        headers["X-HTTP-Method-Override"] = "GET"
        headers["Content-Type"] = "application/x-www-form-urlencoded"

        # Redact authorization before logging
        redacted = re.sub(
            "'Authorization': '.*?'", "'Authorization': '[REDACTED]'", str(kwargs)
        )

        logger.debug(f"Making GET request: {args[0]} (params={redacted})")
        resp = EMuAPIResponse(
            self.session.post(*args, **kwargs),
            api=self,
            select=select,
        )
        if resp.status_code == 401:
            self.get_token(refresh=True)
            return self.get(*args, select=select, **kwargs)
        return resp

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
        # Split irn from API reference notation (emu:{server}/{module}/{irn}))
        if irn.startswith("emu:"):
            irn = irn.split("/")[-1]
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
        cursor_type: str, default="server"
            whether the cursor is stored locally or on the server

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
            cursorType=cursor_type,
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
        else:
            params["filter"] = {}

        for key in ("limit", "cursorType"):
            if kwargs.get(key):
                params[key] = kwargs[key]

        return params


class EMuAPIResponse:
    """Wraps a response from the EMu API response"""

    def __init__(
        self,
        response: requests.Response,
        api: EMuAPI,
        select: list[str] | dict[dict] = None,
    ):
        self.api = api
        self.select = select
        self.resolve_attachments = True
        self._response = response
        self._json = None
        self._cached = []

    def __getattr__(self, attr):
        try:
            return getattr(self._response, attr)
        except AttributeError:
            raise AttributeError(
                f"{repr(self.__class__.__name__)} object has no attribute {repr(attr)}"
            )

    def __len__(self):
        return len(json.loads(self.headers["Next-Offsets"]))

    def __iter__(self):

        if self._cached:
            for rec in self._cached:
                yield rec

        else:
            try:
                rec = self.json()["data"]
                if self.api.parser is not None:
                    rec = self.api.parser.parse(self.module, rec, select=self.select)
                elif self.resolve_attachments:
                    # Resolving attachments individually is slow, so attachments
                    # are deferred until a number of records have been processed
                    # OR the user tries to access a key
                    for key, val in rec.items():
                        if is_ref(key):
                            try:
                                select = self.select[key]
                            except (KeyError, TypeError):
                                select = self.select
                            if isinstance(val, (list, tuple)):
                                rec[key] = [
                                    attachment(
                                        val_, self.api, select=json.dumps(select)
                                    )
                                    for val_ in val
                                ]
                            else:
                                rec[key] = attachment(
                                    val, self.api, select=json.dumps(select)
                                )
                self._cached.append(rec)
                yield rec
            except KeyError:
                resp = self
                count = 0
                while True:
                    try:
                        # Return records in batches to make resolving attachments more
                        # efficient
                        records = []
                        for match in resp.json()["matches"]:
                            rec = match["data"]
                            if resp.api.parser is not None:
                                rec = resp.api.parser.parse(
                                    self.module, rec, select=resp.select
                                )
                            elif self.resolve_attachments:
                                # Resolving attachments individually is slow, so
                                # attachments are deferred until a number of records
                                # have been processed OR the user tries to access a key
                                for key, val in rec.items():
                                    if is_ref(key):
                                        try:
                                            select = resp.select[key]
                                        except (KeyError, TypeError):
                                            select = resp.select
                                        if isinstance(val, (list, tuple)):
                                            rec[key] = [
                                                attachment(
                                                    val_,
                                                    self.api,
                                                    select=json.dumps(select),
                                                )
                                                for val_ in val
                                            ]
                                        else:
                                            rec[key] = attachment(
                                                val, self.api, select=json.dumps(select)
                                            )
                            self._cached.append(rec)
                            records.append(rec)
                            if len(records) >= 1000:
                                for rec in records:
                                    count += 1
                                    yield rec
                                records = []
                        del match  # delete match so that exceptions work as expected
                        for rec in records:
                            count += 1
                            yield rec
                    except Exception as exc:
                        try:
                            raise ValueError(
                                f"Could not parse match: {match} from {repr(resp.text)}"
                            ) from exc
                        except NameError:
                            raise ValueError(
                                f"No records found: {repr(resp.text)} ({resp.request.url}, {resp.params})"
                            ) from exc
                    else:
                        # Get the next page
                        if resp.api.autopage and count < resp.hits:
                            try:
                                resp = resp.next_page()
                            except ValueError:
                                break
                            else:
                                if hasattr(resp, "from_cache") and resp.from_cache:
                                    logger.debug("Response is from cache")
                                else:
                                    logger.debug("Response is from server")
                        else:
                            break

    @cached_property
    def module(self):
        """The EMu module queried to create the response"""
        try:
            return self.json()["id"].split("/")[-2]
        except KeyError:
            return self.json()["matches"][0]["id"].split("/")[-2]

    @cached_property
    def params(self):
        """The query parameters used to make the request"""
        body = self.request.body
        if not body:
            return {}
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
            params[key] = val
        return params

    @cached_property
    def hits(self):
        try:
            return self.json()["hits"]
        except KeyError:
            return 0

    def json(self):
        """Parse JSON from response"""
        if self._json is None:
            try:
                self._json = self._response.json()
            except json.JSONDecodeError:
                raise ValueError(
                    f"Response cannot be decoded: {repr(self.text)} (status_code={self.status_code})"
                )
            else:
                if "@error" in self._json:
                    raise ValueError(f"Error: {self._json}")
        return self._json

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
        for rec in self:
            return rec

    def next_page(self):
        """Gets the next pages of results in the result set

        Returns
        -------
        EMuAPIResponse
            the result from the next page
        """
        try:
            resp = self.api.get(
                self.url,
                data=self.request.body,
                headers={"Next-Search": self.headers["Next-Search"]},
            )
        except KeyError:
            raise ValueError("Next-Search not found in headers")
        return resp


class EMuAPIParser:
    """Parses responses from the EMu API"""

    def __init__(self):
        self.rec_class = dict
        self.api = None

    def parse(self, module: str, rec: dict, select: list | dict[dict] = None):
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
        parsed = _parse_api(module, rec, self.api, select=select)
        if self.rec_class != dict:
            parsed = self.rec_class(parsed, module=module)
        return parsed


class DeferredAttachment:
    """An attached record defined by a module and IRN

    The record itself is loaded when (1) a key is accessed or (2) it is loaded
    manually using the resolve() method. Should be called by the attachment()
    function to allow caching.

    Parameters
    ----------
    val : str
        the EMu attachment string
    api : EMuAPI
        the instance of the EMu API that created the parent record
    select : list | dict
        the fields to retrieve. If omitted, all fields are returned.

    Attributes
    ----------
    verbatim : str
        the EMu attachment string
    module : str
        the backend name of the EMu module
    irn : int
        the IRN of the attached record
    select : list | dict
        the fields to retrieve
    """

    _deferred = {}

    def __init__(self, val, api, select=None):
        self.verbatim = val
        self.module, self.irn = val.split("/")[-2:]
        self.irn = int(self.irn)
        self.select = select
        try:
            key = tuple(sorted(select))
        except TypeError:
            key = select
        self.__class__._deferred.setdefault((self.module, key), {})[self.irn] = self
        self._data = None
        self.api = api

    def __str__(self):
        return f"DeferredAttachment({self._data if self._data else self.verbatim})"

    def __repr__(self):
        return str(self)

    def __int__(self):
        return self.irn

    def __getattr__(self, attr):
        try:
            return getattr(self.data, attr)
        except AttributeError:
            raise AttributeError(
                f"{repr(self.__class__.__name__)} object has no attribute {repr(attr)}"
            )

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, val):
        self.data[key] = val

    @property
    def data(self):
        """The EMu record for the given IRN and select statement"""
        if self._data is None:
            self.resolve()
        return self._data

    def get(self, key, default=None):
        return self.data.get(key, default)

    def items(self):
        return self.data.items()

    def resolve(self):
        """Resolves all deferred records with the same IRN and select statement

        Returns
        -------
        DeferredAttachment
            attachment with data attribute populated
        """
        if not self._data:
            try:
                key = tuple(sorted(self.select))
            except TypeError:
                key = self.select
            deferred = self.__class__._deferred.pop((self.module, key))
            records = self.api.search(
                module=self.module,
                select=self.select,
                filter_={"irn": list(deferred)},
                limit=len(deferred),
            ).records()

            # Convert IRN to integer if records have not been parsed to do so already
            try:
                records = {int(k.split("/")[-1]): v for k, v in records.items()}
            except AttributeError:
                pass

            for irn, rec in deferred.items():
                rec._data = records[irn]

        return self


@cache
def attachment(val, api, select=None):
    """Creates a DeferredAttachment for the given value

    This is the preferred way to create a DeferredAttachment.

    Parameters
    ----------
    val : str
        the EMu attachment string
    api : EMuAPI
        the instance of the EMu API that created the parent record
    select : str
        a JSON-encoded string of the fields to retrieve. If omitted, all fields are
        returned.

    Returns
    -------
    DeferredAttachment

    """
    try:
        return DeferredAttachment(val, api, select=json.loads(select))
    except AttributeError:
        # Some ref fields are not actually attachments
        if not isinstance(val, str) or not val.startswith("emu:"):
            return val
        raise


def attach(obj: dict, api: EMuAPI, select: list | dict, module: str = None):
    """Recursively turns reference fields to attachments that resolve when accessed

    Parameters
    ----------
    obj : dict
        an EMu record returned by the API
    api : EMuAPI
        the instance of the EMu API that created the record
    select : list | dict
        the fields to retrieve. If omitted, all fields are returned.
    module : str

    Returns
    -------
    dict
        the record with top-level references converted to DeferredAttachments
    """
    if isinstance(obj, (dict, DeferredAttachment)):
        for key, val in obj.items():
            if key == "irn":
                obj[key] = val
                module = val.split("/")[-2]
            else:
                if (
                    select
                    and not key.endswith(("_grp", "_subgrp"))
                    and isinstance(select, dict)
                ):
                    select = select.get(EMuAPI.schema.map_short_name(module, key))
                obj[key] = attach(val, api, select, module)
    elif isinstance(obj, (list, tuple)):
        return [attach(v, api, select, module) for v in obj]
    elif isinstance(obj, str) and re.match(r"^emu.*\d$", obj):
        return attachment(obj, api, json.dumps(select))
    return obj


def and_(conds: list[dict]) -> dict:
    """Combines a list of conditions with AND

    Parameters
    ----------
    conds : list[dict]
        list of conditions

    Returns
    -------
    dict
        {"AND": conds}
    """
    return {"AND": conds}


def or_(conds: list[dict]) -> dict:
    """Combines a list of conditions with OR

    Parameters
    ----------
    conds : list[dict]
        list of conditions

    Returns
    -------
    dict
        {"OR": conds}
    """
    return {"OR": conds}


def not_(conds: dict) -> dict:
    """Negates a condition

    Parameters
    ----------
    conds : list[dict] | dict
        list of conditions

    Returns
    -------
    dict
        {"NOT": conds}
    """
    if not isinstance(conds, (list, tuple)):
        conds = [conds]
    return {"NOT": conds}


def contains(val: str | list[str], col: str = None) -> dict:
    """Builds a condition to match fields containing a value

    Equivalent to the basic, text-only search in the EMu client.

    Paramters
    ---------
    val : str | list[str]
        the text to search for or a list of such strings
    col : str
        the name of the column. Typically ommitted.

    Returns
    -------
    dict
        an EMu API contains condition
    """
    return _build_multivalue_cond(val, col=col, op="contains")


def range_(
    gt: str | int | float = None,
    lt: str | int | float = None,
    gte: str | int | float = None,
    lte: str | int | float = None,
    mode: str = None,
    col: str = None,
) -> dict:
    """Builds a condition to match a range of values

    At least one of gt, lt, gte, and lte must be provided. Only one of gt and gte
    can be provided, and only one of lt and lte can be provided.

    Parameters
    ----------
    gt: str | float | int
        the lower bound of the search, not inclusive
    lt: str | float | int
        the upper bound of the search, not inclusive
    gte: str | float | int
        the lower bound of the search, inclusive
    lte: str | float | int
        the upper bound of the search, inclusive
    mode : str
        one of date, time, latitude, or longitude. If omitted, will try to guess
        based on the column or value.
    col : str
        the name of the column. Typically ommitted.

    Returns
    -------
    dict
        an EMu API range condition
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
    return _build_cond(None, col=col, op="range", **op)


def gt(val: str | int | float, mode: str = None, col: str = None):
    """Builds a condition to match values greater than a given value

    This is a helper function based on range_().

    Paramters
    ---------
    val: str | float | int
        the lower bound of the search, not inclusive
    mode : str
        one of date, time, latitude, or longitude. If omitted, will try to guess
        based on the column or value.
    col : str
        the name of the column. Typically ommitted.

    Returns
    -------
    dict
        an EMu API range condition
    """
    return range_(gt=val, mode=mode, col=col)


def gte(val: str | int | float, mode: str = None, col: str = None):
    """Builds a condition to match values greater than or equal to a given value

    This is a helper function based on range_().

    Paramters
    ---------
    val: str | float | int
        the lower bound of the search, inclusive
    mode : str
        one of date, time, latitude, or longitude. If omitted, will try to guess
        based on the column or value.
    col : str
        the name of the column. Typically ommitted.

    Returns
    -------
    dict
        an EMu API range condition
    """
    return range_(gte=val, mode=mode, col=col)


def lt(val: str | int | float, mode: str = None, col: str = None):
    """Builds a condition to match values less than a given value

    This is a helper function based on range_().

    Paramters
    ---------
    val: str | float | int
        the upper bound of the search, not inclusive
    mode : str
        one of date, time, latitude, or longitude. If omitted, will try to guess
        based on the column or value.
    col : str
        the name of the column. Typically ommitted.

    Returns
    -------
    dict
        an EMu API range condition
    """
    return range_(lt=val, mode=mode, col=col)


def lte(val: str | int | float, mode: str = None, col: str = None):
    """Builds a condition to match values less than or equal to a given value

    This is a helper function based on range_().

    Paramters
    ---------
    val: str | float | int
        the upper bound of the search, inclusive
    mode : str
        one of date, time, latitude, or longitude. If omitted, will try to guess
        based on the column or value.
    col : str
        the name of the column. Typically ommitted.

    Returns
    -------
    dict
        an EMu API range condition
    """
    return range_(lte=val, mode=mode, col=col)


def exact(val: str | float | int, col: str = None, mode: str = None) -> dict:
    """Builds a condition to match the complete contents of a column exactly

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
        an EMu API exact condition
    """
    if mode is None:
        mode = _infer_mode(val)
    return _build_cond(val, col=col, op="exact", mode=mode)


def exists(val: bool, col: str = None) -> dict:
    """Builds a condition to test whether a field is populated

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
        an EMu API exists condition
    """
    return _build_cond(val, col=col, op="exists")


def phonetic(val: str | list[str], col: str = None) -> dict:
    """Builds a condition to perform a phonetic search

    Equivalent to \\@smythe in the EMu client.

    Paramters
    ---------
    val : str | list[str]
        the text to search for or a list of such strings

    Returns
    -------
    dict
        an EMu API phonetic condition
    """
    return _build_multivalue_cond(val, col=col, op="phonetic")


def phrase(val: str | list[str], col: str = None) -> dict:
    """Builds a condition to search for a phrase

    Equvalent to \\"the black cat\"" in the EMu client.

    Paramters
    ---------
    val : str | list[str]
        a multiword phrase or a list of such phrases

    Returns
    -------
    dict
        an EMu API phrase condition
    """
    return _build_cond(val, col=col, op="phrase")


def proximity(val: str | list[str], col: str = None, distance: int = 3) -> dict:
    """Builds a condition to search for words within a certain distance of each other

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
        an EMu API phrase condition
    """
    raise NotImplementedError("Condition does not work as expected in API or client")
    # return _build_cond(val, col=col, op="proximity", distance=distance)


def regex(val: str | list[str], col: str = None) -> dict:
    """Builds a condition to perform a regular expression search

    Paramters
    ---------
    val : str | list[str]
        the pattern to search for

    Returns
    -------
    dict
        an EMu API regex condition
    """
    return _build_cond(val, col=col, op="regex")


def stemmed(val: str | list[str], col: str = None) -> dict:
    """Builds a condition to search for words matching the same root

    Equivalent to \\~locate in the EMu client

    Paramters
    ---------
    val : str | list[str]
        the root word to search for. For example, elect would match election,
        elected, electioneering, elects but would not match electricity

    Returns
    -------
    dict
        an EMu API stemmed condition
    """
    return _build_multivalue_cond(val, col=col, op="stemmed")


def is_not_null(col: str = None) -> dict:
    """Builds a condition that matches a non-empty field in the EMu API

    Alias for exists(True).

    Returns
    -------
    dict
        an EMu API exists=True condition
    """
    return exists(True, col=col)


def is_null(col: str = None) -> dict:
    """Builds a condition that matches an empty field in the EMu API

    Returns
    -------
    dict
        an EMu API exists=False condition
    """
    return exists(False, col=col)


def order(val: str = "asc", col: str = None) -> dict:
    """Builds a condition to sort in the given direction

    Paramters
    ---------
    val : str
        sort direction. Must be either "asc" or "desc".

    Returns
    -------
    dict
        an EMu API order condition
    """
    return _build_cond(val, col=col, op="order")


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
    val = val.replace(r">\=", ">=")
    val = val.replace(r"<\=", "<=")
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
    if isinstance(val, (list, tuple)):
        return all((isinstance(_, obj) for _ in val))
    return isinstance(val, obj)


def _type(val: Any) -> type:
    if isinstance(val, (list, tuple)):
        types = [type(_) for _ in val]
        if len(set(types)) != 1:
            raise ValueError(f"Object contains different types: {val}")
        return types[0]
    return type(val)


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
    if isinstance(sort_, (list, tuple)):
        sort_ = {c: "asc" for c in sort_}
    conds = []
    for col, val in sort_.items():
        if not isinstance(val, dict):
            val = order(val, col=col)
        conds.append(val)
    param = json.dumps(conds)
    logger.debug(f"Prepped sort as {repr(param)}")
    return param


def _prep_filter(module: str, filter_: dict, use_emu_syntax: bool = True) -> str:
    """Expands a simple filter to the format used by the EMu API"""
    stmts = []
    for col, val in filter_.items():
        # Add column name to individual conditions if not already there
        if isinstance(val, dict):
            for key in list(val):
                vals = val[key]
                if key in ("AND", "OR"):
                    val[key] = [{_prep_field(col): v} for v in vals]
                elif key == "NOT":
                    val[key] = [_val_to_query(col, v) for v in vals]
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

            # Otherwise base the condition on the type of data supplied
            elif val is None:
                val = exists(False, col=col)
            elif _isinstance(val, bool):
                val = exists(val, col=col)
            elif _isinstance(val, (float, int)):
                val = exact(val, col=col)
            else:
                val = _val_to_query(col, val, use_emu_syntax=use_emu_syntax)

        val = val.get("AND", val)
        if not isinstance(val, (list, tuple)):
            val = [val]
        if len(val) > 1:
            stmts.append(and_(val))
        else:
            stmts.append(val[0])

    # Filter must include a boolean operator even if there is only one element
    if len(stmts) == 1 and list(stmts[0])[0] in ("AND", "OR", "NOT"):
        param = json.dumps(stmts[0])
    else:
        param = json.dumps(and_(stmts))
    logger.debug(f"Prepped filter as {repr(param)}")
    return param


def _build_cond(val: Any, op: str, col: str = None, **kwargs) -> dict:
    """Helper function to build a condition for the EMu API"""

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

        # Complex ranges must be lists, so coerce if needed
        if gt_key and lt_key and _type(gt) != _type(lt):
            raise ValueError(f"{gt_key} and {lt_key} must have the same type")

        # Simplify lists that only include one value
        if isinstance(gt, (list, tuple)) and len(gt) == 1:
            gt = gt[0]
        if isinstance(lt, (list, tuple)) and len(lt) == 1:
            lt = lt[0]

        if isinstance(gt, (list, tuple)) or isinstance(lt, (list, tuple)):

            if gt and lt:
                # If both gt and lt are defined, they must have the same length
                if isinstance(gt, (list, tuple)) and len(gt) != len(lt):
                    raise ValueError(
                        f"{gt_key} and {lt_key} must have the same number of items"
                    )
                vals = []
                for gt, lt in zip(gt, lt):
                    kwargs[gt_key] = gt
                    kwargs[lt_key] = lt
                    vals.append(_build_cond(None, op, col=col, **kwargs))
                cond = or_(vals) if len(vals) > 1 else cond
                logger.debug(f"Built range condition: {cond}")
                return cond

        else:
            cond = {"range": kwargs} if col is None else {col: {"range": kwargs}}
            logger.debug(f"Built range condition: {cond}")
            return cond

    elif isinstance(val, (list, tuple)):
        if len(val) > 1:
            return or_([_build_cond(v, op, col=col, **kwargs) for v in val])
        val = val[0]

    if op != "order":
        val = {"value": val}
        val.update(kwargs)
    cond = {op: val} if col is None else {col: {op: val}}
    logger.debug(f"Built {op} condition: {cond}")
    return cond


def _build_multivalue_cond(val: Any, op: str, col: str = None):
    """Builds conditions for operations that should be split by word"""
    conds = []
    for val in [val] if isinstance(val, str) else val:
        cond = _build_cond(val.split(" "), col=col, op=op)
        try:
            cond = {"AND": cond.pop("OR")}
        except KeyError:
            pass
        conds.append(cond)
    return or_(conds) if len(conds) > 1 else conds[0]


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
    use_emu_syntax : bool = True
        whether the value uses EMu escape syntax
    data_type : str = None
        the EMu data type. Used to ensure that range searches use the correct
        data type.

    Returns
    -------
    dict
        a query corresponding to the given value
    """
    # FIXME: Implement regex

    # Map already defined conditions to the supplied column name
    if isinstance(val, dict):
        return {_prep_field(col): val}

    # Process multiple values
    if isinstance(val, (list, tuple)):
        if len(val) > 1:
            return or_([_val_to_query(col, v, use_emu_syntax, data_type) for v in val])
        else:
            val = val[0]

    # Map booleans and None using exists
    if isinstance(val, bool) or val is None:
        return exists(bool(val), col=col)

    # Coerce to numeric type if data_type hint is numeric
    to_type = {"Float": float, "Integer": int}.get(data_type, str)

    # Simple numeric values can be returned with exact
    if not data_type and isinstance(val, (float, int)):
        return exact(val, col=col)
    elif to_type in (float, int):
        try:
            return exact(to_type(val), col=col)
        except ValueError:
            # Null searches, etc. are valid but non-numeric
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

    # The mode argument controls how the exact and range conditions handle comparisons
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

    conds = []

    # Search for empty fields (null search)
    chars = ["!*", "!+"]
    if use_emu_syntax:
        chars = [emu_escape(n) for n in chars]
    pattern = r"(^|\b|\s)(" + "|".join([re.escape(n) for n in chars]) + r")(\b|\s|$)"
    if re.search(pattern, val):
        conds.append(exists(False, col=col))
    val = re.sub(pattern, "", val).strip()
    if not val:
        return and_(conds) if len(conds) > 1 else conds[0]

    # Search for not
    chars = ["!"]
    if use_emu_syntax:
        chars = [emu_escape(n) for n in chars]
    pattern = r"(^|\b)(" + "|".join([re.escape(n) for n in chars]) + r")([-\w]+)"
    match = re.search(pattern, val)
    if match:
        conds.append(not_(_val_to_query(col, match.group(3))))
    val = re.sub(pattern, "", val).strip()
    if not val:
        return and_(conds) if len(conds) > 1 else conds[0]

    # Search for populated fields
    chars = ["*", "+"]
    if use_emu_syntax:
        chars = [emu_escape(n) for n in chars]
    pattern = r"(^|\b|\s)(" + "|".join([re.escape(n) for n in chars]) + r")(\b|\s|$)"
    if re.search(pattern, val):
        conds.append(exists(True, col=col))
    val = re.sub(pattern, "", val).strip()
    if not val:
        return and_(conds) if len(conds) > 1 else conds[0]

    # Search by stem
    chars = ["~"]
    if use_emu_syntax:
        chars = [emu_escape(n) for n in chars]
    pattern = r"(^|\b|\s)(" + "|".join([re.escape(n) for n in chars]) + r")([-\w]+)"
    match = re.search(pattern, val)
    if match:
        conds.append(stemmed(match.group(3), col=col))
    val = re.sub(pattern, "", val).strip()
    if not val:
        return and_(conds) if len(conds) > 1 else conds[0]

    # Search phonetically
    chars = ["@"]
    if use_emu_syntax:
        chars = [emu_escape(n) for n in chars]
    pattern = r"(^|\b|\s)(" + "|".join([re.escape(n) for n in chars]) + r")([-\w]+)"
    match = re.search(pattern, val)
    if match:
        conds.append(phonetic(match.group(3), col=col))
    val = re.sub(pattern, "", val).strip()
    if not val:
        return and_(conds) if len(conds) > 1 else conds[0]

    # Search case- and diacritic-sensitively
    chars = ["=", "=="]
    if use_emu_syntax:
        chars = [emu_escape(n) for n in chars]
    pattern = r"(^|\b|\s)(" + "|".join([re.escape(n) for n in chars]) + r")([-\w]+)"
    if re.search(pattern, val):
        raise ValueError(
            "Case- and diacritic-sensitive searches are not supported by the API"
        )

    # Search for an exact word or phrase. Most numbers are handled in the Words and
    # Numbers section below, although this pattern should catch phrases containing
    # number, e.g., "Site 123".
    if use_emu_syntax:
        pattern = r'\\\^([^\d\W]+|\\"[^"]+\\")\\\$'
    else:
        pattern = r'\^([^\d\W]+|"[^"]+")\$'
    match = re.match(pattern + "$", val)
    if match:
        conds.append(exact(match.group(1).strip('\\"'), col=col))
        val = re.sub(pattern, "", val).strip()

    # Phrases
    if use_emu_syntax:
        pattern = rf"{ops}?(\\'(?:.*?)\\'|\\\"(?:.*?)\\\")"
    else:
        pattern = rf"{ops}?('(?:.*?)'|\"(?:.*?)\")"
    for op, val_ in re.findall(pattern, val):
        cond = phrase(val_.strip("\"'\\"), col=col)
        conds.append(cond if op.lstrip("\\") != "!" else not_(cond))
    val = re.sub(pattern, "", val).strip()
    if not val:
        return and_(conds) if len(conds) > 1 else conds[0]

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
                cond = exact(to_type(val), col=col, mode=mode)
            else:
                cond = contains(to_type(val), col=col)
            conds.append(cond if op.lstrip("\\") != "!" else not_(cond))

    # Ranges
    if ranges:
        mapping = {">=": "gte", ">": "gt", "<=": "lte", "<": "lt"}
        kwargs = {mapping[k]: to_type(v) for k, v in ranges.items()}
        kwargs["mode"] = mode
        conds.append(range_(col=col, **kwargs))

    return and_(conds) if len(conds) > 1 else conds[0]


def _parse_api(module: str, val: dict, api: EMuAPI, select=None, key=None, mapped=None):
    """Parses API response to remove field groupings"""

    if mapped is None:
        mapped = {}

    if key and not key.endswith(("_grp", "_subgrp")):
        key = EMuAPI.schema.map_short_name(module, key)
        try:
            select = select[key]
        except (KeyError, TypeError):
            pass

    # Iterate dicts
    if isinstance(val, dict):
        for key, val in val.items():
            _parse_api(module, val, api, select=select, key=key, mapped=mapped)

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
                _parse_api(module, vals, api, select=select, key=key, mapped=mapped)

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
                _parse_api(module, vals, api, select=select, key=key, mapped=mapped)

    # Simplify IRNs. Note that multimedia references use Ref fields and IRN-like text.
    elif val and is_ref(key):
        if "/media/" in val:
            mapped[key] = val
        elif isinstance(val, str):
            mapped[key] = attachment(val, api, json.dumps(select))
        elif isinstance(val, (list, tuple)):
            mapped[key] = [
                s if "/media/" in s else attachment(s, api, json.dumps(select))
                for s in val
            ]

    elif key == "irn" and not isinstance(val, int):
        mapped[key] = int(val.split("/")[-1])

    else:
        mapped[key] = val

    return mapped
