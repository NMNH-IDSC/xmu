"""Defines objects used to read and write XML for Axiell EMu"""

import csv
import datetime as dt
import glob
import json
import logging
import mmap
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from joblib import Parallel, delayed
from lxml import etree

from .utils import flatten, is_nesttab, is_ref, is_ref_tab, is_tab, strip_tab


logger = logging.getLogger(__name__)


class EMuReader:
    """Read records from an EMu XML file into dicts

    Parameters
    ----------
    path : str or Path
        path to a file or directory
    json_path : str or Path
        path to a JSON file used to cache records for faster reading

    Attributes
    ----------
    path : str or Path
        path to a file or directory
    json_path : str or Path
        path to a JSON file used to cache records for faster reading
    files : list
        list of file-like objects, each of which is an EMu XML file
    module : str
        the name of an EMu module
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

    def __init__(self, path, rec_class=dict, json_path=None):
        self.path = str(path)
        self._rec_class = rec_class
        self.json_path = json_path
        self.files = []
        self.module = None
        self._fields = None
        self._get_files()
        self._load_schema()

        # Private attributes used to display notifications
        self._job_start = None
        self._job_done = False
        self._notify_start = None
        self._notify_count = 0

    def __iter__(self):
        for rec in self.from_file():
            yield rec

    def __len__(self):
        counts = self.counts()
        if isinstance(counts, int):
            return counts
        raise NotImplementedError("Not implemented when multiple source files provided")

    @property
    def fields(self):
        if self._fields is None:
            self._fields = self._parse_file_schema()
        return self._fields

    def from_file(self):
        """Reads data from file, using JSON if possible

        Yields
        ------
        dict
            EMu record
        """
        if not self.json_path:
            return self.from_xml()

        # If JSON is older than the newest XML file, regenerate it
        try:
            if os.path.getmtime(self.json_path) < self.files[-1].getmtime():
                logger.info("Regenerating JSON (XML is newer)")
                self.to_json()
        except FileNotFoundError:
            logger.info("Generating JSON (JSON not found)")
            self.to_json()

        return self.from_json()

    def from_xml(self, start=0, limit=None):
        """Reads data from XML

        Parameters
        ----------
        start : int
            index of record to start processing
        limit : int
            number of records to process from start. If omitted, all records
            are processed.

        Yields
        ------
        dict
            EMu record
        """
        try:
            for filelike in self.files:
                logger.info("Reading records from %s", filelike)
                self._job_start = None
                self._job_done = False
                self._notify_start = None
                self._notify_count = 0
                with filelike.open("rb") as source:
                    try:
                        context = etree.iterparse(source, events=["end"], tag="tuple")
                        for _, element in context:
                            # Process children of module table only
                            parent = element.getparent().get("name")
                            if parent is not None and parent.startswith("e"):
                                try:
                                    if self._notify_count >= start:
                                        yield self._parse(element)
                                finally:
                                    element.clear()
                                    # while element.getprevious() is not None:
                                    #    del element.getparent()[0]
                                    self._notify_count += 1
                                    if not self._notify_count % 5000:
                                        logger.info(
                                            "Read %s records from %s",
                                            self._notify_count,
                                            filelike,
                                        )
                                    if (
                                        limit is not None
                                        and self._notify_count >= start + limit
                                    ):
                                        break
                    finally:
                        del context
                logger.info("Read %s records total", self._notify_count)
                self._job_done = True
        finally:
            if self._job_start:
                self.report_progress()

    def from_xml_parallel(
        self, callback, num_parts=64, handle_repeated_keys="overwrite"
    ):
        """Reads data from XML in parallel

        Experimental. Works by creating temporary copies of the XML file, then reading from
        those files in parallel. Seems to work best with a small number of copies.

        Parameters
        ----------
        callback : function
            function to run on the import file
        num_parts : int
            number of parts to split the file into
        handle_repeated_keys : str
            defines how to handle keys that repeat across dicts returned by different jobs.
            Must be one of 'combine' (which combines entires in a list), 'keep' (which keeps
            the first key found), 'overwrite' (which overwrites the existing key), 'raise'
            (which raises a KeyError), r 'sum' (which sums integer values). Ignored if
            callback does not return a dict.

        Yields
        ------
        mixed
            result of callback function combined across jobs. If dict, results are combined
            into a single dict. If list, results are combined into a single list. If another
            type, returns a list of results returned by the callback.
        """

        if len(self.files) > 1 or not self.files[0].path.lower().endswith(".xml"):
            raise NotImplementedError(
                "Not implemented when multiple source files provided"
            )

        allowed = ("combine", "keep", "overwrite", "raise", "sum")
        if handle_repeated_keys not in allowed:
            raise ValueError(f"dict_behavior must be one of the following: {allowed}")

        # Create temporary directory
        tmpdir = tempfile.mkdtemp(prefix="xmu-")

        try:

            files = []
            with open(self.files[0].path, "rb") as f:
                with mmap.mmap(
                    f.fileno(), length=0, access=mmap.ACCESS_READ, offset=0
                ) as m:
                    content = m.read()
                    sep = b"<!-- Row"
                    records = content.split(sep)
                    header = records.pop(0)
                    step = int(len(records) / num_parts) + 1
                    for i in range(0, len(records), step):
                        group = records[i : i + step]
                        tmp = tempfile.NamedTemporaryFile(
                            "wb", prefix="xmu-", suffix=".xml", dir=tmpdir, delete=False
                        )
                        with open(tmp.name, "wb") as f:
                            f.write(header)
                            f.write(b"".join((sep + r for r in group)))
                            if not group[-1].rstrip().endswith(b"</table>"):
                                f.write(b"\n</table>")
                        files.append(tmp)

            results = Parallel(n_jobs=-1)(delayed(callback)(tmp.name) for tmp in files)

            if isinstance(results[0], dict):
                result = {}
                for result_ in results:
                    if handle_repeated_keys in ("combine", "sum"):
                        for key, val in result_.items():
                            if not isinstance(val, list):
                                val = [val]
                            result.setdefault(key, []).extend(val)
                    elif handle_repeated_keys == "keep":
                        for key, val in result_.items():
                            result.setdefault(key, val)
                    elif handle_repeated_keys == "overwrite":
                        result.update(result_)
                    else:
                        if set(result_) & set(result):
                            raise KeyError("Duplicate keys returned")
                        result.update(result_)
                if handle_repeated_keys == "sum":
                    result_ = {}
                    for key, vals in result.items():
                        try:
                            result_[key] = sum(vals)
                        except TypeError:
                            result_[key] = vals
                    result = result_
                return result
            elif isinstance(results[0], list):
                result = []
                for result_ in results:
                    result.extend(result_)
                return result

            return results
        finally:
            for tmp in files:
                tmp.close()
                os.remove(tmp.name)
            shutil.rmtree(tmpdir)

    def from_json(self, chunk_size=2097152):
        """Reads data from JSON

        Parameters
        ----------
        chunk_size : int
            size of chunk to use when reading the file

        Yields
        ------
        dict
            EMu record
        """
        logger.info("Reading records from %s", self.json_path)
        self._job_start = None
        self._job_done = False
        self._notify_start = None
        self._notify_count = 0
        with open(self.json_path, encoding="utf-8") as f:
            f.read(1)
            add_to_next_chunk = []
            while True:
                chunk = f.read(chunk_size)
                if add_to_next_chunk:
                    chunk = "".join(add_to_next_chunk[::-1]).lstrip(",") + chunk
                    add_to_next_chunk = []

                if len(chunk) <= 1:
                    break

                while True:
                    try:
                        for rec in json.loads(f"[{chunk.lstrip(',')[:-1]}]"):
                            try:
                                yield rec
                            finally:
                                self._notify_count += 1
                                if not self._notify_count % 5000:
                                    logger.info(
                                        "Read %s records from %s",
                                        self._notify_count,
                                        self.json_path,
                                    )
                        break
                    except json.JSONDecodeError:
                        chunk, trailer = chunk.rsplit("{", 1)
                        add_to_next_chunk.append(f"{{{trailer}")
        logger.info("Read %s records total", self._notify_count)
        self._job_done = True
        if self._job_start:
            self.report_progress()

    def to_csv(self, path, **kwargs):
        """Writes records in reader object to CSV

        Parameters
        ----------
        path : str
            path to write the CSV file
        kwargs :
            any keyword argument accepted by open()
        """
        return write_csv(self, path, **kwargs)

    def to_json(self, path=None, **kwargs):
        """Writes JSON version of XML to file

        Parameters
        ----------
        path : str
            path to write JSON
        kwargs :
            keyword arguments for json.dump()
        """
        if path is None:
            path = self.json_path

        logger.info("Writing records from %s to JSON", self.path)

        params = {
            "ensure_ascii": False,
            "indent": None,
            "sort_keys": False,
            "separators": (",", ":"),
        }
        params.update(**kwargs)

        sep = params["separators"][0]

        with open(path, "w", encoding="utf-8"):
            pass

        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write("[")
                records = []
                for rec in self.from_xml():
                    records.append(rec)
                    if len(records) > 1000:
                        f.write(json.dumps(records, **params)[1:-1] + sep)
                        records = []
                if records:
                    f.write(json.dumps(records, **params)[1:-1])
                f.write("]")
        except KeyboardInterrupt as exc:
            # Remove the partial JSON file if write is interrupted
            os.remove(path)
            raise IOError("Conversion to JSON failed") from exc

    def counts(self):
        """Counts the number of records in each file"""
        counts = {}
        for filelike in self.files:
            with open(filelike.path, mode="r", encoding="utf8") as f:
                with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as m:
                    counts[filelike.path] = len(re.findall(rb"\n  <tuple>", m.read()))
        return counts[list(counts)[0]] if len(counts) == 1 else counts

    def verify_group(self, path, module=None):
        """Verifies that all fields in a group are present in the export

        Raises
        ------
        ValueError
            if one or more fields missing
        """
        if module is None:
            module = self.module
        group = tuple(self.schema.get_field_info(module, path).get("GroupFields", []))
        missing = set(group) - set(self.fields)
        if missing:
            raise ValueError(f"Group including '{path}' is missing fields: {missing}")

    def report_progress(self, by="time", at=5):
        """Prints progress notification messages when reading a file

        Parameters
        ----------
        by : str
            either "count" or "time"
        at : int
            number of seconds (if by time) or number of records (if by count)
        """
        if self._notify_start is None:
            self._job_start = time.time()
            self._notify_start = time.time()

        elapsed = time.time() - (
            self._job_start if self._job_done else self._notify_start
        )
        if (
            self._job_done
            or by == "time"
            and elapsed >= at
            or by == "count"
            and self._notify_count
            and not (self._notify_count % at)
        ):
            print(
                "{:,} records processed (t{}={:.1f}s)".format(
                    self._notify_count, "otal" if self._job_done else "", elapsed
                )
            )
            self._notify_start = time.time()

    def _parse(self, xml):
        """Parses a record from XML

        Parameters
        ----------
        xml : lxml.Element
            XML representing a single record

        Returns
        -------
        dict
           EMu record
        """
        if self._rec_class != dict:
            dct = self._rec_class(module=self.module)
        else:
            dct = self._rec_class()

        elements = [(dct, "", xml)]
        while elements:
            new_elems = []
            for obj, parent_name, elem in elements:
                for child in elem:
                    # Add an empty rows to a nested table, which do not contain
                    # child nodes when exported from EMu
                    if child is None:
                        obj.append(None)
                        continue

                    # Get field name
                    name = child.get("name")
                    if name is None:
                        name = ""

                    # Get field text
                    text = child.text
                    if text is not None:
                        text = text.strip()

                    # Add an atomic field
                    if child.tag == "atom":
                        try:
                            obj[name] = text
                        except TypeError:
                            obj.append(text)

                    # Add a reference
                    elif child.tag == "tuple" and is_ref(name) and not is_tab(name):
                        obj[name] = {}
                        new_elems.append((obj[name], name, child))

                    # Add a table or reference table
                    elif child.tag == "table" or (child.tag == "tuple" and name):
                        try:
                            obj[name] = []
                            new_elems.append((obj[name], name, child))
                        except TypeError:
                            obj.append([])
                            new_elems.append((obj[-1], name, child))

                    # Add a row to a reference table
                    elif (
                        child.tag == "tuple"
                        and is_ref_tab(parent_name)
                        and not is_nesttab(parent_name)
                    ):
                        obj.append({})
                        new_elems.append((obj[-1], name, child))

                    # Add an empty row to an outer nested table
                    elif (
                        child.tag == "tuple"
                        and is_nesttab(parent_name)
                        and not len(child)
                    ):
                        new_elems.append((obj, name, [None]))

                    elif child.tag == "tuple":
                        new_elems.append((obj, name, child))

                elements = new_elems

        return dct

    def _get_files(self):
        """Analyzes source files on self.path"""
        files = []
        zip_file = None
        if self.path:
            if os.path.isdir(self.path):
                files = glob.glob(os.path.join(self.path, "*.xml"))
            elif self.path.lower().endswith(".xml"):
                files = [self.path]
            elif self.path.lower().endswith(".zip"):
                zip_file = zipfile.ZipFile(self.path)
                files = zipfile.ZipFile(self.path).infolist()
            else:
                raise IOError(f"Invalid path: {self.path}")

        # Order source files from oldest to newest
        self.files = [FileLike(obj, zip_file=zip_file) for obj in files]
        self.files.sort(key=lambda f: f.getmtime())

        # Get the module name from the first table tag
        with self.files[0].open(encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("<table"):
                    self.module = line.split("=", 1)[-1].strip('">\r\n')
                    break

    def _load_schema(self):
        """Tries to load the schema based on the rec_class"""
        if self.schema is None:
            try:
                schema = self._rec_class.schema
                if schema is None:
                    # This will also load the configuration
                    schema = self._rec_class(module=self.module).schema
            except (AttributeError, ValueError):
                schema = None
        return self.schema

    def _parse_file_schema(self):
        """Parses top-level fields from header of EMu XML file

        Returns
        -------
        list
            list of top-level fields
        """
        fields = {}
        for filelike in self.files:
            with open(filelike.path, "r", encoding="utf-8") as f:
                lines = []
                for line in f:
                    lines.append(line)
                    if line.startswith("?>"):
                        break
                content = "".join(lines)

            tables = []
            for line in (
                re.search(r"<?schema\s+(.*?)\?>", content, flags=re.DOTALL)
                .group(1)
                .splitlines()
            ):
                line = line.strip()
                try:
                    dtype, field = [s.strip() for s in line.rsplit(" ", 1)]
                except ValueError:
                    dtype = None
                    tables.pop()
                else:
                    if dtype == "table":
                        tables.append(field)
                    else:
                        segments = tables[1:] + [field]
                        segments = [
                            s
                            for i, s in enumerate(segments)
                            if strip_tab(s) not in {strip_tab(s) for s in segments[:i]}
                        ]
                        fields[segments[0]] = 1
        return tuple(fields)


class FileLike:
    """Open text and zip files using the same interface

    Parameters
    ----------
    filelike : mixed
        either the path to an XML file or a ZipInfo object
    zip_file : zipfile.ZipFile
        if filelike is a ZipInfo object, the zip file containing that object

    Attributes
    ----------
    path : str
        path to file
    zip_info : zipfile.ZipInfo
        member of a zip archive
    zip_file : zipfile.ZipFile
        the zip file containing the ZipInfo object
    """

    def __init__(self, filelike, zip_file=None):
        self.path = None
        self.zip_info = None
        self.zip_file = None
        if zip_file:
            self.zip_info = filelike
            self.zip_file = zip_file
        else:
            self.path = os.path.realpath(filelike)

    def __str__(self):
        return f'<FileLike name="{self.filename}">'

    def __repr__(self):
        return str(self)

    @property
    def filename(self):
        """Name of the file-like object"""
        return os.path.basename(self.path) if self.path else self.zip_info.filename

    def open(self, mode="r", encoding=None):
        """Opens a file or ZipInfo object"""
        if not self.zip_info:
            return open(self.path, mode=mode, encoding=encoding)
        stream = self.zip_file.open(self.zip_info, mode.rstrip("b"))
        if encoding:
            return _ByteDecoder(stream, encoding)
        return stream

    def getmtime(self):
        """Returns last modification timestamp from a file or ZipInfo object"""
        try:
            return os.path.getmtime(self.path)
        except TypeError:
            return dt.datetime(*self.zip_info.date_time).timestamp()


class _ByteDecoder:
    """File-like context manager that encodes a binary stream from a zip file"""

    def __init__(self, stream, encoding):
        self._stream = stream
        self._encoding = encoding

    def __iter__(self):
        for line in self._stream:
            yield line.decode(self._encoding)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exception, traceback):
        if exception:
            raise exception
        self._stream.close()


def clean_xml(path, encoding="utf-8"):
    """Removes restricted characters from XML file

    Parameters
    ----------
    path : str
        path to write the CSV file
    encoding : str
        encoding for reading/writing XML

    Returns
    -------
    pathlib.Path
        path to clean XML file
    """

    def _remove_restricted_chars(val):
        # From https://stackoverflow.com/a/64570125
        illegal_unichrs = [
            (0x00, 0x08),
            (0x0B, 0x0C),
            (0x0E, 0x1F),
            (0x7F, 0x84),
            (0x86, 0x9F),
            (0xFDD0, 0xFDDF),
            (0xFFFE, 0xFFFF),
        ]
        if sys.maxunicode >= 0x10000:
            illegal_unichrs.extend(
                [
                    (0x1FFFE, 0x1FFFF),
                    (0x2FFFE, 0x2FFFF),
                    (0x3FFFE, 0x3FFFF),
                    (0x4FFFE, 0x4FFFF),
                    (0x5FFFE, 0x5FFFF),
                    (0x6FFFE, 0x6FFFF),
                    (0x7FFFE, 0x7FFFF),
                    (0x8FFFE, 0x8FFFF),
                    (0x9FFFE, 0x9FFFF),
                    (0xAFFFE, 0xAFFFF),
                    (0xBFFFE, 0xBFFFF),
                    (0xCFFFE, 0xCFFFF),
                    (0xDFFFE, 0xDFFFF),
                    (0xEFFFE, 0xEFFFF),
                    (0xFFFFE, 0xFFFFF),
                    (0x10FFFE, 0x10FFFF),
                ]
            )

        illegal_ranges = [rf"{chr(low)}-{chr(high)}" for (low, high) in illegal_unichrs]
        xml_illegal_character_regex = "[" + "".join(illegal_ranges) + "]"
        illegal_xml_chars_re = re.compile(xml_illegal_character_regex)
        return illegal_xml_chars_re.sub("", val)

    chunks = []
    with open(path, encoding=encoding) as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            chunks.append(_remove_restricted_chars(chunk))

    path = Path(path)
    output = path.parent / f"{path.stem}_clean{path.suffix}"
    with open(output, "w", encoding=encoding) as f:
        f.write("".join(chunks))
    return output


def write_csv(records, path, **kwargs):
    """Writes records to CSV

    Parameters
    ----------
    records : list-like
        list of EMuRecords to be written
    path : str
        path to write the CSV file
    kwargs :
        any keyword argument accepted by open()
    """
    flattened = [flatten(r) for r in records]

    keys = {}
    for rec in flattened:
        keys.update({k: 1 for k in rec})

    # Reorder keys to account for varying grid lengths
    grouped = {}
    for key in keys:
        grouped.setdefault(re.sub(r"\.\d+\.", ".x.", key), []).append(key)

    grouped = {
        k: sorted(v, key=lambda s: ".".join([s.zfill(8) for s in s.split(".")]))
        for k, v in grouped.items()
    }

    fieldnames = []
    for group in grouped.values():
        fieldnames.extend(group)

    kwargs.setdefault("encoding", "utf-8-sig")
    kwargs.setdefault("newline", "")
    with open(path, "w", **kwargs) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(({k: r.get(k) for k in fieldnames} for r in flattened))


def write_import(*args, **kwargs):
    """Writes records to an EMu import file

    Alias for write_xml()
    """
    return write_xml(*args, **kwargs)


def write_xml(records, path, **kwargs):
    """Writes records to an EMu import file

    Parameters
    ----------
    records : list-like
        list of EMuRecords to be imported
    path : str
        path to write the import file
    kwargs :
        any keyword argument accepted by the to_xml() method of the record class
    """
    root = etree.Element("table")
    root.set("name", records[0].module)
    root.addprevious(etree.Comment(" Data "))

    for i, rec in enumerate(records):
        try:
            node = rec.copy().to_xml(root, **kwargs)
            node.addprevious(etree.Comment(f" Row {i + 1} "))
        except AttributeError:
            raise ValueError(f"Could not convert record to XML: {rec}")

    root.getroottree().write(
        path, pretty_print=True, xml_declaration=True, encoding="utf-8"
    )


def write_group(records, path, irn=None, name=None):
    """Writes an import for the egroups module

    Parameters
    ----------
    records : list
        list of EMuRecords, each of which specifies an irn
    path : str
        path to write the import file
    irn : int
        the irn of an existing egroups record (updates only)
    name : str
        the name of the group
    """
    if not irn and not name:
        raise ValueError("Must specify at least one of irn or name for a group")
    rec = records[0].__class__(
        {
            "GroupType": "Static",
            "Module": records[0].module,
            "Keys_tab": [rec["irn"] for rec in records],
        },
        module="egroups",
    )
    if irn:
        rec["irn"] = irn
    if name:
        rec["GroupName"] = name
    write_import([rec], path)
