"""Defines objects used to read and write XML for Axiell EMu"""
import datetime as dt
import glob
import json
import logging
import os
import zipfile

from lxml import etree

from .utils import is_nesttab, is_nesttab_inner, is_ref, is_ref_tab, is_tab


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
    config = None

    #: EMuSchema : info about a specific EMu configuration. Set automatically
    #: when an EMuSchema object is created.
    schema = None

    def __init__(self, path, json_path=None):
        self.path = path
        self._rec_class = dict
        self.json_path = json_path
        self.files = []
        self.module = None
        self._get_files()
        self._load_schema()

    def __iter__(self):
        for rec in self.from_file():
            yield rec

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

    def from_xml(self):
        """Reads data from XML

        Yields
        ------
        dict
            EMu record
        """
        for filelike in self.files:
            logger.info("Reading records from %s", filelike)
            i = 0
            with filelike.open("rb") as source:
                try:
                    context = etree.iterparse(source, events=["end"], tag="tuple")
                    for _, element in context:
                        # Process children of module table only
                        parent = element.getparent().get("name")
                        if parent is not None and parent.startswith("e"):
                            try:
                                yield self._parse(element)
                            finally:
                                element.clear()
                                # while element.getprevious() is not None:
                                #    del element.getparent()[0]
                                i += 1
                                if not i % 5000:
                                    logger.info("Read %s records from %s", i, filelike)
                finally:
                    del context
            logger.info("Read %s records total", i)

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
        i = 0
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
                                i += 1
                                if not i % 5000:
                                    logger.info(
                                        "Read %s records from %s", i, self.json_path
                                    )
                        break
                    except json.JSONDecodeError:
                        chunk, trailer = chunk.rsplit("{", 1)
                        add_to_next_chunk.append(f"{{{trailer}")
        logger.info("Read %s records total", i)

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

                    # Add a row to a table
                    elif (
                        child.tag == "tuple"
                        and is_ref_tab(parent_name)
                        and not is_nesttab(parent_name)
                        and not is_nesttab_inner(parent_name)
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


def write_import(records, path, **kwargs):
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

    for rec in records:
        rec.copy().to_xml(root, **kwargs)

    for i, rec in enumerate(root):
        rec.addprevious(etree.Comment(f" Row {i + 1} "))

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
