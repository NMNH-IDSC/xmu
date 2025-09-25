xmu
===

xmu is a Python utility used to read and write XML for Axiell EMu, a
collections management systems used in museums, galleries, and similar
institutions. This package was developed in the Department of Mineral
Sciences at the Smithsonian National Museum of Natural History to
streamline the process of getting data into and out of EMu.

In addition to the instructions below, a guide and API reference are
available in the [documentation](https://xmu.readthedocs.io/en/latest/).

Install
-------

Install xmu with pip:

    pip install xmu

Or install from the GitHub repository using git and pip:

    git clone https://github.com/NMNH-IDSC/xmu
    cd xmu
    pip install .

Quickstart
----------

``` python
from xmu import EMuReader, EMuRecord, EMuSchema, write_xml

# Loading an EMu schema file allows xmu to validate data, coerce data to
# the proper type, and manage grids
EMuSchema("path/to/schema.pl")

# Read records from an XML export file to dicts using EMuReader
records = []
reader = EMuReader("xmldata.xml")
for rec in reader:

    # Convert dicts to EMuRecords to access some extra functionality
    rec = EMuRecord(rec, module=reader.module)

    rec["EmuRef.irn"]  # use dot paths to retrieve keys
    rec["EmuBadKey"]   # keys not found in the schema throw a special error
    rec["EmuDate"]     # dates use EMuDate wrapper to preserve date format
    rec["EmuFloat"]    # floats use EMuFloat wrapper to preserve precision

    # Access grids defined in the schema using any member field
    grid = rec.grid("EmuTable_tab")
    grid[0]                          # get rows by index
    grid[{"EmuTable_tab": "value"}]  # get rows where EMuGrid_tab == value

    # Use EMuRecords to create or update records in EMu
    update = EMuRecord({
      "irn": rec["irn"],                   # include an irn to update a record
      "EmuString": "String",
      "EmuInteger": 100,
      "EmuFloat": 1.2,
      "EmuDate": "1970-01-01",             # dates are strings or datetime.date
      "EmuRef": {"irn": 1234567},          # references are dicts
      "EmuTable_tab": ["Row 1", "Row 2"],  # tables are lists
      "EmuRef_tab": [{"irn": 1234567}],    # ref tables are lists of dicts
      "EmuNested_nesttab": [["Nested"]],   # nested tables are lists of lists
      "EmuBadKey": ["Bad format"],         # bad keys or formats throw an error
    }, module=reader.module)

    # Create a list of records to import
    records.append(update)

# Write the XML import file from the list of EMu records
write_xml(records, "update.xml")
```

You can use the experimental `from_xml_parallel()` method to read large
XML files more quickly. For example, to create a dict mapping IRNs to
records:

``` python
def callback(path):
    reader = EMuReader(path)
    results = {}
    for rec in reader:
        rec = EMuRecord(rec, module=reader.module)
        results[rec["irn"]] = rec
    return results

results = EMuReader("xmldata.xml").from_xml_parallel(callback)
```
