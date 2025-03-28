Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on [Keep a
Changelog](https://keepachangelog.com/en/1.0.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

0.1b9
-----

- Added support for Python 3.13 and removed support for 3.8 and 3.9
- Added support for reverse attachment fields
- Added `date()` method to EMuDate
- Added `round()` method to EMuFloat
- Added `rec_class` param to EMuReader
- Added type hints to most objects and additional docstrings
- Modified date parsing in `EMuDate` to handle additional formats
- Renamed `query()` to `filter()` on `EMuGrid`
- Fixed bug parsing certain coordinates with trailing zeroes

0.1b8
-----

- Added `clean_xml()` function to removed restricted characters from XML
- Added `sum` keyword for handling repeated keys in
  `from_xml_parallel()`
- Changed handling of reference field views and field groups, including
  group definitions with overlapping fields
- Removed date attribute setters from EMuDate

0.1b7
-----

- Transferred repository to NMNH-IDSC
- Added `from_xml_parallel()` method to EMuReader to allow parallel
  reading of XML files (experimental)
- Added support for pickle and json to EMuRecord objects
- Added check for missing grid fields to the EMuReader and EMuGrid
  objects. These methods are intended to allow users to catch when a
  required grid field is missing from an export.
- Changed custom data types to be essentially immutable. Previously
  attributes could be directly modified.
- Changed loading of schema and config files such that these files are
  explicitly loaded when the module is imported. Previously they were
  lazy loaded when an associated attribute was accessed.

0.1b6
-----

- Added support for Python 3.12
- Added functions to write CSV from XML files
- Changed EMuRecord to coerce NAs to None and to coerce empty nested
  tables to empty lists
- Changed write_import() to write_xml(). The write_import() function
  remains as an alias.
- Changed EMuCoord to catch out-of-bounds minutes and seconds
- Changed EMuDate to handle up to three arguments (date or (year, month,
  day))
- Fixed bug identifying the module in a reference

0.1b5
-----

- Added support for Python 3.11 and removed support for Python 3.7
- Added check for sequence data in atomic fields
- Changed parsing of config files, including autoloading schemas
- Changed EMuCoord to better represent decimal coordinates
- Changed EMuFloat to accept floats with no formatting string
- Fixed bugs with EMuType comparisons, reading/writing XML, and
  populating lookup list parents

0.1b4
-----

- Changed EMuDate to handle dates outside the MINYEAR and MAXYEAR
  allowed in python’s datetime module. Dates outside that range can be
  expressed but do not support most date operations.
- Changed EMuRecord.to_xml() and EMuColumn.to_xml() to (1) include blank
  values when records are created, (2) to add blank values to parent
  fields in a lookup hierarchy based a key in the config file, and (3)
  to check for an emu_str() method when writing values. The emu_str()
  method is used by some EMuType classes to allow greater control in how
  values are depicted in imports.
- Fixed bugs selecting grid rows, parsing IRNs, reporting progress, and
  marking normally inaccessible fields as visible.

0.1b3
-----

- Added EMuReader.report_progress() to notify users about progress
  reading through an XML file
- Changed EMuColumn.to_xml() method so that table rows are numbered
  sequentially using the row attribute if no modifiers are given. This
  prevents the EMu client from skipping empty nested table rows in valid
  XML during import.
- Changed behavior of EMuRow.\_\_getitem\_\_() when a dictionary is
  passed to:
  1.  fix a bug where empty row values matched any query string
  2.  return only the row. Previously the method returned the index and
      row as a tuple, but this was redundant as the index is accessible
      directly from the row object itself.
- Changed modifier pattern to catch additional modifiers supported by
  EMu. Acceptable formats include “=”, “+”, “-”, “1=”, “1+”, and “1-”.
- Renamed EMuRow mod attribute to replace_mod to clarify that it returns
  the replacement modifier

0.1b2
-----

- Modified XML output by write_import to more closely resemble XML
  produced by EMu. XML formatting should now be the same as EMu’s except
  that (1) xmu excludes the DOCTYPE and schema elements near the top of
  the document, (2) xmu does not include separate open/close tags for
  empty tables and tuples, and (3) there can be minor formatting
  differences for certain data types, including times and coordinates.
  These formatting differences should affect only the appearance, not
  the meaning, of the XML.
- Added additional data types for times, latitudes, and longitudes
- Added write_group function
- Fixed errors reading and writing nested tables

0.1b1
-----

Initial release
