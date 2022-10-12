Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on [Keep a
Changelog](https://keepachangelog.com/en/1.0.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

0.1b4
-----

-   Changed EMuDate to handle dates outside the MINYEAR and MAXYEAR
    allowed in python’s datetime module. Dates outside that range can be
    expressed but do not support most date operations.
-   Changed EMuRecord.to_xml() and EMuColumn.to_xml() to (1) include
    blank values when records are created, (2) to add blank values to
    parent fields in a lookup hierarchy based a key in the config file,
    and (3) to check for an emu_str() method when writing values. The
    emu_str() method is used by some EMuType classes to allow greater
    control in how values are depicted in imports.
-   Fixed bugs selecting grid rows, parsing IRNs, reporting progress,
    and marking normally inaccessible fields as visible.

0.1b3
-----

-   Added EMuReader.report_progress() to notify users about progress
    reading through an XML file
-   Changed EMuColumn.to_xml() method so that table rows are numbered
    sequentially using the row attribute if no modifiers are given. This
    prevents the EMu client from skipping empty nested table rows in
    valid XML during import.
-   Changed behavior of EMuRow.\_\_getitem\_\_() when a dictionary is
    passed to:
    1.  fix a bug where empty row values matched any query string
    2.  return only the row. Previously the method returned the index
        and row as a tuple, but this was redundant as the index is
        accessible directly from the row object itself.
-   Changed modifier pattern to catch additional modifiers supported by
    EMu. Acceptable formats include “=”, “+”, “-”, “1=”, “1+”, and “1-”.
-   Renamed EMuRow mod attribute to replace_mod to clarify that it
    returns the replacement modifier

0.1b2
-----

-   Modified XML output by write_import to more closely resemble XML
    produced by EMu. XML formatting should now be the same as EMu’s
    except that (1) xmu excludes the DOCTYPE and schema elements near
    the top of the document, (2) xmu does not include separate
    open/close tags for empty tables and tuples, and (3) there can be
    minor formatting differences for certain data types, including times
    and coordinates. These formatting differences should affect only the
    appearance, not the meaning, of the XML.
-   Added additional data types for times, latitudes, and longitudes
-   Added write_group function
-   Fixed errors reading and writing nested tables

0.1b1
-----

Initial release
