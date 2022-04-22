Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on [Keep a
Changelog](https://keepachangelog.com/en/1.0.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
