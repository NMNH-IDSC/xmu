from datetime import date, datetime, time, timedelta
import os
import re
import zipfile

import pytest
from lxml import etree

from xmu import (
    EMuColumn,
    EMuConfig,
    EMuDate,
    EMuFloat,
    EMuLatitude,
    EMuLongitude,
    EMuReader,
    EMuRecord,
    EMuRow,
    EMuSchema,
    EMuTime,
    get_mod,
    has_mod,
    is_nesttab,
    is_nesttab_inner,
    is_ref,
    is_ref_tab,
    is_tab,
    strip_mod,
    strip_tab,
    write_import,
    write_group,
)
from xmu.types import ExtendedDate


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("output")


@pytest.fixture
def schema_file(output_dir):
    # THis is a partial schema that omits keys not used by the application
    pl = """#
#
#

#
#
#
#
use utf8;

%Schema =
(
	emain =>
	{
		table => 'emain',
		columns =>
		{
            'EmuClientTable_tab' =>
			{
				ColumnName => 'EmuClientTable_tab',
				DataType => 'Text',
                RefLink => 'EmuClientTableRef_tab'

				ItemName => 'Client Table',
			},
            'EmuClientTableRef_tab' =>
			{
				ColumnName => 'EmuRef_tab',
				DataType => 'Integer',
                RefLink => 'eref'

				ItemName => 'Client Reference Table',
			},
			'EmuDate0' =>
			{
				ColumnName => 'EmuDate0',
				DataType => 'Date',

                ItemName => 'Date',
				ItemFields =>
				[
					[ 8, 2, 2 ],
					[ 8, 2, 2 ],
					[ 8, 2, 2 ],
				],
			},
            'EmuEmpty' =>
			{
				ColumnName => 'EmuEmpty',
				DataType => 'Text',

                ItemName => 'Empty',
				ItemCount => 1,
				ItemFields => [ 15 ],
			},
			'EmuFloat' =>
			{
				ColumnName => 'EmuFloat',
				DataType => 'Float',

                ItemName => 'Float',
			},
            'EmuLatitude' =>
			{
				ColumnName => 'EmuLatitude',
				DataType => 'Latitude',

				ItemName => 'Latitude',
			},
            'EmuLongitude' =>
			{
				ColumnName => 'EmuLongitude',
				DataType => 'Longitude',

				ItemName => 'Longitude',
			},
            'EmuNestedTable_nesttab' =>
			{
				ColumnName => 'EmuNestedTable_nesttab',
				DataType => 'Text',

                ItemName => 'Nested Table',
			},
            'EmuNotVisible' =>
			{
				ColumnName => 'EmuNotVisible',
				DataType => 'Text',
			},
            'EmuRef' =>
			{
				ColumnName => 'EmuEmpty',
				DataType => 'Integer',
                RefTable => 'eref',

                ItemName => 'Reference',
				ItemFields =>
				[
					  10,   10,   10,   10,   10,
					  10,   10,   10,   10,   10,
                      10,
				],
			},
            'EmuRef_nesttab' =>
			{
				ColumnName => 'EmuRef_nesttab',
				DataType => 'Integer',
                RefTable => 'eref'

				ItemName => 'Nested Reference Table',
			},
            'EmuRef_tab' =>
			{
				ColumnName => 'EmuRef_tab',
				DataType => 'Integer',
                RefTable => 'eref'

				ItemName => 'Reference Table',
			},
			'EmuTable_tab' =>
			{
				ColumnName => 'EmuTable_tab',
				DataType => 'Text',

                ItemName => 'Table',
			},
            'EmuTableUngrouped_tab' =>
			{
				ColumnName => 'EmuTable_tab',
				DataType => 'Text',

                ItemName => 'Table Ungrouped',
			},
            'EmuText' =>
			{
				ColumnName => 'EmuText',
				DataType => 'Text',

				ItemName => 'Text',
			},
            'EmuTime0' =>
			{
				ColumnName => 'EmuTime0',
				DataType => 'Time',

                ItemName => 'Time',
				ItemFields =>
				[
					[ 8, 2, 2 ],
					[ 8, 2, 2 ],
					[ 8, 2, 2 ],
				],
			},
			'irn' =>
			{
				ColumnName => 'irn',
				DataType => 'Integer',

				ItemName => 'IRN',
			},
		},
        groups =>
        {
            'EmuGrid_grp' =>
            [
                'EmuDate0',
                'EmuNestedTable_nesttab',
                'EmuTable_tab',
                'EmuRef_tab'
            ],
            'EmuClientGrid_grp' =>
            [
                'EmuClientTable_tab',
            ],
        },
	},
	eref =>
	{
		table => 'eref',
		columns =>
		{
            'EmuRefOnly' =>
			{
				ColumnName => 'EmuRefOnly',
				DataType => 'Text',

				ItemName => 'Reference only',
			},
            'EmuTableInRef_tab' =>
			{
				ColumnName => 'EmuTableInRef',
				DataType => 'Text',

				ItemName => 'Table in Reference',
			},
            'irn' =>
			{
				ColumnName => 'irn',
				DataType => 'Integer',

				ItemName => 'IRN',
			},
		},
        groups =>
        {
            'EmuGridInReference_grp' =>
            [
                'EmuTableInRef_tab'
            ],
        },
	},
);

1;
"""
    path = output_dir / "schema.pl"
    with open(path, "w") as f:
        f.write(pl)
    return str(path)


@pytest.fixture
def config_file(output_dir, schema_file):
    config = EMuConfig(".")
    config["schema_path"] = schema_file
    config["groups"]["emain"] = {
        "EmuGrid_tab": [
            "EmuDate0",
            "EmuNestedTable_nesttab",
            "EmuTable_tab",
            "EmuRef_tab",
        ]
    }
    config.save_rcfile(output_dir, overwrite=True)
    return str(output_dir / ".xmurc")


@pytest.fixture
def xml_file(output_dir):
    xml = """<?xml version="1.0" encoding="UTF-8" ?>
<table name="emain">

  <!-- Row 1 -->
  <tuple>
    <atom name="irn">1000000</atom>
    <atom name="EmuText">Text</atom>
    <atom name="EmuFloat">1.0</atom>
    <atom name="EmuLatitude">45 30 15 N</atom>
    <atom name="EmuLongitude">-130 10 5 W</atom>
    <tuple name="EmuRef">
      <atom name="irn">1000000</atom>
      <atom name="EmuRefOnly">Text</atom>
    </tuple>
    <table name="EmuDate0">
      <tuple>
        <atom name="EmuDate">1970-01-01</atom>
      </tuple>
      <tuple>
        <atom name="EmuDate">Jan 1970</atom>
      </tuple>
      <tuple>
        <atom name="EmuDate">1970</atom>
      </tuple>
    </table>
    <table name="EmuTime0">
      <tuple>
        <atom name="EmuTime">9:00</atom>
      </tuple>
      <tuple>
        <atom name="EmuTime">12:00</atom>
      </tuple>
      <tuple>
        <atom name="EmuTime">15:00</atom>
      </tuple>
    </table>
    <table name="EmuTable_tab">
      <tuple>
        <atom name="EmuTable">Text</atom>
      </tuple>
      <tuple>
        <atom name="EmuTable">Text</atom>
      </tuple>
      <tuple>
      </tuple>
    </table>
    <table name="EmuTableUngrouped_tab">
      <tuple>
        <atom name="EmuTableUngrouped">Text</atom>
      </tuple>
    </table>
    <table name="EmuRef_tab">
      <tuple>
      </tuple>
      <tuple>
      </tuple>
      <tuple>
        <atom name="irn">1000000</atom>
        <atom name="EmuRefOnly">Text</atom>
      </tuple>
    </table>
    <table name="EmuNestedTable_nesttab">
      <tuple>
      </tuple>
      <tuple>
        <table name="EmuNestedTable_nesttab_inner">
          <tuple>
            <atom name="EmuNestedTable">Text</atom>
          </tuple>
        </table>
      </tuple>
    </table>
  </tuple>
</table>
"""
    path = output_dir / "xmldata.xml"
    with open(path, "w") as f:
        f.write(xml)
    return str(path)


@pytest.fixture
def rec(xml_file, config_file):
    reader = EMuReader(xml_file)
    reader.config = EMuConfig(config_file)
    for rec in reader:
        return EMuRecord(rec, module=reader.module)


@pytest.fixture
def expected_rec():
    # Expected when using rec_class == dict
    return {
        "irn": "1000000",
        "EmuText": "Text",
        "EmuFloat": "1.0",
        "EmuLatitude": "45 30 15 N",
        "EmuLongitude": "-130 10 5 W",
        "EmuRef": {"irn": "1000000", "EmuRefOnly": "Text"},
        "EmuDate0": ["1970-01-01", "Jan 1970", "1970"],
        "EmuTime0": ["9:00", "12:00", "15:00"],
        "EmuTable_tab": ["Text", "Text"],
        "EmuTableUngrouped_tab": ["Text"],
        "EmuRef_tab": [{}, {}, {"irn": "1000000", "EmuRefOnly": "Text"}],
        "EmuNestedTable_nesttab": [None, ["Text"]],
    }


@pytest.fixture
def grid(rec):
    return rec.grid("EmuTable_tab").pad()


def test_config(config_file, output_dir):
    config = EMuConfig(config_file)
    assert len(config) == 3
    assert [k for k in config] == ["schema_path", "groups", "make_visible"]
    assert config["schema_path"] == str(output_dir / "schema.pl")
    assert config["make_visible"] == []
    del config["make_visible"]
    assert "make_visible" not in config


def test_schema(schema_file):
    schema_pl = EMuSchema(schema_file)
    json_path = os.path.splitext(schema_file)[0] + ".json"
    schema_pl.to_json(json_path)
    assert schema_pl == EMuSchema(json_path)


def test_schema_from_config(config_file, schema_file):
    EMuConfig(config_file)
    assert EMuSchema() == EMuSchema(schema_file)


def test_schema_no_args(schema_file):
    EMuSchema.config = None
    assert EMuSchema() == {}


def test_schema_from_args(schema_file):
    schema = {"Schema": {}}
    assert EMuSchema(schema) == schema


def test_schema_get(schema_file):
    schema = EMuSchema(schema_file)
    assert schema.get("Schema.emain.columns.EmuDate0") == {
        "ColumnName": "EmuDate0",
        "DataType": "Date",
        "ItemName": "Date",
        "ItemFields": [[8, 2, 2], [8, 2, 2], [8, 2, 2]],
        "GroupFields": [
            "EmuDate0",
            "EmuNestedTable_nesttab",
            "EmuTable_tab",
            "EmuRef_tab",
        ],
    }
    assert schema.get("Schema.emain.columns.EmuInvalid") is None


def test_schema_iterfields(schema_file):
    schema = EMuSchema(schema_file)
    fields = {}
    for module, field, info in schema.iterfields():
        fields[(module, field)] = info
    assert list(fields) == [
        ("emain", "EmuClientTable_tab"),
        ("emain", "EmuClientTableRef_tab"),
        ("emain", "EmuDate0"),
        ("emain", "EmuEmpty"),
        ("emain", "EmuFloat"),
        ("emain", "EmuLatitude"),
        ("emain", "EmuLongitude"),
        ("emain", "EmuNestedTable_nesttab"),
        ("emain", "EmuNotVisible"),
        ("emain", "EmuRef"),
        ("emain", "EmuRef_nesttab"),
        ("emain", "EmuRef_tab"),
        ("emain", "EmuTable_tab"),
        ("emain", "EmuTableUngrouped_tab"),
        ("emain", "EmuText"),
        ("emain", "EmuTime0"),
        ("emain", "irn"),
        ("eref", "EmuRefOnly"),
        ("eref", "EmuTableInRef_tab"),
        ("eref", "irn"),
    ]


def test_schema_getitem_bad_module(schema_file):
    match = (
        r"Path not found: \('Schema', 'einvalid', 'columns', 'EmuText'\)"
        r" \(failed at einvalid\)"
    )
    with pytest.raises(KeyError, match=match):
        EMuSchema(schema_file).get_field_info("einvalid", "EmuText")


def test_schema_getitem_not_visible(schema_file):
    with pytest.raises(KeyError, match=r"emain.EmuNotVisible is valid but"):
        EMuSchema(schema_file).get_field_info("emain", "EmuNotVisible")


def test_col_change():
    col = EMuColumn(["Text"], module="emain", field="EmuTable_tab")

    col = col + ["Text"]
    assert col == ["Text"] * 2

    col.insert(0, "Text")
    assert col == ["Text"] * 3

    col.append("Text")
    assert col == ["Text"] * 4

    col.extend(["Text"])
    assert col == ["Text"] * 5

    col += ["Text"]
    assert col == ["Text"] * 6

    assert isinstance(col, EMuColumn)


def test_col_to_xml():
    col = EMuColumn(["Text"], module="emain", field="EmuTable_tab")
    assert (
        etree.tostring(col.to_xml())
        == b'<table name="EmuTable_tab"><tuple row="1"><atom name="EmuTable">Text</atom></tuple></table>'
    )


def test_col_no_module():
    with pytest.raises(ValueError, match=r"Must provide module when schema is used"):
        EMuColumn()


def test_row(rec):
    row = rec.grid("EmuTable_tab")[0]
    assert row["EmuDate0"] == rec["EmuDate0"][0] == EMuDate("1970-01-01")
    del row["EmuDate0"]
    assert row["EmuDate0"] is None and rec["EmuDate0"][0] is None


def test_row_id():
    rec = EMuRecord(
        {
            "irn": 1000000,
            "EmuTable_tab(+)": ["Text", "Text", "Text"],
            "EmuRef_tab(+)": [
                {"irn": 1000000},
                {"irn": 1000000},
                {"irn": 1000001},
            ],
        },
        module="emain",
    )
    row_ids = []
    for i in range(2):
        row_ids.append([r.row_id() for r in rec.grid("EmuTable_tab")])
    assert row_ids[0] == row_ids[1]
    assert row_ids[0][0] != row_ids[0][1]
    assert row_ids[0][0] != row_ids[0][2]
    assert row_ids[0][1] != row_ids[0][2]


def test_row_in_reference(rec):
    rec = rec.copy()
    row1 = EMuRow(rec, "EmuRef.EmuTableInRef_tab", 0)
    row2 = EMuRow(rec["EmuRef"], "EmuTableInRef_tab", 0)
    assert row1 == row2


def test_row_from_ungrouped(rec):
    with pytest.raises(KeyError, match=r"'emain.EmuText is not part of a group"):
        EMuRow(rec, "EmuText", 0)


def test_grid_by_index(grid):
    assert grid[0] == {
        "EmuDate0": EMuDate("1970-01-01"),
        "EmuNestedTable_nesttab": None,
        "EmuRef_tab": {},
        "EmuTable_tab": "Text",
    }
    assert grid[1] == {
        "EmuDate0": EMuDate("Jan 1970"),
        "EmuNestedTable_nesttab": ["Text"],
        "EmuRef_tab": {},
        "EmuTable_tab": "Text",
    }
    assert grid[2] == {
        "EmuDate0": EMuDate("1970"),
        "EmuNestedTable_nesttab": None,
        "EmuRef_tab": {"irn": 1000000, "EmuRefOnly": "Text"},
        "EmuTable_tab": "",
    }


def test_grid_by_str(grid):
    assert grid["EmuDate0"] == [
        EMuDate("1970-01-01"),
        EMuDate("Jan 1970"),
        EMuDate("1970"),
    ]


def test_empty_grid(rec):
    rec = rec.copy()

    rec["EmuRef"] = {"EmuTableInRef_tab": []}
    grid = rec.grid("EmuRef.EmuTableInRef_tab")
    assert len(grid) == 0

    rec["EmuRef"] = {}
    grid = rec.grid("EmuRef.EmuTableInRef_tab")
    assert len(grid) == 0


def test_grid_query(grid):
    results = grid.query("EmuRef_tab", where={"EmuTable_tab": "Text"})
    assert results == [{}, {}]


def test_grid_del_item(rec):
    grid = rec.grid("EmuTable_tab").pad()
    del grid[0]
    assert rec["EmuDate0"] == [EMuDate("Jan 1970"), EMuDate("1970")]
    assert rec["EmuNestedTable_nesttab"] == [["Text"], None]
    assert rec["EmuRef_tab"] == [{}, {"irn": 1000000, "EmuRefOnly": "Text"}]
    assert rec["EmuTable_tab"] == ["Text", ""]


def test_grid_from_client_table(rec):
    assert "GroupFields" in rec.schema.get_field_info(
        rec.module, "EmuClientTableRef_tab"
    )
    assert "GroupFields" not in rec.schema.get_field_info(
        rec.module, "EmuClientTable_tab"
    )


def test_grid_in_reference(rec):
    rec = rec.copy()
    rec["EmuRef"] = {"EmuTableInRef_tab": ["Text", "Text"]}
    grid1 = rec.grid("EmuRef.EmuTableInRef_tab")
    grid2 = rec["EmuRef"].grid("EmuTableInRef_tab")
    assert grid1
    assert grid1 == grid2


def test_grid_inconsistent_modifier():
    rec = EMuRecord(
        {"EmuDate0(+)": ["1970-01-01"], "EmuRef_tab": [1234567]}, module="emain"
    )
    with pytest.raises(ValueError, match="Inconsistent modifier within grid"):
        rec.grid("EmuRef_tab").columns


def test_grid_multiple_modifiers():
    rec = EMuRecord(
        {"EmuDate0(+)": ["1970-01-01"], "EmuRef_tab(-)": [1234567]}, module="emain"
    )
    with pytest.raises(ValueError, match="Inconsistent modifier within grid"):
        rec.grid("EmuRef_tab").columns


def test_grid_set_item(grid):
    with pytest.raises(NotImplementedError, match="Cannot set items on an EMuGrid"):
        grid[0] = None


def test_grid_insert(grid):
    with pytest.raises(NotImplementedError, match="Cannot insert into an EMuGrid"):
        grid.insert(0, None)


def test_grid_from_ungrouped(rec):
    with pytest.raises(KeyError, match=r"'emain.EmuText is not part of a group"):
        rec.grid("EmuText")


def test_rec_from_dir(xml_file, output_dir, expected_rec):
    reader = EMuReader(output_dir)
    for rec in reader:
        assert rec == expected_rec


def test_rec_from_json(xml_file, output_dir, expected_rec):

    reader = EMuReader(output_dir)
    for rec in reader:
        rec_from_xml = EMuRecord(rec, module=reader.module)

    simple_rec = EMuRecord({"irn": 1234567}, module="emain")
    records = [rec_from_xml] + [simple_rec] * 5000
    write_import(records, output_dir / "xmldata_5000.xml", kind="emu")

    xml_path = str(output_dir / "xmldata_5000.xml")
    json_path = str(output_dir / "xmldata_5000.json")

    reader = EMuReader(xml_path, json_path=json_path)
    for rec in reader:
        rec_from_json = EMuRecord(rec, module=reader.module)
        break

    rec_from_json_chunked = None
    for rec in reader.from_json(chunk_size=8192):
        if not rec_from_json_chunked:
            rec_from_json_chunked = EMuRecord(rec, module=reader.module)

    assert rec_from_json == rec_from_xml
    assert rec_from_json_chunked == rec_from_xml


def test_rec_from_zip(xml_file, output_dir, expected_rec):
    path = str(output_dir / "xmldata.zip")
    with zipfile.ZipFile(path, "w") as f:
        f.write(xml_file, arcname="xmldata_1.xml")
        f.write(xml_file, arcname="xmldata_2.xml")
    reader = EMuReader(path)
    for rec in reader:
        assert rec == expected_rec


def test_rec_round_trip(rec, output_dir):
    path = str(output_dir / "import.xml")
    write_import([rec], path, kind="emu")
    reader = EMuReader(path)
    for rec_ in reader:
        assert EMuRecord(rec_, module=reader.module) == rec


def test_report_progress(xml_file, output_dir, capsys):
    reader = EMuReader(output_dir)
    for i, rec in enumerate(reader):
        reader.report_progress("count", 1)
        if i:
            assert re.match(
                r"\d+(,\d{3})* records processed \(t", capsys.readouterr().out
            )
    assert re.match(r"\d+(,\d{3})* records processed \(total=", capsys.readouterr().out)


def test_write_import_invalid_kind(rec, output_dir):
    with pytest.raises(ValueError, match="kind must be one of"):
        write_import([rec], str(output_dir / "import.xml"), kind="invalid")


def test_group(rec, output_dir):
    rec.schema.validate_paths = False
    path = str(output_dir / "group.xml")
    write_group([rec], path, irn=1234567, name="Group")
    reader = EMuReader(path)
    for rec_ in reader:
        assert rec_ == {
            "GroupType": "Static",
            "Module": "emain",
            "Keys_tab": ["1000000"],
            "irn": "1234567",
            "GroupName": "Group",
        }
    rec.schema.validate_paths = True


def test_group_no_metadata(rec, output_dir):
    with pytest.raises(ValueError, match="Must specify at least one of irn or name"):
        write_group([rec], str(output_dir / "group.xml"))


def test_col_not_list(rec):
    with pytest.raises(TypeError, match="Columns must be lists"):
        rec["EmuTable_tab"] = ""


def test_rec_not_dict(rec):
    with pytest.raises(TypeError, match="References must be dicts"):
        rec["EmuRef"] = []


def test_rec_lazy_load_schema():
    EMuRecord.schema = None
    EMuRecord(module="ecatalogue")


def test_rec_update():
    rec = EMuRecord(
        {
            "irn": 1000000,
            "EmuTable_tab(+)": ["Text", "Text", "Text"],
            "EmuRef_tab(+)": [
                {"irn": 1000000},
                {"irn": 1000000},
                {"irn": 1000001},
            ],
        },
        module="emain",
    )
    grid = rec.grid("EmuTable_tab")
    xml = etree.tostring(rec.to_xml()).decode("utf-8")
    assert xml.count('row="+"') == len(grid) * len(grid.columns)
    for row in grid:
        assert xml.count(f'group="{row.row_id()}"') == len(grid.columns)


def test_rec_get(rec):
    assert rec.get("EmuInvalid") is None


def test_rec_getitem_path(rec):
    assert rec["EmuRef.EmuRefOnly"] == "Text"


def test_rec_setdefault(rec):
    irn = rec["irn"]
    rec.setdefault("irn", 0)
    assert rec["irn"] == irn

    del rec["irn"]
    rec.setdefault("irn", 0)
    assert rec["irn"] == 0


def test_rec_irn_from_int(rec):
    irn = 1000000
    rec["EmuRef"] = irn
    rec["EmuRef_tab"] = [irn, irn]
    assert rec["EmuRef"] == irn
    assert rec["EmuRef_tab"] == [irn, irn]


def test_rec_getitem_empty_path(rec):
    with pytest.raises(KeyError, match=r"Path not found but valid"):
        rec["EmuEmpty"]


def test_rec_getitem_invalid_path(rec):
    with pytest.raises(KeyError, match=r"Invalid path"):
        rec["EmuInvalid"]


def test_rec_getitem_no_schema(rec):
    rec.schema = None
    with pytest.raises(KeyError, match=r"Path not found: EmuInvalid"):
        rec["EmuInvalid"]


@pytest.mark.parametrize(
    "key,val,match",
    [
        ("EmuRef", "Text", r"References must be dicts"),
        ("EmuRef_tab", "1234567", r"Columns must be lists"),
        ("EmuRef_tab", ["Text"], r"Could not coerce to Integer"),
        ("EmuTable_tab", "Text", r"Columns must be lists"),
        ("EmuTable_tab", [["Text"]], r"Too many levels in a table"),
        ("EmuNestedTable_nesttab", "Text", r"Columns must be lists"),
        ("EmuNestedTable_nesttab", ["Text"], r"Columns must be lists"),
        ("EmuNestedTable_nesttab", [[["Text"]]], r"Too many levels in a nested table"),
    ],
)
def test_rec_set_invalid_type(key, val, match):
    with pytest.raises(TypeError, match=match):
        EMuRecord(module="emain")[key] = val


def test_rec_set_invalid_field():
    with pytest.raises(KeyError, match=r"Path not found:"):
        EMuRecord(module="emain")["EmuInvalid"] = 1


def test_rec_no_module():
    with pytest.raises(ValueError, match=r"Must provide module when schema is used"):
        EMuRecord()


@pytest.mark.parametrize(
    "date_string,kind,year,month,day,formatted,min_val,max_val",
    [
        ("2022-02-25", "day", 2022, 2, 25, "2022-02-25", "2022-02-25", "2022-02-25"),
        ("Feb 2022", "month", 2022, 2, None, "Feb 2022", "2022-02-01", "2022-02-28"),
        ("2022", "year", 2022, None, None, "2022", "2022-01-01", "2022-12-31"),
    ],
)
def test_dtype_date(date_string, kind, year, month, day, formatted, min_val, max_val):
    val = EMuDate(date_string)
    assert val.kind == kind
    assert val.year == year
    assert val.month == month
    assert val.day == day
    assert str(val) == formatted
    assert val.min_value == date(*[int(n) for n in min_val.split("-")])
    assert val.max_value == date(*[int(n) for n in max_val.split("-")])


@pytest.mark.parametrize(
    "date_string,expected",
    [
        ("2022-01-31", False),
        ("2022-02-01", False),
        ("2022-02-15", False),
        ("2022-02-28", False),
        ("2022-03-01", False),
        ("Jan 2022", False),
        ("Feb 2022", True),
        ("Mar 2022", False),
        ("2021", False),
        ("2022", False),
        ("2023", False),
    ],
)
def test_dtype_date_eq(date_string, expected):
    assert (EMuDate("Feb 2022") == date_string) == expected


@pytest.mark.parametrize(
    "date_string,expected",
    [
        ("2022-01-31", True),
        ("2022-02-01", True),
        ("2022-02-15", True),
        ("2022-02-28", True),
        ("2022-03-01", True),
        ("Jan 2022", True),
        ("Feb 2022", False),
        ("Mar 2022", True),
        ("2021", True),
        ("2022", True),
        ("2023", True),
    ],
)
def test_dtype_date_ne(date_string, expected):
    assert (EMuDate("Feb 2022") != date_string) == expected


@pytest.mark.parametrize(
    "date_string,expected",
    [
        ("2022-01-31", False),
        ("2022-02-01", False),
        ("2022-02-15", False),
        ("2022-02-28", False),
        ("2022-03-01", True),
        ("Jan 2022", False),
        ("Feb 2022", False),
        ("Mar 2022", True),
        ("2021", False),
        ("2022", False),
        ("2023", True),
    ],
)
def test_dtype_lt(date_string, expected):
    assert (EMuDate("Feb 2022") < date_string) == expected


@pytest.mark.parametrize(
    "date_string,expected",
    [
        ("2022-01-31", False),
        ("2022-02-01", True),
        ("2022-02-15", True),
        ("2022-02-28", True),
        ("2022-03-01", True),
        ("Jan 2022", False),
        ("Feb 2022", True),
        ("Mar 2022", True),
        ("2021", False),
        ("2022", True),
        ("2023", True),
    ],
)
def test_dtype_le(date_string, expected):
    assert (EMuDate("Feb 2022") <= date_string) == expected


@pytest.mark.parametrize(
    "date_string,expected",
    [
        ("2022-01-31", True),
        ("2022-02-01", False),
        ("2022-02-15", False),
        ("2022-02-28", False),
        ("2022-03-01", False),
        ("Jan 2022", True),
        ("Feb 2022", False),
        ("Mar 2022", False),
        ("2021", True),
        ("2022", False),
        ("2023", False),
    ],
)
def test_dtype_gt(date_string, expected):
    assert (EMuDate("Feb 2022") > date_string) == expected


@pytest.mark.parametrize(
    "date_string,expected",
    [
        ("2022-01-31", True),
        ("2022-02-01", True),
        ("2022-02-15", True),
        ("2022-02-28", True),
        ("2022-03-01", False),
        ("Jan 2022", True),
        ("Feb 2022", True),
        ("Mar 2022", False),
        ("2021", True),
        ("2022", True),
        ("2023", False),
    ],
)
def test_dtype_ge(date_string, expected):
    assert (EMuDate("Feb 2022") >= date_string) == expected


def test_dtype_invalid_comp():
    assert (EMuFloat(0) == None) == False
    assert (EMuFloat(0) != None) == True


@pytest.mark.parametrize(
    "date_string,expected",
    [
        ("2022-01-31", False),
        ("2022-02-01", True),
        ("2022-02-15", True),
        ("2022-02-28", True),
        ("2022-03-01", False),
        ("Jan 2022", False),
        ("Feb 2022", True),
        ("Mar 2022", False),
        ("2021", False),
        ("2022", False),
        ("2023", False),
        ((2022, 2, 1), True),
    ],
)
def test_dtype_contains(date_string, expected):
    assert (date_string in EMuDate("Feb 2022")) == expected


@pytest.mark.parametrize(
    "date_string,exp_str,exp_emu",
    [
        ("99999-01-01", "99999-01-01", "99999-01-01"),
        ("99999-01-", "Jan 99999", "Jan 99999"),
        ("Jan 99999", "Jan 99999", "Jan 99999"),
        ("99999", "99999", "99999"),
        (99999, "99999", "99999"),
        ("-99999-01-01", "-99999-01-01", "99999-01-01 BC"),
        ("-99999-01-", "Jan -99999", "Jan 99999 BC"),
        ("Jan -99999", "Jan -99999", "Jan 99999 BC"),
        ("-99999", "-99999", "99999 BC"),
        (-99999, "-99999", "99999 BC"),
        ("99999-01-01 BC", "-99999-01-01", "99999-01-01 BC"),
        ("99999-01- b.c.", "Jan -99999", "Jan 99999 BC"),
        ("Jan 99999 bce", "Jan -99999", "Jan 99999 BC"),
        ("99999 B.C.", "-99999", "99999 BC"),
        (99, "0099", "0099 AD"),
        (9, "0009", "0009 AD"),
        (0, "0000", "0000 AD"),
        (-9, "-0009", "0009 BC"),
        (-99, "-0099", "0099 BC"),
    ],
)
def test_dtype_date_out_of_range(date_string, exp_str, exp_emu):
    emu_date = EMuDate(date_string)
    assert str(emu_date) == exp_str
    assert emu_date.emu_str() == exp_emu


def test_dtype_date_parse_same_class():
    emu_date = EMuDate("1970-01-01")
    assert emu_date == EMuDate(emu_date)


def test_dtype_date_operations():
    val = EMuDate("Jan 1970")
    assert val + timedelta(days=1) == (EMuDate("1970-01-02"), EMuDate("1970-02-01"))
    assert val - timedelta(days=1) == (EMuDate("1969-12-31"), EMuDate("1970-01-30"))
    assert EMuDate("1970-01-02") - EMuDate("1970-01-01") == timedelta(days=1)


def test_dtype_date_parse_failed():
    with pytest.raises(ValueError, match="Could not parse date:"):
        EMuDate("Jan 1, 1970")


def test_dtype_date_invalid_directive():
    date = EMuDate("1970-01-")
    with pytest.raises(ValueError, match=r"Invalid directives for \(1970, 1, None\)"):
        date.strftime("%Y-%m-%d")


@pytest.mark.parametrize("attr", ["min_value", "max_value"])
def test_dtype_date_bad_kind(attr):
    date = EMuDate("1970-01-01")
    date.kind = None
    with pytest.raises(ValueError, match="Invalid kind:"):
        getattr(date, attr)


def test_dtype_date_to_datetime():
    date = EMuDate("1970-01-01")
    time = EMuTime("15:00")
    assert date.to_datetime(time) == datetime(1970, 1, 1, 15, 0)
    assert time.to_datetime(date) == datetime(1970, 1, 1, 15, 0)


@pytest.mark.parametrize(
    "date_tuple,expected",
    [
        ((1970, 1, 1), "1970-01-01"),
        ((1970, 1, None), "Jan 1970"),
        ((1970, None, None), "1970"),
    ],
)
def test_dtype_date_tuples(date_tuple, expected):
    assert str(EMuDate(date_tuple)) == expected
    assert str(EMuDate(ExtendedDate(*date_tuple))) == expected


def test_dtype_date_setters():
    date = EMuDate("1970-01-01")
    date.year = 1971
    date.month = 2
    date.day = 2
    assert str(date) == "1971-02-02"


@pytest.mark.parametrize(
    "time_string",
    [
        "1500",
        "15:00",
        "3:00 PM",
        "0300 PM",
        "15:00 UTC-0700",
        "15:00 -0700",
        "3:00 PM UTC-0700",
        "3:00 PM -0700",
        time(hour=15, minute=0),
        EMuTime("1500"),
    ],
)
def test_dtype_time(time_string):
    time = EMuTime(time_string)
    assert time.hour == 15
    assert time.minute == 0


@pytest.mark.parametrize(
    "val,fmt,expected",
    [
        ("1.10", None, "1.10"),
        (1.10, "{:.2f}", "1.10"),
        (EMuFloat("1.10"), None, "1.10"),
        ("1.", None, "1"),
    ],
)
def test_dtype_float(val, fmt, expected):
    assert str(EMuFloat(val, fmt)) == expected


def test_dtype_float_format():
    val = EMuFloat("1.10")
    assert "{}".format(val) == "1.10"
    assert "{:.1f}".format(val) == "1.1"
    assert "{:.3f}".format(val) == "1.100"

    assert f"{val}" == "1.10"
    assert f"{val:.1f}" == "1.1"
    assert f"{val:.3f}" == "1.100"


def test_dtype_float_operations():
    val = EMuFloat("0.1200")

    assert val + 1 == pytest.approx(1.12)
    assert val - 0.12 == pytest.approx(0)
    assert val * 2 == pytest.approx(0.24)
    assert val / 2 == pytest.approx(0.06)
    assert val // 2 == pytest.approx(0)
    assert val % 0.12 == pytest.approx(0)
    assert divmod(val, 0.12) == (1, 0)
    assert val ** 2 == pytest.approx(0.0144)

    assert val == 0.12
    assert val != 0.121
    assert val < 1
    assert val <= 0.12
    assert val > 0
    assert val >= 0.12

    assert str(val) == "0.1200"

    val %= 1
    assert val == pytest.approx(0.12)
    val += 1
    assert val == pytest.approx(1.12)
    val -= 1
    assert val == pytest.approx(0.12)
    val *= 2
    assert val == pytest.approx(0.24)
    val /= 2
    assert val == pytest.approx(0.12)
    val **= 2
    assert val == pytest.approx(0.0144)
    val //= 2
    assert val == pytest.approx(0)


def test_dtype_float_sigfigs():
    val = EMuFloat("0.1200")
    assert str(val + EMuFloat("1.00")) == "1.1200"
    assert str(val * EMuFloat("2.00")) == "0.24"


def test_dtype_float_type_conversions():
    val = EMuFloat("0.1200")
    assert str(val) == str(EMuFloat(0.1200, "{:.4f}"))
    assert int(val) == 0
    assert float(val) == 0.12


def test_dtype_float_no_format():
    with pytest.raises(ValueError, match="Must provide fmt when passing a float"):
        EMuFloat(0.12)


def test_dtype_float_contain_no_range():
    with pytest.raises(ValueError, match="EMuFloat is not a range"):
        1 in EMuFloat("0.12")


@pytest.mark.parametrize(
    "val,fmt",
    [
        ("45°30'15''N", None),
        ("45 30 15 North", None),
        ("N 45 30 15", None),
        ("45 30.25 N", None),
        ("45.5042", None),
        (45.5042, "{:.4f}"),
    ],
)
def test_dtype_latitude(val, fmt):
    lat = EMuLatitude(val, fmt=fmt)
    assert float(lat) == pytest.approx(45.5042)
    assert int(lat) == 45
    assert lat.to_dec() == "45.5042"
    assert lat.to_dms() == "45 30 15 N"


@pytest.mark.parametrize(
    "val,fmt",
    [
        ("45°30'15''W", None),
        ("45 30 15 West", None),
        ("W 45 30 15", None),
        ("45 30.25 W", None),
        ("-45.5042", None),
        (-45.5042, "{:.4f}"),
    ],
)
def test_dtype_longitude(val, fmt):
    lng = EMuLongitude(val, fmt=fmt)
    assert float(lng) == pytest.approx(-45.5042)
    assert int(lng) == -45
    assert lng.to_dec() == "-45.5042"
    assert lng.to_dms() == "45 30 15 W"


@pytest.mark.parametrize(
    "val,unc_m,expected_dms,expected_dec",
    [
        ("45 30 15 N", 10, "45 30 15 N", "45.5042"),
        ("45 30 15 N", 20, "45 30 15 N", "45.5042"),
        ("45 30 15 N", 50, "45 30.3 N", "45.504"),
        ("45 30 15 N", 90, "45 30.3 N", "45.504"),
        ("45 30 15 N", 100, "45 30.3 N", "45.504"),
        ("45 30 15 N", 200, "45 30.3 N", "45.504"),
        ("45 30 15 N", 500, "45 30 N", "45.50"),
        ("45 30 15 N", 900, "45 30 N", "45.50"),
        ("45 30 15 N", 1000, "45 30 N", "45.50"),
    ],
)
def test_dtype_coord_rounding(val, unc_m, expected_dms, expected_dec):
    lat = EMuLatitude(val)
    assert lat.to_dms(unc_m) == expected_dms
    assert lat.to_dec(unc_m) == expected_dec


@pytest.mark.parametrize("coord_class", [EMuLatitude, EMuLongitude])
def test_dtype_coord_invalid(coord_class):
    with pytest.raises(ValueError, match=r"Invalid coordinate"):
        coord_class("45 30 15 7.5 N")


@pytest.mark.parametrize(
    "coord_class,val", [(EMuLatitude, "90.1"), (EMuLongitude, "-180.1")]
)
def test_dtype_coord_out_of_bounds(coord_class, val):
    with pytest.raises(ValueError, match=r"Coordinate out of bounds"):
        coord_class(val)


def test_dtype_coord_to_dms_too_precise():
    with pytest.raises(ValueError, match=r"unc_m cannot be smaller"):
        EMuLatitude("45 30 15 N").to_dms(1)


def test_dtype_coord_to_dec_too_precise():
    with pytest.raises(ValueError, match=r"unc_m cannot be smaller"):
        EMuLongitude("45 30 15 E").to_dec(1)


def test_dtype_coord_unsigned():
    with pytest.raises(ValueError, match=r"Could not parse as EMuLatitude"):
        EMuLatitude("45 30 15")


@pytest.mark.parametrize(
    "field,expected",
    [
        ("AtomField", False),
        ("AtomFieldRef", False),
        ("TableField0", True),
        ("TableField_nesttab", True),
        ("TableField_nesttab_inner", True),
        ("TableFieldRef_tab", True),
        ("TableField_tab(+)", True),
        ("TableField_tab(-)", True),
        ("TableField_tab(=)", True),
        ("TableField_tab(1+)", True),
        ("TableField_tab(12-)", True),
        ("TableField_tab(123=)", True),
    ],
)
def test_is_tab(field, expected):
    assert is_tab(field) == expected


@pytest.mark.parametrize(
    "field,expected",
    [
        ("AtomField", False),
        ("AtomFieldRef", False),
        ("TableField0", False),
        ("TableField_nesttab", True),
        ("TableField_nesttab_inner", False),
        ("TableFieldRef_tab", False),
        ("TableField_tab(+)", False),
        ("TableField_tab(-)", False),
        ("TableField_tab(=)", False),
        ("TableField_tab(1+)", False),
        ("TableField_tab(12-)", False),
        ("TableField_tab(123=)", False),
    ],
)
def test_is_nesttab(field, expected):
    assert is_nesttab(field) == expected


@pytest.mark.parametrize(
    "field,expected",
    [
        ("AtomField", False),
        ("AtomFieldRef", False),
        ("TableField0", False),
        ("TableField_nesttab", False),
        ("TableField_nesttab_inner", True),
        ("TableFieldRef_tab", False),
        ("TableField_tab(+)", False),
        ("TableField_tab(-)", False),
        ("TableField_tab(=)", False),
        ("TableField_tab(1+)", False),
        ("TableField_tab(12-)", False),
        ("TableField_tab(123=)", False),
    ],
)
def test_is_nesttab_inner(field, expected):
    assert is_nesttab_inner(field) == expected


@pytest.mark.parametrize(
    "field,expected",
    [
        ("AtomField", False),
        ("AtomFieldRef", True),
        ("TableField0", False),
        ("TableField_nesttab", False),
        ("TableField_nesttab_inner", False),
        ("TableFieldRef_tab", True),
        ("TableFieldRef_tab(+)", True),
        ("TableFieldRef_tab(-)", True),
        ("TableFieldRef_tab(=)", True),
        ("TableFieldRef_tab(1+)", True),
        ("TableFieldRef_tab(12-)", True),
        ("TableFieldRef_tab(123=)", True),
    ],
)
def test_is_ref(field, expected):
    assert is_ref(field) == expected


@pytest.mark.parametrize(
    "field,expected",
    [
        ("AtomField", False),
        ("AtomFieldRef", False),
        ("TableField0", False),
        ("TableField_nesttab", False),
        ("TableField_nesttab_inner", False),
        ("TableFieldRef_tab", True),
        ("TableField_tab", False),
        ("TableFieldRef_tab(+)", True),
        ("TableFieldRef_tab(-)", True),
        ("TableFieldRef_tab(=)", True),
        ("TableFieldRef_tab(1+)", True),
        ("TableFieldRef_tab(12-)", True),
        ("TableFieldRef_tab(123=)", True),
    ],
)
def test_is_ref_tab(field, expected):
    assert is_ref_tab(field) == expected


@pytest.mark.parametrize(
    "field,expected",
    [
        ("AtomField", "AtomField"),
        ("AtomFieldRef", "AtomFieldRef"),
        ("TableField0", "TableField"),
        ("TableField_nesttab", "TableField"),
        ("TableField_nesttab_inner", "TableField"),
        ("TableFieldRef_tab", "TableFieldRef"),
        ("TableField_tab", "TableField"),
        ("TableField_tab(+)", "TableField"),
        ("TableField_tab(-)", "TableField"),
        ("TableField_tab(=)", "TableField"),
        ("TableField_tab(1+)", "TableField"),
        ("TableField_tab(12-)", "TableField"),
        ("TableField_tab(123=)", "TableField"),
    ],
)
def test_strip_tab(field, expected):
    assert strip_tab(field) == expected


@pytest.mark.parametrize(
    "field",
    [
        "TableField0",
        "TableField_nesttab",
        "TableField_nesttab_inner",
        "TableFieldRef_tab",
        "TableField_tab",
    ],
)
def test_mods(field):
    for mod in ["+", "-", "="]:
        for num in range(5):
            mod = f"{2 ** num if num else ''}{mod}"
            field_with_mod = f"{field}({mod})"
            assert has_mod(field_with_mod)
            assert strip_mod(field_with_mod) == field
            assert get_mod(field_with_mod) == mod


def test_mod_on_atom():
    with pytest.raises(ValueError, match=r"Update modifier found on an atomic"):
        has_mod("AtomField(+)")


def test_mod_invalid():
    with pytest.raises(ValueError, match=r"Invalid modifier"):
        get_mod("AtomField(*)")
