# Integration tests for the EMuAPI class and associated functions

import re
import tomllib
from pathlib import Path

import pytest

from xmu import (
    EMuAPI,
    EMuLatitude,
    EMuRecord,
    EMuSchema,
    contains,
    exact,
    gt,
    gte,
    lt,
    lte,
    is_not_null,
    is_null,
    not_,
    phonetic,
    phrase,
    range_,
    resolve_attachments,
    stemmed,
)


@pytest.fixture(scope="session")
def api():
    config_path = Path(__file__).parent / "emurestapi.toml"
    with open(config_path, "rb") as f:
        kwargs = tomllib.load(f)["params"]
    schema_path = kwargs.pop("schema_path")
    api = EMuAPI(config_path=config_path, autopage=False)
    # Tests need to interface with the API, so hack the test config to use the main
    # config file and reload with the real schema
    api.schema.config["schema_path"] = schema_path
    api.schema.config["groups"].clear()
    api.schema.config["make_visible"].clear()
    api.schema.config["lookup_no_autopopulate"].clear()
    api.schema.config["reverse_attachments"].clear()
    api.schema.config["calculated_fields"].clear()
    api.schema = EMuSchema()
    return api


@pytest.fixture(scope="session")
def emu_record():
    return EMuRecord(
        {
            "LocCountry": "United States",
            "LocProvinceStateTerritory": "Maine",
            "LocDistrictCountyShire": "Androscoggin Co.",
            "LocTownship": "Wales",
            "LatLatitude_nesttab": [["44°10′0″N"]],
            "LatLongitude_nesttab": [["70°3′54″W"]],
            "LatDatum_tab": ["WGS 84 (EPSG:4326)"],
            "NteText0": ["API test record"],
            "NteAttributedToRef_nesttab": [[{"NamFirst": "Adam", "NamLast": "Mansur"}]],
        },
        module="ecollectionevents",
    )


@pytest.fixture(scope="session")
def test_record(api, emu_record):
    return resolve_attachments(api.insert("ecollectionevents", emu_record).first())


def test_refresh_token(api):
    api.get_token(refresh=True)


@pytest.mark.skip("Temporary error searching collection events")
@pytest.mark.parametrize("term", [contains("smith"), "smith"])
def test_contains(term, api):
    resp = api.search(
        "eparties", select=["NamLast"], filter_={"NamLast": term}, limit=100
    )
    assert resp.params["filter"] == {
        "AND": [{"data.NamLast": {"contains": {"value": "smith"}}}]
    }
    assert len(resp)
    for rec in resp:
        assert "smith" in rec["NamLast"].lower()


@pytest.mark.skip("Temporary error searching collection events")
@pytest.mark.parametrize(
    "term,expected",
    [
        (exact("New York"), "New York"),
        (r"\^\"New York\"\$", "New York"),
        (r"\^Maine\$", "Maine"),
    ],
)
def test_exact(term, expected, api):
    resp = api.search(
        "ecollectionevents",
        select=["LocProvinceStateTerritory"],
        filter_={"LocProvinceStateTerritory": term},
        limit=100,
    )
    assert resp.params["filter"] == {
        "AND": [{"data.LocProvinceStateTerritory": {"exact": {"value": expected}}}]
    }
    assert len(resp)
    for rec in resp:
        assert rec["LocProvinceStateTerritory"].lower() == expected.lower()


# @pytest.mark.skip("Slow, skip for dev")
@pytest.mark.parametrize("term", [not_("smith"), r"\!smith"])
def test_not(term, api):
    resp = api.search(
        "eparties", select=["NamLast"], filter_={"NamLast": term}, limit=100
    )
    assert resp.params["filter"] == {
        "NOT": [{"data.NamLast": {"contains": {"value": "smith"}}}]
    }
    assert len(resp)
    for rec in resp:
        assert rec["NamLast"].lower() != "smith"


@pytest.mark.parametrize("term", [is_not_null(), r"\+", r"\*", True])
def test_not_null_search(term, api):
    resp = api.search(
        "eparties", select=["NamLast"], filter_={"NamLast": term}, limit=100
    )
    assert resp.params["filter"] == {
        "AND": [{"data.NamLast": {"exists": {"value": True}}}]
    }
    assert len(resp)
    for rec in resp:
        assert rec["NamLast"]


@pytest.mark.parametrize("term", [is_null(), r"\!\+", r"\!\*", False, None])
def test_null_search(term, api):
    resp = api.search(
        "eparties", select=["NamLast"], filter_={"NamLast": term}, limit=100
    )
    assert resp.params["filter"] == {
        "AND": [{"data.NamLast": {"exists": {"value": False}}}]
    }
    assert len(resp)
    for rec in resp:
        assert "NamLast" not in rec


@pytest.mark.parametrize("term", [phonetic("smith"), r"\@smith"])
def test_phonetic_search(term, api):
    resp = api.search(
        "eparties", select=["NamLast"], filter_={"NamLast": term}, limit=100
    )
    assert resp.params["filter"] == {
        "AND": [{"data.NamLast": {"phonetic": {"value": "smith"}}}]
    }
    assert len(resp)
    for rec in resp:
        if rec["NamLast"] in {"Schmidt", "Smythe"}:
            assert True
            break
    else:
        assert False


@pytest.mark.skip("Filter appears to be correct but request finds no matches")
@pytest.mark.parametrize("term", [phrase("New York"), r"\"New York\""])
def test_phrase(term, api):
    resp = api.search(
        "ecollectionevents",
        select=["LocProvinceStateTerritory"],
        filter_={"LocProvinceStateTerritory": term},
        limit=100,
    )
    assert resp.params["filter"] == {
        "AND": [{"data.LocProvinceStateTerritory": {"phrase": {"value": "New York"}}}]
    }
    assert len(resp)
    for rec in resp:
        assert "new york" in rec["LocProvinceStateTerritory"].lower()


@pytest.mark.skip("Temporary error searching collection events")
@pytest.mark.parametrize("term", [stemmed("locate"), r"\~locate"])
def test_stemmed(term, api):
    # NOTE: The query below is NMNH-specific and therefore brittle. A stemmed search
    # was found to be very, very slow without adding additional parameters, so a
    # state/province known to match a small number of records was added to the query.
    resp = api.search(
        "ecollectionevents",
        select=["LocPreciseLocation"],
        filter_={
            "LocPreciseLocation": term,
            "LocProvinceStateTerritory": "Connecticut",
        },
        limit=100,
    )
    assert resp.params["filter"] == {
        "AND": [
            {"data.LocPreciseLocation": {"stemmed": {"value": "locate"}}},
            {"data.LocProvinceStateTerritory": {"contains": {"value": "Connecticut"}}},
        ]
    }
    assert len(resp)
    for rec in resp:
        for val in re.findall(r"\bloca[a-z]+", rec["LocPreciseLocation"], flags=re.I):
            assert val.lower() in ["local", "located", "location", "locality"]


@pytest.mark.parametrize(
    "term,clause,bounds",
    [
        (range_(gt=80, lt=90), {"gt": 80, "lt": 90}, (80, 90)),
        (">80 <90", {"gt": 80, "lt": 90}, (80, 90)),
        (range_(gte=80, lte=90), {"gte": 80, "lte": 90}, (80, 90)),
        (">=80 <=90", {"gte": 80, "lte": 90}, (80, 90)),
    ],
)
def test_range(term, clause, bounds, api):
    resp = api.search(
        "ecatalogue",
        select=["DarLatitude"],
        filter_={"DarLatitude": term},
        limit=100,
    )
    assert resp.params["filter"] == {"AND": [{"data.DarLatitude": {"range": clause}}]}
    assert len(resp)
    for rec in resp:
        assert min(bounds) <= rec["DarLatitude"] <= max(bounds)


@pytest.mark.parametrize(
    "terms",
    [
        [range_(gt=60, lt=90), range_(gt=-90, lt=-60)],
        range_(gt=[60, -90], lt=[90, -60]),
    ],
)
def test_range_or_search(terms, api):
    resp = api.search(
        "ecatalogue",
        select=["DarLatitude"],
        filter_={"DarLatitude": terms},
        limit=1000,
    )
    assert resp.params["filter"] == {
        "OR": [
            {"data.DarLatitude": {"range": {"gt": 60, "lt": 90}}},
            {"data.DarLatitude": {"range": {"gt": -90, "lt": -60}}},
        ]
    }
    assert len(resp)
    # Must find matches for both conditions
    found = {}
    for rec in resp:
        if 60 < rec["DarLatitude"] <= 90:
            found["N"] = True
        elif -90 < rec["DarLatitude"] < -60:
            found["S"] = True
        else:
            assert False
    assert len(found) == 2


@pytest.mark.parametrize("term", [range_(gt=75), gt(75), r">75"])
def test_gt(term, api):
    resp = api.search(
        "ecatalogue",
        select=["DarLatitude"],
        filter_={"DarLatitude": term},
        limit=100,
    )
    assert resp.params["filter"] == {
        "AND": [{"data.DarLatitude": {"range": {"gt": 75}}}]
    }
    assert len(resp)
    for rec in resp:
        # HACK: The gt range functions treat decimals without leading zeroes as
        # integers, e.g., .112345 as 112345. This affects only a handful of records,
        # and those results are filtered from the test.
        if rec["DarLatitude"] <= 0 or rec["DarLatitude"] >= 1:
            assert rec["DarLatitude"] > 75


@pytest.mark.parametrize("term", [range_(gte=75), gte(75), r">=75"])
def test_gte(term, api):
    resp = api.search(
        "ecatalogue",
        select=["DarLatitude"],
        filter_={"DarLatitude": term},
        sort_=["DarLatitude"],
        limit=100,
    )
    assert resp.params["filter"] == {
        "AND": [{"data.DarLatitude": {"range": {"gte": 75}}}]
    }
    assert len(resp)
    for rec in resp:
        # HACK: The gt range functions treat decimals without leading zeroes as
        # integers, e.g., .112345 as 112345. This affects only a handful of records,
        # and those results are filtered from the test.
        if rec["DarLatitude"] <= 0 or rec["DarLatitude"] >= 1:
            assert rec["DarLatitude"] >= 75


@pytest.mark.parametrize("term", [range_(lt=-75), lt(-75), r"<-75"])
def test_lt(term, api):
    resp = api.search(
        "ecatalogue",
        select=["DarLatitude"],
        filter_={"DarLatitude": term},
        limit=100,
    )
    assert resp.params["filter"] == {
        "AND": [{"data.DarLatitude": {"range": {"lt": -75}}}]
    }
    assert len(resp)
    for rec in resp:
        assert rec["DarLatitude"] < -75


@pytest.mark.parametrize("term", [range_(lte=-75), lte(-75), r"<=-75"])
def test_lte(term, api):
    resp = api.search(
        "ecatalogue",
        select=["DarLatitude"],
        filter_={"DarLatitude": term},
        sort_={"DarLatitude": "desc"},
        limit=100,
    )
    assert resp.params["filter"] == {
        "AND": [{"data.DarLatitude": {"range": {"lte": -75}}}]
    }
    assert len(resp)
    for rec in resp:
        assert rec["DarLatitude"] <= -75


@pytest.mark.parametrize("term", [">80 >=90", "<90 <=80"])
def test_invalid_range(term, api):
    with pytest.raises(ValueError, match=r"Can only provide one"):
        api.search("ecatalogue", filter_={"DarLatitude": term})


def test_next_page(api):
    api.autopage = True
    resp = api.search(
        "ecatalogue",
        select=["DarStateProvince"],
        filter_={"DarStateProvince": "Maine"},
        limit=1000,
    )
    for i, _ in enumerate(resp):
        if i > 1000:
            api.autopage = False
            break
    else:
        api.autopage = False
        assert False


def test_retrieve(api):
    resp_search = api.search(
        "ecatalogue",
        filter_={"DarStateProvince": "Maine"},
        limit=1,
    )
    resp_retrieve = api.retrieve("ecatalogue", resp_search.first()["irn"])
    assert resp_search.first()["irn"] == resp_retrieve.first()["irn"]


@pytest.mark.skip("Temporary error searching collection events")
def test_emu_record_from_response(api):
    resp = api.search(
        "ecollectionevents",
        select=["LatLatitude_nesttab"],
        filter_={"LatCentroidLatitude0": ">80"},
        limit=1,
    )
    rec = EMuRecord(resp.first(), module="ecollectionevents")
    assert isinstance(rec["LatLatitude_nesttab"][0][0], EMuLatitude)


@pytest.mark.skip("Temporary error searching collection events")
def test_deferred_autoresolve(api):
    resp = api.search(
        "ecatalogue",
        select={"BioEventSiteRef": {"ColParticipantRef_tab": ["NamFullName"]}},
        filter_={"CatDepartment": "Mineral Sciences", "DarCollector": r"\+"},
        limit=100,
    )
    assert len(resp)
    for rec in resp:
        assert rec["BioEventSiteRef"]["ColParticipantRole_grp"][0]["ColParticipantRef"]


@pytest.mark.skip("Temporary error searching collection events")
def test_deferred_get(api):
    resp = api.search(
        "ecatalogue",
        select={"BioEventSiteRef": {"ColParticipantRef_tab": ["NamFullName"]}},
        filter_={"CatDepartment": "Mineral Sciences", "DarCollector": r"\+"},
        limit=1,
    )
    rec = resp.first()
    assert rec["BioEventSiteRef"].get("ColParticipantRole_grp")
    assert rec["BioEventSiteRef"].get("LocCountry") is None


@pytest.mark.skip("Temporary error searching collection events")
def test_deferred_int(api):
    resp = api.search(
        "ecatalogue",
        select={"BioEventSiteRef": {"ColParticipantRef_tab": ["NamFullName"]}},
        filter_={"CatDepartment": "Mineral Sciences", "DarCollector": r"\+"},
        limit=1,
    )
    rec = resp.first()
    irn = int(rec["BioEventSiteRef"]["irn"].split("/")[-1])
    assert irn == int(rec["BioEventSiteRef"])


def test_search_kwarg_only(api):
    with pytest.raises(
        TypeError,
        match=r"EMuAPI\.search\(\) takes 2 positional arguments but 4 were given",
    ):
        api.search("ecatalogue", ["CatNumber"], {"CatNumber": 1234})


def test_insert_with_grouped_nested_tables(api, emu_record, test_record):
    emu_record = api.flatten("ecollectionevents", emu_record.group_columns())
    test_record = api.flatten("ecollectionevents", test_record)
    test_record = {k: re.sub(r"\b0(\d)\b", r"\1", test_record[k]) for k in emu_record}
    assert test_record == emu_record


def test_edit_atomic_field(api, test_record):
    patch = {"LocTownship": "Sabattus"}
    resp = api.edit("ecollectionevents", test_record["irn"], patch)
    assert resp.first()["LocTownship"] == "Sabattus"


def test_edit_append_to_empty_table(api, test_record):
    patch = {"LatGeoreferencingNotes0(+)": ["Test"]}
    resp = api.edit("ecollectionevents", test_record["irn"], patch)
    assert resp.first()["LatGeoreferencingNotes0"] == ["Test"]


def test_edit_append_to_table(api, test_record):
    patch = {"LatDatum_tab(+)": ["WGS 84 (EPSG:4326)"]}
    resp = api.edit("ecollectionevents", test_record["irn"], patch)
    assert resp.first()["LatDatum_tab"] == ["WGS 84 (EPSG:4326)", "WGS 84 (EPSG:4326)"]


def test_edit_replace_row_in_table(api, test_record):
    patch = {"LatDatum_tab(2=)": ["NAD 83 (EPSG:4269)"]}
    resp = api.edit("ecollectionevents", test_record["irn"], patch)
    assert resp.first()["LatDatum_tab"] == ["WGS 84 (EPSG:4326)", "NAD 83 (EPSG:4269)"]


def test_edit_replace_table(api, test_record):
    patch = {"LatDatum_tab": ["WGS 84 (EPSG:4326)"]}
    resp = api.edit("ecollectionevents", test_record["irn"], patch)
    assert resp.first()["LatDatum_tab"] == ["WGS 84 (EPSG:4326)"]


def test_edit_insert_replace_table(api, test_record):
    patch = {"LatRadiusVerbatim_tab": ["10 m"]}
    resp = api.edit("ecollectionevents", test_record["irn"], patch)
    assert resp.first()["LatRadiusVerbatim_tab"] == ["10 m"]


def test_edit_append_to_grid(api, test_record):
    patch = {
        "LatLatitude_nesttab(+)": [["44°6′9″N"]],
        "LatLongitude_nesttab(+)": [["70°5′4″W"]],
    }
    resp = api.edit("ecollectionevents", test_record["irn"], patch)
    assert resp.first()["LatComment_grp"] == [
        {
            "LatComment_subgrp": [
                {
                    "LatLatitude": "44 10 00 N",
                    "LatLongitudeDM": "70 03 54 W",
                    "LatLongitudeOrig": 0,
                    "LatLatitudeDM": "44 10 00 N",
                    "LatLatitudeDecimal": 44.1667,
                    "LatLongitudeDecimal": -70.065,
                    "LatLongitude": "70 03 54 W",
                    "LatLatitudeOrig": 0,
                }
            ]
        },
        {
            "LatComment_subgrp": [
                {
                    "LatLatitude": "44 06 09 N",
                    "LatLongitudeDM": "70 05 04 W",
                    "LatLongitudeOrig": 0,
                    "LatLatitudeDM": "44 06 09 N",
                    "LatLatitudeDecimal": 44.1025,
                    "LatLongitudeDecimal": -70.0844,
                    "LatLongitude": "70 05 04 W",
                    "LatLatitudeOrig": 0,
                }
            ]
        },
    ]


@pytest.mark.xfail("Fails unexpectedly")
def test_edit_replace_grid(api, test_record):
    patch = {
        "LatLatitude_nesttab": [["44°6′9″N"]],
        "LatLongitude_nesttab": [["70°5′4″W"]],
    }
    resp = api.edit("ecollectionevents", test_record["irn"], patch)
    assert resp.first()["LatComment_grp"] == [
        {
            "LatComment_subgrp": [
                {
                    "LatLatitude": "44 06 09 N",
                    "LatLongitudeDM": "70 05 04 W",
                    "LatLongitudeOrig": 0,
                    "LatLatitudeDM": "44 06 09 N",
                    "LatLatitudeDecimal": 44.1025,
                    "LatLongitudeDecimal": -70.0844,
                    "LatLongitude": "70 05 04 W",
                    "LatLatitudeOrig": 0,
                }
            ]
        }
    ]


def test_edit_calculated_fields_after_replace(api):
    resp = api.insert("ecollectionevents", {"AquDepthFromMet": 9.2})
    rec = resp.first()
    assert rec["AquDepthFromFath"] == 5
    assert rec["AquDepthFromFt"] == 30
    assert rec["AquDepthFromMet"] == 9.2
    resp = api.edit("ecollectionevents", rec["irn"], {"AquDepthFromMet": 18.4})
    rec = resp.first()
    assert rec["AquDepthFromFath"] == 10.1
    assert rec["AquDepthFromFt"] == 60
    assert rec["AquDepthFromMet"] == 18.4


def test_edit_move(api, test_record):
    patch = [{"op": "move", "from": "/LocTownship", "path": "/LocPreciseLocation"}]
    result = api.edit("ecollectionevents", test_record["irn"], patch).first()
    assert not result.get("LocTownship")
    assert result["LocPreciseLocation"] == "Sabattus"


def test_edit_remove(api, test_record):
    rec = api.retrieve("ecollectionevents", test_record["irn"]).first()
    assert rec.get("LocDistrictCountyShire")
    patch = [{"op": "remove", "path": "/LocDistrictCountyShire"}]
    result = api.edit("ecollectionevents", test_record["irn"], patch).first()
    assert not result.get("LocDistrictCountyShire")
