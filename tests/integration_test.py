# Integration tests for the EMuAPI class and associated functions

import tomllib
import re
from pathlib import Path

import pytest

from xmu import (
    EMuAPI,
    EMuAPIParser,
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
    stemmed,
)


@pytest.fixture
def api():
    config_path = Path(__file__).parent.parent / "emurestapi.toml"
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


def test_refresh_token(api):
    api.get_token(refresh=True)


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
        if rec["NamLast"] == "Smythe":
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
        [range_(gt=80, lt=90), range_(gt=-90, lt=-80)],
        range_(gt=[80, -90], lt=[90, -80]),
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
            {"data.DarLatitude": {"range": {"gt": 80, "lt": 90}}},
            {"data.DarLatitude": {"range": {"gt": -90, "lt": -80}}},
        ]
    }
    assert len(resp)
    # Must find matches for both conditions
    found = {}
    for rec in resp:
        if 80 < rec["DarLatitude"] <= 90:
            found["N"] = True
        elif -90 < rec["DarLatitude"] < -80:
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


def test_search_with_parser(api):
    api.parser = EMuAPIParser()
    resp = api.search(
        "ecollectionevents",
        select=["LatLatitude_nesttab"],
        filter_={"LatCentroidLatitude0": ">80"},
        limit=1,
    )
    rec = EMuRecord(resp.first(), module="ecollectionevents")
    try:
        assert isinstance(rec["LatLatitude_nesttab"][0][0], EMuLatitude)
    except AssertionError:
        raise
    finally:
        api.parser = None


def test_retrieve_with_parser(api):
    api.parser = EMuAPIParser()
    resp_search = api.search(
        "ecollectionevents",
        filter_={"LatCentroidLatitude0": ">80"},
        limit=1,
    )
    resp_retrieve = api.retrieve("ecollectionevents", resp_search.first()["irn"])
    rec = EMuRecord(resp_retrieve.first(), module="ecollectionevents")
    try:
        assert isinstance(rec["LatLatitude_nesttab"][0][0], EMuLatitude)
    except AssertionError:
        raise
    finally:
        api.parser = None


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
