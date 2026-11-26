"""HTTP endpoints of the distributional analysis (FastAPI TestClient, real engine and params)."""
import time

import pytest
from fastapi.testclient import TestClient

from belsim.api.app import create_app

END = 2030
SCEN = {
    "name": "allowance",
    "measures": [{"name": "Tax-free allowance +2000",
                  "changes": [{"param": "pit.basic_allowance", "value": 2000, "op": "add", "start": 2027}]}],
    "options": {"end": END},
}
EMPTY = {"name": "nothing", "measures": [], "options": {"end": END}}


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_distribution_shape(client):
    r = client.post("/api/distribution", json={"scenario": SCEN, "year": END})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["year"] == END and isinstance(d["id"], str)
    assert set(d) >= {"year", "meta", "households", "deciles", "indicators", "summary"}
    assert isinstance(d["meta"]["method"], str) and isinstance(d["meta"]["caveats"], list) and d["meta"]["caveats"]
    assert "microsimulation" in d["meta"]["method"]
    assert 15 <= len(d["households"]) <= 30
    for h in d["households"]:
        assert set(h) >= {"id", "label", "group", "weight", "baseline", "scenario", "delta", "rates"}
        for side in ("baseline", "scenario"):
            assert set(h[side]) >= {"gross", "net", "income_tax", "ssc_employee", "benefits", "indirect_taxes",
                                    "net_after_indirect", "net_2024_prices"}
        assert set(h["delta"]) >= {"eur", "pct", "by_cause"}
        assert set(h["delta"]["by_cause"]) == {"wages", "benefits", "contributions", "direct_taxes",
                                               "indirect_taxes_prices"}
        assert set(h["rates"]) == {"average", "marginal", "participation"}
        assert h["weight"] > 0
    # the short list is the representative subset of the full household set (same row shape)
    assert all(h["representative"] for h in d["households"])
    assert 250 <= len(d["households_all"]) <= 310
    full = {h["id"]: h for h in d["households_all"]}
    assert all(full[h["id"]] == h for h in d["households"])
    assert all(set(h) >= {"representative", "type", "type_weight", "type_delta_eur"} for h in d["households_all"])
    assert sum(h["weight"] for h in d["households_all"]) == pytest.approx(d["meta"]["households_total"], abs=100)
    assert set(d["meta"]["coverage"]) == {"income_tax", "social_contributions"}
    assert [x["decile"] for x in d["deciles"]] == list(range(1, 11))
    for x in d["deciles"]:
        assert set(x) >= {"decile", "baseline_income", "delta_eur", "delta_pct", "gainers_share", "losers_share"}
        assert 0 <= x["gainers_share"] <= 1 and 0 <= x["losers_share"] <= 1
    for k in ("gini", "s80s20", "poverty_rate"):
        assert set(d["indicators"][k]) >= {"baseline", "scenario"}
    assert d["summary"]["delta_eur_mean"] > 0
    # poverty by household type and activity status, with the observed rates
    assert set(d["poverty"]) == {"by_household_type", "by_activity_status", "other_lines"}
    for k in ("by_household_type", "by_activity_status"):
        assert len(d["poverty"][k]) >= 5
        for row in d["poverty"][k]:
            assert set(row) >= {"id", "label", "persons", "population_share", "baseline", "scenario", "delta", "observed"}
            assert 0 <= row["baseline"] <= 1 and 0 <= row["scenario"] <= 1
            assert row["observed"] is None or set(row["observed"]) >= {"value", "year", "source"}
    assert {r["id"] for r in d["poverty"]["by_household_type"]} >= {"single", "single_parent", "couple_no_children",
                                                                   "couple_with_children", "aged_65_plus"}
    assert {r["id"] for r in d["poverty"]["by_activity_status"]} >= {"employed", "unemployed", "retired", "other_inactive"}
    assert set(d["poverty"]["other_lines"]) == {"40", "50", "70"}
    # the run is the one /api/run caches
    run = client.post("/api/run", json=SCEN).json()
    assert run["id"] == d["id"]


def test_distribution_empty_scenario_and_speed(client):
    client.post("/api/distribution", json={"scenario": EMPTY, "year": END})
    t = time.perf_counter()
    r = client.post("/api/distribution", json={"scenario": EMPTY, "year": END - 1})
    assert time.perf_counter() - t < 0.5
    d = r.json()
    assert all(h["delta"]["eur"] == 0 for h in d["households"])
    assert all(x["delta_eur"] == 0 for x in d["deciles"])
    # default scenario = baseline
    r = client.post("/api/distribution", json={"year": 2026})
    assert r.status_code == 200 and r.json()["year"] == 2026


def test_distribution_errors(client):
    r = client.post("/api/distribution", json={"scenario": SCEN, "year": 2099})
    assert r.status_code == 400 and "error" in r.json()
    r = client.post("/api/distribution", json={"scenario": SCEN, "year": "soon"})
    assert r.status_code == 422 and "error" in r.json()
    bad = {"measures": [{"name": "x", "changes": [{"param": "no.such.param", "value": 1, "start": 2027}]}],
           "options": {"end": END}}
    r = client.post("/api/distribution", json={"scenario": bad, "year": END})
    assert r.status_code == 422 and "no.such.param" in r.json()["error"]
    r = client.post("/api/distribution", json={"scenario": {"measures": "many"}, "year": END})
    assert r.status_code == 422 and "error" in r.json()


def test_payslip_shape(client):
    body = {"scenario": SCEN, "year": END, "kind": "emp_private", "household": "single",
            "gross_min": 0, "gross_max": 120000, "points": 25}
    r = client.post("/api/payslip", json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    assert set(d) >= {"grid", "baseline", "scenario", "year", "kind", "household"}
    assert len(d["grid"]) == 25 and d["grid"][0] == 0 and d["grid"][-1] == 120000
    for side in ("baseline", "scenario"):
        s = d[side]
        assert set(s) >= {"net", "marginal", "breakdown_at"}
        assert len(s["net"]) == len(s["marginal"]) == len(s["breakdown_at"]) == 25
        for row in s["breakdown_at"]:
            assert set(row) >= {"gross", "employer_cost", "ssc_employer", "ssc_employee", "taxable_income", "tax",
                                "net", "allowances", "marginal_rate"}
            assert set(row["tax"]) >= {"federal", "regional", "local", "total"}
            assert row["net"] == pytest.approx(row["gross"] - row["ssc_employee"] - row["tax"]["total"]
                                               + row["allowances"]["family_allowance"], abs=0.05)
    # the higher allowance raises net pay where tax is due, never lowers it
    diff = [s - b for b, s in zip(d["baseline"]["net"], d["scenario"]["net"])]
    assert min(diff) >= -0.01 and max(diff) > 400
    # custom household with children and a partner
    body["household"] = {"adults": 2, "children": 2, "partner_kind": "emp_private", "partner_gross": 30000,
                         "region": "WAL"}
    r = client.post("/api/payslip", json=body)
    assert r.status_code == 200
    f = r.json()
    assert f["household"]["children"] == 2 and f["household"]["region"] == "WAL"
    assert f["baseline"]["net"][0] > d["baseline"]["net"][0]


def test_payslip_options_and_errors(client):
    r = client.get("/api/payslip/options")
    assert r.status_code == 200
    o = r.json()
    assert {k["id"] for k in o["kinds"]} >= {"emp_private", "emp_public", "self_employed", "pension",
                                             "unemployment", "disability"}
    assert all({"id", "label"} <= set(h) for h in o["households"]) and len(o["households"]) >= 5
    for h in o["households"]:
        r = client.post("/api/payslip", json={"scenario": EMPTY, "year": END, "household": h["id"], "points": 5})
        assert r.status_code == 200, h
    for bad in ({"kind": "wizard"}, {"household": "castle"}, {"points": 1}, {"gross_min": 10, "gross_max": 5},
                {"household": {"adults": 5}}, {"year": 1999}):
        r = client.post("/api/payslip", json={"scenario": EMPTY, "year": END, **bad})
        assert 400 <= r.status_code < 500 and "error" in r.json(), bad
