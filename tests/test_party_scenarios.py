"""Party-programme scenarios (scenarios/parties/*.yaml): the measures each party submitted to
the Federal Planning Bureau's 2024 costing, as translated by belsim.api.party_build."""
from __future__ import annotations

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from belsim.api import parties as P
from belsim.api import party_build as B
from belsim.api import service as svc
from belsim.api.app import app
from belsim.core import engine
from belsim.core.params import Change, Measure, ParamSet, Scenario

FILES = sorted(B.PARTY_DIR.glob("*.yaml"))
IDS = [p.stem for p in FILES]
END = 2035
STATUSES = set(B.STATUSES)
METHODS = {"pinned", "sized", "baseline", "not_costed", "zero_impulse", "no_lever"}


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def pset():
    return ParamSet.load()


@pytest.fixture(scope="module")
def base(pset):
    return engine.run(None, end=END, params=pset)


def load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def scenario(doc, only=None) -> Scenario:
    ms = [Measure(m["name"], [Change(**c) for c in m["changes"]]) for m in doc["measures"]
          if m["changes"] and (only is None or m["status"] in only)]
    return Scenario(doc["id"], ms)


def test_the_twelve_parties_of_the_costing_are_there():
    assert set(IDS) == {meta["id"] for meta in B.PARTIES.values()}
    assert len(IDS) == 12


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_file_shape(path, pset):
    doc = load(path)
    assert doc["kind"] == "party" and doc["id"] == path.stem
    assert doc["community"] in P.COMMUNITIES
    assert doc["source"]["url"].startswith("https://www.dc2024.be/")
    assert set(doc["official"]["impulse"]) == set(B.PROGRAMME_YEARS)
    fiches = [m["fiche"] for m in doc["measures"]]
    assert len(fiches) == len(set(fiches))
    for m in doc["measures"]:
        where = f"{doc['id']} {m['fiche']}"
        assert m["status"] in STATUSES and m["method"] in METHODS, where
        for field in ("name", "name_fr", "name_nl", "title"):
            assert isinstance(m[field], str) and m[field].strip(), f"{where}: {field}"
        assert m["source"]["fiche"].endswith(str(m["fiche"])) and m["source"]["url"].endswith(".pdf"), where
        # only existing parameters; unmodelled measures carry no change
        for c in m["changes"]:
            assert c["param"] in pset.specs, f"{where}: unknown parameter {c['param']}"
            assert c["start"] >= 2027, where
        if m["status"] in ("not_modelled", "already_in_baseline"):
            assert m["changes"] == [], where
        else:
            assert m["changes"], where
            assert m["official_impulse"] is not None, where
        if m["official_impulse"] is None:
            assert m["status"] == "not_modelled" and m["method"] == "not_costed", where
        if m.get("lever"):
            assert m["lever"] in B.LEVERS, where
        if m["status"] == "already_in_baseline":
            assert m["baseline"] in B.BASELINE_CODES, where
    assert svc.validate_scenario(P.scenario_of(doc), pset, END) == []


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_coverage_metadata_is_consistent(path):
    doc = load(path)
    cov = doc["coverage"]
    assert cov == B.coverage(doc["measures"])
    assert sum(cov["count"].values()) == cov["measures"] == len(doc["measures"])
    assert cov["costed"] == sum(1 for m in doc["measures"] if m["official_impulse"] is not None)
    shares = sum(cov["impulse_share"].values())
    assert shares == pytest.approx(1.0, abs=1e-3) or shares == 0.0
    # the official total is the sum of the fiches
    for year in B.PROGRAMME_YEARS:
        total = sum((m["official_impulse"] or {}).get(year) or 0 for m in doc["measures"])
        assert doc["official"]["impulse"][year] == pytest.approx(total, abs=1)


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_runs_to_2035(path, pset, base):
    doc = load(path)
    res = engine.run(scenario(doc), end=END, baseline=base, params=pset)
    assert res.years[-1] == END
    for year in (2027, 2031, END):
        f = res.at(year).fiscal
        assert np.isfinite(f["balance"]) and np.isfinite(f["debt"]) and np.isfinite(res.at(year).labour["employment"])
    # nothing moves before the first model year of the programmes
    assert res.at(2026).fiscal["balance"] == pytest.approx(base.at(2026).fiscal["balance"], abs=1e-6)


def test_unmodelled_measures_alone_equal_the_baseline(pset, base):
    """A scenario made of the not-modelled and already-in-baseline measures changes nothing."""
    doc = load(FILES[0])
    scen = scenario(doc, only={"not_modelled", "already_in_baseline"})
    assert scen.measures == []
    listed = {"name": "listed only", "measures": [{"name": m["name"], "changes": m["changes"]} for m in doc["measures"]
                                                  if m["status"] in ("not_modelled", "already_in_baseline")]}
    assert listed["measures"], "the first party has measures without changes"
    _, b, res = svc.run_scenario(listed, END)
    for year in (2027, 2031, END):
        assert res.at(year).fiscal["balance"] == b.at(year).fiscal["balance"]
        assert res.at(year).labour["employment"] == b.at(year).labour["employment"]


def test_sized_measures_reproduce_the_official_impulse():
    """The recorded check of a sized measure is close to the scaled impulse unless the file
    says the lever saturates."""
    n = 0
    for path in FILES:
        for m in load(path)["measures"]:
            if m["method"] != "sized" or m.get("sizing_residual") is not None:
                continue
            c = m["check"]
            assert c["model_static"] == pytest.approx(c["official_impulse_scaled"], rel=0.03, abs=1.5), f"{path.stem} {m['fiche']}"
            n += 1
    assert n > 100


def test_api_lists_party_scenarios(client):
    r = client.get("/api/scenarios", params={"kind": "party", "lang": "fr"})
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "party" and "Bureau fédéral du Plan" in body["caveat"]
    assert [s["id"] for s in body["scenarios"]] != [] and {s["id"] for s in body["scenarios"]} == set(IDS)
    # grouped by language community, alphabetical inside a group
    order = [(P.COMMUNITIES.index(s["community"]), s["name"].casefold()) for s in body["scenarios"]]
    assert order == sorted(order)
    for s in body["scenarios"]:
        assert s["kind"] == "party" and s["unknown_params"] == []
        assert s["coverage"]["measures"] == len(s["measures"])
        with_changes = [m for m in s["measures"] if m["changes"]]
        assert len(s["scenario"]["measures"]) == len(with_changes)
        assert all(m["source"]["url"] and m["status"] in STATUSES for m in s["measures"])
    ps = next(s for s in body["scenarios"] if s["id"] == "ps")
    assert ps["measures"][0]["name"] != ps["measures"][0]["name_en"]      # served in French
    # the saved-scenario listing is unchanged, and can carry the parties
    assert "scenarios" in client.get("/api/scenarios").json()
    assert client.get("/api/scenarios", params={"kind": "all"}).json()["parties"]["kind"] == "party"
    assert client.get("/api/scenarios", params={"kind": "nope"}).status_code == 400
    assert client.get("/api/parties/nope").status_code == 404


def test_api_party_scenario_runs_through_the_existing_endpoints(client):
    party = client.get("/api/parties/open-vld").json()
    r = client.post("/api/run", json={**party["scenario"], "options": {"end": 2032}})
    assert r.status_code == 200, r.text
    assert r.json()["years"][-1] == 2032
    r = client.post("/api/run_batch", json={"scenarios": [party["scenario"]], "options": {"end": 2032}})
    assert r.status_code == 200 and "error" not in r.json()["scenarios"][0]


def test_api_effects_and_comparison(client):
    fx = client.get("/api/parties/defi/effects", params={"end": 2031}).json()
    doc = load(B.PARTY_DIR / "defi.yaml")
    assert [r["fiche"] for r in fx["measures"]] == [m["fiche"] for m in doc["measures"]]
    i = fx["years"].index(2031)
    for row, m in zip(fx["measures"], doc["measures"]):
        if not m["changes"]:
            assert "static" not in row and "behavioural" not in row
            continue
        assert "error" not in row, row
        assert row["static"]["fiscal.primary_balance"][i] is not None
        if m["method"] == "sized" and m.get("sizing_residual") is None and m["check"]["year"] == 2031:
            assert row["static"]["fiscal.primary_balance"][i] == pytest.approx(m["check"]["official_impulse_scaled"], rel=0.03, abs=1.5)
    assert fx["package"]["behavioural"]["fiscal.balance"][i] is not None
    cmp_ = client.get("/api/parties/compare", params={"end": 2031}).json()
    assert {s["party"] for s in cmp_["scenarios"]} == set(IDS)
    assert all("series" in s and "official" in s and "coverage" in s for s in cmp_["scenarios"])


def test_generated_validation_entries_match_the_party_files():
    from belsim import validation as V
    ids = {e["id"] for e in V.load()["measures"]}
    for path in FILES:
        for m in load(path)["measures"]:
            if m.get("validation_id"):
                assert m["validation_id"] in ids, f"{path.stem} {m['fiche']}: run party_build --validation"
