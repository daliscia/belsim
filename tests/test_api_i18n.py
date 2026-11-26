"""The `lang` query parameter / Accept-Language header on the endpoints that serve text, the
regional metadata of /api/meta and the regional series of the compact endpoints."""
import pytest
from fastapi.testclient import TestClient

from belsim import i18n
from belsim.api import service as svc
from belsim.api.app import create_app
from belsim.api.units import unit_kind
from belsim.core import dims


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_meta_in_french_and_dutch(client):
    en = client.get("/api/meta").json()
    fr = client.get("/api/meta", params={"lang": "fr"}).json()
    nl = client.get("/api/meta", headers={"Accept-Language": "nl-BE,nl;q=0.9,en;q=0.5"}).json()
    assert (en["lang"], fr["lang"], nl["lang"]) == ("en", "fr", "nl")
    assert en["langs"] == ["en", "fr", "nl"]
    assert fr["labels"]["regions"] == {"FLA": "Flandre", "WAL": "Wallonie", "BRU": "Bruxelles"}
    assert nl["labels"]["regions"] == {"FLA": "Vlaanderen", "WAL": "Wallonië", "BRU": "Brussel"}
    # same ids in every language; group_id keeps the English group as a stable identifier
    assert set(fr["revenue_lines"]) == set(en["revenue_lines"]) == set(nl["revenue_lines"])
    changed = 0
    for lid, meta in en["revenue_lines"].items():
        assert fr["revenue_lines"][lid]["group_id"] == meta["group"] == meta["group_id"]
        changed += fr["revenue_lines"][lid]["label"] != meta["label"]
    assert changed > 0.8 * len(en["revenue_lines"])
    assert set(fr["series_labels"]) == set(en["series_labels"]) == set(en["units"])
    assert fr["series_labels"]["pop.region.WAL"] == "Wallonie"


def test_unknown_language_falls_back_to_english(client):
    en = client.get("/api/meta").json()
    xx = client.get("/api/meta", params={"lang": "de"}, headers={"Accept-Language": "de"}).json()
    assert xx["lang"] == "en" and xx["labels"] == en["labels"]
    # the query parameter wins over the header
    assert client.get("/api/meta", params={"lang": "en"}, headers={"Accept-Language": "fr"}).json()["lang"] == "en"


def test_params_translated_same_values(client):
    en = {p["name"]: p for p in client.get("/api/params").json()["params"]}
    fr = {p["name"]: p for p in client.get("/api/params", params={"lang": "fr"}).json()["params"]}
    assert set(en) == set(fr)
    differ = sum(1 for n in en if en[n]["description"] and fr[n]["description"] != en[n]["description"])
    assert differ > 0.9 * len(en)
    for n in en:
        assert fr[n]["current"] == en[n]["current"] and fr[n]["source"] == en[n]["source"]
        assert fr[n]["description"], n          # never empty: English when untranslated
    regional = [n for n, p in en.items() if p["regional"]]
    for n in regional:
        assert isinstance(en[n]["current"], dict) and en[n]["current"]
    if "labour.nairu" in en:
        assert en["labour.nairu"]["regional"] and set(en["labour.nairu"]["current"]) == set(dims.REGIONS)
    assert not en["vat.rate_standard"]["regional"]


def test_measure_library_translated_keeps_ids(client):
    en = client.get("/api/measures").json()["measures"]
    nl = client.get("/api/measures", params={"lang": "nl"}).json()["measures"]
    assert [m["id"] for m in en] == [m["id"] for m in nl]
    for a, b in zip(en, nl):
        assert b["name_en"] == a["name"] and b["changes"] == a["changes"] and b["tags"] == a["tags"]
        assert set(b["tag_labels"]) == set(b["tags"])
        assert set(b["regions"]) <= set(dims.REGIONS)
    assert sum(a["name"] != b["name"] for a, b in zip(en, nl)) > 0.8 * len(en)
    for m in en:
        if "regional" in m["tags"]:
            assert m["regions"], m["name"]


def test_payslip_options_and_distribution_translated(client):
    en = client.get("/api/payslip/options").json()
    fr = client.get("/api/payslip/options", params={"lang": "fr"}).json()
    assert [k["id"] for k in en["kinds"]] == [k["id"] for k in fr["kinds"]]
    assert [h["id"] for h in en["households"]] == [h["id"] for h in fr["households"]]
    assert any(a["label"] != b["label"] for a, b in zip(en["households"], fr["households"]))
    assert {k: v for k, v in en.items() if k not in ("kinds", "households", "household_fields")} == \
           {k: v for k, v in fr.items() if k not in ("kinds", "households", "household_fields")}

    body = {"scenario": {}, "year": 2030}
    d_en = client.post("/api/distribution", json=body).json()
    d_nl = client.post("/api/distribution", json=body, params={"lang": "nl"}).json()
    assert [h["id"] for h in d_en["households"]] == [h["id"] for h in d_nl["households"]]
    assert sum(a["label"] != b["label"] for a, b in zip(d_en["households"], d_nl["households"])) > 0.8 * len(d_en["households"])
    assert all(b["group_id"] == a["group"] for a, b in zip(d_en["households"], d_nl["households"]))
    assert d_en["deciles"] == d_nl["deciles"] and d_en["summary"] == d_nl["summary"]
    # the cached English answer is not altered by a translated request
    assert client.post("/api/distribution", json=body).json()["households"][0]["label"] == d_en["households"][0]["label"]


def test_translation_helpers_never_fail_on_odd_input():
    assert svc.localise_distribution(None, "fr") is None
    assert svc.localise_distribution({"households": "x", "meta": 3}, "fr") == {"households": "x", "meta": 3}
    assert svc.localise_payslip_options({"kinds": None}, "nl")["kinds"] is None
    m = svc.localise_measure({"name": "No such measure", "description": "d", "tags": ["zzz"], "changes": []}, "fr")
    assert m["name"] == "No such measure" and m["description"] == "d" and m["tag_labels"] == {"zzz": "zzz"}


def test_regional_units():
    for key, kind in {
        "regional.population.FLA": "persons", "regional.unemployment_rate.WAL": "rate",
        "regional.employment_rate_20_64.BRU": "rate", "regional.gdp_per_capita.FLA": "eur",
        "regional.household_income.WAL": "eur_mn", "regional.gov_balance_pct_gdp.BRU": "pct",
        "regional.gov_debt.FLA": "eur_mn", "regional.revenue.pit.WAL": "eur_mn",
        "regional.some_new_amount.FLA": "eur_mn", "regional.some_new_rate.FLA": "rate",
        "fiscal.balance.walloon_region": "eur_mn",
    }.items():
        assert unit_kind(key) == kind, key


def test_meta_regional_metadata(client):
    m = client.get("/api/meta").json()
    regional = [k for k in m["units"] if k.startswith("regional.")]
    assert regional, "the model publishes no regional series"
    metrics = {".".join(k.split(".")[1:-1]) for k in regional}
    assert set(m["regional_metrics"]) == metrics
    assert "population" in metrics
    for eid, info in m["entity_detail"].items():
        assert info["parent"] in dims.ENTITIES and info["label"] and m["labels"]["entities"][eid] == info["label"]


def test_batch_and_effects_carry_regional_headlines(client):
    scen = {"name": "vat", "measures": [{"name": "VAT", "changes": [{"param": "vat.rate_standard", "value": 0.22, "start": 2027}]}]}
    r = client.post("/api/run_batch", json={"scenarios": [scen], "options": {"end": 2030}}).json()
    assert set(svc.HEADLINE_KEYS) & set(r["keys"]) == set(k for k in svc.HEADLINE_KEYS if k in r["baseline"])
    regional = [k for k in r["keys"] if k.startswith("regional.")]
    base_regional = [k for k in svc.baseline_series(2030) if svc.REGIONAL_HEADLINE.match(k)]
    assert sorted(regional) == sorted(base_regional)
    for k in regional:
        assert len(r["scenarios"][0]["series"][k]) == len(r["years"])


def test_region_keyed_change_runs(client):
    """A measure for one region is a Change with `key` = the region code."""
    params = {p["name"]: p for p in client.get("/api/params").json()["params"]}
    name = next((n for n in ("labour.nairu", "family.allowance_per_child") if n in params and params[n]["regional"]), None)
    if name is None:
        pytest.skip("no region-keyed parameter in this model")
    cur = params[name]["current"]["WAL"]
    scen = {"name": "regional", "measures": [{"name": "Wallonia only", "changes": [
        {"param": name, "value": cur * 0.9, "start": 2027, "key": "WAL"}]}], "options": {"end": 2032}}
    r = client.post("/api/run", json=scen)
    assert r.status_code == 200, r.text
    bad = client.post("/api/run", json={**scen, "measures": [{"name": "x", "changes": [
        {"param": name, "value": 1, "start": 2027, "key": "XXX"}]}]})
    assert bad.status_code == 422


def test_translate_is_used_for_validation_rows(monkeypatch):
    rep = {"baseline": [{"id": "a", "description": "A", "unit": "% of GDP"}],
           "measures": [{"id": "b", "description": "B", "unit": "EUR mn", "label": "real GDP, year t"}],
           "not_found": [{"id": "c", "description": "C"}], "problems": [], "meta": {"conventions": "x"}}
    monkeypatch.setattr(svc, "_validation_report", lambda: rep)
    assert svc.validation_report("en") is rep
    fr = svc.validation_report("fr")
    assert fr["baseline"][0]["unit"] == i18n.translate("unit", "% of GDP", "fr", "% of GDP")
    assert fr["baseline"][0]["description"] == "A" and fr["not_found"][0]["description"] == "C"   # unknown ids: English
    assert rep["baseline"][0]["unit"] == "% of GDP"                                               # source untouched
