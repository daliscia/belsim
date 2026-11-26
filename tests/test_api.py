"""HTTP API tests (FastAPI TestClient). Uses the real engine and the real params directory."""
import pytest
from fastapi.testclient import TestClient

from belsim.api import service as svc
from belsim.api.app import create_app
from belsim.api.units import unit_kind
from belsim.core import dims

END = 2030  # short horizon keeps scenario runs fast


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


@pytest.fixture()
def scen_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("BELSIM_SCENARIOS_DIR", str(tmp_path))
    return tmp_path


def _scalar_param(client):
    for p in client.get("/api/params").json()["params"]:
        if p["kind"] == "scalar":
            return p
    return None


def test_meta(client):
    r = client.get("/api/meta")
    assert r.status_code == 200
    m = r.json()
    assert m["dims"]["years"][0] == dims.BASE_YEAR and m["dims"]["years"][-1] == dims.END_YEAR
    for k in ("regions", "sectors", "statuses", "jobtypes", "entities", "cons_cats"):
        assert m["dims"][k] == getattr(dims, k.upper())
        # labels cover every member; entities also label the federated entities (ENTITY_DETAIL)
        assert set(m["labels"][k]) >= set(m["dims"][k])
    assert set(m["labels"]["entities"]) == set(m["dims"]["entities"]) | set(m["entity_detail"])
    for side in ("revenue_lines", "expenditure_lines"):
        assert m[side]
        for line in m[side].values():
            assert {"label", "group", "esa", "entity_shares"} <= set(line)
    assert m["units"]["fiscal.balance"] == "eur_mn"
    assert m["units"]["fiscal.balance_pct_gdp"] == "pct"


def test_unit_kinds():
    assert unit_kind("revenue.anything") == "eur_mn"
    assert unit_kind("fiscal.balance.federal") == "eur_mn"
    assert unit_kind("macro.gdp_real") == "eur_mn_real"
    assert unit_kind("macro.gdp_real_growth") == "rate"
    assert unit_kind("macro.cpi") == "index"
    assert unit_kind("labour.unemployment_rate") == "rate"
    assert unit_kind("labour.avg_wage_private") == "eur"
    assert unit_kind("labour.wage_bill_private") == "eur_mn"
    assert unit_kind("labour.employment") == "persons"
    assert unit_kind("demo.life_expectancy_M") == "years"
    assert unit_kind("pop.total") == "persons"
    assert unit_kind("something.new") == "number"


def test_params(client):
    r = client.get("/api/params")
    assert r.status_code == 200
    body = r.json()
    assert body["years"][0] == dims.BASE_YEAR
    assert sum(g["count"] for g in body["groups"]) == len(body["params"])
    for p in body["params"]:
        assert {"name", "description", "unit", "source", "uprate", "file", "values", "kind",
                "group", "assumption", "current"} <= set(p)
        if p["kind"] == "scalar":
            assert len(p["baseline"]) == len(body["years"])


def test_baseline(client):
    r = client.get("/api/baseline")
    assert r.status_code == 200
    b = r.json()
    assert b["years"] == list(range(dims.BASE_YEAR, dims.END_YEAR + 1))
    for key in ("macro.gdp_nominal", "fiscal.balance", "pop.total"):
        assert len(b["series"][key]) == len(b["years"])
    assert client.get("/api/baseline").json()["series"]["pop.total"] == b["series"]["pop.total"]


def test_run_without_measures_has_zero_deltas(client):
    r = client.post("/api/run", json={"name": "empty", "measures": [], "options": {"end": END}})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["years"] == list(range(dims.BASE_YEAR, END + 1))
    assert set(out) >= {"id", "years", "baseline", "scenario", "delta"}
    assert out["delta"]["series"]
    worst = max(abs(v) for vals in out["delta"]["series"].values() for v in vals if v is not None)
    assert worst < 1e-6
    assert out["scenario"]["series"].keys() == out["baseline"]["series"].keys()


def test_unknown_parameter_is_a_4xx_with_message(client):
    body = {"name": "bad", "measures": [{"name": "m", "changes": [
        {"param": "no.such_parameter", "value": 1, "start": 2026}]}], "options": {"end": END}}
    r = client.post("/api/run", json=body)
    assert 400 <= r.status_code < 500
    assert "no.such_parameter" in r.json()["error"]


def test_malformed_body_is_a_4xx(client):
    r = client.post("/api/run", json={"measures": [{"name": "m", "changes": [{"value": 1}]}]})
    assert 400 <= r.status_code < 500 and "error" in r.json()
    r = client.post("/api/run", json={"measures": [], "options": {"end": 1990}})
    assert r.status_code == 400


def test_bad_values_are_rejected(client):
    p = _scalar_param(client)
    if p is None:
        pytest.skip("no scalar parameter defined yet")
    def post(change):
        return client.post("/api/run", json={"measures": [{"name": "m", "changes": [change]}],
                                             "options": {"end": END}})
    assert post({"param": p["name"], "value": "abc", "start": 2026}).status_code == 422
    assert post({"param": p["name"], "value": 1, "start": 2026, "op": "divide"}).status_code == 422
    assert post({"param": p["name"], "value": 1, "start": 2026, "key": "x"}).status_code == 422
    assert post({"param": p["name"], "value": 1, "start": 2030, "end": 2026}).status_code == 422


def test_run_with_a_measure_and_arrays_for_that_run(client):
    p = _scalar_param(client)
    if p is None:
        pytest.skip("no scalar parameter defined yet")
    body = {"name": "one change", "measures": [{"name": "m", "changes": [
        {"param": p["name"], "value": 1.01, "start": 2026, "op": "mul"}]}],
        "options": {"end": END, "per_measure": True}}
    r = client.post("/api/run", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["per_measure"][0]["name"] == "m"
    assert client.post("/api/run", json=body).json()["id"] == out["id"]  # cached, same id
    a = client.get("/api/arrays", params={"run": out["id"], "year": 2028})
    assert a.status_code == 200 and a.json()["year"] == 2028
    # a disabled measure is ignored -> identical to baseline
    body["measures"][0]["enabled"] = False
    d = client.post("/api/run", json=body).json()["delta"]["series"]
    assert max(abs(v) for vals in d.values() for v in vals if v is not None) < 1e-6


def test_arrays(client):
    r = client.get("/api/arrays", params={"run": "baseline", "year": 2030})
    assert r.status_code == 200
    a = r.json()
    assert len(a["pop"]) == dims.N_AGE and len(a["pop"][0]) == dims.N_SEX
    assert len(a["pop_region"]) == dims.N_REGION
    assert len(a["status_age"]) == dims.N_STATUS and len(a["status_age"][0]) == dims.N_AGE
    assert len(a["emp"]) == dims.N_SECTOR and len(a["emp"][0]) == dims.N_JOBTYPE
    assert len(a["wage"]) == dims.N_SECTOR
    assert len(a["wage_dist"]) == dims.N_JOBTYPE and len(a["wage_dist"][0]) == dims.N_DECILE
    assert len(a["va"]) == dims.N_SECTOR
    total = sum(sum(row) for row in a["pop"])
    base = client.get("/api/baseline").json()
    assert total == pytest.approx(base["series"]["pop.total"][base["years"].index(2030)], rel=1e-4)
    assert client.get("/api/arrays", params={"run": "baseline", "year": 1999}).status_code == 404
    assert client.get("/api/arrays", params={"run": "nope", "year": 2030}).status_code == 404


def test_pyramid(client):
    r = client.get("/api/pyramid")
    assert r.status_code == 200
    p = r.json()
    assert len(p["pop"]) == len(p["years"]) and len(p["pop"][0]) == dims.N_AGE
    assert len(p["pop_region"]) == dims.N_REGION


def test_measures_library_handles_missing_and_present_files(client, scen_dir):
    r = client.get("/api/measures")
    assert r.status_code == 200
    assert {"measures", "scenarios", "errors"} <= set(r.json())
    (scen_dir / "reform.yaml").write_text(
        "name: Reform\nmeasures:\n  - name: First\n    changes:\n"
        "      - {param: no.such_parameter, value: 1, start: 2026}\n")
    (scen_dir / "single.json").write_text(
        '{"name": "Solo", "changes": [{"param": "x.y", "value": 2, "start": 2027, "op": "add"}]}')
    (scen_dir / "broken.yaml").write_text("name: [unclosed\n")
    lib = client.get("/api/measures").json()
    names = {m["name"] for m in lib["measures"] if m["file"].startswith("scenarios/")}
    assert {"First", "Solo"} <= names
    assert any(s["name"] == "Reform" for s in lib["scenarios"])
    assert any(e["file"] == "scenarios/broken.yaml" for e in lib["errors"])
    first = next(m for m in lib["measures"] if m["name"] == "First")
    assert first["unknown_params"] == ["no.such_parameter"]


def test_save_list_delete_scenario(client, scen_dir):
    assert client.get("/api/scenarios").json()["scenarios"] == []
    scen = {"name": "My test / scenario", "description": "d", "measures": [
        {"name": "m", "changes": [{"param": "a.b", "value": 0.5, "start": 2026, "end": 2030}]}]}
    r = client.post("/api/scenarios", json=scen)
    assert r.status_code == 200, r.text
    saved = r.json()
    assert (scen_dir / "user" / f"{saved['slug']}.yaml").is_file()
    assert saved["problems"]  # unknown parameter is reported but the file is still saved
    listed = client.get("/api/scenarios").json()["scenarios"]
    assert listed[0]["scenario"]["name"] == scen["name"]
    assert listed[0]["scenario"]["measures"][0]["changes"][0] == {
        "param": "a.b", "value": 0.5, "start": 2026, "op": "set", "end": 2030}
    # user scenarios are not part of the prebuilt library
    assert all(not m["file"].startswith("scenarios/user") for m in client.get("/api/measures").json()["measures"])
    assert client.delete(f"/api/scenarios/{saved['slug']}").status_code == 200
    assert client.get("/api/scenarios").json()["scenarios"] == []
    assert client.delete(f"/api/scenarios/{saved['slug']}").status_code == 404


def test_baseline_cache_follows_params_signature(client, monkeypatch):
    first = svc.baseline()
    assert svc.baseline() is first
    monkeypatch.setattr(svc, "params_signature", lambda: "changed")
    assert svc.baseline() is not first


def test_static_index_and_unknown_api(client):
    r = client.get("/")
    assert r.status_code == 200 and "<html" in r.text.lower()
    assert client.get("/api/nope").status_code == 404


# ------------------------------------------------------------------ units of newer series keys

def test_unit_kinds_of_newer_keys():
    assert unit_kind("labour.employment_vs_baseline.labour_demand") == "persons"
    assert unit_kind("labour.marginal_wage_factor_private") == "number"
    assert unit_kind("labour.public_staffing_lever_factor") == "number"
    assert unit_kind("macro.cpi_tax_effect") == "rate"
    assert unit_kind("macro.output_gap") == "rate"
    assert unit_kind("macro.tax_threshold_drift") == "index"
    assert unit_kind("macro.real_unit_labour_cost") == "index"
    assert unit_kind("macro.gdp_per_capita") == "eur"
    assert unit_kind("fiscal.stock_flow_adjustment") == "eur_mn"
    assert unit_kind("fiscal.balance_pct_gdp.federal") == "pct"
    assert unit_kind("fiscal.interest_pct_gdp") == "pct"


def test_every_baseline_key_has_a_plausible_unit(client):
    """Fractions must not be typed as money or persons (the UI converts by kind)."""
    b = client.get("/api/baseline").json()["series"]
    units = client.get("/api/meta").json()["units"]
    for key, vals in b.items():
        top = max((abs(v) for v in vals if v is not None), default=0.0)
        if units[key] in ("eur_mn", "persons") and key.split(".")[0] in ("macro", "labour") and top:
            assert top > 50, f"{key} is typed {units[key]} but never exceeds {top}"


def test_wage_linked_parameters_have_a_full_baseline_path(client):
    body = client.get("/api/params").json()
    linked = [p for p in body["params"] if p.get("wage_link") and p["kind"] == "scalar"]
    if not linked:
        pytest.skip("no wage-linked scalar parameter")
    for p in linked:
        assert all(v is not None for v in p["baseline"]), p["name"]


# ------------------------------------------------------------------ batch, band, ranking, validation

def _measure(client, factor=1.01, name="m"):
    p = _scalar_param(client)
    return {"name": name, "changes": [{"param": p["name"], "value": factor, "start": 2026, "op": "mul"}]}


def test_run_batch_returns_headlines_and_isolates_errors(client):
    good = {"name": "good", "measures": [_measure(client)]}
    bad = {"name": "bad", "measures": [{"name": "x", "changes": [{"param": "no.such", "value": 1, "start": 2026}]}]}
    r = client.post("/api/run_batch", json={"scenarios": [good, bad, {"name": "empty", "measures": []}],
                                            "options": {"end": END}})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["years"] == list(range(dims.BASE_YEAR, END + 1))
    assert "fiscal.balance" in out["keys"] and len(out["baseline"]["fiscal.balance"]) == len(out["years"])
    g, b, e = out["scenarios"]
    full = client.post("/api/run", json={**good, "options": {"end": END}}).json()
    assert g["id"] == full["id"]
    assert g["series"]["fiscal.balance"] == full["scenario"]["series"]["fiscal.balance"]
    assert "no.such" in b["error"] and "series" not in b
    assert e["series"]["fiscal.balance"] == out["baseline"]["fiscal.balance"]


def test_run_band_gives_four_runs(client):
    params = {p["name"] for p in client.get("/api/params").json()["params"]}
    if not {"beh.scale", "beh.enabled"} <= params:
        pytest.skip("no behavioural switches in the model")
    scen = {"name": "band", "measures": [_measure(client)], "options": {"end": END}}
    r = client.post("/api/run_band", json=scen)
    assert r.status_code == 200, r.text
    out = r.json()
    assert set(out["runs"]) == {"low", "central", "high", "static"} == set(out["variants"])
    assert "confidence interval" in out["note"]
    full = client.post("/api/run", json=scen).json()
    assert out["id"] == full["id"]
    assert out["runs"]["central"]["series"]["fiscal.balance"] == full["scenario"]["series"]["fiscal.balance"]
    for run in out["runs"].values():
        assert len(run["series"]["fiscal.balance"]) == len(out["years"])
    # a scenario without measures has no behavioural response: the four runs coincide
    flat = client.post("/api/run_band", json={"measures": [], "options": {"end": END}}).json()["runs"]
    assert flat["low"]["series"]["fiscal.balance"] == flat["static"]["series"]["fiscal.balance"]
    bad = client.post("/api/run_band", json={"measures": [{"name": "x", "changes": [
        {"param": "no.such", "value": 1, "start": 2026}]}], "options": {"end": END}})
    assert bad.status_code == 422


def test_library_effects(client, scen_dir):
    p = _scalar_param(client)
    (scen_dir / "mine.yaml").write_text(
        f"measures:\n  - name: Mine\n    changes:\n      - {{param: {p['name']}, value: 1.01, op: mul, start: 2026}}\n"
        "  - name: Broken\n    changes:\n      - {param: no.such_parameter, value: 1, start: 2026}\n")
    r = client.get("/api/measures/effects", params={"end": END})
    assert r.status_code == 200, r.text
    out = r.json()
    by = {m["name"]: m for m in out["measures"]}
    assert len(by["Mine"]["delta"]["fiscal.balance"]) == len(out["years"])
    assert by["Mine"]["delta"]["fiscal.balance"][0] == 0
    assert "no.such_parameter" in by["Broken"]["error"] and "delta" not in by["Broken"]
    assert by["Mine"]["changes"][0]["param"] == p["name"]
    assert client.get("/api/measures/effects", params={"end": 1990}).status_code == 400


def test_validation_endpoint(client, tmp_path, monkeypatch):
    p = _scalar_param(client)
    ref = tmp_path / "validation.yaml"
    ref.write_text(f"""
meta: {{retrieved: 2026-01-01}}
baseline:
  - id: gdp
    description: GDP level
    unit: EUR mn
    values: {{2025: 600000}}
    model: {{num: [macro.gdp_nominal]}}
    source: {{institution: Somebody, title: A report, date: 2026-01}}
    url: https://example.org/report
    confidence: high
  - id: nothing
    description: No model equivalent
    unit: "%"
    values: {{2025: 1.0}}
    confidence: low
measures:
  - id: costing
    description: A costing
    confidence: medium
    url: https://example.org/costing
    measure: {{changes: [{{param: {p['name']}, value: 1.01, op: mul, start: 2026}}]}}
    targets:
      - {{label: balance, ref_value: 100, ref_unit: EUR mn, ref_year: 2027, variant: static,
         model: {{num: [fiscal.balance]}}}}
not_found:
  - {{id: missing, description: Not found, searched: everywhere}}
""")
    monkeypatch.setenv("BELSIM_VALIDATION_FILE", str(ref))
    r = client.get("/api/validation")
    assert r.status_code == 200, r.text
    v = r.json()
    gdp, nothing = v["baseline"]
    assert gdp["status"] == "ok" and gdp["model"] is not None
    assert gdp["gap"] == pytest.approx(gdp["model"] - 600000)
    assert gdp["source"]["url"] == "https://example.org/report" and gdp["confidence"] == "high"
    assert nothing["model"] is None and nothing["status"] != "ok"
    (m,) = v["measures"]
    assert m["variant"] == "static" and m["model"] == m["static"] and m["behavioural"] is not None
    assert m["measure"]["changes"][0]["param"] == p["name"]
    assert v["not_found"][0]["id"] == "missing"
    assert client.get("/api/validation").json() == v          # cached
    monkeypatch.setenv("BELSIM_VALIDATION_FILE", str(tmp_path / "absent.yaml"))
    assert client.get("/api/validation").status_code == 404


def test_version_healthz_and_methodology_page(client):
    assert client.get("/healthz").json()["status"] == "ok"
    v = client.get("/api/version").json()
    assert v["params"] == len(client.get("/api/params").json()["params"])
    assert v["assumptions"] == sum(p["assumption"] for p in client.get("/api/params").json()["params"])
    assert v["base_year"] == dims.BASE_YEAR and v["end_year"] == dims.END_YEAR and v["public"] is False
    assert isinstance(v["modules"], list) and "api" in v["modules"]
    page = client.get("/whitepaper")
    assert page.status_code == 200 and "js/whitepaper.js" in page.text and 'href="css/whitepaper.css"' in page.text
    for lang in ("en", "fr", "nl"):
        frag = client.get(f"/wp/{lang}.html")
        assert frag.status_code == 200 and 'data-v="p.count"' in frag.text
    # the dashboard links to it and sets the theme before first paint, without an inline script
    index = client.get("/").text
    assert 'href="whitepaper"' in index and '<script src="js/theme-init.js"></script>' in index
    assert index.index("js/theme-init.js") < index.index("css/app.css")
    assert 'data-theme="light"' in index and "<noscript" in index and 'property="og:image"' in index
    init = client.get("/js/theme-init.js").text
    assert "'light'" in init and "=== 'dark'" in init
