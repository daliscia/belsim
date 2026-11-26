"""Regional dimension: sums to national totals, 2024 calibration on the regional accounts,
one-region measures, neutrality, idempotency."""
import copy

import numpy as np
import pytest

from belsim.core import engine
from belsim.core.dims import REGIONS
from belsim.core.params import Change, Measure, ParamSet, Scenario
from belsim.core.state import Results
from belsim import regional
from belsim.regional import ENTITIES4, ENTITY_DETAIL, METRICS, OTHER, data

END = 2035
RC = "regions_communities"


@pytest.fixture(scope="module")
def base():
    return engine.run(None, end=END)


def reg(state, metric):
    return np.array([state.regional[f"{metric}.{r}"] for r in REGIONS])


def run(base, *changes):
    return engine.run(Scenario(name="s", measures=[Measure("m", list(changes))]), end=END, baseline=base)


def diff(base, scen, metric, year=END):
    return reg(scen.at(year), metric) - reg(base.at(year), metric)


# ------------------------------------------------------------------ accounting
def test_every_metric_is_published_for_every_region_and_year(base):
    for s in base.states:
        for metric in METRICS:
            v = reg(s, metric)
            assert np.isfinite(v).all(), (s.year, metric)


def test_regional_sums_equal_national_totals(base):
    for s in base.states:
        m, lab, f = s.macro, s.labour, s.fiscal
        checks = {
            "population": s.pop.sum(), "employment_residents": lab["employment"],
            "employment_workplace": s.emp.sum(), "gdp_nominal": m["gdp_nominal"], "gdp_real": m["gdp_real"],
            "wage_bill": m["compensation_employees"], "household_income": m["household_disposable_income"],
            "gov_revenue": f[f"revenue.{RC}"], "gov_expenditure": f[f"expenditure.{RC}"],
            "gov_balance": f[f"balance.{RC}"], "gov_debt": f[f"debt.{RC}"],
            "local_gov_revenue": f["revenue.local"], "local_gov_expenditure": f["expenditure.local"],
            "local_gov_balance": f["balance.local"], "local_gov_debt": f["debt.local"],
            "transfers_from_federal": f[f"transfer.federal_to_{RC}"],
            "net_fiscal_transfer": 0.0,
        }
        for metric, total in checks.items():
            assert reg(s, metric).sum() == pytest.approx(total, rel=1e-9, abs=1e-5), (s.year, metric)
        # unemployed: benefit recipients + excluded job-seekers, as in the national rate
        assert reg(s, "unemployed").sum() == pytest.approx(lab["unemployed"] + lab["unemp_excluded_to_inactive"])
        for r in REGIONS:
            assert s.regional[f"unemployment_rate.{r}"] == pytest.approx(lab[f"unemployment_rate.{r}"])
            assert s.regional[f"employment_rate_20_64.{r}"] == pytest.approx(lab[f"employment_rate_20_64.{r}"])


def test_federated_entities_sum_to_the_subsector(base):
    assert set(ENTITY_DETAIL) == set(ENTITIES4) | {OTHER}
    assert all(d["parent"] == RC and d["label"] for d in ENTITY_DETAIL.values())
    for s in base.states:
        f = s.fiscal
        for item in ("revenue", "expenditure", "balance", "debt"):
            tot = sum(f[f"{item}.{e}"] for e in ENTITY_DETAIL)
            assert tot == pytest.approx(f[f"{item}.{RC}"], rel=1e-10, abs=1e-6), (s.year, item)
        for e in ENTITY_DETAIL:
            assert f[f"balance.{e}"] == pytest.approx(f[f"revenue.{e}"] - f[f"expenditure.{e}"], abs=1e-6)
        assert sum(f[f"federal_grants.{e}"] for e in ENTITIES4) == pytest.approx(f[f"transfer.federal_to_{RC}"])


def test_federal_incidence_adds_up_to_the_federal_and_social_security_balance(base):
    for s in base.states:
        f = s.fiscal
        net = reg(s, "federal_revenue_raised").sum() - reg(s, "federal_spending").sum()
        assert net == pytest.approx(f["balance.federal"] + f["balance.social_security"], rel=1e-8, abs=1e-4)


# ------------------------------------------------------------------ 2024 = data
def test_base_year_matches_regional_accounts(base):
    s = base.at(2024)
    acc = data.accounts(2024)
    for metric, indicator in [("gdp_nominal", "gdp"), ("employment_workplace", "employment_workplace"),
                              ("wage_bill", "compensation_workplace"), ("household_income", "hh_disposable_income")]:
        np.testing.assert_allclose(reg(s, metric) / reg(s, metric).sum(), acc[indicator] / acc[indicator].sum(),
                                   rtol=1e-9, err_msg=metric)
    # levels: the national totals of the model and of the regional accounts are within 1.5%
    np.testing.assert_allclose(reg(s, "gdp_nominal"), acc["gdp"], rtol=0.005)
    np.testing.assert_allclose(reg(s, "employment_workplace"), acc["employment_workplace"] * 1e3, rtol=0.015)
    np.testing.assert_allclose(reg(s, "population"), acc["population_avg"] * 1e3, rtol=0.01)
    np.testing.assert_allclose(reg(s, "gdp_real_growth"), acc["gdp_real_growth"] / 100.0)
    # labour-force survey by region of residence
    np.testing.assert_allclose(reg(s, "unemployment_rate") * 100, acc["lfs_unemployment_rate"], atol=0.5)
    np.testing.assert_allclose(reg(s, "employment_rate_20_64") * 100, acc["lfs_employment_rate_20_64"], atol=1.0)
    # Brussels imports commuters, the two other regions export them
    nc = reg(s, "net_commuting")
    assert nc[2] > 150_000 and nc[0] < 0 and nc[1] < 0
    # GDP per head: Brussels > Flanders > Wallonia; disposable income per head: Flanders highest
    g = reg(s, "gdp_per_capita")
    assert g[2] > g[0] > g[1]
    assert np.argmax(reg(s, "household_income_per_capita")) == 0


def test_base_year_matches_entity_accounts(base):
    s, pset = base.at(2024), ParamSet.load()
    acc = pset.value("regional.entity_accounts_2024", 2024)
    for e in ENTITIES4:
        for item in ("revenue", "expenditure", "debt"):
            assert s.fiscal[f"{item}.{e}"] == pytest.approx(acc[e][item], rel=1e-9), (e, item)
        assert s.fiscal[f"balance.{e}"] == pytest.approx(acc[e]["balance"], abs=0.3), e   # published rounding
    # the consolidation item is what the NBB reports: interregional units +771, debt 3 533 - 5 120
    assert s.fiscal[f"balance.{OTHER}"] == pytest.approx(771.1, abs=1.0)
    assert s.fiscal[f"debt.{OTHER}"] == pytest.approx(3532.6 - 5120.4, abs=1.0)
    # Flanders is a net contributor through the federal budget and social security, Wallonia a net recipient
    nft = reg(s, "net_fiscal_transfer")
    assert nft[0] > 3000 and nft[1] < -3000


# ------------------------------------------------------------------ one-region measures
def test_walloon_structural_unemployment_moves_wallonia_and_national_totals(base):
    scen = run(base, Change("labour.nairu", -0.01, 2027, op="add", key="WAL", phase_in_years=3))
    d_emp = diff(base, scen, "employment_residents")
    assert d_emp[1] > 8000 and d_emp[1] > 20 * max(abs(d_emp[0]), abs(d_emp[2]))
    d_u = diff(base, scen, "unemployment_rate")
    assert d_u[1] < -0.006 and abs(d_u[0]) < 0.0005 and abs(d_u[2]) < 0.0005
    d_gdp = diff(base, scen, "gdp_nominal")
    assert d_gdp[1] > 2 * max(d_gdp[0], d_gdp[2]) and d_gdp.min() > 0     # commuters: others gain a little
    a, b = base.at(END), scen.at(END)
    assert b.labour["employment"] > a.labour["employment"] + 8000
    assert b.macro["gdp_nominal"] > a.macro["gdp_nominal"]
    assert b.fiscal["revenue_total"] > a.fiscal["revenue_total"]
    assert b.fiscal["balance"] > a.fiscal["balance"]


def test_activation_and_disability_levers_are_regional(base):
    act = run(base, Change("labour.activation_factor", 1.25, 2027, key="BRU"))
    d = diff(base, act, "employment_residents")
    assert d[2] > 2000 and d[2] > 20 * max(abs(d[0]), abs(d[1]))
    dis = run(base, Change("labour.disability_rate_factor", 0.9, 2027, key="FLA"))
    d = diff(base, dis, "disabled")
    assert d[0] < -20_000 and abs(d[1]) < 200 and abs(d[2]) < 200
    assert dis.at(END).labour["employment"] > base.at(END).labour["employment"]


def test_regional_productivity_wedge(base):
    scen = run(base, Change("regional.productivity_growth_wedge", 0.0025, 2027, key="FLA"))
    a, b = base.at(END), scen.at(END)
    ratio = reg(b, "gdp_real") / reg(a, "gdp_real")
    years = END - 2027 + 1
    assert ratio[0] == pytest.approx(1.0025 ** years, rel=6e-3)      # + induced demand effects
    assert ratio[0] > 1.0025 ** years - 2e-3
    assert abs(ratio[1] - 1) < 6e-3 and abs(ratio[2] - 1) < 6e-3
    share = a.regional["gdp_real.FLA"] / a.macro["gdp_real"]
    assert b.macro["gdp_real"] / a.macro["gdp_real"] - 1 == pytest.approx(share * (1.0025 ** years - 1), rel=0.45)  # + induced effects


def test_regional_spending_levers_hit_the_right_entity(base):
    cases = [
        (Change("family.allowance_per_child", 1.1, 2027, op="mul", key="WAL"), "walloon_region", -1),
        (Change("invest.regional_extra_pct_gdp", 0.5, 2027, key="FLA"), "flemish_community", -1),
        (Change("educ.community_cost_factor", 0.95, 2027, key="french_community"), "french_community", 1),
        (Change("health.ltc_regional_factor", 1.1, 2027, key="FLA"), "flemish_community", -1),
        (Change("subsidy.service_voucher_factor", 0.8, 2027, key="BRU"), "brussels_region", 1),
        (Change("subsidy.target_group_factor", 0.5, 2027, key="WAL"), "walloon_region", 1),
        (Change("family.childcare_factor", 1.2, 2027, key="flemish_community"), "flemish_community", -1),
        (Change("labour.public_employment_growth", -0.01, 2027, end=2031, key="BRU"), "brussels_region", 1),
    ]
    for change, entity, sign in cases:
        scen = run(base, change)
        a, b = base.at(END), scen.at(END)
        d = {e: b.fiscal[f"balance.{e}"] - a.fiscal[f"balance.{e}"] for e in ENTITIES4}
        assert d[entity] * sign > 20, (change.param, d)
        others = max(abs(v) for e, v in d.items() if e != entity)
        assert abs(d[entity]) > 2 * others, (change.param, d)
        d_rc = b.fiscal[f"balance.{RC}"] - a.fiscal[f"balance.{RC}"]
        d_all = b.fiscal["balance"] - a.fiscal["balance"]
        assert d_rc * sign > 0 and d_all * sign > 0, change.param


def test_regional_investment_is_paid_by_the_regions(base):
    scen = run(base, Change("invest.regional_extra_pct_gdp", 0.5, 2027, key="FLA"))
    a, b = base.at(END), scen.at(END)
    extra = b.expenditure["public_investment"] - a.expenditure["public_investment"]
    # + the national investment line follows the GDP induced by the extra investment
    assert extra == pytest.approx(0.005 * b.regional["gdp_nominal.FLA"], rel=0.06)
    assert b.fiscal[f"own_expenditure.{RC}"] - a.fiscal[f"own_expenditure.{RC}"] > 0.95 * extra
    assert b.fiscal["expenditure.flemish_community"] - a.fiscal["expenditure.flemish_community"] > 0.95 * extra


def test_equal_regional_values_are_the_national_lever(base):
    scalar = run(base, Change("labour.public_employment_growth", -0.01, 2027, end=2031))
    dic = run(base, Change("labour.public_employment_growth", {"FLA": -0.01, "WAL": -0.01, "BRU": -0.01},
                           2027, end=2031))
    d = engine.compare(scalar, dic)
    assert max(abs(v) for vals in d.values() for v in vals if v is not None) < 1e-6
    one = run(base, Change("labour.disability_rate_factor", 0.9, 2027))           # scalar form accepted
    assert one.at(END).status.sum() == pytest.approx(base.at(END).status.sum())
    assert diff(base, one, "disabled").max() < 0


# ------------------------------------------------------------------ neutrality, idempotency
def test_empty_scenario_reproduces_regional_baseline(base):
    scen = engine.run(Scenario(name="empty"), end=2030, baseline=base)
    for year in range(2024, 2031):
        a, b = base.at(year), scen.at(year)
        assert a.regional.keys() == b.regional.keys()
        assert max(abs(a.regional[k] - b.regional[k]) for k in a.regional) < 1e-6
        for e in ENTITY_DETAIL:
            for item in ("revenue", "expenditure", "balance", "debt"):
                assert abs(a.fiscal[f"{item}.{e}"] - b.fiscal[f"{item}.{e}"]) < 1e-6


def test_run_is_idempotent(base):
    pset = ParamSet.load()
    for year in (2024, 2029):
        state = base.at(year)
        prev = base.at(year - 1) if year > 2024 else None
        hist = Results(name="h")
        for s in base.states:
            if s.year < year:
                hist.add(s)
        p = pset.at(year, engine._index_fn(state, hist))
        ctx = engine.Context(year=year, params=pset, history=hist, baseline=None, base_p=None)
        before = (dict(state.regional), dict(state.fiscal), copy.deepcopy(state.extra["regional"]))
        for _ in range(2):
            regional.run(state, prev, p, ctx)
        assert state.regional == before[0]
        assert state.fiscal == before[1]
        for k, v in before[2].items():
            if isinstance(v, dict):
                for kk in v:
                    np.testing.assert_array_equal(v[kk], state.extra["regional"][k][kk])
            else:
                np.testing.assert_array_equal(v, state.extra["regional"][k])


def test_regional_parameters_are_documented():
    specs = ParamSet.load().specs
    names = [n for n, s in specs.items() if s.file == "regional.yaml"] + [
        "labour.activation_factor", "labour.activation_nairu_elasticity", "labour.disability_rate_factor",
        "labour.public_employment_growth", "educ.community_cost_factor", "educ.community_enrolment_factor",
        "family.childcare_factor", "health.ltc_regional_factor", "subsidy.service_voucher_factor",
        "subsidy.target_group_factor", "invest.regional_extra_pct_gdp"]
    assert len(names) > 15
    for n in names:
        s = specs[n]
        assert s.description and s.unit and s.source, n
    for n in ("labour.nairu", "labour.activation_factor", "labour.disability_rate_factor",
              "labour.public_employment_growth", "health.ltc_regional_factor", "subsidy.service_voucher_factor",
              "subsidy.target_group_factor", "invest.regional_extra_pct_gdp", "regional.productivity_growth_wedge",
              "family.allowance_per_child"):
        assert set(specs[n].values[2024]) == set(REGIONS), n


# ------------------------------------------------------------------ regional limits removed
def test_unabsorbed_public_staff_stays_in_the_region_of_the_cut(base):
    scen = run(base, Change("labour.public_employment_growth", -0.01, 2027, end=2031, key="BRU"))
    for year in (2028, 2031):
        lab = scen.at(year).labour
        assert lab["public_staff_unabsorbed.BRU"] > 300
        assert abs(lab["public_staff_unabsorbed.FLA"]) < 5 and abs(lab["public_staff_unabsorbed.WAL"]) < 5
        assert lab["public_staff_unabsorbed"] == pytest.approx(sum(lab[f"public_staff_unabsorbed.{r}"] for r in REGIONS))
    d = diff(base, scen, "unemployed", 2031)
    assert d[2] > 5 * max(abs(d[0]), abs(d[1]))           # the jobless are in Brussels
    # the same norm in the three regions: unabsorbed staff in proportion to the public employees cut
    nat = run(base, Change("labour.public_employment_growth", -0.01, 2027, end=2031))
    lab = nat.at(2031).labour
    assert lab["public_staff_unabsorbed.FLA"] > lab["public_staff_unabsorbed.WAL"] > lab["public_staff_unabsorbed.BRU"] > 0


def test_local_government_by_region_follows_regional_drivers(base):
    for s in (base.at(2024), base.at(2030), base.at(END)):
        for item in ("revenue", "expenditure", "balance", "debt"):
            assert reg(s, f"local_gov_{item}").sum() == pytest.approx(s.fiscal[f"{item}.local"], rel=1e-9)
    # 2024: the regional accounts of the National Bank
    p = ParamSet.load().at(2024, None)["regional.local_government_2024"]
    share = np.array([p["expenditure"][r] for r in REGIONS]) / sum(p["expenditure"].values())
    np.testing.assert_allclose(reg(base.at(2024), "local_gov_expenditure") / base.at(2024).fiscal["expenditure.local"],
                               share, rtol=1e-9)
    # a region that gains inhabitants gains local spending and local tax revenue
    scen = run(base, Change("demo.immigration_factor", 1.3, 2026, key="BRU"))
    a, b = base.at(END), scen.at(END)
    sh = lambda s, item: reg(s, f"local_gov_{item}") / reg(s, f"local_gov_{item}").sum()
    assert sh(b, "expenditure")[2] > sh(a, "expenditure")[2] * 1.02
    assert sh(b, "revenue")[2] > sh(a, "revenue")[2] * 1.01
    assert sh(b, "expenditure")[0] < sh(a, "expenditure")[0]
