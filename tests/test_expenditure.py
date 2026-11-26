"""Tests of belsim.fiscal.expenditure on a synthetic economy (see test_expenditure_world)."""
import time

import numpy as np
import pytest

import test_expenditure_world as w
from belsim.core.dims import ENTITIES, S
from belsim.core.engine import Context
from belsim.core.params import Change, Measure, Scenario
from belsim.core.state import Results
from belsim.fiscal import expenditure
from belsim.fiscal.expenditure import LINES, calc, data

SC = data.scalars()
PENSION_LINES = ["pension_employees", "pension_self_employed", "pension_civil_servants", "pension_grapa"]


@pytest.fixture(scope="module")
def base():
    return w.simulate(2035)


def scen(*changes):
    return Scenario(name="s", measures=[Measure("m", list(changes))])


def test_lines_metadata_complete():
    assert set(LINES) == set(data.lines_2024())
    assert len(LINES) >= 40
    for k, meta in LINES.items():
        assert meta.label and meta.group and meta.esa, k
        assert set(meta.entity_shares) == set(ENTITIES), k
        assert abs(sum(meta.entity_shares.values()) - 1.0) < 1e-9, k


def test_2024_lines_sum_to_official_total(base):
    s = base.at(2024)
    assert set(s.expenditure) == set(LINES)
    assert sum(s.expenditure.values()) == pytest.approx(SC["te_2024"], rel=1e-9)
    assert 0.535 < SC["te_2024"] / SC["gdp_2024"] < 0.55          # ~54% of GDP
    for k, v in s.expenditure.items():
        assert v == pytest.approx(data.lines_2024()[k]["value"], rel=1e-9), k
        assert v > 0, k


def test_2024_key_shares_of_gdp(base):
    e, g = base.at(2024).expenditure, SC["gdp_2024"]
    assert 11.0 < sum(e[k] for k in PENSION_LINES) / g * 100 < 11.6       # pensions ~11.3% of GDP
    assert 1.9 < e["interest"] / g * 100 < 2.4                           # interest ~2.2% of GDP
    assert 6.0 < e["health_care"] / g * 100 < 6.5                        # acute care (SHA: 6.0% of GDP)
    assert 1.8 < e["long_term_care"] / g * 100 < 2.0                     # long-term care, SCA perimeter 1.9%
    assert 8.0 < (e["health_care"] + e["long_term_care"]) / g * 100 < 8.25   # public health spending 8.1% (SHA)
    assert 5.5 < sum(v for k, v in e.items() if k.startswith("educ_")) / g * 100 < 6.5


def test_calibration_holds_with_crude_inputs():
    """Whatever the other modules deliver for 2024, the base year reproduces the accounts."""
    res = w.simulate(2024, retired_factor=0.7, gap=-0.02)
    s = res.at(2024)
    assert sum(s.expenditure.values()) == pytest.approx(SC["te_2024"], rel=1e-9)
    s.labour = {}                              # a labour module providing no aggregates at all
    s.wage_dist = np.zeros_like(s.wage_dist)
    p = w.load_params().at(2024, None)
    expenditure.run(s, None, p, None)
    assert sum(s.expenditure.values()) == pytest.approx(SC["te_2024"], rel=1e-9)


def test_more_retirees_proportionally_more_pensions(base):
    p = w.load_params().at(2024, None)
    s0 = base.at(2024)
    bx = s0.extra["expenditure"]
    s1 = w.make_state(2024, None)
    s1.status[S["retired"]] *= 1.10
    v1, _ = expenditure.compute(s1, None, None, p, bx)
    rcc = p["unemp.rcc_beneficiaries"]
    n0 = s0.status[S["retired"]].sum()
    expected = (1.10 * n0 - rcc) / (n0 - rcc)
    for k in PENSION_LINES[:3]:
        assert v1[k] / s0.expenditure[k] == pytest.approx(expected, rel=1e-9)
    assert expected == pytest.approx(1.10, abs=2e-3)
    assert v1["health_care"] == pytest.approx(s0.expenditure["health_care"])


def test_interest_responds_gradually_to_rate_shock(base):
    shocked = w.simulate(2035, rate_shock=0.01)
    d = []
    for y in range(2024, 2036):
        g = base.at(y).macro["gdp_nominal"]
        d.append((shocked.at(y).expenditure["interest"] - base.at(y).expenditure["interest"]) / g * 100)
    assert d[0] == 0.0
    assert 0.08 < d[1] < 0.22                       # short-term debt + first refinancing tranche
    steps = np.diff(d[1:])
    assert (steps > 0.06).all() and (steps < 0.20).all()   # ~ +0.1% of GDP per year, accumulating
    assert d[-1] < 1.6                              # far from the full pass-through (~1.1-1.3% of GDP) early on
    assert d[6] < 0.85                              # after 6 years still well below debt ratio x 1pt


def test_interest_2025_close_to_actual(base):
    # out-of-sample check: actual 2025 interest (gov_10a_main) with rates near 3%
    assert base.at(2025).expenditure["interest"] == pytest.approx(SC["interest_2025"], rel=0.08)


def test_idempotent(base):
    twice = w.simulate(2035, twice=True)
    a, b = base.series(), twice.series()
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_allclose(a[k], b[k], rtol=1e-12, atol=1e-9, err_msg=k)


def test_empty_scenario_equals_baseline(base):
    s = w.simulate(2035, scenario=Scenario(name="empty"), baseline=base)
    a, b = base.series(), s.series()
    for k in a:
        np.testing.assert_allclose(a[k], b[k], rtol=0, atol=1e-9, err_msg=k)


def test_measure_in_base_year_is_not_calibrated_away(base):
    s = w.simulate(2026, scenario=scen(Change("pension.level_factor", 0.9, 2024, key="employees")), baseline=base)
    for y in (2024, 2026):
        assert s.at(y).expenditure["pension_employees"] == pytest.approx(
            0.9 * base.at(y).expenditure["pension_employees"], rel=1e-9)
        assert s.at(y).expenditure["pension_self_employed"] == pytest.approx(
            base.at(y).expenditure["pension_self_employed"], rel=1e-9)


def test_policy_levers(base):
    ch = [
        Change("defence.target_pct_gdp", 3.0, 2030),
        Change("unemp.generosity_factor", 0.9, 2027),
        Change("health.copayment_share", 0.02, 2027, op="add"),
        Change("invest.public_pct_gdp", 0.5, 2028, op="add"),
        Change("pubwage.staffing_factor", 0.95, 2027, key="general"),
        Change("family.allowance_per_child", 1.1, 2027, op="mul", key="WAL"),
        Change("exp.benefit_indexation_factor", 0.0, 2029, end=2029),
    ]
    s = w.simulate(2032, scenario=scen(*ch), baseline=base)
    b30, s30 = base.at(2030), s.at(2030)
    g = s30.macro["gdp_nominal"]
    assert s30.expenditure["defence"] / g * 100 == pytest.approx(3.0, rel=1e-4)
    assert s30.expenditure["public_investment"] - b30.expenditure["public_investment"] == pytest.approx(0.005 * g, rel=1e-4)
    r = s.at(2027).expenditure["unemployment"] / base.at(2027).expenditure["unemployment"]
    assert 0.90 <= r < 0.95                                     # temporary unemployment not affected
    r = s.at(2027).expenditure["health_care"] / base.at(2027).expenditure["health_care"]
    copay = w.load_params().at(2027, None)["health.copayment_share"]
    assert r == pytest.approx((1 - copay - 0.02) / (1 - copay), rel=1e-9)
    assert s.at(2027).expenditure["general_admin"] < base.at(2027).expenditure["general_admin"]
    assert s.at(2027).expenditure["police_fire"] == pytest.approx(base.at(2027).expenditure["police_fire"])
    r = s.at(2027).expenditure["family_allowances"] / base.at(2027).expenditure["family_allowances"]
    assert 1.02 < r < 1.05                                      # Wallonia ~ a third of children
    # index jump in 2029 (a year of full indexation in the baseline): benefits 2% lower from then on, permanently
    for y in (2029, 2032):
        r = s.at(y).expenditure["disability_allowances"] / base.at(y).expenditure["disability_allowances"]
        assert r == pytest.approx(1 / 1.02, rel=1e-6)
    assert s.at(2032).expenditure["pension_employees"] < base.at(2032).expenditure["pension_employees"]


def test_pension_stock_flow_dynamics(base):
    # replacement-rate cut only bites through new retirees: small at first, growing over time
    s = w.simulate(2035, scenario=scen(Change("pension.replacement_rate", 0.54, 2026, key="employees")), baseline=base)
    r = [s.at(y).expenditure["pension_employees"] / base.at(y).expenditure["pension_employees"] for y in (2025, 2026, 2030, 2035)]
    assert r[0] == pytest.approx(1.0)
    assert 0.99 < r[1] < 1.0
    assert r[3] < r[2] < r[1]
    assert r[3] > 0.93                                          # stock only partly renewed after 10 years
    # under the pre-2025 rules (full indexation, perequation, no reform of new pensions) the average
    # pension rises at least with the health index
    old = w.simulate(2035, baseline=base, scenario=scen(
        Change("exp.benefit_indexation_factor", 1.0, 2025),
        Change("pension.perequation_share", 1.0, 2025),
        Change("pension.bonus_malus_per_year", 0.0, 2025),
        Change("pension.new_pension_factor", 1.0, 2025, key="civil_servants")))
    for y in range(2025, 2036):
        a0, a1 = old.at(y - 1).extra["expenditure"]["pension_avg"], old.at(y).extra["expenditure"]["pension_avg"]
        hg = old.at(y).macro["health_index"] / old.at(y - 1).macro["health_index"]
        for k in a0:
            assert a1[k] / a0[k] >= hg - 1e-9
    # the legislated reforms (capped indexation 2026/2028, end of perequation, longer reference wage of
    # civil servants, malus) lower the baseline average pension relative to those rules
    for k in ("employees", "civil_servants"):
        assert base.at(2035).extra["expenditure"]["pension_avg"][k] < old.at(2035).extra["expenditure"]["pension_avg"][k]


def test_minimum_pension_and_ceiling_matter_for_new_pensions(base):
    up = w.simulate(2035, scenario=scen(Change("pension.min_pension", 1.2, 2026, op="mul", key="employees")), baseline=base)
    assert up.at(2035).expenditure["pension_employees"] > base.at(2035).expenditure["pension_employees"]
    cap = w.simulate(2035, scenario=scen(Change("pension.wage_ceiling", 0.7, 2026, op="mul", key="employees")), baseline=base)
    assert cap.at(2035).expenditure["pension_employees"] < base.at(2035).expenditure["pension_employees"]


def test_unemployment_coverage_and_spillover_to_assistance(base):
    # 30% fewer of the unemployed covered than in the baseline (whose path is 0.75 from 2026)
    s = w.simulate(2028, scenario=scen(Change("unemp.coverage_rate", 0.7, 2027, op="mul")), baseline=base)
    e0, e1 = base.at(2027).expenditure, s.at(2027).expenditure
    assert 0.70 <= e1["unemployment"] / e0["unemployment"] < 0.76
    assert [w.load_params().at(y, None)["unemp.coverage_rate"] for y in (2024, 2025, 2026, 2040)] == [1.0, 0.93, 0.75, 0.75]
    assert e1["social_assistance"] > 1.05 * e0["social_assistance"]              # partial spill-over
    saved = e0["unemployment"] - e1["unemployment"]
    assert 0 < e1["social_assistance"] - e0["social_assistance"] < saved
    # exclusions decided by the labour module (people moved to inactivity) spill over as well
    p = w.load_params().at(2024, None)
    st = w.make_state(2024, None)
    st.labour["unemp_excluded_to_inactive"] = 50_000.0
    v, _ = expenditure.compute(st, None, None, p, base.at(2024).extra["expenditure"])
    assert v["social_assistance"] > base.at(2024).expenditure["social_assistance"] * 1.02
    # RCC beneficiaries delivered by the labour module take precedence over the fallback parameter
    st = w.make_state(2024, None)
    st.labour["rcc_beneficiaries"] = 5175.0
    v, _ = expenditure.compute(st, None, None, p, base.at(2024).extra["expenditure"])
    assert v["rcc_swt"] == pytest.approx(0.5 * base.at(2024).expenditure["rcc_swt"])


def test_ageing_raises_health_and_pensions(base):
    def pct(y, keys):
        s = base.at(y)
        return sum(s.expenditure[k] for k in keys) / s.macro["gdp_nominal"]
    assert pct(2035, PENSION_LINES) > pct(2024, PENSION_LINES)
    assert pct(2035, ["health_care", "long_term_care"]) > pct(2024, ["health_care", "long_term_care"])


def test_replacement_income():
    p = w.load_params().at(2024, None)
    g = np.array([0.0, 20_000.0, 40_000.0, 60_000.0, 200_000.0])
    for status in ("unemployed", "disabled", "retired"):
        b = calc.replacement_income(status, g, p)
        assert b.shape == g.shape and (np.diff(b) >= 0).all() and (b > 0).all()
        assert b[-1] == calc.replacement_income(status, 1e7, p)        # capped
        assert b[-1] < 0.8 * 200_000
    u = calc.replacement_income("unemployed", g, p)
    assert u[0] == pytest.approx(p["unemp.min_benefit"])
    assert u[-1] == pytest.approx(p["unemp.replacement_rate"] * p["unemp.wage_ceiling"])
    r = calc.replacement_income("retired", 50_000.0, p)
    assert r == pytest.approx(0.60 * 0.82 * 50_000)
    assert calc.replacement_income("retired", 50_000.0, p, scheme="civil_servants") > r
    with pytest.raises(ValueError):
        calc.replacement_income("student", g, p)


def test_every_parameter_documented():
    pset = w.load_params()
    prefixes = ("pension.", "unemp.", "disab.", "family.", "assist.", "health.", "educ.", "pubwage.",
                "defence.", "invest.", "subsidy.", "debt.", "transfer.", "exp.", "entity.")
    for name, spec in pset.specs.items():
        assert name.startswith(prefixes), name
        assert spec.source and spec.description and spec.unit, name
        assert min(spec.values) == 2024, name


def test_fast():
    t = time.perf_counter()
    w.simulate(2070)
    per_year = (time.perf_counter() - t) / 47
    assert per_year < 0.02        # target ~1-2 ms per year for expenditure + balance (incl. test world)


def test_bonus_malus_lever(base):
    # baseline: legislated malus/bonus of 2% per year from 2027, 4% from 2031 (law of 30 May 2026)
    assert [w.load_params().at(y, None)["pension.bonus_malus_per_year"] for y in (2026, 2027, 2031, 2040)] == [0.0, 0.02, 0.04, 0.05]
    extra = Change("pension.bonus_malus_per_year", 0.04, 2026, op="add")
    p0 = w.load_params().at(2030, None)
    p1 = w.load_params(scen(extra)).at(2030, None)
    assert calc.pension_accrual_incentive(p1) - calc.pension_accrual_incentive(p0) == pytest.approx(0.04)
    r0, r1 = (calc.replacement_income("retired", 50_000.0, p) for p in (p0, p1))
    expo, b0 = p0["pension.bonus_malus_exposure_years"]["employees"], p0["pension.bonus_malus_per_year"]
    assert expo < 0
    assert r1 / r0 == pytest.approx((1 + expo * (b0 + 0.04)) / (1 + expo * b0))
    s = w.simulate(2035, scenario=scen(extra), baseline=base)
    assert s.at(2035).expenditure["pension_employees"] < base.at(2035).expenditure["pension_employees"]


def test_new_pension_factor_and_legislated_reforms(base):
    ps = w.load_params()
    p26, p27, p45, p70 = (ps.at(y, None) for y in (2026, 2027, 2045, 2070))
    # reference-wage / accrual reform of civil-servant pensions: gradual, by cohort, from 2027
    f = [p["pension.new_pension_factor"]["civil_servants"] for p in (p26, p27, p45, p70)]
    assert f[0] == 1.0 and f[0] > f[1] > f[2] > f[3] == pytest.approx(0.85)
    # employees: cap on assimilated periods and limited fictitious wage (FPB report 13299): -1.9% by 2040
    fe = [p["pension.new_pension_factor"]["employees"] for p in (p26, p27, p45, p70)]
    assert fe[0] == 1.0 and fe[0] > fe[1] > fe[2] == fe[3] == pytest.approx(0.981)
    assert all(p["pension.new_pension_factor"]["self_employed"] == 1.0 for p in (p26, p27, p45, p70))
    r26, r27 = (calc.replacement_income("retired", 60_000.0, p, scheme="civil_servants") for p in (p26, p27))
    assert r27 / r26 == pytest.approx(f[1] * (1 + p27["pension.bonus_malus_per_year"] * p27["pension.bonus_malus_exposure_years"]["civil_servants"]))
    # the factor is a lever on new pensions only: effect builds up with the renewal of the stock
    s = w.simulate(2035, scenario=scen(Change("pension.new_pension_factor", 0.9, 2027, op="mul", key="employees")), baseline=base)
    r = [s.at(y).expenditure["pension_employees"] / base.at(y).expenditure["pension_employees"] for y in (2026, 2027, 2035)]
    assert r[0] == pytest.approx(1.0) and 0.99 < r[1] < 1.0 and 0.9 < r[2] < r[1]
    assert s.at(2035).expenditure["pension_civil_servants"] == pytest.approx(base.at(2035).expenditure["pension_civil_servants"])
    # other dated baseline entries: perequation abolished, welfare envelope suspended 2025-2029,
    # health norm of the legislature, defence 2% until 2033 and 2.5% from 2034
    # perequation: none in 2025-2029, then the capped mechanism of the law of 25 April 2024 (0.3% a year)
    assert [ps.at(y, None)["pension.perequation_share"] for y in (2024, 2025, 2029, 2040)] == [1.0, 0.0, 0.0, 1.0]
    assert [ps.at(y, None)["pension.perequation_cap"] for y in (2024, 2025, 2040)] == [None, 0.003, 0.003]
    assert [ps.at(y, None)["exp.welfare_adjustment"] for y in (2025, 2029, 2030)] == [0.0, 0.0, 0.01]
    assert [ps.at(y, None)["health.growth_norm"] for y in (2025, 2026, 2027, 2028, 2029)] == [0.025, 0.02, 0.02, 0.026, 0.03]
    # defence in ESA terms (equipment at delivery): gradual rise to 2% by 2029, 2.5% from 2034
    assert [ps.at(y, None)["defence.target_pct_gdp"] for y in (2025, 2027, 2029, 2033, 2034)] == [1.40, 1.75, 2.0, 2.0, 2.5]
    assert [ps.at(y, None)["exp.benefit_indexation_factor"] for y in (2025, 2026, 2027, 2028, 2029)] == [1.0, 0.9, 1.0, 0.9, 1.0]
