"""Labour module: accounting consistency, 2024 calibration, policy levers, idempotence."""
import copy
import time

import numpy as np
import pandas as pd
import pytest

from belsim import labour, macro
from belsim.core.dims import BASE_YEAR, EMPLOYED_STATUSES, N_AGE, N_SEX, S, SEC, SECTORS
from belsim.core.params import Change, Measure, Scenario
from belsim.core.state import YearState
from test_labour_support import DATA, baseline_2070, paramset, population, simulate

EMP = [S[n] for n in EMPLOYED_STATUSES]


def scen(*changes: Change) -> Scenario:
    return Scenario(name="s", measures=[Measure(name="m", changes=list(changes))])


def employed_by_age(state) -> np.ndarray:
    return state.status[EMP].sum(axis=(0, 2, 3))


def test_status_sums_to_population_and_is_non_negative():
    for s in baseline_2070().states:
        assert s.status.min() >= 0
        np.testing.assert_allclose(s.status.sum(axis=0), s.pop, rtol=1e-9)
        np.testing.assert_allclose(s.status[S["child"], :15], s.pop[:15], rtol=1e-9)
        assert s.status[EMP, :15].sum() == 0


def test_sector_columns_equal_employed_status_totals():
    for s in baseline_2070().states:
        np.testing.assert_allclose(s.emp.sum(axis=0), s.status[EMP].sum(axis=(1, 2, 3)), rtol=1e-9)
        assert s.emp.min() >= 0


def test_base_year_matches_observed_labour_market():
    lab = simulate(BASE_YEAR).at(BASE_YEAR).labour
    assert 5.0e6 < lab["employment"] < 5.2e6                 # LFS 5.06M, national accounts 5.11-5.19M
    assert lab["employment_rate_20_64"] == pytest.approx(0.723, abs=0.01)     # Eurostat LFS 72.3%
    assert lab["unemployment_rate"] == pytest.approx(0.057, abs=0.006)        # Eurostat LFS 5.7%
    assert lab["activity_rate_20_64"] == pytest.approx(0.765, abs=0.012)
    assert 0.85e6 < lab["employees_public"] < 1.05e6
    assert lab["unemployment_rate.BRU"] > lab["unemployment_rate.WAL"] > lab["unemployment_rate.FLA"]
    assert lab["employment_rate_20_64.FLA"] > lab["employment_rate_20_64.WAL"]


def test_base_year_wage_bill_matches_national_accounts():
    s = simulate(BASE_YEAR).at(BASE_YEAR)
    sector = pd.read_csv(DATA / "labour_sector.csv")
    d11 = sector["d11_meur"].sum()
    lab = s.labour
    assert lab["wage_bill_private"] + lab["wage_bill_public"] == pytest.approx(d11, rel=1e-9)
    assert (s.wage * s.emp).sum(axis=0)[:2].sum() / 1e6 == pytest.approx(d11, rel=1e-9)
    assert 45_000 < lab["avg_wage_private"] < 62_000
    assert 30_000 < lab["avg_income_self_employed"] < 60_000


def test_wage_distribution_mean_equals_average_wage():
    for year in (2024, 2040):
        s = baseline_2070().at(year)
        avg = (s.wage * s.emp).sum(axis=0) / s.emp.sum(axis=0)
        np.testing.assert_allclose(s.wage_dist.mean(axis=1), avg, rtol=1e-9)
        assert (np.diff(s.wage_dist, axis=1) > 0).all()


def test_run_is_idempotent():
    res = simulate(2027)
    prev, last = res.at(2026), res.at(2027)
    p = paramset().at(2027, None)
    st = YearState(year=2027)
    macro.prices(st, prev, p, None)
    st.pop = population(2027)
    st.demo = dict(last.demo)
    st.adjust = {"labour_demand_factor": 0.99, "retirement_shift": 0.2}
    st.macro["demand_gap"] = -0.01
    labour.run(st, prev, p, None)
    macro.output(st, prev, p, None)
    first = copy.deepcopy(st)
    labour.run(st, prev, p, None)
    macro.output(st, prev, p, None)
    for name in ("status", "emp", "wage", "wage_dist", "va"):
        np.testing.assert_array_equal(getattr(st, name), getattr(first, name))
    assert st.labour == first.labour and st.macro == first.macro


def test_higher_pension_age_raises_employment_at_65_to_67():
    base = baseline_2070()
    higher = simulate(2040, scen(Change("labour.legal_pension_age", 68, start=2035)))
    lower = simulate(2040, scen(Change("labour.legal_pension_age", 65, start=2025)))
    b, h, lo = (employed_by_age(r.at(2040)) for r in (base, higher, lower))
    assert h[65:68].sum() > b[65:68].sum() * 1.05
    assert lo[65:68].sum() < b[65:68].sum() * 0.9
    # prime-age employment is unaffected, up to the constant regional unemployment *rate*
    # applied to a larger labour force
    np.testing.assert_allclose(h[:55], b[:55], rtol=3e-3)
    # legislated path: employment at 65-66 jumps when the age goes to 66 (2025) and 67 (2030)
    e = [employed_by_age(base.at(y))[65:67].sum() / base.at(y).pop[65:67].sum() for y in (2024, 2025, 2030)]
    assert e[0] < e[1] < e[2]


def test_early_retirement_and_behavioural_shift_raise_older_employment():
    base = baseline_2070().at(2035)
    early = simulate(2035, scen(Change("labour.early_retirement_age", 65, start=2030))).at(2035)
    beh = simulate(2035, adjust=lambda y: {"retirement_shift": 0.5} if y >= 2030 else {}).at(2035)
    for alt in (early, beh):
        assert employed_by_age(alt)[60:66].sum() > employed_by_age(base)[60:66].sum() * 1.02
        assert alt.status[S["retired"], 60:66].sum() < base.status[S["retired"], 60:66].sum()


def test_unemployment_time_limit_moves_people_out_of_benefits():
    base = baseline_2070()
    no_limit = simulate(2028, scen(Change("labour.unemp_benefit_max_years", 0, start=2025)))
    b, n = base.at(2028), no_limit.at(2028)
    assert n.labour["unemp_excluded_total"] == 0
    assert b.labour["unemp_excluded_total"] > 50_000
    assert b.status[S["unemployed"]].sum() < n.status[S["unemployed"]].sum() - 50_000
    assert b.labour["employment"] > n.labour["employment"]
    assert b.status[S["inactive"]].sum() > n.status[S["inactive"]].sum()
    assert b.status[S["disabled"]].sum() > n.status[S["disabled"]].sum()


def test_public_headcount_norm():
    base = baseline_2070().at(2035)
    cut = simulate(2035, scen(Change("labour.public_employment_growth", -0.01, start=2026),
                              Change("labour.public_employment_pop_elasticity", 0.0, start=2026))).at(2035)
    ratio = cut.labour["employees_public"] / simulate(2025).at(2025).labour["employees_public"]
    # the norm comes on top of the baseline path of the official outlook (2026-2031)
    from belsim.core.params import ParamSet
    ps = ParamSet.load()
    expected = np.prod([0.99 + ps.at(y, None)["labour.public_employment_outlook_growth"] for y in range(2026, 2036)])
    assert ratio == pytest.approx(expected, rel=1e-6)
    assert cut.labour["employees_public"] < base.labour["employees_public"]
    assert cut.labour["employment"] == pytest.approx(base.labour["employment"], rel=1e-9)


def test_behavioural_adjustments():
    # 2032: the exogenous cyclical gap is closed, so the responses can be checked exactly
    base = baseline_2070().at(2032)
    delta = np.zeros((N_AGE, N_SEX))
    delta[30:50, 1] = 0.02
    sup = simulate(2032, adjust=lambda y: {"emp_rate_delta": delta} if y == 2032 else {}).at(2032)
    extra = employed_by_age(sup) - employed_by_age(base)
    np.testing.assert_allclose(extra[30:50].sum(), 0.02 * base.pop[30:50, 1].sum(), rtol=1e-6)
    assert abs(extra[:30]).max() < 1e-6 and abs(extra[50:]).max() < 1e-6

    dem = simulate(2032, adjust=lambda y: {"labour_demand_factor": 0.98} if y == 2032 else {}).at(2032)
    assert dem.labour["employees_private"] == pytest.approx(0.98 * base.labour["employees_private"], rel=1e-6)
    assert dem.labour["employees_public"] == pytest.approx(base.labour["employees_public"], rel=1e-9)
    assert dem.labour["unemployment_rate"] > base.labour["unemployment_rate"]

    okun = paramset().at(2032, None)["labour.okun_coefficient"]
    gap = simulate(2032, gap=lambda y: -0.02 if y == 2032 else 0.0).at(2032)
    assert gap.labour["employees_private"] == pytest.approx(
        (1 - 0.02 * okun) * base.labour["employees_private"], rel=1e-6)

    inc = simulate(2032, adjust=lambda y: {"self_employed_income_factor": 0.9}).at(2032)
    assert inc.labour["self_employed_income"] == pytest.approx(0.9 * base.labour["self_employed_income"], rel=1e-9)


def test_missing_adjust_keys_are_neutral():
    a = simulate(2026).at(2026)
    neutral = {"emp_rate_delta": np.zeros((N_AGE, N_SEX)), "labour_demand_factor": 1.0, "retirement_shift": 0.0}
    b = simulate(2026, adjust=lambda y: neutral).at(2026)
    np.testing.assert_allclose(a.status, b.status, rtol=1e-12, atol=1e-9)


def test_wage_levers():
    base = baseline_2070().at(2030)
    jump = simulate(2030, scen(Change("labour.indexation_factor", 0.0, start=2027, end=2027))).at(2030)
    assert jump.macro["wage_index"] < base.macro["wage_index"] * 0.985
    assert jump.macro["health_index"] == pytest.approx(base.macro["health_index"])
    assert jump.labour["avg_wage_private"] < base.labour["avg_wage_private"]
    norm = simulate(2030, scen(Change("labour.wage_norm_real_cap", 0.0, start=2027))).at(2030)
    assert norm.macro["wage_index"] < base.macro["wage_index"]


def test_sector_structure_follows_demography_and_stays_bounded():
    res = baseline_2070()
    s0, s1 = res.at(2024), res.at(2070)
    sh0, sh1 = s0.emp.sum(axis=1) / s0.emp.sum(), s1.emp.sum(axis=1) / s1.emp.sum()
    assert sh1[SEC["Q"]] > sh0[SEC["Q"]]                         # ageing raises health & care
    assert (sh1 > 0.5 * sh0).all() and (sh1 < 2.0 * sh0).all()   # no sector vanishes or explodes
    sector = pd.read_csv(DATA / "labour_sector.csv").set_index("sector").reindex(SECTORS)
    np.testing.assert_allclose(sh0, sector["emp_ths"] / sector["emp_ths"].sum(), atol=0.004)


def test_long_run_labour_market_is_stable():
    res = baseline_2070()
    for s in res.states:
        lab = s.labour
        assert 0.70 < lab["employment_rate_20_64"] < 0.80
        assert 0.02 < lab["unemployment_rate"] < 0.08
        assert 0.15 < lab["employees_public"] / lab["employment"] < 0.22
    last, first = res.at(2070).labour, res.at(2024).labour
    assert last["employment_rate_20_64"] > first["employment_rate_20_64"]
    # real wages grow with productivity, not explosively
    real = (res.at(2070).macro["wage_index"] / res.at(2070).macro["cpi"]) ** (1 / 46) - 1
    assert 0.007 < real < 0.016


def test_speed():
    res = simulate(2026)
    prev = res.at(2025)
    p = paramset().at(2026, None)
    st = YearState(year=2026)
    macro.prices(st, prev, p, None)
    st.pop = population(2026)
    t0 = time.perf_counter()
    for _ in range(20):
        macro.prices(st, prev, p, None)
        labour.run(st, prev, p, None)
        macro.output(st, prev, p, None)
    assert (time.perf_counter() - t0) / 20 < 0.015
