"""Macro module: 2024 calibration to the national accounts, identities, levers, long-run stability."""
import numpy as np
import pandas as pd
import pytest

from belsim.core.dims import BASE_YEAR, CONS_CATS
from belsim.core.params import Change, Measure, Scenario
from test_labour_support import DATA, baseline_2070, simulate

GDP = pd.read_csv(DATA / "macro_gdp.csv").set_index("key")["value"]
REQUIRED = [
    "inflation", "cpi", "health_index", "wage_index", "interest_rate_10y", "interest_rate_short",
    "demand_gap", "gdp_nominal", "gdp_real", "gdp_real_growth", "gdp_deflator", "consumption_households",
    "investment", "investment_housing", "gov_consumption", "exports", "imports", "compensation_employees",
    "wages_salaries", "gross_operating_surplus", "mixed_income", "corporate_profits_taxable",
    "household_capital_income", "household_financial_wealth", "property_transactions", "inheritance_flow",
    "productivity_growth",
] + [f"cons.{c}" for c in CONS_CATS]
LABOUR_REQUIRED = [
    "employment", "unemployment_rate", "employment_rate_20_64", "activity_rate_20_64", "hours_per_worker",
    "blue_collar_share", "statutory_share", "flexi_jobs", "student_jobs", "service_voucher_workers",
    "cross_border_out", "cross_border_in", "avg_wage_private", "avg_wage_public", "avg_income_self_employed",
    "wage_bill_private", "wage_bill_public", "self_employed_income",
]


def scen(*changes: Change) -> Scenario:
    return Scenario(name="s", measures=[Measure(name="m", changes=list(changes))])


def test_required_keys_present_and_finite():
    for s in baseline_2070().states:
        for k in REQUIRED:
            assert np.isfinite(s.macro[k]), k
        for k in LABOUR_REQUIRED:
            assert np.isfinite(s.labour[k]), k
        assert np.isfinite(s.va).all() and (s.va > 0).all()


def test_base_year_reproduces_national_accounts():
    m = simulate(BASE_YEAR).at(BASE_YEAR).macro
    expected = {
        "gdp_nominal": "B1GQ", "consumption_households": "P31_S14_S15", "gov_consumption": "P3_S13",
        "investment": "P5G", "investment_housing": "GFCF_N111G", "exports": "P6", "imports": "P7",
        "compensation_employees": "D1", "wages_salaries": "D11",
    }
    for key, code in expected.items():
        assert m[key] == pytest.approx(GDP[code], rel=1e-6), key
    assert 600_000 < m["gdp_nominal"] < 640_000              # official 2024 GDP: EUR 619.4bn
    assert m["gdp_real"] == pytest.approx(m["gdp_nominal"]) and m["gdp_deflator"] == 1.0
    assert m["cpi"] == m["health_index"] == m["wage_index"] == 1.0
    assert m["gross_operating_surplus"] + m["mixed_income"] == pytest.approx(GDP["B2A3G"], rel=1e-6)
    assert m["household_financial_wealth"] == pytest.approx(GDP["HH_ASS_F"], rel=1e-9)
    assert simulate(BASE_YEAR).at(BASE_YEAR).va.sum() == pytest.approx(GDP["B1G"], rel=1e-6)


def test_accounting_identities_every_year():
    for s in baseline_2070().states:
        m = s.macro
        gdp = m["gdp_nominal"]
        exp = m["consumption_households"] + m["investment"] + m["gov_consumption"] + m["exports"] - m["imports"]
        assert exp == pytest.approx(gdp, rel=1e-9)
        inc = (m["compensation_employees"] + m["gross_operating_surplus"] + m["mixed_income"]
               + m["taxes_less_subsidies_production"])
        assert inc == pytest.approx(gdp, rel=1e-9)
        assert sum(m[f"cons.{c}"] for c in CONS_CATS) == pytest.approx(m["consumption_households"], rel=1e-9)
        assert m["gdp_real"] * m["gdp_deflator"] == pytest.approx(gdp, rel=1e-9)
        assert m["wages_salaries"] < m["compensation_employees"]
        assert m["investment_housing"] < m["investment"]


def test_price_indices_follow_parameters():
    res = baseline_2070()
    for prev, cur in zip(res.states[:-1], res.states[1:]):
        assert cur.macro["cpi"] == pytest.approx(prev.macro["cpi"] * (1 + cur.macro["inflation"]))
        assert cur.macro["wage_index"] > prev.macro["wage_index"]
    m = res.at(2070).macro
    assert m["inflation"] == pytest.approx(0.02)
    assert m["health_index"] == pytest.approx(m["cpi"], rel=0.01)
    assert 0.02 < m["interest_rate_10y"] < 0.05


def test_long_run_growth_and_stable_ratios():
    res = baseline_2070()
    g = (res.at(2070).macro["gdp_real"] / res.at(2040).macro["gdp_real"]) ** (1 / 30) - 1
    assert 0.012 < g < 0.016
    first = res.at(2024).macro
    for s in res.states:
        m = s.macro
        y = m["gdp_nominal"]
        assert -0.03 < m["gdp_real_growth"] < 0.03
        for key, tol in [("consumption_households", 0.03), ("gov_consumption", 0.03), ("imports", 0.05),
                         ("compensation_employees", 0.03), ("gross_operating_surplus", 0.03),
                         ("corporate_profits_taxable", 0.03), ("household_capital_income", 0.02),
                         ("property_transactions", 0.01), ("inheritance_flow", 0.04)]:
            assert abs(m[key] / y - first[key] / first["gdp_nominal"]) < tol, (s.year, key)
        assert m["household_financial_wealth"] / y == pytest.approx(2.66, abs=0.3)
        assert m["corporate_profits_taxable"] > 0.08 * y


def test_supply_and_demand_levers():
    base = baseline_2070().at(2030).macro
    gap = simulate(2030, gap=lambda y: 0.01 if y == 2030 else 0.0).at(2030).macro
    assert gap["gdp_real"] > base["gdp_real"] * 1.01          # 1% gap plus the Okun employment response
    assert gap["gdp_real"] < base["gdp_real"] * 1.02
    prod = simulate(2030, adjust=lambda y: {"productivity_factor": 1.01} if y == 2030 else {}).at(2030).macro
    assert prod["gdp_real"] == pytest.approx(base["gdp_real"] * 1.01, rel=1e-9)
    fast = simulate(2030, scen(Change("macro.productivity_growth", 0.005, start=2025, op="add"))).at(2030).macro
    assert fast["gdp_real"] > base["gdp_real"] * 1.02
    assert abs(fast["compensation_employees"] / fast["gdp_nominal"]
               - base["compensation_employees"] / base["gdp_nominal"]) < 0.01


def test_tax_base_adjust_factors():
    base = baseline_2070().at(2030).macro
    adj = {"cit_base_factor": 0.9, "capital_income_base_factor": 0.8, "property_transactions_factor": 1.1,
           "cons_factor": {"tobacco": 0.5}}
    alt = simulate(2030, adjust=lambda y: adj if y == 2030 else {}).at(2030).macro
    assert alt["corporate_profits_taxable"] == pytest.approx(0.9 * base["corporate_profits_taxable"], rel=1e-9)
    assert alt["household_capital_income"] == pytest.approx(0.8 * base["household_capital_income"], rel=1e-9)
    assert alt["property_transactions"] == pytest.approx(1.1 * base["property_transactions"], rel=1e-9)
    assert alt["cons.tobacco"] == pytest.approx(0.5 * base["cons.tobacco"], rel=1e-9)
    assert alt["cons.food"] == pytest.approx(base["cons.food"], rel=1e-9)
    assert alt["gdp_nominal"] == pytest.approx(base["gdp_nominal"], rel=1e-12)


def test_savings_rate_lowers_consumption_and_builds_wealth():
    base = baseline_2070().at(2035).macro
    alt = simulate(2035, scen(Change("macro.household_savings_rate", 0.02, start=2030, op="add"))).at(2035).macro
    assert alt["consumption_households"] < base["consumption_households"]
    assert alt["household_financial_wealth"] > base["household_financial_wealth"]
    assert alt["imports"] < base["imports"]


def test_inheritance_flow_grows_with_deaths():
    res = baseline_2070()
    a, b = res.at(2024), res.at(2050)
    ratio_flow = (b.macro["inheritance_flow"] / b.macro["gdp_nominal"]) / (a.macro["inheritance_flow"] / a.macro["gdp_nominal"])
    assert b.demo["deaths"] > a.demo["deaths"]
    assert ratio_flow > 1.0
