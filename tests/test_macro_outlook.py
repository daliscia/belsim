"""Baseline GDP against the official outlook, and the price channels added to macro.

Official figures: Eurostat nama_10_gdp (2025 outturn) and Federal Planning Bureau, Economic
outlook 2026-2031 (June 2026), Tableaux A.2-A.4 (GDP at current prices, real growth, deflator).
The Bureau's 2024 level (EUR 620,272 mn) is an older vintage than the model's base year
(Eurostat, EUR 619,368 mn), so levels are compared as an index, 2024 = 1.
"""
import pytest

from belsim.core import engine
from belsim.core.params import Change, Measure, Scenario

# year: (GDP at current prices EUR mn, real growth %, GDP deflator %) - Planning Bureau, June 2026
FPB = {
    2024: (620_272, 1.1, 1.9), 2025: (642_012, 1.0, 2.5), 2026: (663_326, 0.7, 2.6), 2027: (687_370, 1.0, 2.6),
    2028: (710_640, 1.2, 2.2), 2029: (733_587, 1.4, 1.8), 2030: (756_470, 1.3, 1.8), 2031: (778_846, 1.2, 1.8),
}
EUROSTAT_2025 = (640_852.6, 0.9, 2.5)      # nominal GDP, real growth, deflator (vintage of the base year)


@pytest.fixture(scope="module")
def base():
    return engine.run_baseline()


@pytest.mark.parametrize("year", range(2025, 2032))
def test_nominal_gdp_tracks_the_planning_bureau_outlook(base, year):
    model = base.at(year).macro["gdp_nominal"] / base.at(2024).macro["gdp_nominal"]
    official = FPB[year][0] / FPB[2024][0]
    assert model == pytest.approx(official, rel=0.006), f"{year}: {model:.4f} vs {official:.4f}"


@pytest.mark.parametrize("year", range(2026, 2032))
def test_real_growth_and_deflator_follow_the_planning_bureau_outlook(base, year):
    m, prev = base.at(year).macro, base.at(year - 1).macro
    assert m["gdp_real_growth"] * 100 == pytest.approx(FPB[year][1], abs=0.2)
    assert (m["gdp_deflator"] / prev["gdp_deflator"] - 1.0) * 100 == pytest.approx(FPB[year][2], abs=0.15)


def test_2025_is_the_eurostat_outturn(base):
    m = base.at(2025).macro
    assert m["gdp_nominal"] == pytest.approx(EUROSTAT_2025[0], rel=0.003)
    assert m["gdp_real_growth"] * 100 == pytest.approx(EUROSTAT_2025[1], abs=0.15)
    assert (m["gdp_deflator"] - 1.0) * 100 == pytest.approx(EUROSTAT_2025[2], abs=0.15)


def test_long_run_deflator_is_consumer_price_inflation(base):
    for year in (2035, 2050, 2070):
        m, prev = base.at(year).macro, base.at(year - 1).macro
        assert m["gdp_deflator"] / prev["gdp_deflator"] == pytest.approx(m["cpi"] / prev["cpi"], rel=1e-9)


def _run(base, *changes, end=2031):
    return engine.run(Scenario(name="s", measures=[Measure(name="m", changes=list(changes))]), end=end, baseline=base)


def test_market_price_gdp_equals_gdp_without_indirect_tax_changes(base):
    for year in (2024, 2027, 2040):
        m = base.at(year).macro
        assert m["gdp_nominal_market_prices"] == m["gdp_nominal"]
        assert m["cpi_demand_effect"] == 0.0
        assert m["labour_cost_competitiveness_index"] == m["real_unit_labour_cost"]
    # an income-tax change leaves indirect taxes alone
    s = _run(base, Change("pit.bracket_rates", 1.02, 2027, op="mul")).at(2029).macro
    assert s["gdp_nominal_market_prices"] == s["gdp_nominal"]


def test_vat_rise_lifts_gdp_at_market_prices_only(base):
    s, b = _run(base, Change("vat.rate_standard", 0.22, 2027)).at(2027), base.at(2027)
    extra = s.macro["gdp_nominal_market_prices"] - s.macro["gdp_nominal"]
    d_vat = s.revenue["vat"] - b.revenue["vat"]
    assert 0.5 * d_vat < extra < 1.1 * d_vat               # the tax on household consumption
    assert s.macro["gdp_nominal_market_prices"] / s.macro["gdp_real"] > s.macro["gdp_deflator"]
    assert abs(s.macro["gdp_nominal"] / b.macro["gdp_nominal"] - 1.0) < 0.002


def test_slack_lowers_prices_gradually(base):
    """A cut in public purchases opens a demand gap; prices drift down from the next year on
    (dyn.prices.demand_gap_sensitivity) and stay put when the channel is switched off."""
    cut = Change("exp.function_factor", 0.958, 2027, op="mul")
    s = _run(base, cut)
    assert s.at(2027).macro["demand_gap"] < -0.002 and s.at(2027).macro["cpi_demand_effect"] == 0.0
    effects = [s.at(y).macro["cpi_demand_effect"] for y in (2028, 2029, 2030, 2031)]
    assert all(b < a < 0.0 for a, b in zip(effects, effects[1:]))
    assert -0.0016 < effects[-1] < -0.0006                 # HERMES: consumer prices -0.12% at t+4
    assert s.at(2031).macro["cpi"] < base.at(2031).macro["cpi"]
    off = _run(base, cut, Change("dyn.prices.demand_gap_sensitivity", 0.0, 2025))
    assert off.at(2031).macro["cpi"] == pytest.approx(base.at(2031).macro["cpi"], rel=1e-12)


def test_competitiveness_sees_employer_contributions(base):
    """Labour cost per unit of output includes employer contributions: a cut lowers it although
    gross wages rise a little, and a stronger export response then means more output."""
    cut = Change("ssc.employer_rate", -0.01, 2027, op="add")
    s = _run(base, cut)
    m, b = s.at(2029).macro, base.at(2029).macro
    assert m["labour_cost_competitiveness_index"] < b["real_unit_labour_cost"] * 0.996
    assert m["labour_cost_competitiveness_index"] < m["real_unit_labour_cost"]
    strong = _run(base, cut, Change("dyn.wage_cost.net_export_output_elasticity", 0.35, 2025))
    assert strong.at(2029).macro["gdp_real"] > m["gdp_real"] * 1.0005


def test_output_prices_lead_consumer_prices_after_a_wage_cost_shock(base):
    rise = Change("ssc.employer_rate", 0.0216, 2027, op="add")
    lead = _run(base, rise).at(2027)
    none = _run(base, rise, Change("dyn.wage_cost.deflator_lead", 0.0, 2025)).at(2027)
    b = base.at(2027)
    assert lead.macro["cpi"] == pytest.approx(none.macro["cpi"], rel=1e-12)
    assert lead.macro["gdp_deflator"] > none.macro["gdp_deflator"] > b.macro["gdp_deflator"]
    assert lead.macro["corporate_profits_taxable"] > none.macro["corporate_profits_taxable"]
