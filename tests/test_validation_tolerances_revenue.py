"""Regression guards for the revenue side: static costings of tax measures against official
figures, the tax base by bracket and by VAT rate, and the calibration of the tax lines.

Tolerances are wider than the current gaps (given in the comments); references and sources are
in data/reference/validation.yaml and params/revenue.yaml. Static = behavioural responses off.
"""
import numpy as np
import pytest
import yaml

from belsim import validation as V
from belsim.core import engine
from belsim.core.params import Change, Measure, ParamSet, Scenario

END = 2031
PIT = ("pit_federal", "pit_regional", "pit_local")
STATIC = Measure(name="static", changes=[Change(V.STATIC_SWITCH, 0, 2025)])
IDS = {"dc2024_allowance_13000", "arizona_allowance_15600", "dc2024_pit_45_to_40", "dc2024_abolish_special_ssc",
       "vat_standard_22_press_2026", "vat_reduced_rates_to_9"}


@pytest.fixture(scope="module")
def base():
    return engine.run(None, end=END)


@pytest.fixture(scope="module")
def report(base, tmp_path_factory):
    data = V.load()
    data["measures"] = [m for m in data["measures"] if m["id"] in IDS]
    data["baseline"] = [b for b in data["baseline"] if b["id"] in ("budget2026_excise_total",)]
    data["not_found"] = []
    path = tmp_path_factory.mktemp("validation") / "revenue.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return V.run_validation(path, base=base, end=END)


def static_effect(base, changes, year, lines, reference=()):
    """Scenario minus reference (default: baseline) for the sum of revenue lines, EUR mn, static."""
    def run(ch):
        return engine.run(Scenario(name="t", measures=[Measure(name="m", changes=list(ch)), STATIC]), end=year, baseline=base)
    ref = run(reference) if reference else base
    res = run(changes)
    return sum(res.at(year).revenue[k] - ref.at(year).revenue[k] for k in lines)


def at_2024_scale(base, value, year):
    return value * base.at(2024).macro["gdp_nominal"] / base.at(year).macro["gdp_nominal"]


# (reference id, target label prefix, relative tolerance) - model / official in the comment
MEASURES = [
    ("dc2024_pit_45_to_40", "PIT revenue", 0.16),                     # 1.11 (was 1.25)
    ("dc2024_allowance_13000", "PIT revenue", 0.12),                  # 0.95
    ("arizona_allowance_15600", "PIT revenue (all levels) 2027", 0.08),   # 0.97 (was 0.88)
    ("arizona_allowance_15600", "PIT revenue (all levels) 2029", 0.06),   # 1.00 (was 0.91)
    ("arizona_allowance_15600", "PIT revenue (all levels) 2030", 0.06),   # 0.98 (was 0.89)
    ("dc2024_abolish_special_ssc", "special contribution", 0.25),     # 1.12
    ("vat_standard_22_press_2026", "VAT revenue", 0.20),              # 1.08 (was 1.13)
]


@pytest.mark.parametrize("id_,label,tol", MEASURES, ids=lambda v: str(v))
def test_static_costing_matches_official_figure(report, id_, label, tol):
    r = next(x for x in report.measures if x.id == id_ and x.label.startswith(label))
    assert r.static is not None, r.status
    assert abs(r.static - r.ref) <= tol * abs(r.ref), f"{id_} / {r.label}: model {r.static:.3f} vs official {r.ref:.3f}"


def test_reduced_vat_rates_to_nine_percent_is_inside_the_quoted_range(report):
    """6 % and 12 % merged into 9 %: 'between 1.5 and 3 bn' - the model moves every product and
    sits in the middle (2.2 bn at 2025 scale; 2.75 bn before housing investment was put on the
    Planning Bureau's rate mix, a quarter of it at 6 % instead of half)."""
    low = next(x for x in report.measures if x.id == "vat_reduced_rates_to_9" and "low" in x.label)
    high = next(x for x in report.measures if x.id == "vat_reduced_rates_to_9" and "high" in x.label)
    assert 1.2 * low.ref < low.static < 0.9 * high.ref


def test_planning_bureau_bracket_costings(base):
    """DC2024 fiches of the Planning Bureau (2024 prices and situation): 40 % rate to 30 %
    -6,545 mn, 50 % threshold raised to 61,000 EUR -552 mn, allowance to 11,950 EUR -3,007 mn."""
    cost = lambda ch: at_2024_scale(base, static_effect(base, ch, 2027, PIT), 2027)
    index = base.at(2027).macro["health_index"]
    assert cost([Change("pit.bracket_rates", 0.30, 2027, key="1")]) == pytest.approx(-6545, rel=0.12)          # 1.00
    assert cost([Change("pit.bracket_thresholds", 61000 * index, 2027, key="2")]) == pytest.approx(-552, rel=0.30)   # 1.17


def test_vat_costings_with_the_model_rows(base):
    """Zero rate on food (Planning Bureau DC2024: -2,093 mn in 2024) and hotels at 12 % (budget:
    158 mn a year), with the rows of the current vat.mix."""
    mix = ParamSet.load().at(2024)["vat.mix"]
    food = list(mix["food"])
    zero = [food[0], food[1], 0.0, food[2] + food[3]]
    cost = at_2024_scale(base, static_effect(base, [Change("vat.mix", zero, 2027, key="food")], 2027, ("vat",)), 2027)
    assert cost == pytest.approx(-2093, rel=0.15)                    # 0.95 (was 0.76)
    hotels = -static_effect(base, [Change("vat.mix", list(mix["restaurants"]), 2026, key="restaurants")], 2027, ("vat",))
    assert hotels == pytest.approx(158, rel=0.15)                    # 155


def test_government_spending_cut_loses_its_own_vat(base):
    """A cut of public consumption or investment of 0.5 % of GDP lowers VAT by about 0.06 % of
    GDP: the non-deductible VAT government pays on its purchases (HERMES: 459 and 654 mn of
    indirect taxes at 2024 scale)."""
    gdp = base.at(2027).macro["gdp_nominal"]
    for change in (Change("exp.function_factor", 0.958, 2027, op="mul"), Change("invest.public_pct_gdp", -0.5, 2027, op="add")):
        loss = static_effect(base, [change], 2027, ("vat",))
        assert -0.10 < 100 * loss / gdp < -0.04, (change.param, loss)


def test_excise_level_and_first_year(report):
    """ESA excise lines against federal cash receipts (11,130 mn in 2024, 11,216 mn in 2025):
    within 6 % - the ESA lines hold levies the excise cash figure does not (see params/revenue.yaml)."""
    for r in report.baseline:
        if r.id == "budget2026_excise_total":
            assert r.model == pytest.approx(r.ref, rel=0.06), (r.when, r.model, r.ref)


def test_tax_base_follows_the_tax_returns(base):
    """Calibration of the income-tax and VAT lines, and the structure taken from the data."""
    info = base.at(2024).extra["revenue"]
    calib, pit = info["calib"], info["pit"]
    assert calib["pit_federal"] == pytest.approx(1.0, abs=0.05)       # 1.005
    assert 0.88 < calib["pit_regional"] < 1.0                         # 0.92: surcharges recorded when assessed
    assert 0.85 < calib["pit_local"] < 1.0                            # 0.89: same, municipal
    assert calib["vat"] == pytest.approx(1.0, abs=0.03)               # 1.000
    actual = sum(base.at(2024).revenue[k] for k in PIT)
    assert actual == pytest.approx(49512.3 + 13294.6 + 4105.5, abs=0.5)
    assert base.at(2024).revenue["vat"] == pytest.approx(37832.3, abs=0.1)
    # income taxed at each marginal rate (EUR mn), Statbel fiscal statistics of income 2024:
    # 124 / 67 / 50 / 34 bn at 25 / 40 / 45 / 50 %
    brackets = np.array(pit["brackets"]) / 1e3
    assert brackets == pytest.approx([124, 67, 50, 34], abs=4)
    assert sum(pit["brackets"]) == pytest.approx(275_225, rel=0.01)   # table B.2, net income taxed globally
    for group, fit in pit["distribution_fit"].items():
        assert fit == pytest.approx(1.0, abs=0.01), group
    assert np.all(np.abs(pit["tax_rel"] - 1.0) < 0.08)                # was 0.93 - 1.21


# ---------------------------------------------------------------------------------------------
# revenue by category against the Planning Bureau outlook of June 2026 (Tableaux A.2, C.3, D.1)
# and the observed 2025 accounts
# ---------------------------------------------------------------------------------------------
FPB_GDP = {2024: 620272, 2025: 642012, 2026: 663326, 2027: 687370, 2028: 710640, 2029: 733587, 2030: 756470, 2031: 778846}
FPB = {     # EUR mn, 2024-2031
    "total": [308230, 314849, 328113, 338525, 348524, 356052, 364844, 373989],
    "corporate": [27602, 26852, 28745, 29608, 30202, 30856, 31270, 31638],
    "households": [76906, 79100, 82929, 85912, 88109, 88674, 90819, 93773],          # households + other direct taxes
    "indirect": [73082, 74760, 77106, 79839, 83668, 86484, 88964, 90803],
    "property_income": [6244, 5695, 5808, 5820, 5986, 6146, 6307, 6464],
    "transfers": [6260, 6456, 7114, 6017, 5720, 5589, 5265, 5246],                    # current + capital transfers, subsidies
    "contributions": [80645, 83194, 85362, 89195, 92190, 94776, 97732, 100593],       # effective social contributions
    "capital_taxes": [4258, 4318, 5019, 4763, 4255, 4342, 4478, 4610],
}
CATEGORY = {
    "corporate": ("cit",),
    "households": ("pit_federal", "pit_regional", "pit_local", "special_ssc", "withholding_tax", "capital_gains_tax",
                   "securities_account_tax", "vehicle_taxes", "other_taxes_income_wealth"),
    "property_income": ("interest_received", "dividends_received", "rent_other_property_income"),
    "transfers": ("eu_transfers_current", "other_current_transfers", "eu_transfers_capital", "other_capital_transfers"),
    "contributions": ("ssc_employer_private", "ssc_employer_public", "ssc_employee", "ssc_self_employed", "ssc_replacement_income"),
    "capital_taxes": ("inheritance_gift_tax",),
}
# largest gap, in points of GDP: (2025 = observed accounts, change since 2024 against the
# outlook's own 2025 - what the accounts of October 2026 show above or below the Bureau's June
# estimate; 2026-2031 = change since the observed 2025 against the Bureau's change since its
# 2025). The 2025 level is never overridden: the tax categories follow the Bureau's change by
# category from there (rev.fpb_alignment, computed by belsim.fiscal.revenue.align). Levels are
# not compared with 2024: the categories are not cut the same way there.
CATEGORY_TOL = {"corporate": (0.05, 0.01),          # 0.01, 0.00
                "households": (0.12, 0.01),         # -0.09 (federal tax and other household taxes), 0.00
                "indirect": (0.15, 0.01),           # 0.12 (mostly VAT), 0.00
                "contributions": (0.04, 0.01),      # -0.01, 0.00
                "capital_taxes": (0.05, 0.01),      # 0.03, 0.00
                "property_income": (0.08, 0.02),    # 0.05, 0.00 (rev.line_factor, relative to 2025)
                "transfers": (0.06, 0.04)}          # 0.04, -0.03 (levels of capital transfers taken from the outlook)


def _ratio(base, year, lines=None):
    s = base.at(year)
    rev = s.revenue
    if lines is None:
        value = sum(rev.values())
    elif lines == "indirect":
        from belsim.fiscal.revenue import LINES
        value = sum(v for k, v in rev.items() if LINES[k].esa.startswith("D2"))
    else:
        value = sum(rev[k] for k in lines)
    return 100.0 * value / s.macro["gdp_nominal"]


@pytest.mark.parametrize("category", list(CATEGORY_TOL))
def test_revenue_category_follows_the_planning_bureau(base, category):
    lines = "indirect" if category == "indirect" else CATEGORY[category]
    official = lambda i, year: 100.0 * FPB[category][i] / FPB_GDP[year]
    surprise = _ratio(base, 2025, lines) - _ratio(base, 2024, lines) - (official(1, 2025) - official(0, 2024))
    assert abs(surprise) <= CATEGORY_TOL[category][0], (category, 2025, round(surprise, 3))
    for i, year in enumerate(range(2026, END + 1), start=2):
        model = _ratio(base, year, lines) - _ratio(base, 2025, lines)
        ref = official(i, year) - official(1, 2025)
        assert abs(model - ref) <= CATEGORY_TOL[category][1], (category, year, round(model, 3), round(ref, 3))


def test_total_revenue_ratio_against_the_planning_bureau(base):
    """General-government revenue, % of GDP. 2025 is the observed outturn, 0.18 above the
    outlook's estimate (0.05 of it is the later vintage of the 2024 accounts and GDP). That level
    is kept: 0.07 to 0.16 above the outlook in 2026-2031 (0.09 / 0.07 / 0.08 / 0.10 / 0.13 /
    0.16; sales and imputed contributions, which are not aligned, make the difference with 0.18)."""
    for i, year in enumerate(range(2025, END + 1), start=1):
        official = 100.0 * FPB["total"][i] / FPB_GDP[year]
        assert 0.0 < _ratio(base, year) - official <= 0.20, (year, _ratio(base, year), official)


def test_alignment_on_the_planning_bureau_can_be_switched_off(base):
    """`rev.fpb_alignment_enabled` = 0 leaves the model's own projection from the observed 2025:
    revenue within 0.06 point until 2028, 0.15 to 0.26 point of GDP higher in 2029-2031, nothing
    changed in 2025; the aligned ratio holds after 2031."""
    off = engine.run(Scenario(name="t", measures=[Measure(name="m", changes=[Change("rev.fpb_alignment_enabled", 0, 2025)]), STATIC]),
                     end=END, baseline=base)
    gap = lambda y: _ratio(off, y) - _ratio(base, y)
    assert abs(gap(2025)) < 1e-9
    assert all(abs(gap(y)) < 0.08 for y in (2026, 2027, 2028))            # -0.05 / 0.01 / 0.02
    assert all(0.08 < gap(y) < 0.35 for y in (2029, 2030, 2031))          # 0.15 / 0.20 / 0.26
    long = engine.run(None, end=2034)
    steps = [_ratio(long, y + 1) - _ratio(long, y) for y in range(2031, 2034)]
    assert all(abs(d) < 0.12 for d in steps), steps                       # no jump when the path ends


def test_income_tax_reform_follows_the_official_costing(base):
    """Court of Audit, annex 1 of the commentary on the 2026 budget (FPS Finance): measures of the
    reform outside the calculator -207 / -320 / -319 / -301 / -169 mn in 2026-2030, all levels."""
    official = {2026: -207.4, 2027: -320.1, 2028: -318.6, 2029: -301.0, 2030: -169.1}
    without = engine.run(Scenario(name="t", measures=[Measure(name="m", changes=[Change("pit.reform_other_measures_cost", 0, 2025)]), STATIC]),
                         end=2030, baseline=base)
    for year, ref in official.items():
        model = sum(base.at(year).revenue[k] - without.at(year).revenue[k] for k in PIT)
        assert model == pytest.approx(ref, rel=0.08), (year, model)


def test_baseline_reproduces_the_observed_2025_accounts(base):
    """Every revenue line equals the 2025 outturn (Eurostat / NBB national tax list, EUR mn)."""
    from belsim.fiscal.revenue import outturn
    observed = outturn()[2025]
    rev = base.at(2025).revenue
    for line, value in observed.items():
        assert rev[line] == pytest.approx(value, abs=0.05), line
    assert sum(rev.values()) == pytest.approx(315_402.6, abs=1.0)
    # the 2024 base year is untouched
    assert sum(base.at(2024).revenue.values()) == pytest.approx(308_088.3, abs=0.5)


def test_excise_follows_nominal_duties_and_volume_trends(base):
    """Duties are not indexed before 2032 and fuel volumes fall 1.6 % a year (road-transport
    emissions of the Planning Bureau outlook): fuel excise stays below its 2025 level in euros."""
    fuel = [base.at(y).revenue["excise_fuel"] for y in range(2025, END + 1)]
    assert all(b < a for a, b in zip(fuel, fuel[1:]))
    assert fuel[-1] / fuel[0] == pytest.approx(0.984 ** 6, rel=0.03)
    tobacco = [base.at(y).revenue["excise_tobacco"] for y in range(2025, END + 1)]
    assert tobacco[-1] / tobacco[0] == pytest.approx(0.99 ** 6, rel=0.03)
    # ETS2 receipts from 2028: 46 EUR/t on about 41 Mt
    jump = base.at(2028).revenue["environmental_taxes"] - base.at(2027).revenue["environmental_taxes"]
    assert jump == pytest.approx(1_890, rel=0.06)


def test_corporate_rate_cut_like_for_like(base):
    """2018-2020 reform: the nominal rate from 33.99 to 25 % took the implicit rate from 32.20 to
    24.25 % (Court of Audit, FPS Finance), i.e. -24.7 % of the receipts at the old rate. The
    amount in euros (4.8 bn on the narrow base of 2018) is not comparable with today's base."""
    old = [Change("cit.rate", 0.3399, 2027)]
    year = 2029
    run = lambda ch: engine.run(Scenario(name="t", measures=[Measure(name="m", changes=list(ch)), STATIC]), end=year, baseline=base)
    before, after = run(old).at(year).revenue["cit"], base.at(year).revenue["cit"]
    assert after / before - 1.0 == pytest.approx(24.25 / 32.20 - 1.0, abs=0.025)       # -24.7 % in the model too


def test_household_distribution_against_the_income_survey(base):
    """EU-SILC 2025 (incomes of 2024): Gini 0.234, tenths' income shares, poverty rate 10.9 %,
    median 31,299 EUR. Model, on the household types of the survey: Gini 0.240, shares within
    0.5 point (top tenth 19.6 against 19.9), poverty rate 10.1 %, mean income equal, median
    2.2 % low (it was 0.1 % low on the labour-force-survey household frame, which had too many
    well-off households of three adults and missed the poverty rates by household type by 3 to
    4 points). By household type and activity status: tests/test_distribution.py."""
    from belsim.distribution import distribution
    pset = ParamSet.load()
    out = distribution(base, base, pset, pset, 2024)
    ind = out["indicators"]
    assert ind["gini"]["baseline"] == pytest.approx(0.234, abs=0.010)
    assert ind["gini"]["observed"]["value"] == pytest.approx(0.234, abs=1e-6)
    assert ind["poverty_rate"]["baseline"] == pytest.approx(0.109, abs=0.010)
    assert ind["poverty_rate"]["line_baseline"] / 0.6 == pytest.approx(31_299, rel=0.025)      # -2.2 %
    for d in out["deciles"]:
        assert d["income_share"] == pytest.approx(d["income_share_observed"]["value"], abs=0.01), d["decile"]
    cov = out["meta"]["coverage"]
    assert 0.90 <= cov["income_tax"] <= 1.10 and 0.90 <= cov["social_contributions"] <= 1.10


def test_jobs_of_a_targeted_contribution_cut_are_taxed_where_they_are(base):
    """HERMES 2018, employer contributions cut by 0.5 % of GDP: targeted at low wages the cut
    costs 0.26 point of GDP after feedback at t+4, the jobs it creates being low-wage jobs.
    Labour places the jobs of a scenario in the wage deciles
    (`labour["scenario_jobs_private_by_decile"]`) and re-cuts `wage_dist` into the tenths of the
    workers after the change; the tax units take a tenth of the workers at each of those wages,
    so the new jobs pay the contributions and income tax of their own wage (it was 0.17 point,
    the new jobs taxed as near-average ones)."""
    year = 2031
    run = lambda ch: engine.run(Scenario(name="t", measures=[Measure(name="m", changes=[ch])]), end=year, baseline=base)
    low = run(Change("ssc.structural_reduction_slope", 0.7804, 2027, op="add")).at(year)
    general = run(Change("ssc.employer_rate", -0.02124, 2027, op="add")).at(year)
    b = base.at(year)
    assert not any(b.labour.get("scenario_jobs_private_by_decile", [0.0]))        # baseline: no jobs to place
    returns = {}
    for name, s in (("low", low), ("general", general)):
        jobs = np.array(s.labour["scenario_jobs_private_by_decile"])
        added = s.labour["employees_private"] - b.labour["employees_private"]
        assert jobs.shape == (10,) and jobs.sum() == pytest.approx(added, rel=1e-6)
        # consistent wage bill: a tenth of the workers at each wage of `wage_dist`
        assert s.labour["employees_private"] * np.mean(s.wage_dist[0]) / 1e6 == pytest.approx(s.labour["wage_bill_private"], rel=1e-9)
        paid = sum(s.revenue[k] - b.revenue[k] for k in PIT + ("ssc_employee",))
        returns[name] = paid * 1e6 / added                                   # income tax and employee contributions per job
    assert np.array(low.labour["scenario_jobs_private_by_decile"])[:4].sum() > 0.9 * (low.labour["employees_private"] - b.labour["employees_private"])
    assert returns["low"] < 0.4 * returns["general"]                         # 2,400 against 15,000 EUR a job
    cost = low.fiscal["balance_pct_gdp"] - b.fiscal["balance_pct_gdp"]
    assert cost == pytest.approx(-0.26, abs=0.04)                            # -0.27


def test_minimum_pension_rise_is_taxed_as_low_pensions(base):
    """Vlaams Belang 301 (DC2024): minimum pension +2.96 %, EUR 578 mn. The spending it adds goes
    to pensions at or below the minimum, which pay little or no income tax: 2 to 7 % of it comes
    back (it was 51 %, the rise being spread over all pensions in proportion)."""
    for factor in (1.0296, 0.97):
        s = engine.run(Scenario(name="t", measures=[Measure(name="m", changes=[Change("pension.min_pension", factor, 2027, op="mul")]), STATIC]),
                       end=END, baseline=base)
        for year in (2027, END):
            a, b = s.at(year), base.at(year)
            spending = sum(a.expenditure[k] - b.expenditure[k] for k in a.expenditure if k.startswith("pension_"))
            tax = sum(a.revenue[k] - b.revenue[k] for k in PIT)
            assert abs(spending) > 500
            assert 0.0 <= tax / spending < 0.10, (factor, year, tax / spending)
            assert abs(sum(a.revenue.values()) - sum(b.revenue.values()) - tax) < 0.03 * abs(spending)
