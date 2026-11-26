"""Scenario-vs-baseline dynamics owned by labour + macro + behaviour (params/dynamics.yaml):
marginal workers, real wage cost, indirect taxes -> prices -> indexation, public staffing.

The first part uses the labour+macro mini-engine of test_labour_support (independent of the
fiscal modules); the second part runs library measures through the full engine and checks signs
and orders of magnitude only (no exact values: other parameter files may change).
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import yaml

from belsim.core import engine
from belsim.core.dims import BASE_YEAR, J
from belsim.core.params import Change, Measure, ParamSet, ParamSpec, Scenario
from belsim.labour.dynparams import SOURCES, defaults
from test_labour_support import baseline_2070, paramset, simulate

DYN = defaults()
END = 2032


def scen(*changes: Change) -> Scenario:
    return Scenario(name="s", measures=[Measure(name="m", changes=list(changes))])


@lru_cache(maxsize=1)
def mini_base():
    return simulate(END)


def mini(*changes: Change, **kw):
    return simulate(END, scen(*changes), baseline=mini_base(), **kw)


# ------------------------------------------------------------------ parameters file
def test_dynamics_parameters_are_documented():
    data = yaml.safe_load((Path(__file__).resolve().parents[1] / "params" / "dynamics.yaml").read_text())
    for name, spec in data.items():
        assert name.startswith("dyn."), name
        assert spec["description"] and spec["unit"] and spec["source"], name
    assert set(DYN["dyn.marginal_productivity"]) == set(SOURCES) == set(DYN["dyn.marginal_wage"])
    assert 0.5 <= DYN["dyn.marginal_wage"]["unemp_exclusion"] <= 0.6        # 50-60% of the average wage
    assert sum(DYN["dyn.public_employment.function_weights"].values()) <= 1.0


# ------------------------------------------------------------------ neutrality (mini-engine)
def test_scenario_without_measures_is_exactly_the_baseline():
    base, s = mini_base(), simulate(END, Scenario(name="empty"), baseline=mini_base(), passes=2)
    for a, b in zip(base.states, s.states):
        assert a.series() == b.series()
        np.testing.assert_array_equal(a.wage_dist, b.wage_dist)
        np.testing.assert_array_equal(a.va, b.va)
        assert b.labour["marginal_wage_factor_private"] == 1.0
        assert b.labour["marginal_productivity_factor"] == 1.0
        assert all(b.labour[f"employment_vs_baseline.{k}"] == 0.0 for k in SOURCES)


# ------------------------------------------------------------------ 1. marginal workers
def _ones_pset() -> ParamSet:
    ones = {k: 1.0 for k in SOURCES}
    return _with_specs(ParamSpec("dyn.marginal_wage", {2024: ones}), ParamSpec("dyn.marginal_productivity", {2024: ones}))


def test_jobs_lost_by_abolishing_the_time_limit_are_low_wage_low_productivity():
    change = Change("labour.unemp_benefit_max_years", 0, 2026)
    base, s = mini_base(), mini(change)
    avg = simulate(END, scen(change), baseline=simulate(END, pset=_ones_pset()), pset=_ones_pset())   # average workers
    b, a, o = base.at(2030), s.at(2030), avg.at(2030)
    jobs = a.labour["employment"] - b.labour["employment"]
    assert jobs < -10_000 and jobs == pytest.approx(o.labour["employment"] - b.labour["employment"])
    assert a.labour["employment_vs_baseline.unemp_exclusion"] == pytest.approx(jobs, rel=0.02)
    rel_w, rel_p = (DYN[k]["unemp_exclusion"] for k in ("dyn.marginal_wage", "dyn.marginal_productivity"))
    # each lost job carried rel_w x the wage and rel_p x the output of an average job
    d = lambda st, k: st.labour[k] - b.labour[k]
    assert d(a, "wage_bill_private") / d(o, "wage_bill_private") == pytest.approx(rel_w, abs=0.01)
    assert (a.va.sum() - b.va.sum()) / (o.va.sum() - b.va.sum()) == pytest.approx(rel_p, abs=0.01)
    assert (a.macro["gdp_real"] - b.macro["gdp_real"]) / (o.macro["gdp_real"] - b.macro["gdp_real"]) == pytest.approx(rel_p, abs=0.01)
    assert o.labour["marginal_productivity_factor"] == 1.0
    # the remaining workers are on average better paid; the lower deciles move most
    priv = J["emp_private"]
    assert a.labour["avg_wage_private"] > o.labour["avg_wage_private"]
    ratio = a.wage_dist[priv] / o.wage_dist[priv]
    assert ratio[0] > ratio[-1] > 1.0
    for j in range(3):
        bill = (a.wage[:, j] * a.emp[:, j]).sum()
        assert a.wage_dist[j].mean() * a.emp[:, j].sum() == pytest.approx(bill, rel=1e-9)
    assert a.labour["marginal_wage_factor_private"] > 1.0


def test_later_retirement_adds_near_average_workers():
    base, s = mini_base(), mini(Change("labour.legal_pension_age", 68, 2028))
    b, a = base.at(END), s.at(END)
    jobs = a.labour["employment"] - b.labour["employment"]
    assert jobs > 5_000
    # the structural unemployment rate applied to the larger labour force takes back 6-7%
    assert a.labour["employment_vs_baseline.pension_age"] == pytest.approx(jobs, rel=0.08)
    d_va = a.va.sum() / b.va.sum() - 1.0
    assert 0.8 * jobs / b.labour["employment"] < d_va < 1.15 * jobs / b.labour["employment"]


def test_cyclical_jobs_follow_output_without_adding_to_it():
    gap = -0.01
    s = simulate(END, Scenario(name="gap"), baseline=mini_base(), gap=lambda y: gap if y == 2028 else 0.0)
    b, a = mini_base().at(2028), s.at(2028)
    assert a.labour["employment"] < b.labour["employment"]
    assert a.labour["employment_vs_baseline.cyclical"] == pytest.approx(
        a.labour["employment"] - b.labour["employment"], rel=1e-6)
    # real GDP falls by the gap itself (the Okun jobs are not counted on top)
    assert a.macro["gdp_real"] / b.macro["gdp_real"] - 1.0 == pytest.approx(gap, rel=0.005)
    assert a.labour["wage_bill_private"] < b.labour["wage_bill_private"]
    for k in ("gdp_real", "gdp_nominal", "wages_salaries"):                    # no trace the year after
        assert s.at(2029).macro[k] == pytest.approx(mini_base().at(2029).macro[k], rel=1e-12)
    assert s.at(2029).labour["employment"] == pytest.approx(mini_base().at(2029).labour["employment"], rel=1e-12)


# ------------------------------------------------------------------ 4. public staffing
def _with_specs(*specs: ParamSpec) -> ParamSet:
    all_specs = dict(paramset().specs)
    all_specs.update({s.name: s for s in specs})
    return ParamSet(all_specs)


def _staffing_pset() -> ParamSet:
    functions = DYN["dyn.public_employment.function_weights"]
    return _with_specs(
        ParamSpec("pubwage.staffing_factor", {2024: {k: 1.0 for k in functions}}),
        ParamSpec("educ.enrolment", {2024: {"primary": 0.99, "secondary": 0.98, "tertiary": 0.5}}),
    )


def test_staffing_factor_moves_public_employment_once():
    pset = _staffing_pset()
    base = simulate(END, pset=pset)
    s = simulate(END, scen(Change("pubwage.staffing_factor", 0.9, 2027, key="general")), baseline=base, pset=pset)
    w = DYN["dyn.public_employment.function_weights"]["general"]
    for year in (2027, END):
        b, a = base.at(year), s.at(year)
        assert a.labour["employees_public"] / b.labour["employees_public"] == pytest.approx(1 - 0.1 * w, rel=1e-6)
        # the bill actually paid falls; the contract key (which expenditure multiplies by the
        # staffing factor itself) does not, so the lever is counted once
        assert a.labour["wage_bill_public_actual"] / b.labour["wage_bill_public_actual"] == pytest.approx(1 - 0.1 * w, rel=1e-3)
        assert a.labour["wage_bill_public"] == pytest.approx(b.labour["wage_bill_public"], rel=1e-3)
        assert a.macro["gov_consumption"] < b.macro["gov_consumption"]
        assert a.va[12:15].sum() < b.va[12:15].sum()              # sectors O, P, Q
    # displaced staff are absorbed by the private sector only gradually
    gap = [s.at(y).labour["employment"] - base.at(y).labour["employment"] for y in range(2027, END + 1)]
    assert gap[0] < -0.5 * 0.1 * w * base.at(2027).labour["employees_public"]
    assert all(x < y for x, y in zip(gap, gap[1:])) and gap[-1] > 0.25 * gap[0]
    assert s.at(2026).series() == base.at(2026).series()


def test_enrolment_lever_moves_education_staff():
    pset = _staffing_pset()
    base = simulate(END, pset=pset)
    s = simulate(END, scen(Change("educ.enrolment", 0.55, 2027, key="tertiary")), baseline=base, pset=pset)
    w = (DYN["dyn.public_employment.function_weights"]["education"]
         * DYN["dyn.public_employment.education_level_weights"]["tertiary"])
    ratio = s.at(END).labour["employees_public"] / base.at(END).labour["employees_public"]
    assert ratio == pytest.approx(1 + w * 0.1, rel=1e-6)


def test_headcount_norm_and_baseline_unchanged_by_the_new_keys():
    lab = baseline_2070().at(2040).labour
    assert lab["wage_bill_public"] == lab["wage_bill_public_actual"]
    assert lab["public_staffing_lever_factor"] == 1.0 and lab["public_staff_unabsorbed"] == 0.0


def test_public_headcount_cut_is_reabsorbed_over_time():
    cut = Change("labour.public_employment_growth", -0.01, 2027, end=2031)
    base, s = mini_base(), mini(cut)
    d = lambda y, k: s.at(y).labour[k] - base.at(y).labour[k]
    assert d(2031, "employees_public") < -40_000
    assert d(2031, "employment") < -10_000                       # transition: not all re-employed yet
    assert d(2031, "employment") > 0.8 * d(2031, "employees_public")
    assert d(2031, "unemployed") > 0
    far = simulate(2045, scen(cut), baseline=simulate(2045))
    assert abs(far.at(2045).labour["employment"] - simulate(2045).at(2045).labour["employment"]) < 1_500


# ------------------------------------------------------------------ 2. real wage cost (macro side)
def test_real_unit_labour_cost_index():
    base = mini_base()
    assert base.at(BASE_YEAR).macro["real_unit_labour_cost"] == 1.0
    s = mini(Change("labour.indexation_factor", 0.0, 2027, end=2027))
    for year in (2027, 2030):
        r = s.at(year).macro["real_unit_labour_cost"] / base.at(year).macro["real_unit_labour_cost"]
        w = s.at(year).macro["wage_index"] / base.at(year).macro["wage_index"]
        # wage moderation lowers prices; the real unit labour cost is the wage over those prices
        price = 1.0 + s.at(year).macro["cpi_wage_cost_effect"]
        assert price < 1.0
        assert r == pytest.approx(w / price, rel=1e-12) and 0.955 < r < 0.99
    # faster trend productivity raises wages but not the real unit labour cost
    s = mini(Change("macro.productivity_growth", 0.0025, 2027, op="add"))
    r = s.at(END).macro["real_unit_labour_cost"] / base.at(END).macro["real_unit_labour_cost"]
    assert s.at(END).macro["wage_index"] > base.at(END).macro["wage_index"] * 1.005
    assert r == pytest.approx(1.0, abs=1e-9)


# ====================================================================== full engine
LIB = {m["name"]: Measure.from_dict(m) for m in yaml.safe_load(
    (Path(engine.__file__).resolve().parents[1] / "measures" / "library" / "measures.yaml").read_text())["measures"]}
FULL_END = 2035


@lru_cache(maxsize=1)
def full_base():
    return engine.run(None, end=FULL_END)


@lru_cache(maxsize=16)
def full(*names: str):
    return engine.run(Scenario(name="+".join(names), measures=[LIB[n] for n in names]), end=FULL_END, baseline=full_base())


def diff(s, year: int, section: str, key: str) -> float:
    return getattr(s.at(year), section)[key] - getattr(full_base().at(year), section)[key]


def test_full_engine_time_limit_costs_less_than_average_jobs():
    s = full("Abolish the unemployment benefit time limit")
    b = full_base().at(2030)
    jobs, gdp = diff(s, 2030, "labour", "employment"), diff(s, 2030, "macro", "gdp_nominal")
    assert jobs < -15_000 and gdp < 0
    per_worker = b.macro["gdp_nominal"] / b.labour["employment"]
    assert 0.35 < gdp / (jobs * per_worker) < 0.80               # was about 1.3 with average workers
    assert diff(s, 2030, "fiscal", "revenue_total") < 0 and diff(s, 2030, "fiscal", "balance") < 0


def test_full_engine_index_jump_creates_jobs_and_improves_the_balance():
    s = full("Index jump in 2027")
    b = full_base()
    assert s.at(2030).macro["wage_index"] / b.at(2030).macro["wage_index"] < 0.99
    # NBB / FPB: +15 000 to +33 000 jobs for 2% of wage moderation, i.e. 7 500 to 16 500 per 1%
    cut_pct = (1 - s.at(2030).macro["wage_index"] / b.at(2030).macro["wage_index"]) * 100
    assert 5_000 < diff(s, 2030, "labour", "employment") / cut_pct < 17_000
    assert 5_000 < diff(s, 2035, "labour", "employment") / cut_pct < 17_000
    assert diff(s, 2030, "fiscal", "balance") > 0 and diff(s, 2035, "fiscal", "balance") > 0
    # benefits are cut like wages: the gain from work, hence labour supply, barely moves
    assert s.at(2030).extra["behaviour"]["benefit_level"] < 0.99
    assert abs(s.at(2030).extra["behaviour"]["x"]["ext"]).max() < 4e-3
    # an index jump is not a change in the tax wedge: no bargained compensation
    assert s.at(2030).extra["macro_prices"]["bargaining"] == 0.0


def test_output_prices_do_not_run_ahead_after_skipped_indexation():
    """Index jump: producers pass the lower wage cost on no faster than consumer prices fall
    (dyn.wage_cost.deflator_lead_indexation = 0), so margins and corporate tax rise in the
    first years; with a lead of 1 output prices fall at once and the gain is smaller. Consumer
    prices, wages and the long-run level are the same."""
    jump = [Change("labour.indexation_factor", 0.5, 2027, end=2027)]
    b = full_base()
    slow = engine.run(scen(*jump), end=END, baseline=b)
    fast = engine.run(scen(*jump, Change("dyn.wage_cost.deflator_lead_indexation", 1.0, 2026)), end=END, baseline=b)
    for y in (2027, 2028):
        assert slow.at(y).macro["cpi"] < b.at(y).macro["cpi"]
        assert fast.at(y).macro["gdp_deflator"] < slow.at(y).macro["gdp_deflator"] < b.at(y).macro["gdp_deflator"]
        assert slow.at(y).macro["corporate_profits_taxable"] > fast.at(y).macro["corporate_profits_taxable"]
        assert slow.at(y).revenue["cit"] > fast.at(y).revenue["cit"] > b.at(y).revenue["cit"]
    assert slow.at(2027).macro["cpi"] == pytest.approx(fast.at(2027).macro["cpi"], rel=1e-12)
    px = slow.at(2027).extra["macro_prices"]
    assert px["wage_moderation"] < 0.0 and px["wage_price_target_moderation"] < px["wage_price_shift_moderation"] < 0.0
    assert slow.at(END).macro["gdp_deflator"] == pytest.approx(fast.at(END).macro["gdp_deflator"], rel=2e-3)
    # a contribution change is not skipped indexation: nothing changes with the new lead
    cut = [Change("ssc.employer_rate", -0.01, 2027, op="add")]
    a = engine.run(scen(*cut), end=2028, baseline=b)
    c = engine.run(scen(*cut, Change("dyn.wage_cost.deflator_lead_indexation", 1.0, 2026)), end=2028, baseline=b)
    assert a.at(2028).extra["macro_prices"]["wage_moderation"] == 0.0
    assert a.at(2028).macro["gdp_deflator"] == c.at(2028).macro["gdp_deflator"]


def test_full_engine_vat_is_passed_into_prices_and_indexation():
    s, b = full("VAT standard rate 21% -> 22%"), full_base()
    assert s.at(2026).series() == b.at(2026).series()
    cpi = s.at(2027).macro["cpi"] / b.at(2027).macro["cpi"] - 1.0
    tax = s.at(2027).macro["cpi_tax_effect"]
    assert 0.002 < tax < 0.0083                                   # at most the full 1/1.21 on every category
    # plus the first pass-through of the indexed wages into prices, in the same year
    second = s.at(2027).macro["cpi_wage_cost_effect"]
    assert 0 < second < 0.5 * tax
    assert cpi == pytest.approx((1 + tax) * (1 + second) - 1, rel=1e-9)
    assert s.at(2027).macro["inflation"] == pytest.approx(b.at(2027).macro["inflation"] + cpi, rel=0.02)
    # a level shift plus a damped second round (indexed wages feed prices), not permanent inflation
    later = s.at(2031).macro["cpi"] / b.at(2031).macro["cpi"] - 1.0
    assert 0.9 * cpi < later < 1.6 * cpi   # stays near the first-year shift (slack and wage costs offset)
    # the tax shift is indexed in the year of the rise (dyn.indexation.tax_shift_lag_weight)
    h27 = s.at(2027).macro["health_index"] / b.at(2027).macro["health_index"] - 1.0
    w27 = s.at(2027).macro["wage_index"] / b.at(2027).macro["wage_index"] - 1.0
    assert 0.7 * h27 < w27 <= h27
    assert s.at(2031).macro["cpi_wage_cost_effect"] > 0
    assert s.at(2035).macro["inflation"] == pytest.approx(b.at(2035).macro["inflation"], abs=2e-4)
    health = s.at(2031).macro["health_index"] / b.at(2031).macro["health_index"] - 1.0
    wages = s.at(2031).macro["wage_index"] / b.at(2031).macro["wage_index"] - 1.0
    # automatic indexation, part of which bargaining takes back (dyn.wage_bargaining.*)
    assert health > 0.002 and 0.5 * health < wages < health
    assert s.at(2031).macro["wage_bargaining_index"] < 1.0
    assert s.at(2031).macro["gdp_deflator"] >= b.at(2031).macro["gdp_deflator"]   # wage costs only
    assert not s.at(2031).macro.get("cons_reflects_tax_changes")
    # still a net gain for the budget, smaller than the mechanical yield because spending is indexed
    assert diff(s, 2030, "fiscal", "balance") > 0
    assert diff(s, 2030, "fiscal", "expenditure_total") > 0
    static = full("VAT standard rate 21% -> 22%", "Static costing (no behavioural responses)")
    assert static.at(2030).macro["cpi"] == b.at(2030).macro["cpi"]
    assert diff(static, 2030, "fiscal", "balance") > diff(s, 2030, "fiscal", "balance")


def test_full_engine_tobacco_excise_moves_cpi_but_not_the_health_index():
    s, b = full("Tobacco excise +25%"), full_base()
    assert s.at(2030).macro["cpi"] > b.at(2030).macro["cpi"] * 1.0005
    # not in the health index: no indexation, no bargained compensation; wages and prices only
    # follow the (small) demand effect (dyn.prices.demand_gap_sensitivity)
    health = s.at(2030).macro["health_index"] / b.at(2030).macro["health_index"]
    m = s.at(2030).macro
    assert health == pytest.approx((1.0 + m["cpi_demand_effect"]) * (1.0 + m["cpi_wage_cost_effect"]), rel=1e-9)
    assert health == pytest.approx(1.0, abs=5e-5)     # the price rise lowers real household income a little
    assert s.at(2030).extra["macro_prices"]["bargaining"] == 0.0
    assert s.at(2030).macro["wage_index"] == pytest.approx(b.at(2030).macro["wage_index"], rel=2e-4)


def test_full_engine_public_employment_cut():
    s = full("Public employment -1% a year for 5 years")
    assert diff(s, 2031, "labour", "employees_public") < -30_000
    assert diff(s, 2031, "labour", "employment") < -5_000
    assert diff(s, 2035, "fiscal", "expenditure_total") < 0 and diff(s, 2035, "fiscal", "balance") > 0


# ------------------------------------------------------------------ 5. wage bargaining, prices
@lru_cache(maxsize=16)
def full_changes(*changes: tuple):
    return engine.run(Scenario(name="c", measures=[Measure(name="m", changes=[Change(*c[:3], **dict(c[3:])) for c in changes])]),
                      end=2032, baseline=full_base())


def rel(s, year: int, key: str) -> float:
    return s.at(year).macro[key] / full_base().at(year).macro[key] - 1.0


def test_new_price_and_wage_series_are_neutral_in_the_baseline():
    for st in full_base().states:
        assert st.macro["wage_bargaining_index"] == 1.0
        px = st.extra["macro_prices"]
        assert px.get("bargaining", 0.0) == 0.0 and px.get("activity_wage", 0.0) == 0.0
        assert px.get("employer_ssc_ratio", 1.0) == 1.0


def test_employee_contribution_rise_is_partly_recouped_in_gross_wages():
    s = full_changes(("ssc.employee_rate", 0.01, 2027, ("op", "add")))
    comp = DYN["dyn.wage_bargaining.compensation_employee_taxes"]
    # running agreements: nothing in the first year, then wages rise towards the target
    assert s.at(2027).macro["wage_bargaining_index"] == 1.0
    target = s.at(2027).extra["macro_prices"]["bargaining_target"]
    assert 0.3 * comp * 0.01 < target < 2.5 * comp * 0.01          # net wage loss of 1 point of gross
    assert 1.0 < s.at(2028).macro["wage_bargaining_index"] < s.at(2031).macro["wage_bargaining_index"]
    assert np.log(s.at(2028).extra["macro_prices"]["bargaining"]
                  / target) == pytest.approx(np.log(DYN["dyn.wage_bargaining.speed"]), abs=1e-9)
    # higher labour cost: jobs are lost through labour demand, prices rise a little
    assert rel(s, 2031, "real_unit_labour_cost") > 0.001
    assert s.at(2031).labour["employment_vs_baseline.labour_demand"] < -1_000
    assert s.at(2031).labour["employment_vs_baseline.labour_demand"] < 2 * s.at(2031).labour["employment_vs_baseline.labour_supply"]
    assert s.at(2031).macro["cpi_wage_cost_effect"] > 0
    # first year: hardly any job loss (HERMES: -1.5 thousand for 0.5% of GDP)
    assert abs(diff(s, 2027, "labour", "employment")) < 0.35 * abs(diff(s, 2031, "labour", "employment"))


def test_employer_contribution_cut_raises_wages_and_profits_not_household_income():
    s = full_changes(("ssc.employer_rate", -0.01, 2027, ("op", "add")))
    b = full_base()
    assert s.at(2027).extra["macro_prices"]["employer_ssc_ratio"] < 1.0
    assert s.at(2027).macro["wage_bargaining_index"] == 1.0 < s.at(2029).macro["wage_bargaining_index"]
    # the cut is in compensation of employees, hence in profits and corporate tax ...
    cut = -diff(s, 2027, "revenue", "ssc_employer_private")
    assert cut > 0
    d_profit = diff(s, 2027, "macro", "corporate_profits_taxable")
    assert 0.3 * cut < d_profit < 1.2 * cut
    assert diff(s, 2027, "revenue", "cit") > 0
    # ... not in the income of households beyond the jobs created
    jobs = diff(s, 2027, "labour", "employment") / b.at(2027).labour["employment"]
    assert rel(s, 2027, "household_disposable_income") < 3 * jobs + 1e-4
    # prices fall in the first year already
    assert rel(s, 2027, "cpi") < -0.0003


def test_static_costing_has_no_bargaining_or_price_response():
    s = full_changes(("ssc.employee_rate", 0.01, 2027, ("op", "add")), ("vat.rate_standard", 0.22, 2027),
                     ("beh.enabled", 0, 2025))
    for year in (2027, 2031):
        st, b = s.at(year), full_base().at(year)
        assert st.macro["wage_bargaining_index"] == 1.0
        assert st.macro["cpi"] == b.macro["cpi"] and st.macro["wage_index"] == b.macro["wage_index"]
        assert st.macro["gdp_nominal"] == b.macro["gdp_nominal"]
        assert st.labour["employment"] == b.labour["employment"]


def test_public_investment_cut_lowers_public_capital_and_output():
    s = full_changes(("invest.public_pct_gdp", -0.5, 2027, ("op", "add")))
    assert s.at(2027).extra["behaviour"]["public_capital_gap"] == 0.0       # stock at the start of the year
    k28, k31 = (s.at(y).extra["behaviour"]["public_capital_gap"] for y in (2028, 2031))
    assert k28 == pytest.approx(-0.005, rel=0.1) and k31 < 3.4 * k28
    pf = s.at(2031).adjust["productivity_factor"]
    assert pf == pytest.approx(np.exp(defaults_beh()["beh.pubcap.output_per_capital"] * k31), rel=5e-5)
    # the loss of output grows over time instead of fading with the demand effect
    assert rel(s, 2031, "gdp_real") < rel(s, 2027, "gdp_real") < 0


def defaults_beh():
    from belsim.behaviour._params import defaults as d
    return d()


def test_wages_follow_a_regional_productivity_gain():
    s = full_changes(("regional.productivity_growth_wedge", 0.0025, 2027, ("key", "FLA")))
    b = full_base()
    share = b.at(2027).regional["gdp_real.FLA"] / b.at(2027).macro["gdp_real"]
    years = 2031 - 2027 + 1
    gain = rel(s, 2031, "wage_index")
    assert gain == pytest.approx(share * (1.0025 ** years - 1.0), rel=0.25)
    assert gain > 0.8 * (rel(s, 2031, "gdp_real") - s.at(2031).macro["demand_gap"]) * 0.8
    # productivity and wages move together: unit labour costs and prices do not
    assert rel(s, 2031, "real_unit_labour_cost") == pytest.approx(0.0, abs=2e-4)
    assert rel(s, 2031, "cpi") == pytest.approx(0.0, abs=2e-4)


def test_government_output_is_valued_at_cost():
    """An index jump lowers public pay: nominal GDP falls with it, corporate profits only gain
    from private wages."""
    s = full_changes(("labour.indexation_factor", 0, 2027, ("end", 2027)), ("beh.enabled", 0, 2025))
    b = full_base().at(2027)
    st = s.at(2027)
    saved_public = b.labour["wage_bill_public"] - st.labour["wage_bill_public"]
    saved_private = b.labour["wage_bill_private"] - st.labour["wage_bill_private"]
    assert saved_public > 0 and saved_private > 0
    d_gdp = st.macro["gdp_nominal"] - b.macro["gdp_nominal"]
    assert -1.6 * saved_public < d_gdp < -0.8 * saved_public
    # profits gain the private labour cost saved (wages plus contributions, and the indexed
    # income of the self-employed), not the public pay saved
    d_profit = st.macro["corporate_profits_taxable"] - b.macro["corporate_profits_taxable"]
    assert 0.5 * saved_private < d_profit < 2.0 * saved_private
    assert st.macro["gdp_real"] == b.macro["gdp_real"]


def test_cpi_weights_leave_out_imputed_rents():
    from belsim import macro
    full_w = macro._cons_shares(0.5)
    w = macro._cpi_weights(0.5, DYN["dyn.cpi.imputed_rent_share_of_housing"])
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["housing"] < full_w["housing"] and w["other_goods"] > full_w["other_goods"]
    assert macro._cpi_weights(0.5, 0.0)["housing"] == pytest.approx(full_w["housing"])
