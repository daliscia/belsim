"""Unit tests of the behavioural module, independent of the sibling modules: fake parameter
views and fake calculators are injected instead of the real tax/benefit code."""
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from belsim import behaviour
from belsim.behaviour import demand, responses
from belsim.behaviour._params import defaults
from belsim.core.dims import CONS_CATS, J, N_AGE, N_SEX, S
from belsim.core.engine import Context
from belsim.core.params import ParamSet
from belsim.core.state import LineMeta, Results, YearState
from belsim.labour.dynparams import defaults as dyn_defaults

BASE_POLICY = {
    "t.pit": 0.30, "t.ssc_er": 0.25, "t.se": 0.40, "t.tobacco": 3.0, "t.vat": 0.21,
    "t.cit": 0.25, "t.cap": 0.30, "b.unemp": 0.5, "b.dis": 0.6, "b.pension": 0.5, "b.assist": 0.0,
}
_W = defaults()["beh.ext.outside_option_weights"]
OUT_SHARE = _W["unemployed"] * 0.5 + _W["disabled"] * 0.6      # outside-option weights x benefit rates
ALL_LAMBDA_1 = {k: 1.0 for k in defaults()["beh.lambda"]}


class FakeParams:
    def __init__(self, **over):
        self._v = {**BASE_POLICY, **over}

    def get(self, name, default=None):
        return self._v.get(name, default)

    def __getitem__(self, name):
        return self._v[name]


class FakeRev:
    """Flat taxes: wage net = g (1 - pit), employer cost = g (1 + ssc); benefits untaxed."""

    @staticmethod
    def tax_on_income(gross, p, kind="emp_private"):
        g = np.asarray(gross, dtype=float)
        rate = {"emp_private": p["t.pit"], "emp_public": p["t.pit"], "self_employed": p["t.se"]}.get(kind, 0.0)
        er = g * (p["t.ssc_er"] if kind.startswith("emp_") else 0.0)
        return {"gross": g, "employer_cost": g + er, "ssc_employer": er,
                "ssc_employee": g * 0.0, "pit": g * rate, "net": g * (1 - rate)}

    @staticmethod
    def consumption_tax_rates(p):
        r = dict.fromkeys(CONS_CATS, p["t.vat"])
        r["tobacco"] = p["t.tobacco"]
        return r

    @staticmethod
    def cit_effective_rate(p):
        return p["t.cit"]

    @staticmethod
    def capital_income_tax_rate(p):
        return p["t.cap"]


class FakeExp:
    @staticmethod
    def replacement_income(status, prev_gross, p):
        if status == "assisted":                      # flat amount, whatever the wage
            return np.full(np.shape(prev_gross), float(p["b.assist"]))
        rate = {"unemployed": p["b.unemp"], "disabled": p["b.dis"], "retired": p["b.pension"]}[status]
        return np.asarray(prev_gross, dtype=float) * rate


@pytest.fixture(autouse=True)
def fake_calculators(monkeypatch):
    monkeypatch.setattr(responses, "_calculators", lambda: (FakeRev, FakeExp))
    monkeypatch.setattr(demand, "_line_meta", lambda: {
        "wages_public": LineMeta("Compensation of employees", "Operating", esa="D1"),
        "pensions": LineMeta("Pensions", "Social benefits", esa="D62"),
        "interest": LineMeta("Interest", "Interest", esa="D41"),
    })


def make_state(year=2025, gdp=600_000.0, revenue=None, expenditure=None):
    s = YearState(year=year)
    s.pop[:] = 1000.0
    s.status[S["emp_private"], 20:65] = 500.0
    s.status[S["emp_public"], 20:65] = 100.0
    s.status[S["self_employed"], 20:65] = 100.0
    s.status[S["inactive"]] = s.pop - s.status.sum(axis=0)
    s.wage_dist[:] = np.linspace(20_000, 110_000, 10)
    s.macro.update(wage_index=1.02, gdp_nominal=gdp, demand_gap=0.0)
    s.revenue = dict(revenue or {"pit": 100_000.0, "vat": 60_000.0, "other": 140_000.0})
    s.expenditure = dict(expenditure or {"wages_public": 80_000.0, "pensions": 70_000.0,
                                         "interest": 12_000.0, "other": 160_000.0})
    s.fiscal["primary_balance"] = sum(s.revenue.values()) - sum(s.expenditure.values()) + s.expenditure["interest"]
    return s


def make_ctx(base_state, base_p, year=2025):
    baseline = Results("baseline")
    baseline.add(base_state)
    return Context(year=year, params=None, history=Results("scen"), baseline=baseline, base_p=base_p)


def run_once(prev=None, year=2025, **over):
    """One behaviour.run for a scenario differing from the baseline by `over`."""
    base = make_state(year)
    state = make_state(year)
    p = FakeParams(**over)
    ctx = make_ctx(base, FakeParams(), year)
    behaviour.run(state, prev, p, ctx)
    return state, p, ctx


def assert_neutral(adj):
    assert not np.any(adj["emp_rate_delta"])
    assert adj["emp_rate_delta"].shape == (N_AGE, N_SEX)
    for k in ("labour_demand_factor", "cit_base_factor", "capital_income_base_factor",
              "self_employed_income_factor", "property_transactions_factor", "productivity_factor"):
        assert adj[k] == 1.0, k
    assert adj["retirement_shift"] == 0.0
    assert all(v == 1.0 for v in adj["cons_factor"].values())
    assert set(adj["cons_factor"]) == set(CONS_CATS)


# ------------------------------------------------------------------------- neutrality
def test_equal_parameters_are_exactly_neutral():
    prev = None
    for year in range(2025, 2030):
        state, p, ctx = run_once(prev, year)
        assert_neutral(state.adjust)
        assert behaviour.demand_gap(state, prev, p, ctx) == 0.0
        assert state.extra["behaviour"]["errors"] == []
        prev = state


def test_disabled_gives_no_adjustment_and_no_gap():
    state, p, ctx = run_once(**{"beh.enabled": 0, "t.pit": 0.5, "t.tobacco": 5.0})
    assert state.adjust == {}
    state.revenue["pit"] += 10_000
    assert behaviour.demand_gap(state, None, p, ctx) == 0.0


def test_baseline_run_or_missing_baseline_is_neutral():
    state = make_state()
    ctx = Context(year=2025, params=None, history=Results("x"), baseline=None, base_p=None)
    behaviour.run(state, None, FakeParams(), ctx)
    assert state.adjust == {}
    assert behaviour.demand_gap(state, None, FakeParams(), ctx) == 0.0


# ------------------------------------------------------------------------ labour supply
def test_income_tax_rise_lowers_employment_rate_by_the_elasticity():
    state, p, _ = run_once(**{"t.pit": 0.35, "beh.lambda": ALL_LAMBDA_1})
    delta = state.adjust["emp_rate_delta"]
    assert delta[20:65].max() < 0 and not delta[:18].any() and not delta[67:].any()

    # gain from work per euro of gross: (1 - pit) - out-of-work share, same in every decile;
    # bargaining gives the worker back part of the loss in net wage (dyn.wage_bargaining.*)
    comp = dyn_defaults()["dyn.wage_bargaining.compensation_employee_taxes"]
    assert 0 < comp < 1
    ln_gain = math.log((1 - 0.35 + comp * 0.05 - OUT_SHARE) / (1 - 0.30 - OUT_SHARE))
    d = defaults()
    eps = np.array(d["beh.ext.elasticity_by_decile"])
    tilt = d["beh.ext.decile_tilt"]["prime"] + d["beh.ext.decile_tilt"]["M"]
    w = np.exp(-tilt * np.linspace(-1, 1, 10))
    eta = d["beh.ext.sex_multiplier"]["M"] * d["beh.ext.age_multiplier"]["prime"] * (w / w.sum()) @ eps * ln_gain
    assert delta[40, 0] == pytest.approx(0.7 * math.expm1(eta), rel=1e-9)     # baseline rate 0.7
    # order of magnitude: -5 points of net wage with elasticity ~0.2 -> about -1.5 points
    assert -0.03 < delta[40, 0] < -0.004
    # women, young and older workers respond more than prime-age men
    assert delta[40, 1] < delta[40, 0]
    assert delta[60, 0] < delta[40, 0]
    assert delta[22, 0] < delta[40, 0]


def test_more_generous_benefits_lower_employment_and_pension_cut_delays_retirement():
    state, _, _ = run_once(**{"b.unemp": 0.6})
    assert state.adjust["emp_rate_delta"][40, 0] < 0
    assert state.adjust["labour_demand_factor"] == 1.0

    state, _, _ = run_once(**{"b.pension": 0.45, "beh.lambda": ALL_LAMBDA_1})
    expected = -defaults()["beh.ret.years_per_pct_replacement"] * 100 * math.log(0.45 / 0.5)
    assert state.adjust["retirement_shift"] == pytest.approx(expected)
    assert 0.2 < state.adjust["retirement_shift"] < 0.5
    assert not state.adjust["emp_rate_delta"].any()


def test_marginal_rate_affects_self_employed_income_and_productivity():
    state, _, _ = run_once(**{"t.se": 0.46, "beh.lambda": ALL_LAMBDA_1})
    wd = np.linspace(20_000, 110_000, 10)
    eps = np.array(defaults()["beh.eti.self_employed_by_decile"])
    expected = math.exp((wd / wd.sum()) @ eps * math.log(0.54 / 0.60))
    assert state.adjust["self_employed_income_factor"] == pytest.approx(expected, rel=1e-6)
    assert state.adjust["productivity_factor"] == 1.0

    state, _, _ = run_once(**{"t.pit": 0.35})
    assert 0.99 < state.adjust["productivity_factor"] < 1.0
    assert state.adjust["self_employed_income_factor"] == 1.0


# ------------------------------------------------------------------------ labour demand
def test_employer_ssc_rise_lowers_labour_demand():
    state, _, _ = run_once(**{"t.ssc_er": 0.30, "beh.lambda": ALL_LAMBDA_1})
    eps = np.mean(defaults()["beh.ld.elasticity_by_decile"])
    assert state.adjust["labour_demand_factor"] == pytest.approx((1.30 / 1.25) ** eps, rel=1e-9)
    assert state.adjust["labour_demand_factor"] < 1.0
    assert not state.adjust["emp_rate_delta"].any()        # workers' net income unchanged


def test_targeted_employer_cut_counts_cost_as_a_whole_and_part_of_the_headcount_premium(monkeypatch):
    """A cut on the bottom deciles only: the response is the cost-weighted one plus the share
    `beh.ld.composition_net_share` of the headcount premium, the premium at its own speed; a cut
    that is uniform across deciles has no premium."""

    class LowWageRev(FakeRev):
        @staticmethod
        def tax_on_income(gross, p, kind="emp_private"):
            r = FakeRev.tax_on_income(gross, p, kind)
            if kind == "emp_private":
                g = np.asarray(gross, dtype=float)
                er = g * np.where(g < 45_000, p["t.ssc_er"] - p.get("t.ssc_cut_low", 0.0), p["t.ssc_er"])
                r.update(employer_cost=g + er, ssc_employer=er)
            return r

    monkeypatch.setattr(responses, "_calculators", lambda: (LowWageRev, FakeExp))
    d = defaults()
    grid = np.linspace(20_000, 110_000, 10)
    cost = grid * 1.25
    resp = np.asarray(d["beh.ld.elasticity_by_decile"]) * np.log(np.where(grid < 45_000, 1.15 / 1.25, 1.0))
    whole = float(cost / cost.sum() @ resp)
    premium = d["beh.ld.composition_net_share"] * (resp.mean() - whole)
    assert 0.0 < whole < resp.mean() and premium > 0.0

    state, _, _ = run_once(**{"t.ssc_cut_low": 0.10, "beh.lambda": ALL_LAMBDA_1})
    x = state.extra["behaviour"]["x"]
    assert x["labour_demand"] == pytest.approx(whole, rel=1e-9)
    assert x["labour_composition"] == pytest.approx(premium, rel=1e-9)
    assert state.adjust["labour_demand_factor"] == pytest.approx(math.exp(whole + premium), rel=1e-9)
    assert math.exp(whole) < state.adjust["labour_demand_factor"] < math.exp(resp.mean())

    # default speeds: the premium builds up more slowly than the general effect
    state, _, _ = run_once(**{"t.ssc_cut_low": 0.10})
    x, lam = state.extra["behaviour"]["x"], d["beh.lambda"]
    assert lam["labour_composition"] < lam["labour_demand"]
    assert x["labour_demand"] == pytest.approx(lam["labour_demand"] * whole, rel=1e-9)
    assert x["labour_composition"] == pytest.approx(lam["labour_composition"] * premium, rel=1e-9)

    state, _, _ = run_once(**{"t.ssc_er": 0.20, "beh.lambda": ALL_LAMBDA_1})
    assert state.extra["behaviour"]["x"]["labour_composition"] == pytest.approx(0.0, abs=1e-12)


def test_benefit_level_scales_amounts_not_wage_related_benefits():
    """An index jump on benefits lowers ceilings and flat amounts; a benefit that is a
    percentage of the wage follows the wage (no second cut)."""
    grid = np.linspace(20_000, 110_000, 10)
    p = FakeParams()
    same = responses._income_side(FakeRev, FakeExp, p, grid, grid, 1.0, 0.05, [], 1.0)
    cut = responses._income_side(FakeRev, FakeExp, p, grid, grid, 1.0, 0.05, [], 0.98)
    assert np.allclose(cut["out"]["unemployed"], same["out"]["unemployed"])     # proportional benefit

    class CappedExp(FakeExp):
        @staticmethod
        def replacement_income(status, prev_gross, p):
            return np.minimum(FakeExp.replacement_income(status, prev_gross, p), 30_000.0)

    same = responses._income_side(FakeRev, CappedExp, p, grid, grid, 1.0, 0.05, [], 1.0)
    cut = responses._income_side(FakeRev, CappedExp, p, grid, grid, 1.0, 0.05, [], 0.98)
    capped = same["out"]["unemployed"] >= 30_000.0 - 1e-9
    assert capped.any() and not capped.all()
    assert np.allclose(cut["out"]["unemployed"][capped], 0.98 * same["out"]["unemployed"][capped])
    assert np.allclose(cut["out"]["unemployed"][~capped][:1], same["out"]["unemployed"][~capped][:1])
    assert np.allclose(cut["out"]["retired"], same["out"]["retired"])           # pensions: own formula


# -------------------------------------------------------------------------- consumption
def test_tobacco_excise_rise_lowers_tobacco_volume():
    state, _, _ = run_once(**{"t.tobacco": 3.4, "beh.lambda": ALL_LAMBDA_1})
    d = defaults()
    eps = d["beh.cons.elasticity"]["tobacco"] + d["beh.cons.cross_border"]["tobacco"]
    price = 1 + d["beh.cons.pass_through"]["tobacco"] * (4.4 / 4.0 - 1)
    cf = state.adjust["cons_factor"]
    assert cf["tobacco"] == pytest.approx(price ** eps, rel=1e-9)
    assert cf["tobacco"] < 1.0
    assert all(v == 1.0 for c, v in cf.items() if c != "tobacco")


def test_vat_pass_through_is_partial_for_restaurants():
    state, _, _ = run_once(**{"t.vat": 0.25, "beh.lambda": ALL_LAMBDA_1})
    d = defaults()
    price = 1 + d["beh.cons.pass_through"]["restaurants"] * (1.25 / 1.21 - 1)
    assert state.adjust["cons_factor"]["restaurants"] == pytest.approx(
        price ** d["beh.cons.elasticity"]["restaurants"], rel=1e-9)


# ------------------------------------------------------------------- corporate / capital
def test_cit_and_capital_income_tax():
    state, _, _ = run_once(**{"t.cit": 0.30, "t.cap": 0.35, "beh.lambda": ALL_LAMBDA_1})
    d = defaults()
    a = state.adjust
    assert a["cit_base_factor"] == pytest.approx(math.exp(d["beh.cit.base_semi_elasticity"] * 0.05))
    assert a["productivity_factor"] == pytest.approx(math.exp(d["beh.cit.productivity_semi_elasticity"] * 0.05))
    assert a["capital_income_base_factor"] == pytest.approx((0.65 / 0.70) ** d["beh.cap.net_of_tax_elasticity"])
    assert a["property_transactions_factor"] == 1.0        # optional calculator absent


def test_optional_property_calculator_is_used_when_present(monkeypatch):
    class Rev(FakeRev):
        @staticmethod
        def property_transaction_tax_rate(p):
            return {"FLA": p.get("t.reg", 0.03), "WAL": 0.125, "BRU": 0.125}

    monkeypatch.setattr(responses, "_calculators", lambda: (Rev, FakeExp))
    state, _, _ = run_once(**{"t.reg": 0.06, "beh.lambda": ALL_LAMBDA_1})
    semi = defaults()["beh.prop.transactions_semi_elasticity"]
    assert state.adjust["property_transactions_factor"] == pytest.approx(math.exp(semi * 0.01))


# ----------------------------------------------------------------------------- dynamics
def test_phase_in_converges_to_long_run_value():
    lam = defaults()["beh.lambda"]["labour_demand"]
    long_run = math.log(1.30 / 1.25) * np.mean(defaults()["beh.ld.elasticity_by_decile"])
    prev, path = None, []
    for year in range(2025, 2060):
        state, _, _ = run_once(prev, year, **{"t.ssc_er": 0.30})
        path.append(math.log(state.adjust["labour_demand_factor"]))
        prev = state
    assert path[0] == pytest.approx(lam * long_run)
    assert path[1] == pytest.approx((1 - (1 - lam) ** 2) * long_run)
    assert all(b < a for a, b in zip(path, path[1:6]))      # monotone approach
    assert path[-1] == pytest.approx(long_run, rel=1e-4)

    # the measure is withdrawn: the response decays back towards neutral
    state, _, _ = run_once(prev, 2060)
    assert long_run < math.log(state.adjust["labour_demand_factor"]) < 0
    assert math.log(state.adjust["labour_demand_factor"]) == pytest.approx((1 - lam) * path[-1])


def test_scale_scales_responses():
    logs = {}
    for sc in (0.5, 1.0, 1.5):
        state, _, _ = run_once(**{"t.ssc_er": 0.30, "t.tobacco": 3.4, "t.pit": 0.33, "beh.scale": sc})
        logs[sc] = (math.log(state.adjust["labour_demand_factor"]),
                    math.log(state.adjust["cons_factor"]["tobacco"]),
                    math.log1p(state.adjust["emp_rate_delta"][40, 0] / 0.7))
    for i in range(3):
        assert logs[1.0][i] < 0
        assert logs[0.5][i] == pytest.approx(0.5 * logs[1.0][i], rel=1e-9)
        assert logs[1.5][i] == pytest.approx(1.5 * logs[1.0][i], rel=1e-9)


# --------------------------------------------------------------- robustness and bounds
def test_failing_calculators_make_channels_neutral(monkeypatch):
    class Boom:
        def __getattr__(self, name):
            def f(*a, **k):
                raise RuntimeError("broken sibling")
            return f

    monkeypatch.setattr(responses, "_calculators", lambda: (Boom(), Boom()))
    state, _, _ = run_once(**{"t.pit": 0.5})
    assert_neutral(state.adjust)
    assert state.extra["behaviour"]["errors"]

    monkeypatch.setattr(responses, "_calculators", lambda: (None, None))
    state, _, _ = run_once(**{"t.pit": 0.5})
    assert_neutral(state.adjust)

    # benefit calculator broken: tax-only channels still work, participation is neutral
    monkeypatch.setattr(responses, "_calculators", lambda: (FakeRev, None))
    state, _, _ = run_once(**{"t.pit": 0.5, "t.ssc_er": 0.3})
    assert not state.adjust["emp_rate_delta"].any()
    assert state.adjust["labour_demand_factor"] < 1.0


def test_extreme_measures_are_bounded():
    state, _, _ = run_once(**{"t.pit": 0.69, "t.ssc_er": 5.0, "t.tobacco": 500.0, "t.cit": 0.99,
                              "t.cap": 0.999, "b.pension": 0.001, "beh.lambda": ALL_LAMBDA_1})
    clip = defaults()["beh.clip"]
    a = state.adjust
    assert np.abs(a["emp_rate_delta"]).max() <= clip["emp_rate_delta_abs"] + 1e-12
    assert a["labour_demand_factor"] == pytest.approx(math.exp(-clip["labour_demand"]))
    assert a["cons_factor"]["tobacco"] == pytest.approx(math.exp(-clip["consumption"]))
    assert a["cit_base_factor"] == pytest.approx(math.exp(-clip["cit_base"]))
    assert a["capital_income_base_factor"] == pytest.approx(math.exp(-clip["capital_income"]))
    assert a["retirement_shift"] == pytest.approx(clip["retirement"])
    assert all(np.isfinite(v) for v in a["cons_factor"].values())


def test_indexation_differences_are_not_policy_changes(monkeypatch):
    """Scenario wages 3% higher and a wage-indexed allowance 3% higher: no real change."""
    class Rev(FakeRev):
        @staticmethod
        def tax_on_income(gross, p, kind="emp_private"):
            g = np.asarray(gross, dtype=float)
            pit = np.maximum(g - p["t.allow"], 0) * p["t.pit"] if kind.startswith("emp_") else g * 0.0
            return {"gross": g, "employer_cost": g * 1.25, "net": g - pit}

    monkeypatch.setattr(responses, "_calculators", lambda: (Rev, FakeExp))
    base, state = make_state(), make_state()
    state.macro["wage_index"] = base.macro["wage_index"] * 1.03
    ctx = make_ctx(base, FakeParams(**{"t.allow": 10_000.0}))
    behaviour.run(state, None, FakeParams(**{"t.allow": 10_300.0, "beh.lambda": ALL_LAMBDA_1}), ctx)
    assert np.abs(state.adjust["emp_rate_delta"]).max() < 1e-12
    assert state.adjust["labour_demand_factor"] == pytest.approx(1.0, abs=1e-12)
    assert state.adjust["productivity_factor"] == pytest.approx(1.0, abs=1e-12)


def test_higher_social_assistance_lowers_employment_most_at_low_wages():
    """Social assistance is a flat amount in the out-of-work income (beh.ext.outside_option_weights)."""
    state, _, _ = run_once(**{"b.assist": 6_000.0, "beh.lambda": ALL_LAMBDA_1})
    delta = state.adjust["emp_rate_delta"]
    assert delta[20:65].max() < 0 and not delta[:18].any()
    more, _, _ = run_once(**{"b.assist": 9_000.0, "beh.lambda": ALL_LAMBDA_1})
    assert more.adjust["emp_rate_delta"][40, 0] < delta[40, 0]
    assert state.extra["behaviour"]["errors"] == []


# ------------------------------------------------------------------------ demand feedback
def demand_year(prev, year, d_rev=0.0, d_exp=None, pass_no=1, gdp_scen=None, **over):
    base = make_state(year)
    state = make_state(year, gdp=gdp_scen or 600_000.0)
    state.revenue["pit"] += d_rev
    for k, v in (d_exp or {}).items():
        state.expenditure[k] += v
    p = FakeParams(**over)
    ctx = make_ctx(base, FakeParams(), year)
    ctx.pass_no = pass_no
    return state, behaviour.demand_gap(state, prev, p, ctx), p, ctx


def test_tax_rise_gives_negative_gap_that_builds_up_then_fades():
    d = defaults()
    m, rho, mu = d["beh.demand.mult_transfers"], d["beh.demand.persistence"], d["beh.demand.income_first_year_share"]
    prev, gaps, eff, gap = None, [], 0.0, 0.0
    for year in range(2025, 2100):
        prev, g, _, _ = demand_year(prev, year, d_rev=6_000.0)         # +1% of GDP, permanent
        gaps.append(g)
        new_eff = eff + mu * (-m * 0.01 - eff)                         # households adjust gradually
        gap = rho * gap + (new_eff - eff)
        eff = new_eff
        assert g == pytest.approx(gap, rel=1e-9)
    assert gaps[0] == pytest.approx(-m * mu * 0.01)
    assert gaps[0] < 0 and abs(gaps[2]) > abs(gaps[0])                 # builds up over the first years
    assert 0.6 * m * 0.01 < abs(gaps[4]) < m * 0.01                    # persists (HERMES) ...
    assert abs(gaps[-1]) < 0.1 * abs(gaps[4])                          # ... but fades in the long run

    # withdrawal of the measure: rebound
    _, g, _, _ = demand_year(prev, 2100)
    assert g == pytest.approx(rho * gaps[-1] + mu * m * 0.01)
    assert g > 0


def test_taxes_paid_by_firms_have_their_own_multiplier(monkeypatch):
    d = defaults()
    monkeypatch.setattr(demand, "_revenue_meta", lambda: {
        "ssc_er": LineMeta("Employer contributions, private sector", "Social contributions", esa="D611"),
        "cit": LineMeta("Corporate income tax", "Corporate income tax", esa="D51"),
    })

    def year_with(line):
        base, state = make_state(2025), make_state(2025)
        for st in (base, state):
            st.revenue.update(ssc_er=50_000.0, cit=20_000.0)
        state.revenue[line] += 6_000.0
        ctx = make_ctx(base, FakeParams(), 2025)
        return behaviour.demand_gap(state, None, FakeParams(), ctx)

    # corporate tax: its own small multiplier; employer contributions: none of their own (jobs,
    # wages, prices and competitiveness carry their effect)
    assert year_with("cit") == pytest.approx(-d["beh.demand.mult_taxes_firms"]["corporate"] * 0.01)
    assert d["beh.demand.mult_taxes_firms"]["employer"] == 0.0
    assert year_with("ssc_er") == 0.0
    assert year_with("pit") == pytest.approx(
        -d["beh.demand.mult_transfers"] * d["beh.demand.income_first_year_share"] * 0.01)


def test_indexation_of_public_pay_is_income_not_public_demand():
    """Wages 1% above the baseline and public pay indexed accordingly: no change in the volume
    of purchases; the extra pay is household income (transfer multiplier, phased in)."""
    d = defaults()
    base, state = make_state(2025), make_state(2025)
    state.macro["wage_index"] = base.macro["wage_index"] * 1.01
    w = d["beh.demand.purchases_pay_share"]
    price = 1.01 ** w                                      # health index unchanged
    state.expenditure["wages_public"] *= price
    ctx = make_ctx(base, FakeParams(), 2025)
    gap = behaviour.demand_gap(state, None, FakeParams(), ctx)
    mem = state.extra["behaviour"]["demand"]
    assert mem["d_purchases"] == pytest.approx(0.0, abs=1e-6)
    extra = 80_000.0 * (price - 1.0)
    assert mem["indexation_public_pay"] == pytest.approx(extra)
    assert gap == pytest.approx(d["beh.demand.mult_transfers"] * d["beh.demand.income_first_year_share"]
                                * extra / 600_000.0)


def test_price_of_goods_bought_is_neither_demand_nor_household_income(monkeypatch):
    """Health index 1% above the baseline: the non-pay part of purchases costs more, which is
    no change in volume and no income of households. A line indexed a year late
    (beh.demand.lagged_price_lines) is deflated with last year's index: unchanged spending in
    the year of the price rise is no cut in volume, and its later indexation is income of the
    care providers."""
    d = defaults()
    w = d["beh.demand.purchases_pay_share"]
    base, state = make_state(2025), make_state(2025)
    base.macro["health_index"], state.macro["health_index"] = 1.0, 1.01
    state.expenditure["wages_public"] *= 1.01 ** (1.0 - w)
    gap = behaviour.demand_gap(state, None, FakeParams(), make_ctx(base, FakeParams(), 2025))
    mem = state.extra["behaviour"]["demand"]
    assert mem["d_purchases"] == pytest.approx(0.0, abs=1e-6)
    assert mem["indexation_public_pay"] == pytest.approx(0.0, abs=1e-9)
    assert gap == pytest.approx(0.0, abs=1e-12)

    monkeypatch.setattr(demand, "_line_meta", lambda: {
        "health_care": LineMeta("Health care", "Health", esa="D632"),
        "interest": LineMeta("Interest", "Interest", esa="D41")})
    spend = {"health_care": 40_000.0, "interest": 12_000.0, "other": 100_000.0}
    lag = FakeParams(**{"health.indexation_lag_share": 1.0})
    # year 1: prices up, fees not yet indexed -> nothing happens
    b1, s1 = make_state(2025, expenditure=spend), make_state(2025, expenditure=spend)
    b1.macro["health_index"], s1.macro["health_index"] = 1.0, 1.01
    assert behaviour.demand_gap(s1, None, lag, make_ctx(b1, lag, 2025)) == pytest.approx(0.0, abs=1e-12)
    # year 2: fees indexed -> income of the providers, no change in volume
    b2, s2 = make_state(2026, expenditure=spend), make_state(2026, expenditure={**spend, "health_care": 40_400.0})
    b2.macro["health_index"], s2.macro["health_index"] = 1.0, 1.01
    ctx = make_ctx(b2, lag, 2026)
    ctx.baseline.add(b1)
    gap = behaviour.demand_gap(s2, s1, lag, ctx)
    mem = s2.extra["behaviour"]["demand"]
    assert mem["d_purchases"] == pytest.approx(0.0, abs=1e-6)
    assert mem["indexation_public_pay"] == pytest.approx(400.0)
    assert gap == pytest.approx(d["beh.demand.mult_transfers"] * d["beh.demand.income_first_year_share"] * 400.0 / 600_000.0)


def test_multipliers_differ_by_instrument_and_interest_is_excluded():
    d = defaults()
    _, g_purch, _, _ = demand_year(None, 2025, d_exp={"wages_public": 6_000.0})
    _, g_transf, _, _ = demand_year(None, 2025, d_exp={"pensions": 6_000.0})
    _, g_other, _, _ = demand_year(None, 2025, d_exp={"other": 6_000.0})
    _, g_int, _, _ = demand_year(None, 2025, d_exp={"interest": 6_000.0})
    assert g_purch == pytest.approx(d["beh.demand.mult_purchases"] * 0.01)
    mu = d["beh.demand.income_first_year_share"]                 # households adjust gradually
    assert g_transf == pytest.approx(d["beh.demand.mult_transfers"] * mu * 0.01)
    assert g_other == pytest.approx(d["beh.demand.mult_transfers"] * mu * 0.01)   # unclassified -> transfers
    assert g_int == 0.0
    assert g_purch > g_transf > 0


def test_demand_gap_scale_switch_bounds_and_second_pass():
    m = defaults()["beh.demand.mult_transfers"] * defaults()["beh.demand.income_first_year_share"]
    _, g, _, _ = demand_year(None, 2025, d_rev=6_000.0, **{"beh.scale": 1.5})
    assert g == pytest.approx(-1.5 * m * 0.01)
    _, g, _, _ = demand_year(None, 2025, d_rev=6_000.0, **{"beh.demand.enabled": 0})
    assert g == 0.0
    _, g, _, _ = demand_year(None, 2025, d_rev=600_000.0)
    assert g == -defaults()["beh.demand.max_gap"]

    # pass 2: budget now reflects the feedback, the stored pass-1 gap is returned unchanged
    state, g1, p, ctx = demand_year(None, 2025, d_rev=6_000.0)
    state.revenue["pit"] -= 3_000.0
    ctx.pass_no = 2
    assert behaviour.demand_gap(state, None, p, ctx) == g1
    assert state.extra["behaviour"]["demand"]["stance"] == pytest.approx(-m * 0.01)


def test_supply_driven_revenue_is_not_a_fiscal_impulse():
    """Higher GDP from supply responses raises revenue; that part is not a tightening."""
    sigma = defaults()["beh.demand.budget_semi_elasticity"]
    _, g, _, _ = demand_year(None, 2025, d_rev=sigma * 6_000.0, gdp_scen=606_000.0)
    assert g == pytest.approx(0.0, abs=1e-12)


def test_fallback_on_primary_balance_when_lines_are_missing():
    base, state = make_state(), make_state()
    state.revenue, state.expenditure = {}, {}
    state.fiscal["primary_balance"] = base.fiscal["primary_balance"] + 6_000.0
    g = behaviour.demand_gap(state, None, FakeParams(), make_ctx(base, FakeParams()))
    assert g == pytest.approx(-defaults()["beh.demand.mult_default"] * 0.01)


# ------------------------------------------------------------------- params and speed
def test_parameter_file_is_complete_and_documented():
    pset = ParamSet.load()
    specs = {n: s for n, s in pset.specs.items() if n.startswith("beh.")}
    assert set(specs) == set(defaults())
    for name, spec in specs.items():
        assert spec.file == "behaviour.yaml"
        assert spec.source and spec.description, name
    d = defaults()
    assert d["beh.enabled"] == 1 and d["beh.scale"] == 1.0
    assert sum(d["beh.ext.outside_option_weights"].values()) == pytest.approx(1.0)
    for name in ("beh.cons.elasticity", "beh.cons.cross_border", "beh.cons.pass_through"):
        assert set(d[name]) == set(CONS_CATS)
    assert set(responses._LAMBDA_KEY.values()) == set(d["beh.lambda"])
    assert set(responses._CLIP_KEY.values()) | {"emp_rate_delta_abs"} == set(d["beh.clip"])


def test_run_is_fast():
    base, state = make_state(), make_state()
    p = FakeParams(**{"t.pit": 0.35, "t.tobacco": 3.4})
    ctx = make_ctx(base, FakeParams())
    behaviour.run(state, None, p, ctx)
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        behaviour.run(state, None, p, ctx)
        behaviour.demand_gap(state, None, p, ctx)
    per_year = (time.perf_counter() - t0) / n
    assert per_year < 3e-3, f"{per_year * 1e3:.2f} ms per year"
