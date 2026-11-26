"""Levers added in November 2026: minimum pension on pensions in payment, growth norm as a budget,
social assistance in the benefit calculator."""
import numpy as np
import pytest

from belsim.core import engine
from belsim.core.params import Change, Measure, ParamSet, Scenario
from belsim.fiscal.expenditure import calc


def _static(changes, end=2029):
    base = engine.run(None, end=end)
    scen = engine.run(Scenario(name="s", measures=[Measure(name="m", changes=changes + [Change("beh.enabled", 0, 2026)])]),
                      end=end, baseline=base)
    return base, scen


def test_minimum_pension_rise_moves_pensions_in_payment():
    """+2.96% on the minimum of employees and self-employed costs EUR 567-588 mn (DC2024, VB 301):
    the share of each scheme's spending at the minimum follows it."""
    base, scen = _static([Change("pension.min_pension", 1.0296, 2027, op="mul")])
    share = ParamSet.load().value("pension.min_pension_spending_share", 2027)
    for y in (2027, 2029):
        s, b = scen.at(y), base.at(y)
        for k in ("employees", "self_employed", "civil_servants"):
            d = s.expenditure[f"pension_{k}"] / b.expenditure[f"pension_{k}"] - 1.0
            assert d == pytest.approx(share[k] * 0.0296, rel=0.2)     # plus the floor in the formula of new pensions
        cost = sum(s.expenditure[f"pension_{k}"] - b.expenditure[f"pension_{k}"] for k in ("employees", "self_employed"))
        cost_2024 = cost / (b.macro["cpi"])
        assert 500 < cost_2024 < 720


def test_growth_norm_as_a_budget_binds_in_both_directions():
    base = engine.run(None, end=2031)
    grow = lambda r, y: r.at(y).extra["expenditure"]["health_real"] / r.at(y - 1).extra["expenditure"]["health_real"] - 1.0
    for norm in (0.015, 0.035):
        ceiling = [Change("health.growth_norm", norm, 2027)]
        budget = ceiling + [Change("health.norm_room_spent", 1.0, 2027)]
        _, c = _static(ceiling, 2031)
        _, b = _static(budget, 2031)
        assert grow(b, 2030) == pytest.approx(norm, abs=1e-9)
        assert grow(c, 2030) == pytest.approx(min(norm, grow(base, 2030)), abs=1e-9)
    assert ParamSet.load().value("health.norm_room_spent", 2030) == 0.0      # baseline: a ceiling


def test_social_assistance_in_the_benefit_calculator():
    p = ParamSet.load().at(2024, lambda kind, y: 1.0)
    out = calc.replacement_income("assisted", np.array([20_000.0, 60_000.0]), p)
    assert out == pytest.approx(np.full(2, p["assist.amount"] * p["assist.generosity_factor"]))
