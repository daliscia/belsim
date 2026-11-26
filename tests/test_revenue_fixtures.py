"""Synthetic 2024 state for the revenue tests: approximate official 2024 magnitudes, built
without the other modules (they are developed in parallel)."""
from types import SimpleNamespace

import numpy as np

from belsim.core.dims import CONS_CATS, J, N_AGE, N_SECTOR, S
from belsim.core.params import ParamSet
from belsim.core.state import YearState

GDP_2024 = 614_000.0          # EUR mn, approx. national accounts
CONS_SHARES = dict(food=.125, alcohol=.018, tobacco=.015, fuel=.03, energy_home=.05, housing=.19,
                   restaurants=.07, renovation=.02, other_goods=.222, other_services=.26)
DECILES = np.array([.35, .55, .68, .78, .88, .98, 1.1, 1.25, 1.5, 1.93])
WORKERS = {"emp_private": 3.33e6, "emp_public": 0.87e6, "self_employed": 0.85e6}
AVG_WAGE = {"emp_private": 55_000.0, "emp_public": 60_000.0, "self_employed": 42_000.0}


def params(year: int = 2024, pset: ParamSet | None = None):
    return (pset or ParamSet.load()).at(year)


def make_state(year: int = 2024, scale: float = 1.0) -> YearState:
    st = YearState(year=year)
    for name, n in WORKERS.items():
        st.emp[:, J[name]] = n / N_SECTOR
        st.wage[:, J[name]] = AVG_WAGE[name]
        st.wage_dist[J[name]] = AVG_WAGE[name] * DECILES / DECILES.mean()
        st.status[S[name], 30, 0, 0] = n
    st.status[S["retired"], 70, 0, 0] = 2.4e6
    st.status[S["unemployed"], 40, 0, 0] = 0.35e6
    st.status[S["disabled"], 50, 0, 0] = 0.5e6
    st.pop = st.status.sum(axis=0)
    st.pop[np.arange(N_AGE) < 18] += 2.3e6 / 18 / 6
    gdp = GDP_2024 * scale
    cons = 0.515 * gdp
    st.macro = dict(
        gdp_nominal=gdp, cpi=1.0, health_index=1.0, wage_index=1.0,
        consumption_households=cons, investment=0.245 * gdp, investment_housing=0.05 * gdp,
        gov_consumption=147_436.0 * scale, corporate_profits_taxable=115_000.0 * scale,
        household_capital_income=30_000.0 * scale, household_financial_wealth=1_550_000.0 * scale,
        property_transactions=60_000.0 * scale, inheritance_flow=55_000.0 * scale,
    )
    for c in CONS_CATS:
        st.macro[f"cons.{c}"] = cons * CONS_SHARES[c]
    st.expenditure = {"pensions": 68_000.0 * scale, "unemployment": 6_000.0 * scale,
                      "disability": 13_000.0 * scale, "health": 40_000.0 * scale}
    return st


def ctx(base_p=None):
    return SimpleNamespace(base_p=base_p, year=2024)
