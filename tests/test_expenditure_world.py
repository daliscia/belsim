"""Synthetic mini-economy used by the expenditure / balance tests (no tests in this file).

It mimics the engine loop for the expenditure-owned modules only, so the tests do not depend on
the other modules: crude prices, population, statuses, GDP and a revenue stub scaled so that
2024 revenue equals the actual total.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from belsim.core.dims import BASE_YEAR, N_AGE, N_DECILE, N_JOBTYPE, N_REGION, N_SECTOR, N_SEX, S
from belsim.core.engine import Context, _baseline_index_fn, _index_fn
from belsim.core.params import ParamSet, ParamSpec, Scenario
from belsim.core.state import Results, YearState
from belsim.fiscal import balance, expenditure
from belsim.fiscal.expenditure import data

GDP_2024 = data.scalars()["gdp_2024"]
TR_2024 = data.scalars()["tr_2024"]


def load_params(scenario: Scenario | None = None) -> ParamSet:
    path = Path(__file__).resolve().parents[1] / "params" / "expenditure.yaml"
    raw = yaml.safe_load(path.read_text())
    specs = {
        name: ParamSpec(name=name, values={int(y): v for y, v in spec["values"].items()},
                        description=spec.get("description", ""), unit=str(spec.get("unit", "")),
                        source=spec.get("source", ""), uprate=spec.get("uprate", "none") or "none",
                        file=path.name)
        for name, spec in raw.items()
    }
    return ParamSet(specs, scenario)


def make_state(year: int, prev: YearState | None, *, rate_shock: float = 0.0, retired_factor: float = 1.0,
               gap: float = 0.0, rev_factor: float = 1.0) -> YearState:
    s = YearState(year=year)
    n = year - BASE_YEAR
    infl, real_w = 0.02, 0.008
    shock = rate_shock if year > BASE_YEAR else 0.0
    s.macro.update(inflation=infl, cpi=1.02 ** n, health_index=1.02 ** n, wage_index=(1.02 * 1.008) ** n,
                   interest_rate_10y=0.03 + shock, interest_rate_short=0.03 + shock, demand_gap=gap)
    age = np.arange(N_AGE)
    age_w = np.clip(1.0 - age / 105.0, 0.02, None) ** 0.7
    # ageing: older cohorts grow over time
    age_w = age_w * (1 + 0.012 * n * (age >= 65)) * (1 + 0.002 * n)
    reg = np.array([0.578, 0.316, 0.106])
    pop = 11_800_000 * (age_w / age_w.sum() if n == 0 else age_w / (np.clip(1.0 - age / 105.0, 0.02, None) ** 0.7).sum())
    s.pop = pop[:, None, None] * 0.5 * reg[None, None, :] * np.ones((N_AGE, N_SEX, N_REGION))
    st = np.zeros((len(S),) + s.pop.shape)
    st[S["child"]][age < 15] = s.pop[age < 15]
    young = (age >= 15) & (age < 23)
    st[S["student"]][young] = s.pop[young] * 0.7
    st[S["emp_private"]][young] = s.pop[young] * 0.3
    work = (age >= 23) & (age < 66)
    for name, sh in dict(emp_private=.47, emp_public=.14, self_employed=.12, disabled=.08,
                         unemployed=.05, inactive=.14).items():
        st[S[name]][work] = s.pop[work] * sh
    old = age >= 66
    st[S["retired"]][old] = s.pop[old] * min(retired_factor, 1.0)
    st[S["inactive"]][old] = s.pop[old] * (1 - min(retired_factor, 1.0))
    if retired_factor > 1.0:  # extra early retirees taken from the inactive of working age
        extra = (retired_factor - 1.0) * s.pop[old].sum()
        share = extra / st[S["inactive"]][work].sum()
        st[S["retired"]][work] += st[S["inactive"]][work] * share
        st[S["inactive"]][work] *= 1 - share
    s.status = st
    tot = np.array([st[S[k]].sum() for k in ("emp_private", "emp_public", "self_employed")])
    s.emp = np.ones((N_SECTOR, N_JOBTYPE)) * tot[None, :] / N_SECTOR
    w = s.macro["wage_index"]
    means = np.array([52_000, 55_000, 40_000])
    s.wage = np.ones((N_SECTOR, N_JOBTYPE)) * means[None, :] * w
    dec = np.array([.35, .55, .68, .78, .88, .98, 1.1, 1.25, 1.5, 2.43])
    assert dec.size == N_DECILE
    s.wage_dist = means[:, None] * dec[None, :] / dec.mean() * w
    s.labour = {
        "employment": float(tot.sum()), "unemployment_rate": 0.055,
        "avg_wage_private": means[0] * w, "avg_wage_public": means[1] * w,
        "avg_income_self_employed": means[2] * w,
        "wage_bill_private": tot[0] * means[0] * w / 1e6, "wage_bill_public": tot[1] * means[1] * w / 1e6,
        "service_voucher_workers": 150_000.0, "statutory_share": 0.5,
    }
    gdp = GDP_2024 * (1.02 * 1.012) ** n * (1 + gap)
    s.macro.update(gdp_nominal=gdp, gdp_real=gdp / s.macro["cpi"], gdp_deflator=s.macro["cpi"])
    s.va = np.full(N_SECTOR, gdp * 0.89 / N_SECTOR)
    return s


def stub_revenue(s: YearState, factor: float = 1.0) -> None:
    g = s.macro["gdp_nominal"] / GDP_2024 * TR_2024 * factor
    s.revenue = {"ssc": g * 0.31, "pit": g * 0.24, "cit": g * 0.085, "vat": g * 0.13, "other": g * 0.235}


def simulate(end: int = 2030, scenario: Scenario | None = None, baseline: Results | None = None,
             twice: bool = False, **world) -> Results:
    pset = load_params(scenario)
    base_pset = pset.with_scenario(None)
    res = Results(name="test")
    prev = None
    for year in range(BASE_YEAR, end + 1):
        s = make_state(year, prev, **{k: v for k, v in world.items() if k != "rev_factor"})
        ctx = Context(year=year, params=pset, history=res, baseline=baseline, base_p=None)
        p = pset.at(year, _index_fn(s, res))
        if baseline is not None:
            ctx.base_p = base_pset.at(year, _baseline_index_fn(baseline))
        for _ in range(2 if twice else 1):
            expenditure.run(s, prev, p, ctx)
            stub_revenue(s, world.get("rev_factor", 1.0))
            balance.run(s, prev, p, ctx)
        res.add(s)
        prev = s
    return res
