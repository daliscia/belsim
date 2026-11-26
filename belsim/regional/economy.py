"""Regional economy by place of work: employment, GDP, compensation of employees.

Top-down and share based: the national totals of the labour and macro modules are distributed
over the three regions, so regional values always add up to the national ones.

    workplace employment_r  = national employment x share( k_r x sum_q residents employed_q x C[q][r] )
    GDP_r                   = national GDP x share( g_r x effective workplace employment_r x P_r )
    compensation D1_r       = national D1  x share( w_r x employees at workplace_r )

`C` is the commuting matrix `regional.commuting_matrix`; `k, g, w` are calibration factors
fitted in the base year on the regional accounts (Eurostat nama_10r_3empers, nama_10r_2gdp,
nama_10r_2coe: workplace employment, productivity level and wage differential of each region)
and kept constant; `P_r` is the cumulated regional productivity wedge
(`regional.productivity_growth_wedge`), which also raises national GDP by the region's weight
(applied by `macro.output`). Jobs gained or lost relative to the baseline carry the national
marginal productivity (labour module), attributed to the regions where the jobs are.

`macro.output` calls `workplace()` before it computes GDP (it needs the regional GDP weights
for the productivity wedge) and `finish()` afterwards; the result is carried in
`state.extra["regional_economy"]` for the expenditure module and `regional.run`.
"""
from __future__ import annotations

import numpy as np

from ..core.dims import REGIONS, S
from . import data

NS = "regional_economy"
_EMPLOYED = [S["emp_private"], S["emp_public"], S["self_employed"]]
_EMPLOYEES = [S["emp_private"], S["emp_public"]]
_FALLBACK_COMMUTING = {q: {r: (1.0 if q == r else 0.0) for r in REGIONS} for q in REGIONS}


def by_region(v) -> np.ndarray:
    if isinstance(v, dict):
        return np.array([float(v[r]) for r in REGIONS])
    return np.full(len(REGIONS), float(v))


def _share(a: np.ndarray) -> np.ndarray:
    tot = float(a.sum())
    return a / tot if tot > 0 else np.full(len(a), 1.0 / len(a))


def _commuting(p) -> np.ndarray:
    """[residence q, workplace r] share of the employed residents of q working in r."""
    c = (p.get("regional.commuting_matrix") if p is not None else None) or _FALLBACK_COMMUTING
    return np.array([[float(c[q][r]) for r in REGIONS] for q in REGIONS])


def calibrate(state, p) -> dict:
    """Base-year factors: data share / model share (regional accounts of the base year)."""
    res = state.status[_EMPLOYED].sum(axis=(0, 1, 2))
    res_emp = state.status[_EMPLOYEES].sum(axis=(0, 1, 2))
    com = _commuting(p)
    raw, raw_emp = res @ com, res_emp @ com
    k_wp = data.shares("employment_workplace") / np.maximum(_share(raw), 1e-12)
    k_emp = data.shares("employees_workplace") / np.maximum(_share(raw_emp), 1e-12)
    wp, emp = _share(k_wp * raw), _share(k_emp * raw_emp)
    return {
        "k_wp": k_wp, "k_emp": k_emp,
        "g": data.shares("gdp") / np.maximum(wp, 1e-12),
        "w": data.shares("compensation_workplace") / np.maximum(emp, 1e-12),
    }


def workplace(state, p, calib: dict, base_x: dict | None = None, efficiency: float = 1.0) -> dict:
    """Employment by place of work and regional GDP weights (before the productivity wedge).

    `base_x`: the baseline's `extra["regional_economy"]` of the same year; `efficiency`: national
    effective labour / headcount (1 without marginal jobs).
    """
    res = state.status[_EMPLOYED].sum(axis=(0, 1, 2))
    res_emp = state.status[_EMPLOYEES].sum(axis=(0, 1, 2))
    com = _commuting(p)
    n_emp = float(state.emp.sum()) or float(res.sum())
    n_employees = float(state.emp[:, :2].sum()) or float(res_emp.sum())
    emp_wp = n_emp * _share(calib["k_wp"] * (res @ com))
    employees_wp = n_employees * _share(calib["k_emp"] * (res_emp @ com))
    # efficiency units: the productivity discount (or premium) of the jobs that differ from the
    # baseline is charged to the regions where those jobs are
    eff = emp_wp
    if base_x is not None and efficiency != 1.0 and base_x.get("emp_wp") is not None:
        diff = emp_wp - np.asarray(base_x["emp_wp"])
        d_tot = float(diff.sum())
        if abs(d_tot) > 1.0:
            m = float(np.clip(n_emp * (1.0 - efficiency) / d_tot, -1.0, 1.5))
            eff = np.maximum(emp_wp - m * diff, 0.0)
    return {
        "emp_res": res, "employees_res": res_emp, "emp_wp": emp_wp, "employees_wp": employees_wp,
        "gdp_share0": _share(calib["g"] * eff),
        "d1_share": _share(calib["w"] * employees_wp),
    }


def finish(reg: dict, prod_index: np.ndarray, gdp: float, gdp_real: float, d1: float) -> dict:
    """Adds regional GDP and compensation of employees (EUR mn) once the national ones are known."""
    share = _share(reg["gdp_share0"] * prod_index)
    out = dict(reg)
    out.update(gdp_share=share, gdp=gdp * share, gdp_real=gdp_real * share, d1=d1 * reg["d1_share"],
               prod_index=prod_index)
    return out
