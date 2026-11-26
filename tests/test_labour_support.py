"""Shared helpers for the labour and macro tests: a self-contained mini-engine that needs only
the labour+macro parameter files and data tables (other modules may be mid-rewrite)."""
from __future__ import annotations

import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from belsim import labour, macro
from belsim.core.dims import BASE_YEAR, N_AGE, N_REGION, N_SEX, SEXES
from belsim.core.params import PARAMS_DIR, ParamSet, Scenario
from belsim.core.state import Results, YearState

DATA = Path(__file__).resolve().parents[1] / "data" / "base"
REGION_SHARE = np.array([0.577, 0.317, 0.106])      # Flanders, Wallonia, Brussels


@lru_cache(maxsize=1)
def paramset() -> ParamSet:
    tmp = Path(tempfile.mkdtemp(prefix="belsim_params_"))
    for name in ("labour.yaml", "macro.yaml"):
        shutil.copy(PARAMS_DIR / name, tmp / name)
    return ParamSet.load(tmp)


@lru_cache(maxsize=1)
def _pop_2024() -> np.ndarray:
    """Population by age, sex, region built from the LFS population totals (labour_lfs_status.csv):
    uniform inside 5-year groups, children and the 75+ tail filled with plausible profiles."""
    d = pd.read_csv(DATA / "labour_lfs_status.csv")
    d = d[d.year == BASE_YEAR]
    pop = np.zeros((N_AGE, N_SEX, N_REGION))
    for si, sex in enumerate(SEXES):
        x = d[d.sex == sex].set_index("age_lo")["POP"] * 1e3
        by_age = np.zeros(N_AGE)
        by_age[:15] = 63_000.0
        for lo in range(15, 75, 5):
            by_age[lo:lo + 5] = x[lo] / 5
        tail = np.exp(-np.arange(26) / 7.0)
        by_age[75:] = x[75] * tail / tail.sum()
        pop[:, si, :] = by_age[:, None] * REGION_SHARE[None, :]
    return pop


def population(year: int) -> np.ndarray:
    """Synthetic ageing: the 67+ population grows 1.2% a year until 2054, the rest is constant."""
    pop = _pop_2024().copy()
    pop[67:] *= 1.0 + 0.012 * min(year - BASE_YEAR, 30)
    return pop


def simulate(end: int = BASE_YEAR, scenario: Scenario | None = None, adjust=None, gap=None,
             passes: int = 1, baseline: Results | None = None, pset: ParamSet | None = None) -> Results:
    """prices -> (synthetic demographics) -> labour -> output, year by year.

    With `baseline` (a finished run of this function) the modules receive a context like the
    engine's in a scenario run: `baseline`, `base_state`, `base_p`, `params`, `history`."""
    pset = (pset or paramset()).with_scenario(scenario)
    base_pset = pset.with_scenario(None)
    res = Results(name="test")
    prev = None
    for year in range(BASE_YEAR, end + 1):
        st = YearState(year=year)
        ctx = None
        if baseline is not None:
            ctx = SimpleNamespace(year=year, params=pset, history=res, baseline=baseline, is_baseline=False,
                                  base_state=baseline.at(year), base_p=base_pset.at(year, None), pass_no=1)
        macro.prices(st, prev, pset.at(year, None), ctx)
        p = pset.at(year, None)
        st.pop = population(year)
        st.demo = {"deaths": 112_000.0 * st.pop[67:].sum() / _pop_2024()[67:].sum()}
        st.adjust = adjust(year) if adjust else {}
        st.macro["demand_gap"] = float(gap(year)) if gap else 0.0
        for _ in range(passes):
            labour.run(st, prev, p, ctx)
            macro.output(st, prev, p, ctx)
        res.add(st)
        prev = st
    return res


@lru_cache(maxsize=1)
def baseline_2070() -> Results:
    return simulate(2070)
