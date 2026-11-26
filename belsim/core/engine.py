"""Simulation loop. Calls the modules in a fixed order each year (see docs/ARCHITECTURE.md)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .dims import BASE_YEAR, END_YEAR
from .params import ParamSet, ParamView, Scenario
from .state import Results, YearState

INDEX_KEYS = {"cpi": "cpi", "health_index": "health_index", "wages": "wage_index",
              "wage_link": "tax_threshold_drift"}


@dataclass
class Context:
    """Everything a module may need beyond the current and previous YearState."""
    year: int
    params: ParamSet                 # parameters of the run in progress (scenario applied)
    history: Results                 # years already simulated in this run (excludes current year)
    baseline: Results | None         # finished baseline run; None while running the baseline itself
    base_p: ParamView | None         # baseline parameters for this year (None in a baseline run)
    pass_no: int = 1                 # 1 = first pass, 2 = re-run after demand feedback

    @property
    def is_baseline(self) -> bool:
        return self.baseline is None

    @property
    def base_state(self) -> YearState | None:
        """Baseline state of the same year (None in a baseline run)."""
        if self.baseline is not None and self.baseline.has(self.year):
            return self.baseline.at(self.year)
        return None


def _index_fn(state: YearState, history: Results):
    def index(kind: str, year: int) -> float:
        key = INDEX_KEYS[kind]
        if year == state.year:
            return state.macro[key]
        if year < BASE_YEAR:
            return 1.0
        return history.at(year).macro[key]
    return index


def _baseline_index_fn(baseline: Results):
    def index(kind: str, year: int) -> float:
        if year < BASE_YEAR:
            return 1.0
        return baseline.at(year).macro[INDEX_KEYS[kind]]
    return index


def run(
    scenario: Scenario | None = None,
    *,
    end: int = END_YEAR,
    baseline: Results | None = None,
    params: ParamSet | None = None,
) -> Results:
    """Run one simulation from BASE_YEAR to `end`.

    Pass `baseline` (a finished run without measures) when running a scenario with measures:
    behavioural responses and demand feedback are computed relative to it.
    """
    from .. import behaviour, demographics, labour, macro, regional
    from ..fiscal import balance, expenditure, revenue

    pset = (params or ParamSet.load()).with_scenario(scenario)
    base_pset = pset.with_scenario(None)
    results = Results(name=pset.scenario.name)
    prev: YearState | None = None

    for year in range(BASE_YEAR, end + 1):
        state = YearState(year=year)
        ctx = Context(year=year, params=pset, history=results, baseline=baseline, base_p=None)

        # 1. prices and indices, from un-uprated parameters
        p_plain = pset.at(year, None)
        macro.prices(state, prev, p_plain, ctx)
        # Cumulative real drift of wage-linked tax thresholds: share of real wage growth
        # (wage index over health index) passed on to thresholds on top of legal indexation.
        link = float(p_plain.get("core.tax_threshold_wage_link", 0.0) or 0.0)
        if prev is None:
            state.macro["tax_threshold_drift"] = 1.0
        else:
            real = (state.macro["wage_index"] / prev.macro["wage_index"]) / (
                state.macro["health_index"] / prev.macro["health_index"])
            state.macro["tax_threshold_drift"] = prev.macro["tax_threshold_drift"] * real ** link

        # 2. full parameter views (uprated with this year's indices)
        p = pset.at(year, _index_fn(state, results))
        if baseline is not None and baseline.has(year):
            ctx.base_p = base_pset.at(year, _baseline_index_fn(baseline))

        demographics.run(state, prev, p, ctx)

        # 3. behavioural adjustments relative to baseline (never in the base year / baseline run)
        state.adjust = {}
        state.macro.setdefault("demand_gap", 0.0)
        if baseline is not None and year > BASE_YEAR:
            behaviour.run(state, prev, p, ctx)

        def economy_and_budget() -> None:
            labour.run(state, prev, p, ctx)
            macro.output(state, prev, p, ctx)
            expenditure.run(state, prev, p, ctx)
            revenue.run(state, prev, p, ctx)
            balance.run(state, prev, p, ctx)
            regional.run(state, prev, p, ctx)

        economy_and_budget()

        # 4. short-run demand feedback of the fiscal impulse (one extra pass)
        if baseline is not None and year > BASE_YEAR:
            gap = float(behaviour.demand_gap(state, prev, p, ctx))
            if abs(gap) > 1e-7:
                state.macro["demand_gap"] = gap
                ctx.pass_no = 2
                economy_and_budget()

        results.add(state)
        prev = state

    return results


@lru_cache(maxsize=4)
def run_baseline(end: int = END_YEAR) -> Results:
    return run(None, end=end)


def run_scenario(scenario: Scenario, end: int = END_YEAR) -> tuple[Results, Results]:
    """Returns (baseline, scenario) results."""
    base = run_baseline(end)
    return base, run(scenario, end=end, baseline=base)


def compare(base: Results, scen: Results) -> dict[str, list[float | None]]:
    """Scenario minus baseline for every scalar series."""
    a, b = base.series(), scen.series()
    out: dict[str, list[float | None]] = {}
    for k, vb in b.items():
        va = a.get(k)
        if va is None:
            continue
        out[k] = [(y - x) if (x is not None and y is not None) else None for x, y in zip(va, vb)]
    return out
