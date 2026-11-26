"""Calibration helper: trend productivity that reproduces official real GDP growth.

    uv run python -m belsim.macro.outlook

Real GDP is employment x productivity. `macro.productivity_growth` for 2025-2031 is the path
that, with the employment the labour module produces, gives the official real growth (Eurostat
outturn for 2025, Federal Planning Bureau outlook of June 2026 for 2026-2031). When the
employment path of the baseline changes, run this again and copy the printed values into
`params/macro.yaml`; `tests/test_macro_outlook.py` fails when the two have drifted apart.
Nothing here is used at run time.
"""
from __future__ import annotations

from ..core import engine
from ..core.params import Change, Measure, ParamSet, Scenario

# real GDP growth, % a year: 2025 Eurostat nama_10_gdp; 2026-2031 Federal Planning Bureau,
# Economic outlook 2026-2031 (June 2026), Tableau A.3
OFFICIAL_REAL_GROWTH = {2025: 0.9, 2026: 0.7, 2027: 1.0, 2028: 1.2, 2029: 1.4, 2030: 1.3, 2031: 1.2}
PARAM = "macro.productivity_growth"


def solve_productivity(targets: dict[int, float] | None = None, rounds: int = 4) -> dict[int, float]:
    """Trend productivity growth by year (fractions, 4 decimals) such that baseline real GDP
    growth equals `targets` (% a year). Fixed-point iteration: growth reacts one for one."""
    targets = dict(targets or OFFICIAL_REAL_GROWTH)
    pset = ParamSet.load()
    g = {y: float(pset.value(PARAM, y)) for y in targets}
    end = max(targets)
    for _ in range(rounds):
        changes = [Change(PARAM, round(v, 4), y, end=y) for y, v in g.items()]
        run = engine.run(Scenario(name="calibration", measures=[Measure(name="p", changes=changes)]), end=end)
        for y, target in targets.items():
            g[y] += target / 100.0 - run.at(y).macro["gdp_real_growth"]
    return {y: round(v, 4) for y, v in g.items()}


if __name__ == "__main__":
    for year, value in solve_productivity().items():
        print(f"    {year}: {value}")
