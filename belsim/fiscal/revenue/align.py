"""Calibration helper: the multipliers of `rev.fpb_alignment`.

    uv run python -m belsim.fiscal.revenue.align            # print the rows for params/revenue.yaml
    uv run python -m belsim.fiscal.revenue.align --write    # and replace them in the file

Rule. The observed 2025 accounts are the starting point and are never overridden: for every
category of the Planning Bureau's account and every year t of the outlook after 2025

    model ratio(t) - model ratio(2025)  =  Bureau ratio(t) - Bureau ratio(2025)        (% of GDP)

the model's 2025 being the outturn (data/base/revenue_outturn.csv) and the ratios being taken
on each one's own GDP. Only the *change* since 2025 is the Bureau's; what the 2025 accounts
show above or below the Bureau's June estimate of 2025 stays in the level. Household and
indirect taxes are aligned by subsector (federal government with social security, regions and
communities, local government): a revenue line is shared between subsectors
(`LINES[line].entity_shares`), so the three multipliers of a group solve a 3 x 3 system. The
multipliers are computed on the baseline *as it is without the alignment* (legislation, tax
bases, `rev.outturn_persistence`), so they hold what is left between that projection and the
Bureau's change.

Official figures: data/base/revenue_fpb_outlook.csv - Federal Planning Bureau, Perspectives
economiques 2026-2031 (June 2026), Tableaux D.1 (corporate tax, capital taxes, effective social
contributions, GDP), D.2, D.3, D.5 and D.6 (household and other direct taxes, indirect taxes by
subsector), EUR mn, transcribed from the publication (no machine-readable release).

When the baseline tax bases change, run this again; `tests/test_revenue_align.py` fails when
the stored multipliers have drifted from what this computes. Nothing here is used at run time.
"""
from __future__ import annotations

import csv
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import DATA, LINES, alignment_categories

PARAM = "rev.fpb_alignment"
START = 2025                                 # observed year the changes are counted from
# subsectors of the Bureau's accounts -> entities of the model whose receipts they hold
SUBSECTORS = {"federal": ("federal", "social_security"), "regions": ("regions_communities",), "local": ("local",)}
GROUPS = ("households", "indirect")          # aligned by subsector
SINGLE = ("corporate", "capital_taxes", "contributions")
PARAMS_FILE = Path(__file__).resolve().parents[3] / "params" / "revenue.yaml"


@lru_cache(maxsize=1)
def outlook() -> dict[tuple[str, str], dict[int, float]]:
    """(series, subsector) -> year -> EUR mn of the Planning Bureau's outlook; GDP under
    ("gdp", "general_government")."""
    out: dict[tuple[str, str], dict[int, float]] = {}
    with (DATA / "revenue_fpb_outlook.csv").open() as f:
        for r in csv.DictReader(f):
            out.setdefault((r["series"], r["subsector"]), {})[int(r["year"])] = float(r["amount"])
    return out


def official_change(series: str, entities: tuple[str, ...], year: int) -> float:
    """Change of the Bureau's ratio to GDP between `START` and `year` (fraction of GDP)."""
    data = outlook()
    gdp = data[("gdp", "general_government")]
    value = lambda y: sum(data[(series, e)][y] for e in entities)
    return value(year) / gdp[year] - value(START) / gdp[START]


def _unaligned(state) -> dict[str, float]:
    """Revenue lines of a baseline year without the alignment multipliers it was run with."""
    applied = state.extra.get("revenue_alignment") or {}
    return {line: value / applied.get(line, 1.0) for line, value in state.revenue.items()}


def compute(base=None) -> dict[int, dict[str, float]]:
    """year -> category -> multiplier (4 decimals) for the years of the outlook after `START`,
    from a baseline run (default: a fresh one to the last year of the outlook)."""
    years = [y for y in sorted(outlook()[("gdp", "general_government")]) if y > START]
    if base is None:
        from ...core import engine
        base = engine.run(None, end=years[-1])
    cats = alignment_categories()
    start = base.at(START)
    gdp0, rev0 = float(start.macro["gdp_nominal"]), _unaligned(start)
    share = lambda line, entities: sum(LINES[line].entity_shares[e] for e in entities)
    out: dict[int, dict[str, float]] = {}
    for year in years:
        state = base.at(year)
        gdp, rev = float(state.macro["gdp_nominal"]), _unaligned(state)
        f: dict[str, float] = {}
        for group in GROUPS:
            names = [f"{group}_{s}" for s in SUBSECTORS]
            # receipts of each subsector (rows) from the lines of each category (columns)
            a = np.array([[sum(rev[k] * share(k, ents) for k in cats[n]) for n in names] for ents in SUBSECTORS.values()])
            a0 = np.array([sum(rev0[k] * share(k, ents) for n in names for k in cats[n]) for ents in SUBSECTORS.values()])
            target = np.array([official_change(group, ents, year) for ents in SUBSECTORS.values()]) * gdp + a0 * gdp / gdp0
            f.update(zip(names, np.linalg.solve(a, target).tolist()))
        for cat in SINGLE:
            ratio = lambda r, g: sum(r[k] for k in cats[cat]) / g
            f[cat] = (ratio(rev0, gdp0) + official_change(cat, ("general_government",), year)) / ratio(rev, gdp)
        out[year] = {c: round(float(f[c]), 4) for c in cats}
    return out


def rows(factors: dict[int, dict[str, float]]) -> str:
    """The dated values of the parameter as they stand in params/revenue.yaml."""
    names = list(alignment_categories())
    lines = ["    2024: {" + ", ".join(f"{c}: 1.0" for c in names) + "}"]
    lines += [f"    {y}: {{" + ", ".join(f"{c}: {f[c]}" for c in names) + "}" for y, f in sorted(factors.items())]
    return "\n".join(lines) + "\n"


def write(factors: dict[int, dict[str, float]], path: Path = PARAMS_FILE) -> None:
    """Replace the dated values of `rev.fpb_alignment` in the parameter file."""
    text = path.read_text()
    pattern = re.compile(r"(^rev\.fpb_alignment:\n(?:  .*\n)*?  values:\n)(?:    \d{4}: \{.*\}\n)+", re.M)
    new, n = pattern.subn(lambda m: m.group(1) + rows(factors), text)
    if n != 1:
        raise RuntimeError(f"values of {PARAM} not found in {path}")
    path.write_text(new)


if __name__ == "__main__":
    result = compute()
    print(rows(result), end="")
    if "--write" in sys.argv[1:]:
        write(result)
        print(f"written to {PARAMS_FILE}")
