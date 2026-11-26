"""Population of tax units behind the aggregate personal income tax and social contributions.

The aggregate lines are the household calculator (`calc.tax_household`) summed over a grid

    region of residence (3) x tax-unit type (18) x income group (24)

for each income kind (private and public employees, self-employed, pensions, unemployment and
sickness benefits). Head counts and incomes come from the simulation every year
(`state.emp`, `state.wage_dist`, `state.status` by region, benefit spending), so the tax base
moves with the scenario. The structure that the simulation does not have comes from the
Statbel fiscal income statistics of income year 2024 (data/base/revenue_pit_statbel.csv and
revenue_pit_distribution.csv, built by `belsim.data.fetch_revenue_pit`) and is fixed in the
base year:

  income groups    the ten tenths of each kind (`state.wage_dist`, stylised profiles for
                   benefits) are refined into 24 groups (`FINE_EDGES`: halves of the first nine
                   tenths, six groups in the top tenth up to the top 0.1 %), see `refine`.
  distribution     taxable income is then put on the distribution of the tax returns: the
                   model's taxpayers of a region are ranked by taxable income and each receives
                   the income of the same rank in the fiscal statistics (persons behind the
                   returns of the region by decile, the top ten percentiles and type of return,
                   `fiscal_distribution`). The correction is stored as a ratio by income level
                   (`map_x`, `map_ratio`: more very low and very high incomes than the earnings
                   deciles of the model, fewer in the middle) and applied every year to incomes
                   expressed relative to the average of their kind, so the amount of income taxed
                   at each marginal rate is that of the tax returns in the base year and follows
                   the simulated incomes afterwards. Social contributions keep the model's own
                   wages.
  tax-unit types   shares of persons by region: taxed alone with 0-5 dependants (individual
                   returns, table B.5), partner in a two-income joint return with 0-5
                   dependants, sole income of a joint return with 0-5 dependants (table B.3).
                   A person in a two-income couple is taxed with a representative partner
                   (the median income of the same kind in the region): the household rules
                   (unused allowance of a low-income partner used by the other, supplements
                   for children, partial marital quotient) are evaluated for that pair and
                   half of their effect is attributed to each partner; everybody keeps his
                   or her own place in the income distribution. The kind of tax unit follows
                   income as in the tax returns (`_family_tilt`: the sole income of a joint
                   return is on average far above an individual return, table B.3), the
                   number of units of each kind staying that of the statistics. The special
                   social security contribution is due per household until income 2027: on
                   the joint income for joint returns (`_special_ssc`).
  taxpayers        persons with income in the tax returns (8.8 million) over the model's head
                   count of workers and benefit recipients at one moment (8.4 million), by
                   region: more people earn income during a year than hold a status at one time,
                   so the same income is spread over more taxpayers (`persons_region`; social
                   contributions keep the model's head counts and wages).
  taxable share    net taxable income of each income kind (table B.2, plus the property,
                   capital and miscellaneous income taxed with it) over what the model's
                   gross incomes give: wages of the national accounts (D.11) include untaxed
                   components (meal vouchers, exempt bonuses, student and flexi-job pay, pay of
                   international staff), mixed income is more than declared profits.
  separate income  income taxed separately (arrears, termination pay, pension capital: table C
                   total less table B.2). It is not in the distribution above and does not
                   follow the progressive schedule: its tax is the base-year average rate of
                   the region on an amount that grows with taxable income (`separate_tax`), so
                   a change of the rates or of the allowance does not move it.
  regional level   net taxable income per recipient of each kind by region relative to the
                   national average (table B.2 over the model's head counts by residence), and
                   the regions' shares of the state tax (table C; `tax_rel`: what the regional
                   distributions leave unexplained).

Nothing here is a parameter of legislation: the law is in `calc` and params/revenue.yaml.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

from ...core.dims import J, REGIONS, S
from . import calc

DATA = Path(__file__).resolve().parents[3] / "data" / "base"
N_DEP = 5                                    # dependants 1..5 (5 = five or more)
TYPES = ([f"alone_{k}" for k in range(N_DEP + 1)] + [f"couple_{k}" for k in range(N_DEP + 1)]
         + [f"sole_{k}" for k in range(N_DEP + 1)])
T = {name: i for i, name in enumerate(TYPES)}
CHILDREN = np.array([int(t.split("_")[1]) for t in TYPES])[:, None]           # [type, 1]
SINGLE = np.array([float(t.startswith("alone") and not t.endswith("_0")) for t in TYPES])[:, None]
COUPLE = np.array([float(t.startswith("couple")) for t in TYPES])[:, None]   # partner with own income
JOINT = np.array([float(not t.startswith("alone")) for t in TYPES])[:, None]  # taxed jointly
HALF = 1.0 - 0.5 * COUPLE                    # share of the household-rule effect attributed to the person
FAMILIES = ("alone", "couple", "sole")       # individual return, two-income joint return, sole income of a joint return
FAMILY = np.array([FAMILIES.index(t.split("_")[0]) for t in TYPES])          # family of each type
CAR = np.array([0.0, 1.0]).reshape(2, 1, 1, 1)
# income groups inside a kind: cumulative shares of persons (halves of the first nine tenths, then
# 90-94-97-99-99.5-99.9-100 %)
FINE_EDGES = np.concatenate([np.arange(19) * 0.05, [0.94, 0.97, 0.99, 0.995, 0.999, 1.0]])
FINE_W = np.diff(FINE_EDGES)                 # share of the persons of a kind in each group
N_FINE = FINE_W.size
FINE_DECILE = np.minimum((FINE_EDGES[:-1] * 10.0 + 1e-9).astype(int), 9)
_TOP = (FINE_EDGES[18:] - 0.9) / 0.1         # the same edges inside the top tenth
_MID = (9, 10)                               # the two groups around the median
_TILT_WINDOW = 0.05                          # half-width, in ranks, of the window of `_family_tilt`

# income kind -> (status holding the head count, component of the fiscal statistics)
STATUS = {"emp_private": "emp_private", "emp_public": "emp_public", "self_employed": "self_employed",
          "pension": "retired", "unemployment": "unemployed", "disability": "disabled"}
GROUP = {"emp_private": "wages", "emp_public": "wages", "self_employed": "self_employed",
         "pension": "pensions", "unemployment": "unemployment", "disability": "sickness"}
GROUPS = ("wages", "self_employed", "pensions", "unemployment", "sickness")
REGIONAL_KEYS = ("state_tax", "child_credit", "gross", "ssc_employee", "special_ssc",
                 "ssc_employer_private", "ssc_employer_public", "ssc_self_employed",
                 "ssc_replacement_income", "wage_bill_public", "taxable")
RETURN_TYPES = ("individual", "joint_one_income", "joint_two_incomes")


@lru_cache(maxsize=1)
def statbel() -> dict[tuple[str, str, str], float]:
    """(table, region, key) -> value of the Statbel fiscal income statistics. Read once."""
    path = DATA / "revenue_pit_statbel.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `uv run python -m belsim.data.fetch_revenue_pit`")
    with path.open() as f:
        return {(r["table"], r["region"], r["key"]): float(r["value"]) for r in csv.DictReader(f)}


def unit_shares(p, pensioners: float) -> tuple[np.ndarray, np.ndarray]:
    """([region, type] shares of working-age persons with income, same for pensioners).

    Counts of returns by type and number of dependants are official; not observed and assumed:
    which single-income joint returns are pensioner couples (`pit.sole_income_pension_share`),
    how many dependants the working-age ones have (same distribution as all joint returns), and
    the share of pensioners living in a two-income couple (same as the whole population).
    Dependants are attributed to working-age tax units."""
    sb = statbel()
    ps = float(p["pit.sole_income_pension_share"])
    work, pens = np.zeros((len(REGIONS), len(TYPES))), np.zeros((len(REGIONS), len(TYPES)))
    for i, reg in enumerate(REGIONS):
        ind = np.array([sb[("B5", reg, f"individual.dependants_{k}")] for k in range(N_DEP + 1)])
        jnt = np.array([sb[("B5", reg, f"joint.dependants_{k}")] for k in range(N_DEP + 1)])
        one, two = sb[("B3", reg, "joint_one_income.returns")], sb[("B3", reg, "joint_two_incomes.returns")]
        persons = ind.sum() + 2.0 * two + one
        share = sb[("B2", reg, "pensions")] / sb[("B2", "BE", "pensions")]
        old = pensioners * share if pensioners > 0 else 0.28 * persons
        old = min(old, 0.6 * persons)
        young = persons - old
        coupled = 2.0 * two / persons                      # persons in a two-income joint return
        sole_work = (1.0 - ps) * one * jnt / jnt.sum()
        two_dep = np.maximum(jnt - sole_work, 0.0)         # two-income returns by number of dependants
        two_dep[0] = max(two - two_dep[1:].sum() - 0.5 * coupled * old, 0.0)
        for k in range(N_DEP + 1):
            work[i, T[f"couple_{k}"]] = 2.0 * two_dep[k] / young
            work[i, T[f"sole_{k}"]] = sole_work[k] / young
            if k:
                work[i, T[f"alone_{k}"]] = ind[k] / young
        work[i, T["alone_0"]] = max(1.0 - work[i].sum(), 0.0)
        work[i] /= work[i].sum()
        pens[i, T["sole_0"]] = min(ps * one / old, 0.5)
        pens[i, T["couple_0"]] = coupled
        pens[i, T["alone_0"]] = 1.0 - pens[i].sum()
    return work, pens


def population(state, p, benefits: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """kind -> (persons by region of residence [region], gross income of each tenth [decile]).

    Workers: national head count from `state.emp`, split by residence with `state.status`;
    wages from `state.wage_dist`, a tenth of the workers at each of its ten wages. Jobs that a
    scenario adds to or takes from particular wage deciles
    (`labour["scenario_jobs_private_by_decile"]`) are already in `wage_dist`: labour re-cuts it
    into the tenths of the workers after the change, with the wage bill of the job type, so the
    head count per tenth stays uniform here and the key is not applied a second time (weighting
    the heads by decile on top of the re-cut wages would count the jobs twice). Replacement income: recipients from `state.status`, benefit
    spending of the year (`benefits`: kind -> (EUR mn, recipients)) spread over a stylised
    distribution."""
    by_status = state.status.sum(axis=(1, 2))                       # [status, region]
    pop = state.pop.sum(axis=(0, 1))
    fallback = pop / pop.sum() if pop.sum() > 0 else np.full(len(REGIONS), 1.0 / len(REGIONS))
    workers = state.emp.sum(axis=0)
    out = {}
    for kind in calc.KINDS:
        n = by_status[S[STATUS[kind]]]
        split = n / n.sum() if n.sum() > 0 else fallback
        if kind in J:
            out[kind] = (workers[J[kind]] * split, np.asarray(state.wage_dist[J[kind]], dtype=float))
        else:
            total, count = benefits.get(kind, (0.0, 0.0))
            prof = np.asarray(p["pit.pension_decile_profile" if kind == "pension" else "pit.benefit_decile_profile"],
                              dtype=float)
            level = total * 1e6 / count if count > 0 else 0.0
            extra = benefits.get("pension_minimum_increment") if kind == "pension" else None
            if extra and count > 0:
                # spending added (or cut) through the guaranteed minimum pension is not spread
                # over all pensions: it is taxed apart, at the rate of low pensions (`evaluate`)
                level = (total - extra[0]) * 1e6 / count
            out[kind] = (count * split, level * prof / prof.mean())
    return out


def _relative(rel: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Regional income levels rescaled so the national average income is unchanged."""
    mean = float((n * rel).sum() / n.sum())
    return rel / mean if mean > 0 else np.ones_like(rel)


def refine(deciles) -> np.ndarray:
    """Mean income of the `N_FINE` groups of `FINE_EDGES` from the means of the ten tenths
    ([..., 10] -> [..., 24]); the mean of every tenth is kept. The two halves of a tenth lie a
    quarter of the gap to the nearest neighbouring tenth away from its mean; the top tenth is a
    Pareto tail above the ninth decile (index from its mean over that threshold)."""
    g = np.asarray(deciles, dtype=float)
    gap = np.diff(g, axis=-1)
    below = np.concatenate([g[..., :1], gap], axis=-1)[..., :9]
    d = np.clip(np.minimum(below, gap), 0.0, None) / 4.0
    body = np.stack([g[..., :9] - d, g[..., :9] + d], axis=-1).reshape(g.shape[:-1] + (18,))
    m = g[..., 9]
    threshold = g[..., 8] + 2.0 * d[..., 8]
    a = np.clip(m / np.maximum(m - threshold, 1e-9), 1.5, 1e6)[..., None]
    e = (1.0 - _TOP) ** (1.0 - 1.0 / a)
    top = m[..., None] * (e[..., :-1] - e[..., 1:]) / np.diff(_TOP)
    return np.concatenate([body, top], axis=-1)


@lru_cache(maxsize=1)
def _returns() -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """(region, type of return) -> (income per return, number of returns): the distribution of
    globally taxable income of the fiscal statistics. Read once."""
    path = DATA / "revenue_pit_distribution.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `uv run python -m belsim.data.fetch_revenue_pit`")
    rows: dict[tuple[str, str], list[tuple[float, float]]] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.setdefault((r["region"], r["type"]), []).append((float(r["income"]), float(r["returns"])))
    return {k: (np.array([x for x, _ in v]), np.array([n for _, n in v])) for k, v in rows.items()}


@lru_cache(maxsize=16)
def fiscal_distribution(region: str, lower_shares: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray, float]:
    """Distribution of taxable income between the persons behind the tax returns of a region:
    (cumulative share of persons, cumulative income per person of the region up to that rank
    [EUR], number of persons). An individual return and a joint return with one professional
    income are one person; a joint return with two incomes is two persons, the partner with
    the lower income holding one of the equally likely `lower_shares` of the joint income
    (`pit.two_income_lower_share` - the split inside couples is not published)."""
    data = _returns()
    xs, ws = [], []
    for kind in RETURN_TYPES[:2]:
        x, n = data[(region, kind)]
        xs.append(x)
        ws.append(n)
    x, n = data[(region, RETURN_TYPES[2])]
    for share in lower_shares:
        xs += [x * share, x * (1.0 - share)]
        ws += [n / len(lower_shares)] * 2
    x, w = np.concatenate(xs), np.concatenate(ws)
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    total = float(w.sum())
    u = np.concatenate([[0.0], np.cumsum(w) / total])
    income = np.concatenate([[0.0], np.cumsum(x * w) / total])
    return u, income, total


@lru_cache(maxsize=16)
def fiscal_families(region: str, lower_shares: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    """The persons of `fiscal_distribution` by kind of return: (cumulative share of persons
    ranked by income, cumulative share of all persons who are below that rank and come from an
    individual return / a two-income joint return / a joint return with one income
    [point, family], in the order of `FAMILIES`)."""
    data = _returns()
    xs, ws, fam = [], [], []
    for kind, f in ((RETURN_TYPES[0], 0), (RETURN_TYPES[1], 2)):
        x, n = data[(region, kind)]
        xs.append(x)
        ws.append(n)
        fam.append(np.full(x.size, f))
    x, n = data[(region, RETURN_TYPES[2])]
    for share in lower_shares:
        xs += [x * share, x * (1.0 - share)]
        ws += [n / len(lower_shares)] * 2
        fam += [np.full(x.size, 1)] * 2
    x, w, fam = np.concatenate(xs), np.concatenate(ws), np.concatenate(fam)
    order = np.argsort(x, kind="stable")
    w, fam = w[order], fam[order]
    total = float(w.sum())
    u = np.concatenate([[0.0], np.cumsum(w) / total])
    cum = np.zeros((w.size + 1, len(FAMILIES)))
    for f in range(len(FAMILIES)):
        cum[1:, f] = np.cumsum(w * (fam == f)) / total
    return u, cum


def _family_tilt(rows: list[dict], cal: dict, region_names: list[str], lower: tuple[float, ...]) -> list[np.ndarray]:
    """Kind of tax unit by income, one array [region, family, group] per row: the factor on the
    shares of the tax-unit types of a family (`FAMILIES`) for the persons of an income group.
    The persons behind the tax returns are ranked by income like the model's taxpayers
    (`_rank_incomes`); a group takes the mix of returns found at its rank (the single income of
    a joint return is on average much higher than an individual return's), expressed relative
    to the mix of all returns and adjusted so that the persons of each family, summed over the
    income groups, stay what `unit_shares` gives - separately for pensions and for the other
    kinds of income, which have their own shares."""
    n_reg, n_fam = len(REGIONS), len(FAMILIES)
    out = [np.ones((n_reg, n_fam, r["taxable"].shape[-1])) for r in rows]
    onehot = (FAMILY[:, None] == np.arange(n_fam)[None, :]).astype(float)     # [type, family]
    sets = ([i for i, r in enumerate(rows) if r["kind"] == "pension"], [i for i, r in enumerate(rows) if r["kind"] != "pension"])
    for i in range(n_reg):
        t = np.concatenate([r["taxable"][i].ravel() for r in rows])
        w = np.concatenate([r["w"][i].ravel() for r in rows])
        if w.sum() <= 0:
            continue
        u, cum = fiscal_families(region_names[i], lower)
        order = np.argsort(t, kind="stable")
        share = w[order] / w.sum()
        hi = np.cumsum(share)
        mid = hi - 0.5 * share
        # the mix of returns in a window of ranks around the group (a twentieth of the persons
        # on each side, narrower towards the top): the statistics are points, not a density
        half = np.minimum(_TILT_WINDOW, 0.5 * (1.0 - mid) + 0.002)
        lo, hi = np.maximum(mid - half, 0.0), np.minimum(mid + half, 1.0)
        mix = np.ones((t.size, n_fam))
        ok = share > 0
        for f in range(n_fam):
            total = cum[-1, f]
            if total > 0:
                mix[order[ok], f] = (np.interp(hi[ok], u, cum[:, f]) - np.interp(lo[ok], u, cum[:, f])) / (hi[ok] - lo[ok]) / total
        size = rows[0]["taxable"][i].size
        mix = mix.reshape(len(rows), size, n_fam)                              # [row, group, family]
        for members, units in zip(sets, (cal["units_pension"], cal["units_work"])):
            if not members:
                continue
            base = units[i] @ onehot                                           # family shares of the kind set
            ww = np.stack([rows[k]["w"][i].ravel() for k in members])          # [row, group]
            if ww.sum() <= 0:
                continue
            m = mix[members]
            a = np.ones(n_fam)
            for _ in range(30):                                                # proportional fitting
                cell = base * m * a
                cell = cell / np.maximum(cell.sum(axis=-1, keepdims=True), 1e-300)
                got = (cell * ww[..., None]).sum(axis=(0, 1)) / ww.sum()
                a = a * np.where(got > 0, base / np.maximum(got, 1e-300), 1.0)
            cell = base * m * a
            cell = cell / np.maximum(cell.sum(axis=-1, keepdims=True), 1e-300)
            factor = np.divide(cell, base, out=np.ones_like(cell), where=base > 0)   # [row, group, family]
            for k, row in enumerate(members):
                out[row][i] = factor[k].T
    return out


def _units(cal: dict, rows: list[dict]) -> list[np.ndarray]:
    """Shares of the tax-unit types for every row, [region, type, group]: the shares of the kind
    of income (`units_work`, `units_pension`) with the kind of tax unit following income
    (`family_tilt`, when the calibration holds it for these rows)."""
    tilt = cal.get("family_tilt")
    out = []
    for k, r in enumerate(rows):
        base = cal["units_pension"] if r["kind"] == "pension" else cal["units_work"]
        if tilt is None or len(tilt) != len(rows) or tilt[k].shape[-1] != r["taxable"].shape[-1]:
            out.append(base[:, :, None] + np.zeros(r["taxable"].shape[-1]))
        else:
            out.append(base[:, :, None] * tilt[k][:, FAMILY, :])
    return out


def fiscal_persons(region: str = "BE") -> float:
    """Persons with income in the tax returns: individual returns, both partners of two-income
    joint returns, one partner of single-income joint returns."""
    sb = statbel()
    ind = sum(sb[("B5", region, f"individual.dependants_{k}")] for k in range(N_DEP + 1))
    return ind + 2.0 * sb[("B3", region, "joint_two_incomes.returns")] + sb[("B3", region, "joint_one_income.returns")]


def _rows(pop, p, cal: dict) -> list[dict]:
    """One row per income kind (private employees: two, without and with a company car) with
    arrays [region, 1, group]: `taxable` (net taxable income before the fiscal distribution),
    `reduction`, `w` (taxpayers), and `kind`, `allowance`, `wage` (gross income per person of
    the model), `heads` (persons of the model), `employee`."""
    out = []
    persons = cal["persons_region"][:, None, None]
    for kind in calc.KINDS:
        n, g = pop[kind]
        if n.sum() <= 0.0 or not g.any():
            continue
        group = GROUP[kind]
        wage = refine(g) * _relative(cal["rel"][group], n)[:, None, None]      # [region, 1, group]
        heads = n[:, None, None] * FINE_W                                      # model persons per cell
        wp = heads * persons                                                   # taxpayers per cell
        common = {"kind": kind, "wage": wage, "heads": heads, "employee": kind in ("emp_private", "emp_public")}
        if kind == "emp_private":
            car = np.asarray(p["pit.company_car_share_by_decile"], dtype=float)[FINE_DECILE]
            base = calc.tax_base((wage / persons)[None], p, kind, company_car=CAR, taxable_share=cal["share"][group])
            out.append({**common, "taxable": base["taxable"][0], "reduction": base["reduction"][0],
                        "w": wp * (1.0 - car), "allowance": base["allowance"]})
            out.append({**common, "taxable": base["taxable"][1], "reduction": base["reduction"][0],
                        "w": wp * car, "allowance": base["allowance"], "wage": None})
        else:
            base = calc.tax_base(wage / persons, p, kind, taxable_share=cal["share"][group])
            allowance = base["allowance"]
            if kind in calc.REPLACEMENT_KINDS:
                # people who draw a pension or benefit next to earned income keep part of the
                # 2026-2030 increase of the allowance (the cut in the reduction for replacement
                # income is pro rata of that income): the neutralisation binds on a share only
                allowance += (1.0 - float(p.get("pit.allowance_neutralised_population_share", 1.0))) \
                    * min(float(p["pit.allowance_neutralised_replacement"]), float(p["pit.basic_allowance"]))
            out.append({**common, "taxable": base["taxable"], "reduction": base["reduction"] + np.zeros_like(wage),
                        "w": wp + np.zeros_like(wage), "allowance": allowance})
    return out


def _kind_means(rows: list[dict]) -> dict[str, np.ndarray]:
    """kind -> mean taxable income per taxpayer by region (before the fiscal distribution)."""
    num: dict[str, np.ndarray] = {}
    den: dict[str, np.ndarray] = {}
    for r in rows:
        num[r["kind"]] = num.get(r["kind"], 0.0) + (r["taxable"] * r["w"]).sum(axis=(1, 2))
        den[r["kind"]] = den.get(r["kind"], 0.0) + r["w"].sum(axis=(1, 2))
    return {k: np.divide(num[k], den[k], out=np.zeros_like(num[k]), where=den[k] > 0) for k in num}


def _mapped(rows: list[dict], cal: dict) -> list[np.ndarray]:
    """Taxable income of every row on the fiscal distribution: income x the ratio stored for
    its level, the level being measured relative to the base-year average of the kind in the
    region (`mean0`), so the correction follows the nominal growth of each kind of income."""
    if "map_x" not in cal:
        return [r["taxable"] for r in rows]
    means = _kind_means(rows)
    out = []
    for r in rows:
        t = r["taxable"]
        mean0 = cal["mean0"].get(r["kind"])
        res = np.empty_like(t)
        for i in range(t.shape[0]):
            scale = means[r["kind"]][i] / mean0[i] if mean0 is not None and mean0[i] > 0 else 1.0
            ti = t[i]
            res[i] = ti * np.interp(ti / scale if scale > 0 else ti, cal["map_x"][i], cal["map_ratio"][i])
        out.append(res)
    return out


def _bracket_base(p):
    """`pit.bracket_base_factor` when it is not neutral (else None): the multiplier on the
    income taxed in each bracket of the schedule, for the population of tax units only."""
    get = getattr(p, "get", None)
    factors = get("pit.bracket_base_factor") if get else None
    if not factors or all(float(k) == 1.0 for k in factors):
        return None
    return tuple(float(k) for k in factors)


def _rank_incomes(t: np.ndarray, w: np.ndarray, dist: tuple) -> np.ndarray:
    """Income of the fiscal distribution `dist` for taxpayers with incomes `t` and weights `w`:
    each one gets the mean income of the persons of the same rank."""
    u, income, _ = dist
    order = np.argsort(t, kind="stable")
    share = w[order] / w.sum()
    hi = np.cumsum(share)
    lo = hi - share
    out = np.zeros_like(t)
    ok = share > 0
    out[order[ok]] = (np.interp(hi[ok], u, income) - np.interp(lo[ok], u, income)) / share[ok]
    return out


def calibrate(state, p, benefits: dict) -> dict:
    """Base-year structure of the population of tax units (see the module docstring):
    `persons` (taxpayers per person of the model) and `persons_region`, `share` (taxable share
    by group), `rel` (regional income level by group), `map_x`, `map_ratio` and `mean0` (the
    fiscal distribution of taxable income, by region), `separate` (separately taxed income over
    net income) and `separate_tax_rate` (its tax per euro of globally taxable income, by region),
    `units_work`, `units_pension` ([region, type] shares), `family_tilt` (kind of tax unit by
    income group, `_family_tilt`), `employee_share`, `tax_rel` (regional correction of the
    state tax) and, for information, `brackets` (income taxed at each marginal rate, EUR mn) and
    `distribution_fit` (taxable income by group after the fiscal distribution over table B.2)."""
    sb = statbel()
    pop = population(state, p, benefits)
    heads = np.sum([v[0] for v in pop.values()], axis=0)                       # [region]
    total = float(heads.sum())
    persons = float(np.clip(fiscal_persons() / total, 0.8, 1.3)) if total > 0 else 1.0
    regional = bool(heads.min() > 0)                   # every region populated (not a synthetic state)
    if regional:
        persons_region = np.clip(np.array([fiscal_persons(r) for r in REGIONS]) / heads, 0.8, 1.4)
    else:
        persons_region = np.full(len(REGIONS), persons)
    net = sb[("B2", "BE", "net_income")]
    other = net / sum(sb[("B2", "BE", g)] for g in GROUPS)      # property, capital, miscellaneous income
    target = {g: sb[("B2", "BE", g)] * other * 1e6 for g in GROUPS}
    rel = {}
    for group in GROUPS:
        n = sum(pop[k][0] for k in calc.KINDS if GROUP[k] == group)
        if n.min() > 0:
            income = np.array([sb[("B2", reg, group)] for reg in REGIONS])
            rel[group] = np.clip((income / n) / (income.sum() / n.sum()), 0.5, 2.0)
        else:
            rel[group] = np.ones(len(REGIONS))
    cal = {"persons": persons, "persons_region": persons_region, "share": dict.fromkeys(GROUPS, 1.0), "rel": rel}

    def group_totals(rows, taxable):
        out = dict.fromkeys(GROUPS, 0.0)
        for r, t in zip(rows, taxable):
            out[GROUP[r["kind"]]] += float((t * r["w"]).sum())
        return out

    raw = group_totals(_rows(pop, p, cal), [r["taxable"] for r in _rows(pop, p, cal)])
    # the fiscal distribution of each region, at the level of table B.2
    lower = tuple(float(x) for x in p["pit.two_income_lower_share"])
    dists = []
    for i, reg in enumerate(REGIONS):
        u, income, n = fiscal_distribution(reg if regional else "BE", lower)
        level = sb[("B2", reg if regional else "BE", "net_income")] * 1e6 / (income[-1] * n)
        dists.append((u, income * level, n))

    def rank(rows):
        """Incomes of the fiscal distribution for every row, region by region."""
        out = [np.zeros_like(r["taxable"]) for r in rows]
        for i in range(len(REGIONS)):
            t = np.concatenate([r["taxable"][i].ravel() for r in rows])
            w = np.concatenate([r["w"][i].ravel() for r in rows])
            if w.sum() <= 0:
                continue
            y = _rank_incomes(t, w, dists[i])
            k = 0
            for r, o in zip(rows, out):
                size = r["taxable"][i].size
                o[i] = y[k:k + size].reshape(r["taxable"][i].shape)
                k += size
        return out

    # taxable share of each group: its income after the fiscal distribution equals table B.2
    # (flat-rate expenses and the ranking make this non-linear: fixed-point iteration)
    for _ in range(25):
        rows = _rows(pop, p, cal)
        got = group_totals(rows, rank(rows))
        worst = 0.0
        for g in GROUPS:
            if got[g] > 0:
                step = target[g] / got[g]
                cal["share"][g] = float(np.clip(cal["share"][g] * step, 0.2, 5.0))
                worst = max(worst, abs(step - 1.0))
        if worst < 2e-3:
            break
    rows = _rows(pop, p, cal)
    ranked = rank(rows)
    fit = group_totals(rows, ranked)
    cal["distribution_fit"] = {g: (fit[g] / target[g] if target[g] > 0 else 1.0) for g in GROUPS}
    cal["model_share_1"] = {g: raw[g] / 1e6 for g in GROUPS}
    cal["mean0"] = _kind_means(rows)
    # the correction as a ratio by income level, region by region
    map_x, map_ratio = [], []
    for i in range(len(REGIONS)):
        t = np.concatenate([r["taxable"][i].ravel() for r in rows])
        y = np.concatenate([o[i].ravel() for o in ranked])
        w = np.concatenate([r["w"][i].ravel() for r in rows])
        keep = (w > 0) & (t > 0)
        if not keep.any():
            map_x.append(np.array([0.0, 1.0]))
            map_ratio.append(np.array([1.0, 1.0]))
            continue
        t, y, w = t[keep], y[keep], w[keep]
        x, inverse = np.unique(np.round(t, 2), return_inverse=True)
        num, den = np.bincount(inverse, y * w), np.bincount(inverse, t * w)
        map_x.append(x)
        map_ratio.append(np.clip(num / den, 0.02, 50.0))
    populated = [i for i in range(len(REGIONS)) if map_x[i].size > 2]
    for i in range(len(REGIONS)):                      # an empty region takes the first populated one
        if i not in populated and populated:
            map_x[i], map_ratio[i] = map_x[populated[0]], map_ratio[populated[0]]
    cal["map_x"], cal["map_ratio"] = map_x, map_ratio

    work, pens = unit_shares(p, float(pop["pension"][0].sum()) * persons)
    cal["units_work"], cal["units_pension"] = work, pens
    # share of the working-age persons with income who are employees: how often the partner in a
    # two-income couple owes the special social security contribution too
    working = sum(float(pop[k][0].sum()) for k in calc.KINDS if k != "pension")
    cal["employee_share"] = sum(float(pop[k][0].sum()) for k in calc.EMPLOYEE_KINDS) / working if working > 0 else 1.0
    if p.get("pit.tax_unit_type_follows_income", 1):
        cal["family_tilt"] = _family_tilt(rows, cal, [reg if regional else "BE" for reg in REGIONS], lower)
    separate = sb[("C", "BE", "net_taxable_income")] / net - 1.0
    cal["separate"] = separate
    # state tax of the progressive schedule by region, then the regions' shares of the state tax
    # on the fiscal statistics (only if every region is populated) and the separately taxed income
    res = evaluate(state, p, benefits, cal)
    tax, taxable = res["state_tax"], res["taxable"]
    sep = np.full(len(REGIONS), separate)
    if regional and tax.min() > 0:
        sep = np.array([sb[("C", reg, "net_taxable_income")] / sb[("B2", reg, "net_income")] - 1.0 for reg in REGIONS])
        official = np.array([sb[("C", reg, "state_tax")] for reg in REGIONS])
        whole = tax * (1.0 + sep)
        cal["tax_rel"] = (official / official.sum()) / (whole / whole.sum())
    rate = np.divide(tax * cal.get("tax_rel", 1.0), taxable, out=np.zeros_like(tax), where=taxable > 0)
    cal["separate_tax_rate"] = sep * rate
    thr = p["pit.bracket_thresholds"]
    edges = np.array([0.0, *thr, np.inf])
    cal["brackets"] = [float(sum(((np.clip(t, lo, hi) - lo) * r["w"]).sum() for r, t in zip(rows, ranked)) / 1e6)
                       for lo, hi in zip(edges[:-1], edges[1:])]
    return cal


def _special_ssc(t, partner, units, employees: float, p) -> np.ndarray:
    """Special social security contribution per employee with taxable income `t`
    [region, 1, group]. While the contribution is due per household (`calc.special_joint_scale`),
    persons taxed alone pay on their own income; the sole income of a joint return pays on the
    joint scale; a partner in a two-income couple pays the household's contribution on the joint
    income with the representative partner (`partner` [region, 1, 1]) - half of it when the
    partner is an employee too (share `employees`), all of it otherwise, the partner then
    paying none. `units` [region, type, group]: shares of the tax-unit types."""
    own = calc.special_ssc(t, p)
    if calc.special_joint_scale(p) is None:
        return own
    joint = (units * JOINT).sum(axis=1, keepdims=True)
    couple = (units * COUPLE).sum(axis=1, keepdims=True)
    return (own * (1.0 - joint) + calc.special_ssc(t, p, joint=True) * (joint - couple)
            + calc.special_ssc(t + partner, p, joint=True) * couple * (1.0 - 0.5 * employees))


def evaluate(state, p, benefits: dict, cal: dict) -> dict[str, np.ndarray]:
    """Taxes and contributions of the population of tax units, EUR, by region of residence
    (`REGIONAL_KEYS` -> [region]). `state_tax` is the state tax before the split into federal
    and regional tax and before surcharges; `child_credit` the refundable credit; `taxable` the
    net taxable income taxed at the progressive rates.

    One call of the household calculator on arrays [row, region, type, group], a row being an
    income kind (private employees: two rows, without and with a company car)."""
    pop = population(state, p, benefits)
    out = {k: np.zeros(len(REGIONS)) for k in REGIONAL_KEYS}
    rows = _rows(pop, p, cal)
    if not rows:
        return out
    quotient = p["pit.marital_quotient_max"]
    taxable = _mapped(rows, cal)
    base_factor = _bracket_base(p)
    employees = cal.get("employee_share", 1.0)
    reduction, weight, allowance, rep_t, rep_r, qmax = [], [], [], [], [], []
    private = None                               # (taxable, reduction) of the median private employee
    lo, hi = _MID
    units = _units(cal, rows)                    # [region, type, group] per row
    for r, t, share in zip(rows, taxable, units):
        kind, red, w = r["kind"], r["reduction"], r["w"]
        # representative partner of a two-income couple: the median income of the kind
        # (pensions, private employees) or a median private employee (other working-age kinds)
        median = (0.5 * (t[..., lo:lo + 1] + t[..., hi:hi + 1]), 0.5 * (red[..., lo:lo + 1] + red[..., hi:hi + 1]))
        if kind == "emp_private":
            if private is None:
                private = median
            else:
                median = private
        elif kind != "pension" and private is not None:
            median = private
        reduction.append(red)
        weight.append(w)
        allowance.append(r["allowance"])
        rep_t.append(median[0])
        rep_r.append(median[1])
        qmax.append(quotient["pension" if kind == "pension" else "work"])
        out["taxable"] += (t * w).sum(axis=(1, 2))
        if r["employee"]:
            out["special_ssc"] += (_special_ssc(t, median[0], share, employees, p) * w).sum(axis=(1, 2))
        wage, heads = r["wage"], r["heads"]
        if wage is None:                         # second row of the private employees
            continue
        out["gross"] += (wage * heads).sum(axis=(1, 2))
        # social contributions: the model's head counts and wages
        if kind in ("unemployment", "disability"):
            continue
        if kind == "self_employed":
            out["ssc_self_employed"] += (calc.self_employed_ssc(wage, p) * heads).sum(axis=(1, 2))
        elif kind == "pension":
            out["ssc_replacement_income"] += (calc.pension_health_contribution(wage, p) * heads).sum(axis=(1, 2))
        else:
            er, ssc, _, _ = calc.employee_contributions(wage, p, kind)
            out["ssc_employee"] += (ssc * heads).sum(axis=(1, 2))
            if kind == "emp_private":
                out["ssc_employer_private"] += (er * heads).sum(axis=(1, 2))
            else:
                out["ssc_employer_public"] += (er * heads).sum(axis=(1, 2))
                out["wage_bill_public"] += (wage * heads).sum(axis=(1, 2))

    col = (len(taxable), 1, 1, 1)
    t, red, w = np.stack(taxable), np.stack(reduction), np.stack(weight)       # [row, region, 1, group]
    al = np.array(allowance).reshape(col)
    pt, pr = np.stack(rep_t) * COUPLE, np.stack(rep_r) * COUPLE                # [row, region, type, 1]
    person = {"kind": "tax_unit", "gross": 0.0, "ssc_own": 0.0, "special_ssc": 0.0, "ssc_employer": 0.0,
              "taxable": t, "reduction": red, "allowance": al}
    partner = {**person, "taxable": pt, "reduction": pr}
    r = calc.tax_household(p, [person, partner], CHILDREN, single_parent=SINGLE * float(p["pit.single_parent_eligible_share"]),
                           joint=JOINT, quotient_max=np.array(qmax).reshape(col), levels=False, base_factor=base_factor)
    alone = calc.state_tax(t, al, red, p, base_factor)
    alone_rep = calc.state_tax(pt, al, pr, p, base_factor)
    w = w * np.stack(units)                                                   # taxpayers [row, region, type, group]
    cells = (alone + HALF * (r["state_tax"] - alone - alone_rep)) * w
    progressive = cells.sum(axis=(0, 2, 3))
    extra = benefits.get("pension_minimum_increment")
    if extra:
        # a change of the guaranteed minimum pension: the amount goes to the pensions at or
        # below the minimum and bears their average rate of state tax (little or none), by
        # region in proportion to pensions
        amount, minimum = extra
        k = next((i for i, x in enumerate(rows) if x["kind"] == "pension"), None)
        if k is not None:
            gross = rows[k]["wage"] * rows[k]["heads"]                          # [region, 1, group]
            income = (gross.sum(axis=(0, 1)) / np.maximum(rows[k]["heads"].sum(axis=(0, 1)), 1e-9))
            low = income <= minimum
            if not low.any():
                low = income <= income.min()
            paid = float(cells[k].sum(axis=(0, 1))[low].sum())
            rate = paid / max(float(gross.sum(axis=(0, 1))[low].sum()), 1e-9)
            region = gross.sum(axis=(1, 2)) / max(float(gross.sum()), 1e-9)
            progressive = progressive + amount * 1e6 * rate * region
            out["gross"] += amount * 1e6 * region
    out["state_tax"] = progressive * cal.get("tax_rel", 1.0) + out["taxable"] * cal.get("separate_tax_rate", 0.0)
    out["child_credit"] = (r["child_credit"] * HALF * w).sum(axis=(0, 2, 3))
    return out
