"""Who wins and who loses: scenario against baseline for the stylised households of a year.

`distribution(base_results, scen_results, base_pset, scen_pset, year)` evaluates every household
(`households_all`: every household type in every income group; `households`: the short list of
representative cases, same rows) three times:
    E0  baseline rules, baseline economy                      (baseline)
    E1  baseline rules, scenario wage levels                  (-> cause "wages")
    E2  scenario rules, scenario economy                      (scenario)
and adds the loss of purchasing power from the scenario's consumer prices on the household's
baseline basket. The change shown is  net(E2) - net(E0) - price effect, in euros of that year
at baseline prices; it is split exactly into wages, benefits, contributions, direct taxes and
indirect taxes / prices.

This is an approximation with about 290 synthetic households, not a microsimulation: household
types repeated over the income groups of the tax units behind the aggregate income tax; the
weights are the household structure of the income survey (persons by household type and people
in households without work, EU-SILC 2025 on incomes of 2024) rescaled to the head counts of the
baseline simulation of the year. `meta.coverage` says how much of the budget's income tax and contributions the
households add up to.

Income is that of the income survey (EU-SILC): the rules of the model, plus the cash benefits of
the budget the rules do not produce (`budget_benefit_keys`), capital income and the recurrent
property tax (`survey_items`). The poverty rates, the Gini coefficient and S80/S20 count the
people of a household as spread around its income (`dispersion`, `spread`); `poverty` in the
result gives the at-risk-of-poverty rate by household type and by activity status next to the
rates published by Eurostat (data/base/distribution_reference.csv).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erf as _erf, ndtri as _ndtri

from ..core.dims import BASE_YEAR, CONS_CATS, REGIONS
from ..core.engine import INDEX_KEYS
from ..fiscal.revenue import calc as rcalc
from . import households as hh_mod
from dataclasses import replace

from .calc import BenefitEnv, grapa_topup, net_income, _person

DATA = Path(__file__).resolve().parents[2] / "data" / "base"
CAUSES = ("wages", "benefits", "contributions", "direct_taxes", "indirect_taxes_prices")
WORK = ("emp_private", "emp_public", "self_employed")
EPS = 0.5      # EUR/year below which a household neither gains nor loses

METHOD = ("Synthetic-household approximation: the tax-benefit rules of the model are applied to "
          "household types (2024 household statistics) repeated over the income groups of the tax "
          "returns, so the households add up to the income tax of the budget. It is not a "
          "microsimulation on survey or administrative data.")
CAVEATS = [
    "Incomes vary between income groups; inside a group they are spread around the group's income by an "
    "assumed dispersion (in the poverty rates, the Gini coefficient and S80/S20 only) and partners' incomes are "
    "combined independently; capital and property income and the recurrent property tax are spread by tenth of "
    "income on assumed shares (the amount an income survey records, not the national-accounts total): "
    "inequality and poverty indicators are indicative; read their change between baseline and scenario "
    "rather than their level. By household type the poverty rates are within about 2 points of the observed "
    "ones: somewhat too high for people living alone and single parents, too low for families with children "
    "and households of three or more adults.",
    "Weights are the 2024 household structure rescaled to the baseline head counts of the year (workers, "
    "pensioners, unemployed, long-term sick), the same in both runs: people who find or lose a job because "
    "of the scenario are not shown (see the employment results for that). In the baseline the unemployed "
    "who no longer receive a benefit and the people excluded by the time limit who claim it live on the "
    "integration income.",
    "The change in income is net of the scenario's effect on consumer prices, valued on the household's "
    "baseline basket; it is expressed in euros of the year at baseline prices.",
    "Earnings are those of all employees (part-time included) from the simulated earnings distribution; the "
    "income taxed is that of the same income group in the tax returns (untaxed pay components, part-year work).",
    "Benefits use average rules (one cohabitant rate, no head-of-household rate); the income guarantee for the "
    "elderly and the social supplement to family allowances follow simplified rules; the cash benefits of the "
    "budget that the rules do not produce (sickness and career-break benefits, allowances for persons with "
    "disabilities, family allowances of students, integration income of people living with others) are shared "
    "between the adults or children concerned on assumed keys; social tariffs, housing costs, regional tax "
    "differences, company cars and wealth are not modelled.",
    "Pensioner households receive the model's average pension income per pensioner (pensions in payment, "
    "all pensions of a person together), which reacts slowly to rules that only apply to new pensioners.",
]


def index_fn(results):
    """Index levels of a finished run, as the engine supplies them to `ParamSet.at`."""
    def index(kind: str, year: int) -> float:
        if year < BASE_YEAR:
            return 1.0
        return results.at(year).macro[INDEX_KEYS[kind]]
    return index


def param_view(pset, results, year: int):
    """The engine's resolved parameters of `year` for a finished run, with the personal income
    tax amounts on their legal indexation as in the revenue module (`revenue.legal_params`)."""
    from ..fiscal import revenue
    p = pset.at(year, index_fn(results))
    paths = (results.at(year).extra or {}).get("revenue_paths") or {}
    lag = paths.get("lag_run", paths.get("lag"))
    return revenue.legal_params(p, pset, lag)


# ---------------------------------------------------------------------------------------------
# consumption
# ---------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _hbs() -> dict:
    df = pd.read_csv(DATA / "distribution_hbs_structure.csv")
    return {int(q): dict(zip(g["category"], g["share"])) for q, g in df.groupby("quintile")}


def consumption_shares(quintile: int, p) -> dict[str, float]:
    """Cash consumption basket of an income quintile over the model's consumption categories
    (Household Budget Survey; imputed rents excluded)."""
    raw = dict(_hbs()[quintile])
    raw.pop("imputed_rent", None)
    transport = raw.pop("transport_operation", 0.0)
    fuel = float(p["macro.fuel_share_of_cp072"])
    raw["fuel"] = raw.get("fuel", 0.0) + fuel * transport
    raw["other_services"] = raw.get("other_services", 0.0) + (1.0 - fuel) * transport
    total = sum(raw.get(c, 0.0) for c in CONS_CATS)
    return {c: raw.get(c, 0.0) / total for c in CONS_CATS}


@lru_cache(maxsize=1)
def _propensity() -> dict:
    df = pd.read_csv(DATA / "distribution_propensity.csv")
    return {(int(r.year), int(r.quintile)): float(r.propensity) for r in df.itertuples()}


def cash_propensity(quintile: int, p) -> float:
    """Cash consumption as a share of disposable income for an income quintile: survey propensity
    to consume (Eurostat icw_sr_10), less imputed rents."""
    prop = _propensity()[(int(p["dist.consumption_propensity_wave"]), quintile)]
    if p["dist.propensity_includes_imputed_rent"]:
        prop *= 1.0 - float(_hbs()[quintile].get("imputed_rent", 0.0))
    return prop * float(p["dist.consumption_propensity_scale"])


@lru_cache(maxsize=1)
def reference_indicators() -> dict:
    """Observed EU-SILC indicators (yardstick for the approximation)."""
    path = DATA / "distribution_reference.csv"
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    return {r.indicator: {"value": float(r.value), "year": int(r.year), "source": f"Eurostat {r.dataset}"}
            for r in df.itertuples()}


# ---------------------------------------------------------------------------------------------
# weighted statistics on mass points
# ---------------------------------------------------------------------------------------------
def quantile_membership(y, mass, n: int) -> np.ndarray:
    """[n, len(y)] mass of each point falling in each of `n` equal groups ordered by `y` (a point
    that straddles a boundary is split)."""
    y, mass = np.asarray(y, dtype=float), np.asarray(mass, dtype=float)
    order = np.argsort(y, kind="stable")
    hi = np.cumsum(mass[order])
    lo = hi - mass[order]
    total = hi[-1]
    out = np.zeros((n, len(y)))
    for g in range(n):
        a, b = total * g / n, total * (g + 1) / n
        out[g, order] = np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0.0, None)
    return out


def gini(y, mass) -> float:
    y, mass = np.asarray(y, dtype=float), np.asarray(mass, dtype=float)
    mean = float((y * mass).sum() / mass.sum())
    if mean <= 0:
        return 0.0
    diff = np.abs(y[:, None] - y[None, :])
    return float((mass[:, None] * mass[None, :] * diff).sum() / (2.0 * mean * mass.sum() ** 2))


def s80s20(y, mass) -> float:
    m = quantile_membership(y, mass, 5)
    bottom = float((m[0] * y).sum())
    return float((m[4] * y).sum() / bottom) if bottom > 0 else float("nan")


def weighted_median(y, mass) -> float:
    y, mass = np.asarray(y, dtype=float), np.asarray(mass, dtype=float)
    order = np.argsort(y, kind="stable")
    cum = np.cumsum(mass[order])
    return float(y[order][np.searchsorted(cum, 0.5 * cum[-1])])


def share_below(y, line: float, sigma=0.0) -> np.ndarray:
    """Share of the persons of each mass point `y` with an income below `line`. With `sigma` > 0
    (a number or one value per point) the incomes of the persons of a point are spread around it
    (log-normal with that standard deviation of the logarithm, mean `y`): a point just above the
    line holds some persons below it, and the count moves smoothly when the line or the incomes
    move."""
    y = np.asarray(y, dtype=float)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), y.shape)
    sharp = sigma <= 0
    sd = np.where(sharp, 1.0, sigma)
    z = np.log(max(line, 1e-9) / np.maximum(y, 1e-9)) / sd + 0.5 * sd                # same mean as the point
    return np.where(sharp, (y < line).astype(float), 0.5 * (1.0 + _erf(z / np.sqrt(2.0))))


def poverty_rate(y, mass, share: float, sigma=0.0) -> tuple[float, float]:
    """(share of persons below `share` x median, the poverty line); `sigma`: see `share_below`
    (the median is then that of the incomes spread around the points, see `spread`)."""
    y, mass = np.asarray(y, dtype=float), np.asarray(mass, dtype=float)
    line = share * weighted_median(*spread(y, mass, sigma, 15))
    return float((mass * share_below(y, line, sigma)).sum() / mass.sum()), line


# poverty breakdowns published by EU-SILC: id -> (label, row of data/base/distribution_reference.csv)
POVERTY_HOUSEHOLD_TYPES = {
    "single": ("Single person", "poverty_rate.household.A1"),
    "single_parent": ("Single parent", "poverty_rate.household.A1_DCH"),
    "couple_no_children": ("Two adults without children", "poverty_rate.household.A2"),
    "couple_with_children": ("Two or more adults with children", "poverty_rate.household.A_GE2_DCH"),
    "three_adults_no_children": ("Three or more adults without children", "poverty_rate.household.A_GE3"),
    "aged_65_plus": ("People aged 65 and over", "poverty_rate.age.Y_GE65"),
}
POVERTY_ACTIVITY_STATUSES = {
    "employed": ("In work", "poverty_rate.activity.EMP"),
    "employee": ("Employees", "poverty_rate.activity.SAL"),
    "self_employed": ("Self-employed", "poverty_rate.activity.NSAL"),
    "unemployed": ("Unemployed", "poverty_rate.activity.UNE"),
    "retired": ("Retired", "poverty_rate.activity.RET"),
    "other_inactive": ("Other people out of work (long-term sick, at home, students)", "poverty_rate.activity.INAC_OTH"),
}


def adult_counts(h) -> dict[str, float]:
    """Number of adults of one household by income kind (`work` = the three kinds of workers
    together; a third adult counts in each kind for its share of `h.mix`)."""
    out: dict[str, float] = {}
    for a in h.adults:
        if a.kind != "third":
            out[a.kind] = out.get(a.kind, 0.0) + 1.0
    for kind, _, share in h.mix:
        out[kind] = out.get(kind, 0.0) + share
    out["work"] = sum(out.get(k, 0.0) for k in WORK)
    return out


def person_groups(h) -> dict[str, float]:
    """Number of persons of one household in each group of the published poverty breakdowns
    (`POVERTY_HOUSEHOLD_TYPES`: all members; `POVERTY_ACTIVITY_STATUSES`: adults by income kind;
    a third adult counts in each kind for the share of `h.mix`)."""
    adults, kinds = len(h.adults), [a.kind for a in h.adults]
    mix: dict[str, float] = {}
    for kind, _, share in h.mix:
        mix[kind] = mix.get(kind, 0.0) + share

    def count(kind):
        return kinds.count(kind) + mix.get(kind, 0.0)

    if h.children:
        kind = "single_parent" if adults == 1 else "couple_with_children"
    else:
        kind = {1: "single", 2: "couple_no_children"}.get(adults, "three_adults_no_children")
    out = {kind: float(h.persons)}
    # people of 65 and over: pensioners, and the partner without income of a pensioner
    elderly = adults if all(k in ("pension", "none") for k in kinds) and "pension" in kinds else count("pension")
    work = sum(count(k) for k in WORK)
    out.update(aged_65_plus=float(elderly), employed=float(work), self_employed=float(count("self_employed")),
               employee=float(work - count("self_employed")), unemployed=float(count("unemployment")),
               retired=float(count("pension")))
    out["other_inactive"] = adults - out["employed"] - out["unemployed"] - out["retired"]
    return out


def poverty_breakdown(hhs, eq_b, eq_s, line_b: float, line_s: float, ref: dict, sigma=0.0) -> dict:
    """At-risk-of-poverty rate by household type and by activity status of the adults, baseline
    and scenario (each against its own line), with the rate observed in the income survey."""
    groups = [person_groups(h) for h in hhs]
    weight = np.array([h.weight for h in hhs])
    total = float(sum(w * h.persons for w, h in zip(weight, hhs)))
    poor_b, poor_s = share_below(eq_b, line_b, sigma), share_below(eq_s, line_s, sigma)
    out = {}
    for name, spec in (("by_household_type", POVERTY_HOUSEHOLD_TYPES), ("by_activity_status", POVERTY_ACTIVITY_STATUSES)):
        rows = []
        for key, (label, row) in spec.items():
            n = weight * np.array([g.get(key, 0.0) for g in groups])
            if n.sum() <= 0:
                continue
            b, s = float((n * poor_b).sum() / n.sum()), float((n * poor_s).sum() / n.sum())
            item = {"id": key, "label": label, "persons": round(float(n.sum())), "population_share": _round(n.sum() / total, 4),
                    "baseline": _round(b, 4), "scenario": _round(s, 4), "delta": _round(s - b, 4), "observed": None,
                    "median_income": _round(weighted_median(np.asarray(eq_b), n)), "median_income_observed": None}
            if row in ref:
                item["observed"] = {**ref[row], "value": ref[row]["value"] * 0.01}
            income = row.replace("poverty_rate.", "median_equivalised_income_eur.")
            if income in ref and income != row:
                item["median_income_observed"] = ref[income]
            rows.append(item)
        out[name] = rows
    return out


# ---------------------------------------------------------------------------------------------
# household evaluation
# ---------------------------------------------------------------------------------------------
def _f(x) -> float:
    return float(np.asarray(x).reshape(-1)[0])


def evaluate(hh, p, wages: dict, env: BenefitEnv, rates: bool = True, ref_wages: dict | None = None,
             cache: dict | None = None) -> dict:
    """Incomes, taxes and (optionally) effective tax rates of one household. `cache` (a dict
    shared by the households evaluated with the same rules and incomes) keeps the cases of the
    third adult, which are the same in every household of three adults."""
    adults = hh_mod.resolve(hh, p, wages, env, ref_wages)
    keys = ("gross", "market_income", "replacement_income", "assistance", "family_allowance",
            "ssc_employee", "ssc_employer", "employer_cost", "income_tax", "child_credit", "marital_quotient", "net")
    third = None
    if hh.mix:
        # third adult (taxed on their own): the average over the cases of `hh.mix`, each
        # evaluated as a person; the rest of the household is evaluated without them
        third = dict.fromkeys(keys, 0.0)
        for share, kind, gross, ratio in hh_mod.resolve_third(hh, p, wages, env, ref_wages):
            if share > 0 and kind != "none":
                one = None if cache is None else cache.get((kind, gross, ratio))
                if one is None:
                    r = net_income(p, [(kind, gross, ratio)], 0, env)
                    one = {k: _f(r[k]) for k in keys}
                    if cache is not None:
                        cache[(kind, gross, ratio)] = one
                for k in keys:
                    third[k] += share * one[k]
        adults = [a for a in adults if a[0] != "third"]
    kind, g, ratio = adults[0]
    rest = adults[1:]
    # marginal effective rate on the main income (not meaningful for means-tested assistance):
    # the household is evaluated in the same call with the main income one step higher
    marginal = rates and kind != "assistance" and g > 0
    step = float(p["dist.marginal_step"])
    r = net_income(p, ([(kind, np.array([g, g + step]), ratio)] + rest) if marginal else adults, hh.children, env)
    out = {k: _f(r[k]) + (third[k] if third else 0.0) for k in keys}
    out["benefits"] = out["replacement_income"] + out["assistance"] + out["family_allowance"]
    if not rates:
        return out
    income = out["gross"]
    out["average_rate"] = (out["ssc_employee"] + out["income_tax"]) / income if income > 0 else None
    net = out["net"] - (third["net"] if third else 0.0)          # of the adults evaluated together
    out["marginal_rate"] = 1.0 - (float(np.asarray(r["net"]).reshape(-1)[1]) - net) / step if marginal else None
    # participation tax rate: share of the gross wage lost to tax, contributions and withdrawn
    # benefits when moving from benefit to work
    out["participation_rate"] = None
    if kind in ("emp_private", "emp_public"):
        ben = hh_mod.benefit("unemployment", "", p, env, g)
        idle = net_income(p, [("unemployment", ben, wages.get("ratio", {}).get("unemployment", 1.0))] + rest, hh.children, env)
        out["participation_rate"] = 1.0 - (net - _f(idle["net"])) / g
    elif kind in ("unemployment", "disability", "assistance"):
        w = hh_mod.wage("ref", p, wages)
        work = net_income(p, [("emp_private", w)] + rest, hh.children, env)
        out["participation_rate"] = 1.0 - (_f(work["net"]) - net) / w
    return out


PIT_LINES = ("pit_federal", "pit_regional", "pit_local")
SSC_LINES = ("ssc_employee", "special_ssc", "ssc_self_employed", "ssc_replacement_income")


def grapa_takeup(hhs, p, wages: dict, env: BenefitEnv, state) -> float:
    """Share of the income guarantee for the elderly that the pensioner households of the set
    would receive at full take-up which the budget actually pays (expenditure line of the year
    over the potential of the set), at most 1."""
    spending = float((state.expenditure or {}).get("pension_grapa", 0.0)) * 1e6
    potential = 0.0
    for h in hhs:
        if all(a.kind in ("pension", "none") for a in h.adults):
            persons = [_person(g, k, p, r) for k, g, r in hh_mod.resolve(h, p, wages, env)]
            potential += h.weight * float(np.sum(grapa_topup(p, persons)))
    return float(np.clip(spending / potential, 0.0, 1.0)) if potential > 0 else 0.0


def survey_items(hhs, e0: list[dict], scale: np.ndarray, p) -> dict[str, np.ndarray]:
    """Per household, the keys by which the items of the survey income concept that the model
    has no rule for are spread: `capital` (capital and property income before tax, EUR, a share
    `dist.capital_income_share` of the disposable income of the set, by tenth of equivalised
    income) and `property_key` (share of the households' recurrent property tax borne by the
    household). Both are fixed on the baseline, inside a tenth in proportion to household size."""
    weight = np.array([h.weight for h in hhs])
    persons = weight * np.array([h.persons for h in hhs])
    net = np.array([e["net"] for e in e0])
    member = quantile_membership(net / scale, persons, 10)          # [decile, household] persons
    frac = member / np.where(persons > 0, persons, 1.0)             # share of the household in each tenth
    size = weight * scale
    total = float(p["dist.capital_income_share"]) * float((weight * net).sum())
    out = {}
    for key, name, amount in (("capital", "dist.capital_income_by_decile", total), ("property_key", "dist.property_tax_by_decile", 1.0)):
        shares = np.asarray(p[name], dtype=float)
        shares = shares / shares.sum()
        per_unit = np.array([amount * shares[d] / max(float((frac[d] * size).sum()), 1e-9) for d in range(10)])
        out[key] = (frac * per_unit[:, None]).sum(axis=0) * scale
    return out


def add_survey_items(e: dict, capital: float, property_key: float, p, state) -> None:
    """Add capital income (net of the withholding tax of `p`) and take off the recurrent
    property tax of `state` in the evaluation `e` of a household (in place)."""
    tax = capital * rcalc.capital_income_tax_rate(p)
    prop = property_key * float(p["dist.property_tax_household_share"]) * float((state.revenue or {}).get("property_tax", 0.0)) * 1e6
    prop = min(prop, max(e["net"] + capital - tax, 0.0))          # never more than the income left
    e["capital_income"], e["capital_tax"], e["property_tax"] = capital, tax, prop
    e["gross"] += capital
    e["market_income"] += capital
    e["income_tax"] += tax + prop
    e["net"] += capital - tax - prop


def budget_benefit_keys(hhs, e0: list[dict], p, state) -> dict:
    """How the cash benefits of the budget that the household rules do not produce reach the
    households (fixed on the baseline, same in the scenario):

    `family_factor`  family allowances of all the entitled children of the model (under 18, and
                     the share `family.student_share` of the 18-24, as in the expenditure module)
                     over those the rules give the children of the set: the household statistics
                     count fewer children than there are entitled ones, and the families of the
                     set stand for them all. The rest of the budget line (maternity and
                     parental-leave benefits) is left out like the other benefits replacing wages;
    `lines`          line -> (households' share of the line per household [array], taxed or not,
                     part of the line not already paid by the rules of the set): each line of
                     `dist.other_benefit_recipients` is shared equally between the adults of the
                     set of the kinds listed (`work`, `unemployment`, `disability`, `none` ...).
    """
    weight = np.array([h.weight for h in hhs])
    spending = state.expenditure or {}
    rules = float(sum(w * e["family_allowance"] for w, e in zip(weight, e0)))
    amounts, pop = p["family.allowance_per_child"], np.asarray(state.pop, dtype=float)
    budget = sum((float(pop[:18, :, r].sum()) + float(p["family.student_share"]) * float(pop[18:25, :, r].sum())) * float(amounts[name])
                 for r, name in enumerate(REGIONS)) * float(((state.extra or {}).get("expenditure") or {}).get("benefit_index", 1.0))
    out = {"family_factor": budget / rules if rules > 0 and budget > 0 else 1.0, "lines": {}}
    counts = [adult_counts(h) for h in hhs]
    for line, spec in (p.get("dist.other_benefit_recipients") or {}).items():
        n = np.array([sum(c.get(k, 0.0) for k in spec["kinds"]) for c in counts])
        if line == "social_assistance":
            # not in the households of pensioners (their minimum is the income guarantee for the elderly)
            n = n * np.array([0.0 if c.get("pension") and not c.get("work") and c.get("pension", 0) + c.get("none", 0) == len(h.adults)
                              else 1.0 for c, h in zip(counts, hhs)])
        total = float((weight * n).sum())
        if total <= 0 or line not in spending:
            continue
        # part of the line the rules already pay to the households of the set
        # (integration income of the households on it and of the unemployed without benefit)
        inside = sum(w * e["assistance"] for w, e, c in zip(weight, e0, counts)
                     if c.get("assistance") or (c.get("unemployment") and not c.get("pension"))) if line == "social_assistance" else 0.0
        amount = float(spending[line]) * 1e6
        out["lines"][line] = (n / total, bool(spec["taxed"]), float(np.clip(1.0 - inside / amount, 0.0, 1.0)) if amount > 0 else 0.0)
    return out


def add_budget_benefits(e: dict, i: int, keys: dict, state) -> None:
    """Add to the evaluation `e` of household `i` (in place) the benefits of `budget_benefit_keys`
    at the amounts of the budget of `state`: family allowances scaled to the budget line, the
    other lines per adult; the taxed ones bear the household's average income-tax rate."""
    extra = (keys["family_factor"] - 1.0) * e["family_allowance"]
    e["family_allowance"] += extra
    rate = min(max(e["income_tax"] / e["gross"], 0.0), 1.0) if e["gross"] > 0 else 0.0
    other = tax = 0.0
    for line, (share, taxed, part) in keys["lines"].items():
        amount = share[i] * part * float((state.expenditure or {}).get(line, 0.0)) * 1e6
        other += amount
        tax += amount * rate if taxed else 0.0
    e["other_benefits"], e["other_benefits_tax"] = other, tax
    e["gross"] += other
    e["employer_cost"] += other
    e["income_tax"] += tax
    e["benefits"] += extra + other
    e["net"] += extra + other - tax


def spread(y, mass, sigma, points: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Mass points with the incomes of each one spread around it (log-normal, standard deviation
    `sigma` of the logarithm - a number or one value per point -, same mean): every point becomes
    `points` points of equal mass at the centres of equal-probability slices. `sigma` = 0 returns
    the points unchanged."""
    y, mass = np.asarray(y, dtype=float), np.asarray(mass, dtype=float)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), y.shape)
    if not np.any(sigma > 0):
        return y, mass
    f = np.exp(sigma[:, None] * _ndtri((np.arange(points) + 0.5) / points)[None, :])
    f = f / f.mean(axis=1, keepdims=True)
    return (y[:, None] * f).ravel(), np.repeat(mass / points, points)


def children_dispersion(p, child: float) -> dict[int, float]:
    """Spread of equivalised income inside a household type with children that comes from the
    number of children (the set gives every family of a type the same number): number of adults
    (1, 2, 3) -> standard deviation of the logarithm of the equivalence scale over the families
    with 1, 2 and 3 or more children of the household statistics, weighted by persons, less the
    part offset by family allowances (`dist.children_number_spread`). Only for three adults
    (one type with one child): with one and two adults the families with 1, 2 and 3 or more
    children are types of their own. `child` is the weight of a child on the scale."""
    d = p.get("dist.children_number_spread") or {}
    if not d:
        return {}
    cells, scale_p = hh_mod._cells(), p["dist.equivalence_scale"]
    out = {}
    for adults, comp in ((3, "OTH_CH"),):
        numbers = {"1": 1.0, "2": 2.0, "GE3": float(d["large_family_children"])}
        n = np.array(list(numbers.values()))
        w = np.array([cells.get(("TOTAL", k, comp), 0.0) for k in numbers]) * (adults + n)       # persons
        if w.sum() <= 0:
            continue
        logs = np.log(scale_p["first_adult"] + scale_p["other_adult"] * (adults - 1) + child * n)
        mean = float((w * logs).sum() / w.sum())
        out[adults] = (1.0 - float(d["allowance_offset"])) * float(np.sqrt((w * (logs - mean) ** 2).sum() / w.sum()))
    return out


def dispersion(hhs, e0: list[dict], p, state) -> np.ndarray:
    """Spread of the incomes of the people of each household around its income (standard
    deviation of the logarithm): that of earnings and that of benefits
    (`dist.within_group_dispersion`), weighted by the share of earnings and capital income in the
    gross income of the household (baseline), plus - for the households with children - the
    spread that comes from the number of children (`children_dispersion`)."""
    d = p.get("dist.within_group_dispersion") or {}
    market = np.array([e["market_income"] / e["gross"] if e["gross"] > 0 else 0.0 for e in e0])
    market = np.clip(market, 0.0, 1.0)
    sigma = float(d.get("benefits", 0.0)) + (float(d.get("market", 0.0)) - float(d.get("benefits", 0.0))) * market
    if not np.any(sigma > 0):
        return sigma
    kids = children_dispersion(p, child_weight(p, state))
    extra = np.array([kids.get(min(len(h.adults), 3), 0.0) if h.children else 0.0 for h in hhs])
    return np.sqrt(sigma ** 2 + extra ** 2)


def child_weight(p, state) -> float:
    """Weight of an average child on the equivalence scale: 0.3 under 14, 0.5 from 14 - the share
    of the under-18s of `state` aged 14 to 17."""
    s = p["dist.equivalence_scale"]
    pop = np.asarray(state.pop, dtype=float).reshape(state.pop.shape[0], -1).sum(axis=1)
    teens = float(pop[14:18].sum() / pop[:18].sum()) if pop[:18].sum() > 0 else 0.0
    return float(s["child"] + (s["other_adult"] - s["child"]) * teens)


def equivalence_scale(hhs, p, state) -> np.ndarray:
    """Equivalent adults of each household on the scale of the income survey: first adult 1,
    other persons of 14 and over 0.5, children under 14 0.3. The children of a household are an
    average child: the share of the under-18s of `state` aged 14 to 17 counts as 0.5. A family
    of the types with three children stands for the families with three or more: it counts
    for their average number of children (`dist.children_number_spread`: large_family_children,
    less the share of an extra child that family allowances make up, allowance_offset)."""
    s, child = p["dist.equivalence_scale"], child_weight(p, state)
    d = p.get("dist.children_number_spread") or {}
    more = max(float(d.get("large_family_children", 3.0)) - 3.0, 0.0) * (1.0 - float(d.get("allowance_offset", 0.0)))
    return np.array([s["first_adult"] + s["other_adult"] * (len(h.adults) - 1) + child * (h.children + (more if h.children >= 3 else 0.0))
                     for h in hhs])


def coverage(hhs, evaluated: list[dict], state) -> dict:
    """What the households add up to, over the budget lines of the year: personal income tax
    (three levels; the tax of the household rules and that on the benefits taken from the
    budget, without the withholding tax on capital income and the property tax) and personal
    social contributions (employees incl. the special contribution, self-employed, pensioners)."""
    rev = state.revenue or {}
    tax = sum(h.weight * (e["income_tax"] - e.get("capital_tax", 0.0) - e.get("property_tax", 0.0))
              for h, e in zip(hhs, evaluated)) / 1e6
    ssc = sum(h.weight * e["ssc_employee"] for h, e in zip(hhs, evaluated)) / 1e6
    pit, contributions = sum(rev.get(k, 0.0) for k in PIT_LINES), sum(rev.get(k, 0.0) for k in SSC_LINES)
    return {"income_tax": _round(tax / pit, 4) if pit > 0 else None,
            "social_contributions": _round(ssc / contributions, 4) if contributions > 0 else None}


def _round(x, nd=2):
    return None if x is None else round(float(x), nd) + 0.0      # + 0.0: no negative zero


def distribution(base_results, scen_results, base_pset, scen_pset, year: int) -> dict:
    """Distributional effect of a scenario in `year` (see the module docstring).

    base_pset / scen_pset: `ParamSet` without and with the scenario's measures; they are resolved
    with the indices of the corresponding run, exactly as the engine did.
    """
    sb, ss = base_results.at(year), scen_results.at(year)
    pb, ps = param_view(base_pset, base_results, year), param_view(scen_pset, scen_results, year)
    hhs = hh_mod.household_set(pb, base_results.at(BASE_YEAR), sb)
    cal = (base_results.at(BASE_YEAR).extra.get("revenue") or {}).get("pit")
    wages_b = hh_mod.income_levels(sb, pb, cal)
    wages_s = hh_mod.income_levels(ss, ps, cal, ratios=wages_b["ratio"])
    env_b, env_s = hh_mod.benefit_env(sb), hh_mod.benefit_env(ss, pop_state=sb)
    # income guarantee for the elderly: take-up from the baseline budget, same in the scenario
    takeup = grapa_takeup(hhs, pb, wages_b, env_b, sb)
    env_b, env_s = replace(env_b, grapa_takeup=takeup), replace(env_s, grapa_takeup=takeup)

    c0, c1, c2 = {}, {}, {}
    e0 = [evaluate(h, pb, wages_b, env_b, cache=c0) for h in hhs]
    e1 = [evaluate(h, pb, wages_s, env_b, rates=False, ref_wages=wages_b, cache=c1) for h in hhs]
    e2 = [evaluate(h, ps, wages_s, env_s, cache=c2) for h in hhs]

    # cash benefits of the budget that the rules do not produce (survey income concept)
    keys = budget_benefit_keys(hhs, e0, pb, sb)
    for i in range(len(hhs)):
        add_budget_benefits(e0[i], i, keys, sb)
        add_budget_benefits(e1[i], i, keys, sb)
        add_budget_benefits(e2[i], i, keys, ss)
    scale = equivalence_scale(hhs, pb, sb)
    # survey income concept: capital and property income, recurrent property tax
    items = survey_items(hhs, e0, scale, pb)
    for i in range(len(hhs)):
        add_survey_items(e0[i], items["capital"][i], items["property_key"][i], pb, sb)
        add_survey_items(e1[i], items["capital"][i], items["property_key"][i], pb, sb)
        add_survey_items(e2[i], items["capital"][i], items["property_key"][i], ps, ss)
    weight = np.array([h.weight for h in hhs])
    persons = weight * np.array([h.persons for h in hhs])
    net_b = np.array([e["net"] for e in e0])
    net_s = np.array([e["net"] for e in e2])
    eq_b = net_b / scale

    # --- consumption and indirect taxes: baseline basket of the household's income quintile ---
    quint = quantile_membership(eq_b, persons, 5).argmax(axis=0) + 1
    tau_b, tau_s = rcalc.consumption_tax_rates(pb), rcalc.consumption_tax_rates(ps)
    cons_tot = float(sum(sb.macro.get(f"cons.{c}", 0.0) for c in CONS_CATS))
    agg = {c: (sb.macro.get(f"cons.{c}", 0.0) / cons_tot if cons_tot > 0 else 1.0 / len(CONS_CATS)) for c in CONS_CATS}
    rel = {c: (1.0 + tau_s[c]) / (1.0 + tau_b[c]) - 1.0 for c in CONS_CATS}     # full pass-through
    pi_agg = sum(agg[c] * rel[c] for c in CONS_CATS)
    cpi_b, cpi_s = float(sb.macro["cpi"]), float(ss.macro["cpi"])
    cons = np.zeros(len(hhs))
    indirect_b, indirect_s, price_effect = np.zeros(len(hhs)), np.zeros(len(hhs)), np.zeros(len(hhs))
    for i in range(len(hhs)):
        shares = consumption_shares(int(quint[i]), pb)
        cons[i] = cash_propensity(int(quint[i]), pb) * net_b[i]
        pretax = {c: cons[i] * shares[c] / (1.0 + tau_b[c]) for c in CONS_CATS}
        indirect_b[i] = sum(pretax[c] * tau_b[c] for c in CONS_CATS)
        indirect_s[i] = sum(pretax[c] * tau_s[c] for c in CONS_CATS)
        # price change of the household's basket: the model's CPI change, tilted by how much more
        # (or less) than the average basket this one is hit by the indirect-tax changes
        pi_h = sum(shares[c] * rel[c] for c in CONS_CATS)
        price_effect[i] = cons[i] * ((cpi_s / cpi_b) * (1.0 + pi_h) / (1.0 + pi_agg) - 1.0)

    real_s = net_s - price_effect                  # scenario income at baseline prices
    delta = real_s - net_b
    eq_s = real_s / scale

    rows, causes = [], {k: np.zeros(len(hhs)) for k in CAUSES}
    for i, h in enumerate(hhs):
        a, m, z = e0[i], e1[i], e2[i]
        cause = {
            "wages": m["net"] - a["net"],
            "benefits": z["benefits"] - m["benefits"],
            "contributions": -(z["ssc_employee"] - m["ssc_employee"]),
            "direct_taxes": -(z["income_tax"] - m["income_tax"]),
            "indirect_taxes_prices": -price_effect[i],
        }
        # market income that changes with the rules (minimum wage, student earnings) counts as wages
        cause["wages"] += z["market_income"] - m["market_income"]
        for k in CAUSES:
            causes[k][i] = cause[k]

        def side(e, indirect, real):
            return {
                "gross": _round(e["gross"]), "market_income": _round(e["market_income"]),
                "benefits": _round(e["benefits"]), "family_allowance": _round(e["family_allowance"]),
                "ssc_employee": _round(e["ssc_employee"]), "ssc_employer": _round(e["ssc_employer"]),
                "employer_cost": _round(e["employer_cost"]), "income_tax": _round(e["income_tax"]),
                "other_benefits": _round(e["other_benefits"]), "other_benefits_tax": _round(e["other_benefits_tax"]),
                "capital_income": _round(e["capital_income"]), "capital_tax": _round(e["capital_tax"]),
                "property_tax": _round(e["property_tax"]),
                "net": _round(e["net"]), "indirect_taxes": _round(indirect),
                "net_after_indirect": _round(e["net"] - indirect),
                "net_real": _round(real), "net_2024_prices": _round(real / cpi_b),
                "equivalised_net": _round(real / scale[i]),
                "rates": {"average": _round(e["average_rate"], 5), "marginal": _round(e["marginal_rate"], 5),
                          "participation": _round(e["participation_rate"], 5)},
            }

        rows.append({
            "id": h.id, "label": h.label, "group": h.group, "representative": h.representative,
            "type": h.base, "adults": len(h.adults), "children": h.children, "persons": h.persons,
            "weight": round(h.weight), "quintile": int(quint[i]), "consumption": _round(cons[i]),
            "baseline": side(a, indirect_b[i], net_b[i]),
            "scenario": side(z, indirect_s[i], real_s[i]),
            "delta": {
                "eur": _round(delta[i]),
                "pct": _round(100.0 * delta[i] / net_b[i], 4) if net_b[i] > 0 else None,
                "eur_2024_prices": _round(delta[i] / cpi_b),
                "nominal_eur": _round(net_s[i] - net_b[i]),
                "by_cause": {k: _round(v) for k, v in cause.items()},
            },
            "rates": {k: _round(z[f"{k}_rate"], 5) for k in ("average", "marginal", "participation")},
        })

    # --- deciles of persons by baseline equivalised disposable income ---
    member = quantile_membership(eq_b, persons, 10)                 # persons
    n_persons = np.array([h.persons for h in hhs], dtype=float)
    gain, loss = delta > EPS, delta < -EPS
    deciles = []
    ref = reference_indicators()
    total_eq_b, total_eq_s = float((persons * eq_b).sum()), float((persons * eq_s).sum())
    member_s = quantile_membership(eq_s, persons, 10)
    for d in range(10):
        m = member[d]
        hh_mass = m / n_persons
        tot = hh_mass.sum()
        base_inc = float((hh_mass * net_b).sum() / tot)
        d_eur = float((hh_mass * delta).sum() / tot)
        deciles.append({
            "decile": d + 1,
            "baseline_income": _round(base_inc),
            "baseline_equivalised_income": _round(float((m * eq_b).sum() / m.sum())),
            "delta_eur": _round(d_eur),
            "delta_pct": _round(100.0 * d_eur / base_inc, 4),
            "by_cause": {k: _round(float((hh_mass * causes[k]).sum() / tot)) for k in CAUSES},
            "gainers_share": _round(float(m[gain].sum() / m.sum()), 4),
            "losers_share": _round(float(m[loss].sum() / m.sum()), 4),
            "households": round(float(tot)), "persons": round(float(m.sum())),
            # share of this tenth in total equivalised income (EU-SILC: ilc_di01)
            "income_share": _round(float((m * eq_b).sum() / total_eq_b), 4),
            "income_share_scenario": _round(float((member_s[d] * eq_s).sum() / total_eq_s), 4),
            "income_share_observed": ({**ref[f"decile_share_{d + 1}"], "value": ref[f"decile_share_{d + 1}"]["value"] * 0.01}
                                      if f"decile_share_{d + 1}" in ref else None),
        })

    share = float(pb["dist.poverty_line_share"])
    sigma = dispersion(hhs, e0, pb, sb)
    pov_b, line_b = poverty_rate(eq_b, persons, share, sigma)
    pov_s, line_s = poverty_rate(eq_s, persons, share, sigma)
    pov_fixed = float((persons * share_below(eq_s, line_b, sigma)).sum() / persons.sum())
    median_b = line_b / share
    lines = {}                                       # rates at other lines (share of the median), with the observed ones
    for pct in (40, 50, 70):
        row = ref.get(f"poverty_rate_line{pct}.age.TOTAL")
        lines[str(pct)] = {"baseline": _round(float((persons * share_below(eq_b, 0.01 * pct * median_b, sigma)).sum() / persons.sum()), 4),
                           "observed": ({**row, "value": row["value"] * 0.01} if row else None)}
    g_b, g_s = gini(*spread(eq_b, persons, sigma)), gini(*spread(eq_s, persons, sigma))
    r_b, r_s = s80s20(*spread(eq_b, persons, sigma)), s80s20(*spread(eq_s, persons, sigma))

    def pair(b, s, nd=4, key=None, factor=1.0):
        out = {"baseline": _round(b, nd), "scenario": _round(s, nd), "delta": _round(s - b, nd)}
        if key in ref:
            out["observed"] = {**ref[key], "value": ref[key]["value"] * factor}
        return out

    total_delta = float((weight * delta).sum())
    # per household type: number of households and mean change over all its income groups,
    # repeated on every row of the type (the short list shows one row per type)
    by_type: dict[str, list[float]] = {}
    for i, h in enumerate(hhs):
        acc = by_type.setdefault(h.base, [0.0, 0.0, 0.0])
        acc[0] += weight[i]
        acc[1] += weight[i] * delta[i]
        acc[2] += weight[i] * net_b[i]
    for row, h in zip(rows, hhs):
        n, d, y = by_type[h.base]
        row["type_weight"] = round(n)
        row["type_delta_eur"] = _round(d / n) if n > 0 else None
        row["type_delta_pct"] = _round(100.0 * d / y, 4) if y > 0 else None
    return {
        "year": year,
        "meta": {
            "method": METHOD,
            "caveats": CAVEATS,
            "income_concept": "Disposable income per household, EUR per year, in the concept of the income survey "
                              "(EU-SILC): gross incomes incl. capital and property income and the other cash benefits "
                              "of the budget + family allowances - "
                              "employee social contributions - personal income tax, withholding tax on capital "
                              "income and recurrent property tax. 'delta' is the change in disposable income "
                              "less the extra cost of the baseline basket at scenario prices.",
            "prices": {"cpi_baseline": _round(cpi_b, 6), "cpi_scenario": _round(cpi_s, 6),
                       "note": "net_2024_prices = amounts of the year divided by the baseline consumer price index"},
            "causes": {
                "wages": "wage levels and indexation from the macro simulation, net of tax under baseline rules",
                "benefits": "replacement incomes, assistance and family allowances (rules and indexation)",
                "contributions": "employee social contributions",
                "direct_taxes": "personal income tax incl. local surcharges and child credit, withholding tax on "
                                "capital income, recurrent property tax",
                "indirect_taxes_prices": "VAT, excise and the resulting consumer prices on the baseline basket",
            },
            "wage_levels": {"baseline": {k: _round(hh_mod.wage(k, pb, wages_b)) for k in hh_mod.WAGE_LEVELS},
                            "scenario": {k: _round(hh_mod.wage(k, ps, wages_s)) for k in hh_mod.WAGE_LEVELS}},
            "households_total": round(float(weight.sum())), "persons_total": round(float(persons.sum())),
            "weights_year": BASE_YEAR,
            "households_shown": sum(1 for h in hhs if h.representative),
            "coverage": coverage(hhs, e0, sb),
        },
        "poverty": {**poverty_breakdown(hhs, eq_b, eq_s, line_b, line_s, ref, sigma), "other_lines": lines},
        "households": [r for r in rows if r["representative"]],
        "households_all": rows,
        "deciles": deciles,
        "summary": {
            "delta_eur_mean": _round(total_delta / weight.sum()),
            "delta_pct": _round(100.0 * total_delta / float((weight * net_b).sum()), 4),
            "delta_total_meur": _round(total_delta / 1e6, 1),
            "gainers_share": _round(float(persons[gain].sum() / persons.sum()), 4),
            "losers_share": _round(float(persons[loss].sum() / persons.sum()), 4),
        },
        "indicators": {
            "gini": pair(g_b, g_s, key="gini", factor=0.01),
            "s80s20": pair(r_b, r_s, key="s80s20"),
            "poverty_rate": {**pair(pov_b, pov_s, key="poverty_rate", factor=0.01),
                             "line_baseline": _round(line_b), "line_scenario": _round(line_s),
                             "scenario_fixed_line": _round(pov_fixed, 4)},
        },
    }


# ---------------------------------------------------------------------------------------------
# payslip explorer over a grid of gross incomes
# ---------------------------------------------------------------------------------------------
PAYSLIP_KEYS = ("employer_cost", "ssc_employer", "gross", "ssc_employee", "taxable_income", "income_tax",
                "family_allowance", "net", "net_earner_only", "marginal_rate", "average_rate")


def payslip_grid(base_results, scen_results, base_pset, scen_pset, year: int, kind: str, household,
                 gross_min: float, gross_max: float, points: int) -> dict:
    """Payslip of a household over a grid of gross incomes, under baseline and scenario rules of
    `year` (each with its own indices; the grid is in euros of that year)."""
    from .calc import normalise_household, payslip

    sb, ss = base_results.at(year), scen_results.at(year)
    pb, ps = param_view(base_pset, base_results, year), param_view(scen_pset, scen_results, year)
    grid = np.linspace(float(gross_min), float(gross_max), int(points))
    out = {"year": year, "kind": kind, "grid": [round(float(g), 2) for g in grid]}
    for name, p, state in (("baseline", pb, sb), ("scenario", ps, ss)):
        wages = hh_mod.wage_levels(state)
        env = hh_mod.benefit_env(state, pop_state=sb)
        r = payslip(p, grid, kind, household, env=env, median_wage=wages["median"])
        out.setdefault("household", normalise_household(household, hh_mod.wage_levels(sb)["median"]))
        rows = []
        for i in range(len(grid)):
            rows.append({
                "gross": _round(r["gross"][i]), "employer_cost": _round(r["employer_cost"][i]),
                "ssc_employer": _round(r["ssc_employer"][i]), "ssc_employee": _round(r["ssc_employee"][i]),
                "taxable_income": _round(r["taxable_income"][i]),
                "tax": {"federal": _round(r["tax_federal"][i]), "regional": _round(r["tax_regional"][i]),
                        "local": _round(r["tax_local"][i]), "child_credit": _round(r["child_credit"][i]),
                        "total": _round(r["income_tax"][i])},
                "allowances": {"tax_free_allowance": _round(r["allowance"][i]),
                               "marital_quotient": _round(r["marital_quotient"][i]),
                               "family_allowance": _round(r["family_allowance"][i])},
                "household_gross": _round(r["household_gross"][i]),
                "net": _round(r["net"][i]), "net_earner_only": _round(r["net_earner_only"][i]),
                "marginal_rate": _round(r["marginal_rate"][i], 5), "average_rate": _round(r["average_rate"][i], 5),
            })
        out[name] = {
            "net": [x["net"] for x in rows],
            "marginal": [x["marginal_rate"] for x in rows],
            "average": [x["average_rate"] for x in rows],
            "employer_cost": [x["employer_cost"] for x in rows],
            "breakdown_at": rows,
            "reference_wages": {k: _round(hh_mod.wage(k, p, wages)) for k in hh_mod.WAGE_LEVELS},
        }
    return out
