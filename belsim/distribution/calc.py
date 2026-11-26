"""Net income of a household: the household tax calculator of `belsim.fiscal.revenue.calc`
(`tax_household`: marital quotient, supplements to the tax-free allowance for children and
single parents, transfer of the unused allowance between partners, refundable child credit,
regional and municipal surcharges - the same code that produces the aggregate personal income
tax), plus what only the household analysis needs:
  - family allowances (regional amounts, `family.allowance_per_child`), untaxed, with the
    income-tested social supplement (`dist.family_social_supplement`): families under the income
    ceiling receive it, the others the average less what the supplement costs, so the average
    over all children stays that of the budget;
  - the income guarantee for the elderly (GRAPA/IGO, `dist.grapa_amounts`) for households of
    pensioners, at the take-up rate handed over in `BenefitEnv.grapa_takeup`;
  - student-job earnings (solidarity contribution) and untaxed assistance benefits.

Everything is vectorised over numpy arrays of gross income, pure, and reads only the parameter
view `p`. Amounts are EUR per year.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.dims import REGIONS
from ..fiscal.revenue import calc as rcalc

WORK_KINDS = ("emp_private", "emp_public", "self_employed")
REPLACEMENT_KINDS = rcalc.REPLACEMENT_KINDS
EXTRA_KINDS = ("student", "assistance", "none")
ALL_KINDS = rcalc.KINDS + EXTRA_KINDS


@dataclass
class BenefitEnv:
    """State-dependent levels the household calculator needs besides the parameters."""
    welfare_index: float = 1.0            # real welfare adjustments of ceilings and minima
    benefit_index: float = 1.0            # indexation of benefits in payment relative to prices
    pension_avg: dict = field(default_factory=dict)   # scheme -> average pension in payment
    child_region_shares: dict = field(default_factory=lambda: {r: 1.0 / len(REGIONS) for r in REGIONS})
    pension_income_ratio: float = 1.0     # pension income per pensioner / average pension per scheme
    grapa_takeup: float = 0.0             # share of the income guarantee for the elderly actually paid


def level_shares(p, region: str | None = None) -> tuple[float, float, float]:
    """(federal, regional, local) personal income tax per euro of state tax, for a region or the
    national average (`revenue.calc.level_shares`)."""
    return rcalc.level_shares(p, region)


def child_supplement(children: int, p) -> float:
    """Supplement to the tax-free allowance for `children` dependent children
    (`revenue.calc.child_allowance`)."""
    return float(rcalc.child_allowance(int(children), p))


def family_allowance_per_child(p, env: BenefitEnv | None = None, region: str | None = None) -> float:
    """Average family allowance per child: one region, or the average over regions weighted by
    where the children live."""
    env = env or BenefitEnv()
    amounts = p["family.allowance_per_child"]
    if region in amounts:
        amount = float(amounts[region])
    else:
        amount = float(sum(env.child_region_shares[r] * amounts[r] for r in REGIONS))
    return amount * env.benefit_index


def family_allowance(children: int, household_gross, p, env: BenefitEnv | None = None, region: str | None = None):
    """Family allowances of a household with `children` children and gross income
    `household_gross` (array): the average amount per child, less the average cost of the social
    supplement, plus the supplement under the income ceiling."""
    env = env or BenefitEnv()
    avg = family_allowance_per_child(p, env, region)
    sup = p.get("dist.family_social_supplement") if hasattr(p, "get") else None
    if not sup or children <= 0:
        return children * avg + 0.0 * np.asarray(household_gross, dtype=float)
    amount = float(sup["amount"]) * env.benefit_index
    share = float(p.get("dist.family_supplement_child_share", 0.0) or 0.0)
    entitled = np.asarray(household_gross, dtype=float) < float(sup["ceiling"])
    return children * (avg - share * amount + amount * entitled)


def grapa_topup(p, persons: list[dict]):
    """Income guarantee for the elderly (GRAPA/IGO) of a household made only of pensioners, at
    full take-up: the guaranteed amount (increased amount for a person living alone, basic
    amount per person otherwise) less the counted share of the pensions. Zero for any other
    household."""
    kinds = [x["kind"] for x in persons]
    amounts = p.get("dist.grapa_amounts") if hasattr(p, "get") else None
    if not amounts or "pension" not in kinds or any(k not in ("pension", "none") for k in kinds):
        return 0.0
    guarantee = float(amounts["increased"]) if len(persons) == 1 else float(amounts["basic"]) * len(persons)
    pensions = sum(x["gross"] for x in persons if x["kind"] == "pension")
    return np.maximum(guarantee - float(p["dist.grapa_pension_counted"]) * pensions, 0.0)


def _person(gross, kind: str, p, ratio: float = 1.0) -> dict:
    """One adult before the household rules, in the shape of `revenue.calc.tax_base`: that
    function for the kinds it knows, simple rules for student jobs and untaxed assistance.
    `ratio` scales the net taxable income (income groups of the household set: taxable income
    of the group in the fiscal statistics over what the payslip rules give); the contributions
    of a self-employed person are then those of the rules on the declared income."""
    if kind in rcalc.KINDS:
        out = rcalc.tax_base(gross, p, kind)
        if ratio != 1.0:
            out["taxable"] = out["taxable"] * ratio
            if kind in ("emp_private", "emp_public"):
                out["special_ssc"] = rcalc.special_ssc(out["taxable"], p)
            elif kind == "self_employed":
                # contributions are levied on the declared income, not on the income of the
                # national accounts: the income before contributions that leaves the taxable
                # income of the tax returns (declared = taxable + contributions on declared)
                declared = out["gross"] * ratio
                for _ in range(4):
                    declared = out["taxable"] + rcalc.self_employed_ssc(declared, p)
                out["ssc_own"] = np.minimum(rcalc.self_employed_ssc(declared, p), out["ssc_own"])
        return out
    g = np.atleast_1d(np.asarray(gross, dtype=float))
    zero = np.zeros_like(g)
    if kind == "none":
        return rcalc.tax_base(zero, p, "none")
    out = rcalc.tax_base(zero, p, "none")            # basic allowance, no reductions
    out["kind"] = kind
    if kind == "student":
        rate = p["dist.student_solidarity_rate"]
        ssc = rate["employee"] * g
        income = g - ssc
        out.update(gross=g, ssc_own=ssc, ssc_employer=rate["employer"] * g,
                   taxable=income - np.minimum(p["pit.expense_deduction_rate"] * income, p["pit.expense_deduction_max"]))
    elif kind == "assistance":
        out.update(gross=g)
    else:
        raise ValueError(f"unknown income kind '{kind}' (expected one of {ALL_KINDS})")
    return out


def net_income(p, adults, children: int = 0, env: BenefitEnv | None = None, region: str | None = None) -> dict:
    """Disposable income of a household.

    adults: list of (kind, gross) or (kind, gross, taxable ratio) - gross is a float or an
    array (arrays must broadcast); the ratio (default 1) scales the adult's net taxable income.
    The first two adults are partners (married or legally cohabiting, taxed jointly); further
    adults are taxed on their own. `children` are dependent children of the first adult(s).
    `region` (FLA, WAL, BRU) selects that region's surcharges and family allowance; without it
    the national averages apply.

    Returns arrays: gross (all incomes before tax), market_income, replacement_income,
    assistance, family_allowance, ssc_employee, ssc_employer, employer_cost, taxable_income,
    pit_federal, pit_regional, pit_local, child_credit, income_tax (= the three levels less the
    refundable credit), marital_quotient (income attributed to the partner), allowance_children
    (supplements to the tax-free allowance), net (disposable income), plus `adults`: the
    per-adult detail.
    """
    if not adults:
        raise ValueError("a household needs at least one adult")
    persons = [_person(a[1], a[0], p, *a[2:3]) for a in adults]
    children = int(children)
    r = rcalc.tax_household(p, persons, children, region=region if region in REGIONS else None)
    shape = np.broadcast_shapes(*[x["gross"].shape for x in persons])
    zero = np.zeros(shape)

    def total(kinds):
        return sum((x["gross"] for x in persons if x["kind"] in kinds), zero)

    grapa = zero + (grapa_topup(p, persons) * env.grapa_takeup if env is not None and env.grapa_takeup > 0 else 0.0)
    gross = zero + r["gross"] + grapa
    fam = zero + family_allowance(children, gross, p, env, region)
    ssc_ee, ssc_er, income_tax = zero + r["ssc_employee"], zero + r["ssc_employer"], zero + r["income_tax"]
    return {
        "gross": gross,
        "market_income": total(WORK_KINDS + ("student",)),
        "replacement_income": total(REPLACEMENT_KINDS),
        "assistance": total(("assistance",)) + grapa,
        "family_allowance": fam,
        "ssc_employee": ssc_ee, "ssc_employer": ssc_er, "employer_cost": gross + ssc_er,
        "taxable_income": zero + r["taxable_income"],
        "pit_federal": zero + r["pit_federal"], "pit_regional": zero + r["pit_regional"],
        "pit_local": zero + r["pit_local"], "child_credit": zero + r["child_credit"],
        "income_tax": income_tax,
        "marital_quotient": zero + r["marital_quotient"],
        "allowance_children": zero + r["allowance_children"],
        "net": gross - ssc_ee - income_tax + fam,
        "adults": [{"kind": a["kind"], "gross": zero + a["gross"], "ssc_employee": zero + a["ssc_employee"],
                    "ssc_employer": zero + a["ssc_employer"], "taxable_income": zero + a["taxable_income"],
                    "income_tax": zero + a["income_tax"]} for a in r["adults"]],
    }


# ---------------------------------------------------------------------------------------------
# payslip explorer
# ---------------------------------------------------------------------------------------------
HOUSEHOLD_PRESETS = {
    "single": {"label": "Single, no children", "adults": 1, "children": 0},
    "single_parent_1": {"label": "Single parent, 1 child", "adults": 1, "children": 1},
    "single_parent_2": {"label": "Single parent, 2 children", "adults": 1, "children": 2},
    "couple_one_earner": {"label": "Couple, one income, no children", "adults": 2, "children": 0},
    "couple_one_earner_2": {"label": "Couple, one income, 2 children", "adults": 2, "children": 2},
    "couple_two_earners": {"label": "Couple, partner at the median wage, no children", "adults": 2,
                           "children": 0, "partner_kind": "emp_private", "partner_gross": "median"},
    "couple_two_earners_2": {"label": "Couple, partner at the median wage, 2 children", "adults": 2,
                             "children": 2, "partner_kind": "emp_private", "partner_gross": "median"},
}


def normalise_household(household, median_wage: float | None = None) -> dict:
    """Preset id or {adults, children, partner_kind, partner_gross, region} -> full description."""
    if household is None:
        household = "single"
    if isinstance(household, str):
        if household not in HOUSEHOLD_PRESETS:
            raise ValueError(f"unknown household '{household}' (expected one of {list(HOUSEHOLD_PRESETS)})")
        household = HOUSEHOLD_PRESETS[household]
    if not isinstance(household, dict):
        raise ValueError("household must be a preset name or an object")
    base = {}
    if isinstance(household.get("preset"), str):
        if household["preset"] not in HOUSEHOLD_PRESETS:
            raise ValueError(f"unknown household preset '{household['preset']}'")
        base = HOUSEHOLD_PRESETS[household["preset"]]
    h = {**base, **{k: v for k, v in household.items() if v is not None}}
    try:
        adults, children = int(h.get("adults", 1)), int(h.get("children", 0))
    except (TypeError, ValueError):
        raise ValueError("household.adults and household.children must be numbers") from None
    if adults not in (1, 2):
        raise ValueError("household.adults must be 1 or 2")
    if not 0 <= children <= 12:
        raise ValueError("household.children must be between 0 and 12")
    partner_kind = h.get("partner_kind") or "none"
    if adults == 1:
        partner_kind = "none"
    if partner_kind not in rcalc.KINDS + ("none",):
        raise ValueError(f"unknown partner_kind '{partner_kind}'")
    pg = h.get("partner_gross", 0.0)
    if pg == "median":
        pg = median_wage if median_wage is not None else 0.0
    try:
        pg = float(pg or 0.0)
    except (TypeError, ValueError):
        raise ValueError("household.partner_gross must be a number") from None
    if pg < 0:
        raise ValueError("household.partner_gross cannot be negative")
    if partner_kind == "none":
        pg = 0.0
    region = h.get("region") or "average"
    if region != "average" and region not in REGIONS:
        raise ValueError(f"unknown region '{region}' (expected average or one of {REGIONS})")
    return {"adults": adults, "children": children, "partner_kind": partner_kind, "partner_gross": pg,
            "region": region}


def payslip(pset_view, gross, kind: str = "emp_private", household=None, env: BenefitEnv | None = None,
            median_wage: float | None = None) -> dict:
    """Full payslip of a household whose first adult earns `gross` (float or array) of `kind`.

    household: a preset name of HOUSEHOLD_PRESETS or {adults, children, partner_kind,
    partner_gross, region}. Returns arrays over `gross`: employer_cost, ssc_employer, gross,
    ssc_employee, taxable_income, tax_federal, tax_regional, tax_local, child_credit, income_tax,
    family_allowance, marital_quotient, allowance (tax-free allowance of the earner incl. child
    supplements), net (disposable income of the household), net_earner_only (without the
    partner's own net income), marginal_rate (share of an extra euro of gross lost to tax and
    contributions) and average_rate (tax and employee contributions over gross income).
    """
    p = pset_view
    if kind not in rcalc.KINDS:
        raise ValueError(f"unknown kind '{kind}' (expected one of {rcalc.KINDS})")
    h = normalise_household(household, median_wage)
    g = np.atleast_1d(np.asarray(gross, dtype=float))
    region = None if h["region"] == "average" else h["region"]

    def run(x):
        adults = [(kind, x)]
        if h["adults"] == 2:
            adults.append((h["partner_kind"], h["partner_gross"]))
        return net_income(p, adults, h["children"], env, region)

    r = run(g)
    step = float(p["dist.marginal_step"])
    up = run(g + step)
    marginal = 1.0 - (up["net"] - r["net"]) / step
    gross_all = r["gross"]
    average = np.where(gross_all > 0, (r["ssc_employee"] + r["income_tax"]) / np.where(gross_all > 0, gross_all, 1.0), 0.0)
    basic = float(p["pit.basic_allowance"])
    if kind in REPLACEMENT_KINDS:
        basic = max(basic - p["pit.allowance_neutralised_replacement"], 0.0)
    partner_net = 0.0
    if h["adults"] == 2 and h["partner_gross"] > 0:
        partner_net = float(net_income(p, [(h["partner_kind"], h["partner_gross"])])["net"][0])
    return {
        "household": h,
        "gross": g, "employer_cost": g + r["adults"][0]["ssc_employer"],
        "ssc_employer": r["adults"][0]["ssc_employer"],
        "ssc_employee": r["ssc_employee"], "taxable_income": r["taxable_income"],
        "tax_federal": r["pit_federal"], "tax_regional": r["pit_regional"], "tax_local": r["pit_local"],
        "child_credit": r["child_credit"], "income_tax": r["income_tax"],
        "family_allowance": r["family_allowance"], "marital_quotient": r["marital_quotient"],
        "allowance": basic + r["allowance_children"],
        "household_gross": gross_all, "net": r["net"], "net_earner_only": r["net"] - partner_net,
        "marginal_rate": marginal, "average_rate": average,
    }
