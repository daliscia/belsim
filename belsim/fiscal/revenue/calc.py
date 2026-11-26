"""Pure micro-calculators of the Belgian tax and contribution system.

Vectorised over numpy arrays, no state, no data access: everything comes from the parameter
view `p` (params/revenue.yaml). The aggregate revenue lines in `belsim.fiscal.revenue` and the
household analysis in `belsim.distribution` call these same functions, so a reform changes the
payslip, the household and the budget consistently.

Personal income tax, in the order of the Income Tax Code:
  `tax_base`        one person: contributions, net taxable income, own tax-free allowance and
                    federal tax reductions (work bonus, replacement income);
  `tax_household`   a tax unit: marital quotient (art. 87-88), supplements to the tax-free
                    allowance for dependent children and single parents (art. 132-133), transfer
                    of the unused allowance between partners and refundable child credit
                    (art. 134), then the split into federal tax, regional surcharges and
                    municipal surcharges;
  `tax_on_income`   the payslip contract: one person without dependants by default, any tax unit
                    with `household=`.

Amounts are EUR per year. Regional parameters are dicts keyed by region (a scalar is accepted);
without a region the national average is used (weights `pit.region_tax_share`).
"""
from __future__ import annotations

import numpy as np

from ...core.dims import CONS_CATS, REGIONS

KINDS = ("emp_private", "emp_public", "self_employed", "pension", "unemployment", "disability")
REPLACEMENT_KINDS = ("pension", "unemployment", "disability")
EMPLOYEE_KINDS = ("emp_private", "emp_public")
VAT_RATE_KEYS = ("vat.rate_standard", "vat.rate_intermediate", "vat.rate_reduced")
# excise-type lines levied on each consumption category: (line, duty parameter, pre-tax price parameter)
EXCISE_COMPONENTS = {
    "fuel": [("excise_fuel", "excise.fuel_duty", "excise.fuel_pretax_price")],
    "tobacco": [("excise_tobacco", "excise.tobacco_duty", "excise.tobacco_pretax_price")],
    "alcohol": [("excise_alcohol", "excise.alcohol_duty", "excise.alcohol_pretax_price")],
    "energy_home": [("excise_energy", "excise.energy_duty", "excise.energy_pretax_price"),
                    ("energy_green_levies", "excise.energy_green_levy", "excise.energy_pretax_price")],
    "food": [("excise_beverages_packaging", "excise.beverage_packaging_rate", None)],
}


# ---------------------------------------------------------------------------------------------
# regional parameters
# ---------------------------------------------------------------------------------------------
def by_region(value) -> np.ndarray:
    """A regional parameter ({FLA, WAL, BRU} dict, or a scalar applying everywhere) as an array
    in the order of `REGIONS`."""
    if isinstance(value, dict):
        return np.array([float(value[r]) for r in REGIONS])
    return np.full(len(REGIONS), float(value))


def region_weights(p) -> np.ndarray:
    """Shares of the regions in the state tax (used for national-average surcharges)."""
    w = by_region(p["pit.region_tax_share"])
    return w / w.sum()


def surcharge_rates(p, region: str | None = None) -> tuple[float, float]:
    """(regional surcharge on the reduced state tax, municipal surcharge on federal + regional
    tax) of one region, or the national average (same total tax as the regional rates applied
    to the regions' shares of the state tax)."""
    s, m = by_region(p["pit.regional_surcharge"]), by_region(p["pit.municipal_surcharge"])
    if region is not None:
        i = REGIONS.index(region)
        return float(s[i]), float(m[i])
    if s.min() == s.max() and m.min() == m.max():
        return float(s[0]), float(m[0])
    w = region_weights(p)
    s_avg = float(w @ s)
    return s_avg, float(w @ ((1.0 + s) * m)) / (1.0 + s_avg)


def level_shares(p, region: str | None = None) -> tuple[float, float, float]:
    """(federal, regional, municipal) personal income tax per euro of state tax: a share
    `pit.autonomy_factor` of the state tax is handed to the regions, which levy their surcharge
    on the remaining ("reduced") state tax; municipalities add theirs on the total."""
    s, m = surcharge_rates(p, region)
    fed = 1.0 - p["pit.autonomy_factor"]
    reg = fed * s
    return fed, reg, (fed + reg) * m


# ---------------------------------------------------------------------------------------------
# personal income tax building blocks
# ---------------------------------------------------------------------------------------------
def progressive_tax(income, thresholds, rates) -> np.ndarray:
    """Tax of a progressive schedule: rates[i] applies between thresholds[i-1] and thresholds[i]."""
    x = np.maximum(np.asarray(income, dtype=float), 0.0)
    tax = rates[0] * x
    for t, r_lo, r_hi in zip(thresholds, rates[:-1], rates[1:]):
        tax = tax + (r_hi - r_lo) * np.maximum(x - t, 0.0)
    return tax


def allowance_scale(p) -> tuple:
    """(thresholds, rates) of the schedule that values the tax-free allowance: the special
    scale of art. 134 while `pit.allowance_special_scale` is on - the rates of the ordinary
    brackets with a band at `pit.allowance_second_band_rate` (30 %) inserted above the first
    limit - and the ordinary schedule otherwise (from income 2029)."""
    thr, rates = p["pit.bracket_thresholds"], p["pit.bracket_rates"]
    if not p["pit.allowance_special_scale"]:
        return thr, rates
    return p["pit.allowance_scale_thresholds"], (rates[0], p["pit.allowance_second_band_rate"], rates[1], rates[2])


def state_tax(taxable, allowance, reduction, p, base_factor=None) -> np.ndarray:
    """State tax (belasting Staat) = schedule(taxable) - tax on the tax-free allowance - federal
    reductions, not below zero. The allowance is relieved from the bottom of its own scale
    (`allowance_scale`), so supplements stacked on the basic allowance are worth more than the
    bottom rate. `base_factor` (aggregate tax base only, `pit.bracket_base_factor`): multiplier
    on the income taxed in each bracket of the schedule."""
    thr, rates = p["pit.bracket_thresholds"], p["pit.bracket_rates"]
    a_thr, a_rates = allowance_scale(p)
    if base_factor is not None:
        rates = [r * k for r, k in zip(rates, base_factor)]
    taxable = np.maximum(taxable, 0.0)
    base = progressive_tax(taxable, thr, rates) - progressive_tax(np.minimum(taxable, allowance), a_thr, a_rates)
    return np.maximum(base - reduction, 0.0)


def child_allowance(children, p) -> np.ndarray:
    """Supplement to the tax-free allowance for `children` dependent children (art. 132);
    vectorised over the number of children."""
    n = np.asarray(children)
    table = np.concatenate(([0.0], np.asarray(p["pit.child_allowance_supplement"], dtype=float)))
    k = np.clip(n, 0, len(table) - 1).astype(int)
    return table[k] + np.maximum(n - (len(table) - 1), 0) * p["pit.child_allowance_extra_per_child"]


def marital_quotient(taxable_a, taxable_b, p, maximum="work") -> np.ndarray:
    """Income attributed to the partner with the lower income (art. 87-88): up to
    `pit.marital_quotient_rate` of the joint income less that partner's own income, capped at
    `maximum`: an amount, or the key of `pit.marital_quotient_max` ('work', or 'pension' for
    couples of pensioners)."""
    if isinstance(maximum, str):
        maximum = float(p["pit.marital_quotient_max"][maximum])
    low = np.minimum(taxable_a, taxable_b)
    return np.clip(p["pit.marital_quotient_rate"] * (taxable_a + taxable_b) - low, 0.0, maximum)


def work_bonus_parts(gross, p) -> tuple[np.ndarray, np.ndarray]:
    """Annual social work bonus (werkbonus / bonus a l'emploi), parts A and B.

    Legal formula (NSSO, since April 2024), on the monthly reference wage S:
      part B (very low wages): maximum up to S_b, falling linearly to zero at S_a;
      part A (low wages):      maximum up to S_a, falling linearly to zero at S_zero.
    """
    monthly = np.asarray(gross, dtype=float) / p["ssc.pay_months"]
    s_b, s_a = p["ssc.workbonus_b_full_wage_monthly"], p["ssc.workbonus_a_full_wage_monthly"]
    s_zero = p["ssc.workbonus_zero_wage_monthly"]
    frac_a = np.clip((s_zero - monthly) / max(s_zero - s_a, 1e-9), 0.0, 1.0)
    frac_b = np.clip((s_a - monthly) / max(s_a - s_b, 1e-9), 0.0, 1.0)
    on = monthly > 0
    return (12.0 * p["ssc.workbonus_a_max_monthly"] * frac_a * on,
            12.0 * p["ssc.workbonus_b_max_monthly"] * frac_b * on)


def work_bonus(gross, p) -> np.ndarray:
    """Annual reduction of employee contributions for low wages (parts A + B, before the cap
    at the contributions due)."""
    a, b = work_bonus_parts(gross, p)
    return a + b


def special_joint_scale(p):
    """(thresholds, rates) of the special contribution for households with a joint assessment
    (`ssc.special_joint_thresholds`, `ssc.special_joint_rates`), or None when the contribution
    is individual (income 2028 onwards: empty lists) or the parameters are absent."""
    get = getattr(p, "get", None)
    thr = get("ssc.special_joint_thresholds") if get else None
    if not thr:
        return None
    return thr, p["ssc.special_joint_rates"]


def special_ssc(taxable, p, joint: bool = False) -> np.ndarray:
    """Special social security contribution of employees and civil servants: the annual scale
    of the law of 30 March 1994 (art. 108) on the taxable income of the household
    (piecewise-linear, capped; amounts are not indexed). Until income 2027 the household is the
    unit: a person taxed alone pays on his or her own income, spouses and legal cohabitants
    (joint assessment) pay one contribution on their joint income, on a scale of their own -
    `joint=True` with the joint income. From income 2028 the contribution is individual and
    `joint` changes nothing. The monthly NSSO withholding on wages is only an advance."""
    x = np.asarray(taxable, dtype=float)
    scale = special_joint_scale(p) if joint else None
    thr, rates = scale if scale is not None else (p["ssc.special_thresholds"], p["ssc.special_rates"])
    amount = np.zeros_like(x)
    for i, rate in enumerate(rates):
        amount = amount + rate * np.clip(x - thr[i], 0.0, thr[i + 1] - thr[i])
    return np.minimum(amount, p["ssc.special_max"])


def employer_ssc_private(gross, p) -> np.ndarray:
    """Employer contributions of a private-sector (category 1) employer on annual gross wage.

    Basic rate (incl. wage moderation) + other contributions, less
      - the structural reduction R = a x (S0 - S) + c x (S2 - S)   (low and very low wages),
        capped at the basic contributions. The law computes R on the full-time-equivalent
        reference wage S and scales it by the fraction worked: annual earnings below the
        full-time minimum wage (`ssc.fte_floor_wage`) are read as part-time work at that wage
        (fraction = earnings / minimum wage), so they get that fraction of the reduction of a
        full-time minimum wage instead of a reduction that wipes out their contributions;
      - from mid-2025, the exemption of basic contributions on the part of the wage above the
        very-high-wage ceiling (`ssc.employer_cap_exempt_share` of it).
    """
    g = np.asarray(gross, dtype=float)
    basic = p["ssc.employer_rate"] * g
    exempt = p["ssc.employer_cap_exempt_share"] * p["ssc.employer_rate"] * np.maximum(
        g - p["ssc.employer_cap_wage"], 0.0)
    floor = float(p.get("ssc.fte_floor_wage", 0.0) or 0.0) if hasattr(p, "get") else 0.0
    s = np.maximum(g, floor)                         # full-time-equivalent wage
    fraction = np.divide(g, s, out=np.ones_like(g), where=s > 0)
    structural = (p["ssc.structural_reduction_slope"] * np.maximum(p["ssc.structural_reduction_threshold"] - s, 0.0)
                  + p["ssc.structural_reduction_low_slope"]
                  * np.maximum(p["ssc.structural_reduction_low_threshold"] - s, 0.0)) * fraction * (g > 0)
    return p["ssc.employer_other_rate"] * g + np.maximum(basic - exempt - structural, 0.0)


def self_employed_ssc(income, p) -> np.ndarray:
    """Social contributions of a self-employed person in main occupation."""
    g = np.asarray(income, dtype=float)
    c1, c2 = p["ssc.self_employed_ceiling_1"], p["ssc.self_employed_ceiling_2"]
    base = np.maximum(g, p["ssc.self_employed_min_income"])
    ssc = (p["ssc.self_employed_rate"] * np.minimum(base, c1)
           + p["ssc.self_employed_rate_upper"] * np.clip(base - c1, 0.0, c2 - c1))
    return np.where(g > 0, np.minimum(ssc, g), 0.0)


def employee_contributions(gross, p, kind: str = "emp_private") -> tuple:
    """(employer contributions, employee contributions net of the work bonus, work bonus part A,
    part B) on the gross wage of a private (`emp_private`) or public (`emp_public`) employee."""
    g = np.asarray(gross, dtype=float)
    if kind == "emp_private":
        rate_ee = p["ssc.employee_rate"]
        ssc_er = employer_ssc_private(g, p)
    else:
        s = p["ssc.public_statutory_share"]
        rate_ee = s * p["ssc.employee_rate_statutory"] + (1.0 - s) * p["ssc.employee_rate"]
        ssc_er = p["ssc.employer_rate_public"] * g
    bonus_a, bonus_b = work_bonus_parts(g, p)
    # the bonus cannot exceed the contributions due (part B is cut first)
    bonus_a = np.minimum(bonus_a, rate_ee * g)
    bonus_b = np.minimum(bonus_b, rate_ee * g - bonus_a)
    return ssc_er, rate_ee * g - bonus_a - bonus_b, bonus_a, bonus_b


def pension_health_contribution(gross, p) -> np.ndarray:
    """Health-insurance contribution withheld on a pension (it may not push the pension below
    the threshold)."""
    g = np.asarray(gross, dtype=float)
    return np.clip(g - p["ssc.pension_health_threshold"], 0.0, p["ssc.pension_health_rate"] * g)


def tax_base(gross, p, kind: str = "emp_private", company_car=False, taxable_share=1.0) -> dict:
    """One person before the household rules.

    Returns arrays `gross, ssc_employer, ssc_own` (personal contributions without the special
    contribution), `special_ssc, taxable` (net taxable income), `reduction` (federal tax
    reductions: fiscal work bonus, replacement income), and the scalar `allowance` (tax-free
    allowance of the person: the basic allowance, less the neutralised part for replacement
    income) and `kind`.

    kind: one of `KINDS`, or 'none' (no income).
    company_car: adds the taxable benefit in kind of a company car (employees; 0/1, may be an array).
    taxable_share: share of income that is taxable remuneration (population calibration: wages
      in the national accounts include untaxed components); 1 for a payslip.
    """
    g = np.atleast_1d(np.asarray(gross, dtype=float))
    zero = np.zeros_like(g)
    ssc_er, special, reduction = zero, zero, zero
    allowance = float(p["pit.basic_allowance"])

    if kind in ("emp_private", "emp_public"):
        ssc_er, ssc_own, bonus_a, bonus_b = employee_contributions(g, p, kind)
        income = g - ssc_own
        if company_car is not False:
            income = income + np.asarray(company_car, dtype=float) * p["pit.company_car_benefit"]
        if taxable_share != 1.0:
            income = income * taxable_share
        expenses = np.minimum(p["pit.expense_deduction_rate"] * income, p["pit.expense_deduction_max"])
        taxable = income - expenses
        special = special_ssc(taxable, p)
        fiscal = p["pit.workbonus_tax_reduction_rate"]
        reduction = fiscal["a"] * bonus_a + fiscal["b"] * bonus_b
    elif kind == "self_employed":
        ssc_own = self_employed_ssc(g, p)
        taxable = (g - ssc_own) * taxable_share
    elif kind in REPLACEMENT_KINDS:
        ssc_own = pension_health_contribution(g, p) if kind == "pension" else zero
        taxable = (g - ssc_own) * taxable_share
        amount = p["pit.reduction_replacement"][kind]
        if kind == "unemployment":
            amount = amount * p["pit.reduction_unemployment_factor"]
            start, end = p["pit.reduction_unemployment_phaseout"]
            floor = 0.0
        else:
            start, end = p["pit.reduction_phaseout_start"], p["pit.reduction_phaseout_end"]
            floor = p["pit.reduction_floor_share"][kind]
        keep = 1.0 - (1.0 - floor) * np.clip((taxable - start) / max(end - start, 1e-9), 0.0, 1.0)
        if kind == "pension":
            # from income 2026: the remaining third is withdrawn from high pensions
            lo, hi = p["pit.reduction_pension_top_limits"]
            keep = keep * (1.0 - p["pit.reduction_pension_top_phaseout"]
                           * np.clip((taxable - lo) / max(hi - lo, 1e-9), 0.0, 1.0))
        reduction = amount * keep
        # the increases of the tax-free allowance from 2026 are neutralised for pensions and
        # replacement income by equal cuts in their reductions: same as a lower allowance
        allowance = max(allowance - p["pit.allowance_neutralised_replacement"], 0.0)
    elif kind == "none":
        ssc_own, taxable = zero, zero
    else:
        raise ValueError(f"unknown kind '{kind}' (expected one of {KINDS})")
    return {"kind": kind, "gross": g, "ssc_employer": ssc_er, "ssc_own": ssc_own, "special_ssc": special,
            "taxable": np.maximum(taxable, 0.0), "reduction": reduction, "allowance": allowance}


def tax_household(p, adults, children=0, *, region: str | None = None, single_parent=None, joint=1.0,
                  quotient_max=None, levels: bool = True, base_factor=None) -> dict:
    """Personal income tax of a tax unit.

    adults: list of `(kind, gross)` or of dicts shaped like the result of `tax_base` (so callers
      can add income kinds of their own). The first two adults are partners; `joint` (1 or 0,
      may be an array) says whether they are taxed jointly (married or legally cohabiting):
      only then do the marital quotient and the transfer of the unused allowance apply. Further
      adults are taxed on their own.
    children: number of dependent children (int or array). Their supplement to the tax-free
      allowance goes to the partner with the higher taxable income; the part that cannot be
      used gives a refundable credit (bottom rate, capped per child).
    single_parent: weight (0..1, may be an array) of the single-parent supplement of the first
      adult; default 1 for one adult with children, 0 with a partner.
    region: FLA, WAL, BRU for that region's surcharges; None for the national average.
    quotient_max: maximum of the marital quotient (default `pit.marital_quotient_max`: 'pension'
      for a couple of pensioners, 'work' otherwise).
    levels=False returns only `state_tax`, `child_credit`, `taxable_income` and
      `marital_quotient` (for callers that apply the surcharges themselves).
    base_factor: see `state_tax` (the population of tax units only).

    Returns arrays: `taxable_income` (after the quotient), `state_tax, pit_federal,
    pit_regional, pit_local, child_credit, income_tax` (= the three levels less the credit),
    `marital_quotient, allowance_children` (supplements for children and single parents),
    `ssc_employee` (incl. the special contribution), `ssc_employer, special_ssc, gross`, and
    `adults`: per adult `kind, gross, ssc_employee, ssc_employer, taxable_income, allowance,
    reduction, state_tax, income_tax` (before the credit).
    """
    if not adults:
        raise ValueError("a household needs at least one adult")
    persons = [a if isinstance(a, dict) else tax_base(a[1], p, a[0]) for a in adults]
    T = [x["taxable"] for x in persons]
    A = [x["allowance"] for x in persons]
    zero = 0.0
    quotient = unused = supp_extra = zero
    has_children = np.ndim(children) > 0 or children > 0
    supp = child_allowance(children, p) if has_children else 0.0
    couple = len(persons) >= 2

    if couple:
        j = joint if np.ndim(joint) == 0 else np.asarray(joint, dtype=float)
        a, b = persons[0], persons[1]
        kinds = {a["kind"], b["kind"]}
        if quotient_max is None:
            quotient_max = float(p["pit.marital_quotient_max"][
                "pension" if ("pension" in kinds and kinds <= {"pension", "none"}) else "work"])
        quotient = j * marital_quotient(T[0], T[1], p, quotient_max)
        a_high = T[0] >= T[1]
        T = [T[0] + np.where(a_high, -quotient, quotient), T[1] + np.where(a_high, quotient, -quotient)] + T[2:]
        first = T[0] >= T[1]                       # holder of the supplements for children
        if single_parent is not None:
            supp_extra = np.asarray(single_parent, dtype=float) * p["pit.single_parent_allowance"] * (np.asarray(children) > 0)
        own0, own1 = A[0] + first * supp + supp_extra, A[1] + ~first * supp
        # unused allowance of one partner is transferred to the other (joint taxation only)
        A = [own0 + j * np.maximum(own1 - T[1], 0.0), own1 + j * np.maximum(own0 - T[0], 0.0)] + A[2:]
        if has_children:
            alone = np.where(first, own0 - T[0], own1 - T[1])
            unused = np.clip(j * (own0 + own1 - T[0] - T[1]) + (1.0 - j) * alone, 0.0, supp)
    elif has_children:
        w = 1.0 if single_parent is None else np.asarray(single_parent, dtype=float)
        supp_extra = w * p["pit.single_parent_allowance"] * (np.asarray(children) > 0)
        A = [A[0] + supp + supp_extra]
        unused = np.clip(A[0] - T[0], 0.0, supp)

    # special contribution: one amount on the joint income of partners taxed jointly (until
    # income 2027), carried by the partner(s) who are employees
    special = [x["special_ssc"] for x in persons]
    if couple and special_joint_scale(p) is not None:
        paying = [x["kind"] in EMPLOYEE_KINDS for x in persons[:2]]
        if any(paying):
            household = special_ssc(persons[0]["taxable"] + persons[1]["taxable"], p, joint=True) / sum(paying)
            special = [j * household * pays + (1.0 - j) * own for pays, own in zip(paying, special)] + special[2:]

    if not levels:
        total_state = sum(state_tax(t, al, x["reduction"], p, base_factor) for x, t, al in zip(persons, T, A))
        credit = 0.0
        if has_children:
            credit = np.minimum(p["pit.bracket_rates"][0] * unused, p["pit.child_credit_max_per_child"] * np.asarray(children))
        return {"state_tax": total_state, "child_credit": credit, "taxable_income": sum(T), "marital_quotient": quotient}

    f, r, m = level_shares(p, region)
    k = f + r + m
    out_adults, total_state = [], zero
    for x, t, al, sp in zip(persons, T, A, special):
        st = state_tax(t, al, x["reduction"], p, base_factor)
        total_state = total_state + st
        out_adults.append({"kind": x["kind"], "gross": x["gross"], "ssc_employee": x["ssc_own"] + sp,
                           "ssc_employer": x["ssc_employer"], "taxable_income": t, "allowance": al,
                           "reduction": x["reduction"], "state_tax": st, "income_tax": k * st})
    if has_children:
        credit = np.minimum(p["pit.bracket_rates"][0] * unused, p["pit.child_credit_max_per_child"] * np.asarray(children))
    else:
        credit = np.zeros_like(total_state)
    fed, reg, mun = f * total_state, r * total_state, m * total_state
    return {
        "gross": sum(x["gross"] for x in persons),
        "taxable_income": sum(T),
        "state_tax": total_state, "pit_federal": fed, "pit_regional": reg, "pit_local": mun,
        "child_credit": credit, "income_tax": fed + reg + mun - credit,
        "marital_quotient": quotient + np.zeros_like(total_state),
        "allowance_children": supp + supp_extra + np.zeros_like(total_state),
        "ssc_employee": sum(x["ssc_own"] for x in persons) + sum(special),
        "ssc_employer": sum(x["ssc_employer"] for x in persons),
        "special_ssc": sum(special),
        "adults": out_adults,
    }


def tax_on_income(gross, p, kind: str = "emp_private", company_car: bool = False, household: dict | None = None) -> dict:
    """Simplified Belgian payslip for annual gross income `gross` of the given `kind`.

    Returns arrays `gross, employer_cost, ssc_employer, ssc_employee, pit, net` with
    net = gross - ssc_employee - pit and employer_cost = gross + ssc_employer, plus the detail
    `special_ssc` (included in ssc_employee), `pit_federal, pit_regional, pit_local,
    taxable_income, state_tax`, and what the tax was computed with: `allowance` (tax-free
    allowance incl. supplements), `reduction` (federal tax reductions), `child_credit`,
    `marital_quotient`.

    kind:
      emp_private / emp_public  gross wage; employer SSC (face rate less structural reduction
                                and the very-high-wage exemption), employee SSC less work bonus
                                (parts A and B), special SSC on taxable income, flat-rate expenses.
      self_employed             income before own contributions (20.5 % with ceilings).
      pension / unemployment / disability   gross benefit; health contribution on pensions,
                                tax reduction for replacement income.
    company_car=True adds the taxable benefit in kind of a company car (employees only).
    household: None for a single person without dependants at the national-average surcharges
      (the contract other modules rely on), or a dict with any of `children` (number),
      `partner` ((kind, gross) of a partner taxed jointly, or None), `single_parent` (weight of
      the single-parent supplement), `region` (FLA / WAL / BRU). With a partner, `gross`,
      contributions, `pit` and `net` are those of the couple.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind '{kind}' (expected one of {KINDS})")
    base = tax_base(gross, p, kind, company_car)
    h = household or {}
    adults = [base]
    if h.get("partner") is not None:
        adults.append(h["partner"])
    r = tax_household(p, adults, h.get("children", 0), region=h.get("region"),
                      single_parent=h.get("single_parent"), joint=h.get("joint", 1.0))
    g, pit = r["gross"], r["income_tax"]
    return {
        "gross": g, "employer_cost": g + r["ssc_employer"], "ssc_employer": r["ssc_employer"],
        "ssc_employee": r["ssc_employee"], "pit": pit, "net": g - r["ssc_employee"] - pit,
        "special_ssc": r["special_ssc"], "pit_federal": r["pit_federal"] - r["child_credit"],
        "pit_regional": r["pit_regional"], "pit_local": r["pit_local"],
        "taxable_income": r["taxable_income"], "state_tax": r["state_tax"],
        "allowance": r["adults"][0]["allowance"] + np.zeros_like(pit), "reduction": r["adults"][0]["reduction"],
        "child_credit": r["child_credit"], "marital_quotient": r["marital_quotient"],
    }


# ---------------------------------------------------------------------------------------------
# consumption taxes
# ---------------------------------------------------------------------------------------------
def vat_rate(p, key: str) -> float:
    """Average statutory VAT rate of a base (`vat.mix` key): shares x (standard, 12 %, 6 %, 0)."""
    mix = p["vat.mix"][key]
    return sum(share * p[name] for share, name in zip(mix, VAT_RATE_KEYS))


def vat_rate_parts(p, key: str) -> tuple[float, float]:
    """(standard-rate part, reduced-rate part) of the average statutory VAT rate of a base:
    their sum is `vat_rate`. The compliance gap differs between the two (see `vat.compliance_gap`)."""
    mix = p["vat.mix"][key]
    return mix[0] * p["vat.rate_standard"], mix[1] * p["vat.rate_intermediate"] + mix[2] * p["vat.rate_reduced"]


def vat_collected_rate(p, key: str, pb=None) -> float:
    """VAT collected per euro of a base before VAT: the statutory rate (`vat_rate`) less the
    compliance gap. The gap differs between supplies at the standard rate
    (`vat.compliance_gap`) and at the reduced rates (`vat.compliance_gap_reduced`). With `pb`
    (baseline legislation), supplies that a scenario takes out of the standard rate keep the
    gap of standard-rated supplies on what they now owe (the share leaving the standard rate
    goes to the rates whose shares rise): a lower rate on new housing does not make the
    undeclared part of construction disappear. Everything else has the gap of its rate, as
    without `pb`."""
    mix = p["vat.mix"][key]
    rates = [p[name] for name in VAT_RATE_KEYS]
    gap_s, gap_r = p["vat.compliance_gap"], p["vat.compliance_gap_reduced"]
    due = sum(share * rate for share, rate in zip(mix, rates))
    standard = mix[0] * rates[0]                 # due with the gap of standard-rated supplies
    if pb is not None and pb is not p:
        moved = pb["vat.mix"][key][0] - mix[0]
        if moved > 0.0:
            rises = [max(a - b, 0.0) for a, b in zip(mix[1:], pb["vat.mix"][key][1:])]
            total = sum(rises)
            if total > 0.0:
                standard += moved * sum(x * rate for x, rate in zip(rises, rates[1:] + [0.0])) / total
    return due - gap_s * standard - gap_r * (due - standard)


def excise_rates(p, cat: str) -> dict[str, float]:
    """Excise-type lines on a category -> ad-valorem equivalent on the pre-tax price
    (duty per unit / pre-tax price per unit; the duty is the policy lever)."""
    out = {}
    for line, duty, price in EXCISE_COMPONENTS.get(cat, ()):
        rate = p[duty] / p[price] if price else p[duty]
        if duty == "excise.energy_duty":
            rate *= p["excise.energy_shift_factor"]     # legislated electricity -> gas shift
        out[line] = rate
    return out


def consumption_tax_rates(p) -> dict:
    """VAT + excise as a fraction of the pre-tax price, per consumption category.

    VAT is charged on the price including excise: rate = (1 + excise) * (1 + vat) - 1.
    """
    return {c: (1.0 + sum(excise_rates(p, c).values())) * (1.0 + vat_rate(p, c)) - 1.0 for c in CONS_CATS}


# ---------------------------------------------------------------------------------------------
# corporate and capital taxes
# ---------------------------------------------------------------------------------------------
def cit_effective_rate(p) -> float:
    """Average corporate income tax rate on taxable profits before the deduction levers.

    Blend of the standard and SME rates, applied to the share of profits left after the
    innovation-income and investment deductions, plus the minimum-tax top-up.
    """
    sme = p["cit.sme_base_share"]
    statutory = sme * p["cit.sme_rate"] + (1.0 - sme) * p["cit.rate"]
    deducted = (p["cit.innovation_income_share"] * p["cit.innovation_deduction_rate"]
                + p["cit.investment_deduction_base_share"])
    topup = p["cit.minimum_tax_base_share"] * max(p["cit.minimum_tax_rate"] - p["cit.minimum_tax_low_etr"], 0.0)
    return float(statutory * max(1.0 - deducted, 0.0) + topup)


def capital_income_tax_rate(p) -> float:
    """Average withholding-tax rate on household interest and dividends."""
    exempt, reduced = p["cap.income_share_exempt"], p["cap.income_share_reduced"]
    return float(max(1.0 - exempt - reduced, 0.0) * p["cap.withholding_rate"]
                 + reduced * p["cap.withholding_reduced_rate"])


def _base_shares(p) -> np.ndarray:
    w = by_region(p["prop.region_base_share"])
    return w / w.sum()


def property_transaction_tax_by_region(p) -> np.ndarray:
    """Registration duties per euro of national real-estate sales, by region: the region's share
    of the base x its average rate (standard and sole-own-home). Sums to the national rate."""
    h = p["prop.sole_home_share"]
    std, sole = by_region(p["prop.registration_rate"]), by_region(p["prop.registration_rate_sole_home"])
    return _base_shares(p) * ((1.0 - h) * std + h * sole)


def inheritance_tax_by_region(p) -> np.ndarray:
    """Inheritance and gift tax per euro of the national flow of inheritances and gifts, by region."""
    return _base_shares(p) * by_region(p["prop.inheritance_rate"])


def property_tax_by_region(p) -> np.ndarray:
    """Recurrent property tax per euro of national indexed cadastral income, by region: regional
    rate x (1 + provincial and municipal surcharges)."""
    return _base_shares(p) * by_region(p["prop.property_tax_rate"]) * (1.0 + by_region(p["prop.property_tax_local_multiplier"]))


def property_transaction_tax_rate(p) -> float:
    """Average registration-duty rate on the value of real-estate sales (regional rates,
    standard and sole-own-home, weighted by the regions' share of the base)."""
    return float(property_transaction_tax_by_region(p).sum())


def inheritance_tax_rate(p) -> float:
    """Average effective inheritance and gift tax rate on the flow of inheritances and gifts."""
    return float(inheritance_tax_by_region(p).sum())


def property_tax_rate(p) -> float:
    """Average recurrent property tax (precompte immobilier / onroerende voorheffing) as a share
    of indexed cadastral income: regional rate x (1 + provincial and municipal surcharges)."""
    return float(property_tax_by_region(p).sum())
